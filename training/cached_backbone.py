"""Train MobileNetV2 from cached activations on a CPU."""

import numpy as np
import tensorflow as tf


def fit_cached(
    model, backbone, train_ds, validation_ds, *, batch_size, head_epochs,
    fine_tune_epochs, fine_tune_layers, class_weights, seed, model_path,
):
    """Fit the same deployable model, with one cached flip per training photo."""
    layers = tf.keras.layers
    cut = backbone.get_layer("block_12_add").output
    prefix = tf.keras.Model(backbone.input, cut)
    tail = tf.keras.Model(cut, backbone.output)
    if fine_tune_epochs and not 1 <= fine_tune_layers <= len(tail.layers) - 1:
        raise ValueError("Cached mode can fine-tune only layers after MobileNetV2 block_12_add.")

    normalize = model.get_layer("mobilenet_rgb")
    pooling = model.get_layer("mobile_pool")
    dropout = model.get_layer("mobile_dropout")
    scores = model.get_layer("class_scores")
    tail.trainable = False

    @tf.function(reduce_retracing=True)
    def frozen_features(images):
        features = prefix(normalize(images), training=False)
        # Store exactly what the fine-tuning phase will consume.
        packed_features = tf.cast(features, tf.float16)
        features = tf.cast(packed_features, tf.float32)
        return packed_features, pooling(tail(features, training=False))

    def cache(dataset, horizontal_flip):
        activations, vectors, labels = [], [], []
        for batch, (images, targets) in enumerate(dataset):
            if horizontal_flip:
                images = tf.concat([images, tf.image.flip_left_right(images)], axis=0)
                targets = tf.concat([targets, targets], axis=0)
            packed_features, pooled_vectors = frozen_features(images)
            activations.append(packed_features.numpy())
            vectors.append(pooled_vectors.numpy())
            labels.append(targets.numpy())
            if (batch + 1) % 40 == 0:
                print(f"Cached {batch + 1} batches", flush=True)
        return np.concatenate(activations), np.concatenate(vectors), np.concatenate(labels)

    train_features, train_vectors, train_labels = cache(train_ds, horizontal_flip=True)
    validation_features, validation_vectors, validation_labels = cache(
        validation_ds, horizontal_flip=False
    )

    def batches(values, labels, shuffle=False):
        dataset = tf.data.Dataset.from_tensor_slices((values, labels))
        if shuffle:
            dataset = dataset.shuffle(min(2048, len(labels)), seed=seed)
        return dataset.batch(batch_size).prefetch(1)

    def fit(network, train, validation, epochs, learning_rate):
        network.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        return network.fit(
            train, validation_data=validation, epochs=epochs,
            class_weight=class_weights, verbose=2,
            callbacks=[
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.4),
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=3, restore_best_weights=True
                ),
            ],
        ).history

    vector = tf.keras.Input(shape=train_vectors.shape[1:])
    head = tf.keras.Model(vector, scores(dropout(vector)))
    histories = {"head": fit(
        head, batches(train_vectors, train_labels, shuffle=True),
        batches(validation_vectors, validation_labels), head_epochs, 1e-3,
    )}
    best_validation_loss = min(histories["head"]["val_loss"])
    if not np.isfinite(best_validation_loss):
        raise ValueError("Training did not produce a finite validation loss.")
    model.save(model_path)
    selected_phase = "head"

    if fine_tune_epochs:
        tail.trainable = True
        for layer in tail.layers:
            layer.trainable = False
        for layer in tail.layers[-fine_tune_layers:]:
            layer.trainable = not isinstance(layer, layers.BatchNormalization)
        features = tf.keras.Input(shape=train_features.shape[1:], dtype=tf.float16)
        fine_tuner = tf.keras.Model(features, head(pooling(tail(features, training=False))))
        histories["fine_tune"] = fit(
            fine_tuner, batches(train_features, train_labels, shuffle=True),
            batches(validation_features, validation_labels), fine_tune_epochs, 1e-4,
        )
        fine_tune_loss = min(histories["fine_tune"]["val_loss"])
        if np.isfinite(fine_tune_loss) and fine_tune_loss < best_validation_loss:
            model.save(model_path)
            best_validation_loss = fine_tune_loss
            selected_phase = "fine_tune"

    return histories, best_validation_loss, selected_phase
