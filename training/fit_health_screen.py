"""Fit a normal-versus-symptomatic head to a saved paddy classifier."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


MIN_DISEASE_CLASS_SCORE = 0.60


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def decision(scores, symptoms, normal_index, policy):
    """Use -1 for unclear photos, the normal index for normal, or a disease index."""
    predicted = scores.argmax(axis=1)
    possible_disease = (
        (predicted != normal_index)
        & (scores.max(axis=1) >= policy["min_disease_class_score"])
        & (symptoms > policy["disease_above"])
    )
    return np.where(symptoms < policy["normal_below"], normal_index,
                    np.where(possible_disease, predicted, -1))


def summarize(labels, guesses, normal_index):
    normal = labels == normal_index
    symptomatic = ~normal
    uncertain = guesses == -1
    flags = ~uncertain & (guesses != normal_index)
    looks_normal = guesses == normal_index
    return {
        "normal_images": int(normal.sum()),
        "symptomatic_images": int(symptomatic.sum()),
        "normal_correct": int((normal & looks_normal).sum()),
        "normal_false_alarms": int((normal & flags).sum()),
        "normal_uncertain": int((normal & uncertain).sum()),
        "symptomatic_flagged": int((symptomatic & flags).sum()),
        "symptomatic_marked_normal": int((symptomatic & looks_normal).sum()),
        "symptomatic_uncertain": int((symptomatic & uncertain).sum()),
        "symptomatic_correct_label": int((symptomatic & (labels == guesses)).sum()),
    }


def select_policy(symptoms, labels, normal_index):
    # The thresholds are selected on validation only. Ties at a boundary abstain.
    normal = np.sort(symptoms[labels == normal_index])
    diseased = np.sort(symptoms[labels != normal_index])
    if not len(normal) or not len(diseased):
        raise ValueError("Validation needs both normal and symptomatic photos.")
    return {
        "normal_below": float(min(0.5, diseased[int(0.01 * len(diseased))])),
        "disease_above": float(max(0.5, normal[-1 - int(0.02 * len(normal))])),
        "min_disease_class_score": MIN_DISEASE_CLASS_SCORE,
    }


def read_dataset(rows, image_size):
    paths = [item["path"] for item in rows]
    labels = [item["label"] for item in rows]

    def read(path, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape((None, None, 3))
        return tf.image.resize_with_pad(image, *image_size), label

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    return dataset.map(read, num_parallel_calls=2).batch(64).prefetch(2)


def fit_health_screen(run_dir):
    metadata = load_json(run_dir / "labels.json")
    splits = load_json(run_dir / "splits.json")["splits"]
    paths = [item["path"] for rows in splits.values() for item in rows]
    if len(set(paths)) != len(paths):
        raise ValueError("The supplied splits contain repeated image paths.")
    normal_index = metadata["class_names"].index("normal")
    base_path = run_dir / "model.keras"
    base = tf.keras.models.load_model(base_path, compile=False)
    pooled = base.get_layer("mobile_pool").output
    extractor = tf.keras.Model(base.input, [pooled, base.output])

    @tf.function(reduce_retracing=True)
    def predict(images):
        return extractor(images, training=False)

    def features(split, training=False):
        vectors, scores, labels = [], [], []
        for images, targets in read_dataset(splits[split], metadata["image_size"]):
            if training:
                images = tf.concat([images, tf.image.flip_left_right(images)], axis=0)
                targets = tf.concat([targets, targets], axis=0)
            vectors_batch, scores_batch = predict(images)
            vectors.append(vectors_batch.numpy())
            scores.append(scores_batch.numpy())
            labels.append(targets.numpy())
        print(f"Extracted {split} features", flush=True)
        return np.concatenate(vectors), np.concatenate(scores), np.concatenate(labels)

    train, _, train_labels = features("train", training=True)
    validation, validation_scores, validation_labels = features("validation")
    scaler = StandardScaler().fit(train)
    train = scaler.transform(train)
    validation = scaler.transform(validation)
    best = None
    for strength in (0.003, 0.03, 0.3, 3.0):
        classifier = LogisticRegression(C=strength, class_weight="balanced", solver="liblinear")
        classifier.fit(train, train_labels != normal_index)
        values = classifier.predict_proba(validation)[:, 1].astype(np.float32)
        policy = select_policy(values, validation_labels, normal_index)
        guesses = decision(validation_scores, values, normal_index, policy)
        stats = summarize(validation_labels, guesses, normal_index)
        rank = (
            stats["normal_correct"] / stats["normal_images"]
            + stats["symptomatic_flagged"] / stats["symptomatic_images"]
        )
        if best is None or rank > best[0]:
            best = rank, strength, classifier, policy, stats

    _, strength, classifier, policy, validation_stats = best
    print(f"Validation policy: {policy}; {validation_stats}", flush=True)
    kernel = classifier.coef_[0] / scaler.scale_
    bias = classifier.intercept_[0] - np.dot(kernel, scaler.mean_)
    health = tf.keras.layers.Dense(1, activation="sigmoid", name="symptom_score")
    output = tf.keras.layers.Concatenate(name="paddy_outputs")([base.output, health(pooled)])
    screen_model = tf.keras.Model(base.input, output, name="paddy_disease_with_health_screen")
    health.set_weights([kernel[:, None].astype(np.float32), np.array([bias], dtype=np.float32)])

    # Freeze the policy before looking at the held-out images.
    test, test_scores, test_labels = features("test")
    symptom_scores = health(test).numpy()[:, 0]
    test_guesses = decision(test_scores, symptom_scores, normal_index, policy)
    baseline = summarize(test_labels, test_scores.argmax(axis=1), normal_index)
    test_stats = summarize(test_labels, test_guesses, normal_index)
    with base_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "base_checkpoint_sha256": digest,
        "normal_class": "normal",
        "fit": {"training_images": len(splits["train"]), "regularization_c": strength},
        "policy": policy,
        "validation": validation_stats,
        "test_source": metadata["test_source"],
        "test_baseline": baseline,
        "test": test_stats,
    }
    (run_dir / "health_screen.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    screen_model.save(run_dir / "health_model.keras")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Existing MobileNetV2 run with its fixed splits")
    fit_health_screen(parser.parse_args().run_dir)
