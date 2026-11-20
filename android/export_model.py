"""Export a notebook run to the Android app's LiteRT assets."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def export(run_dir, assets_dir):
    checkpoint = run_dir / "model.keras"
    labels = read_json(run_dir / "labels.json")
    height, width = labels["image_size"]
    classes = labels["class_names"]
    if not classes or len(set(classes)) != len(classes) or labels["pixel_range"] != [0, 255]:
        raise ValueError("Expected unique classes and float RGB values in the range 0–255.")

    model = tf.keras.models.load_model(checkpoint, compile=False)
    if model.input_shape[1:] != (height, width, 3) or model.output_shape[-1] != len(classes):
        raise ValueError("The checkpoint dimensions do not agree with labels.json.")

    @tf.function(input_signature=[tf.TensorSpec((1, height, width, 3), tf.float32, name="rgb")])
    def predict(rgb):
        return model(rgb, training=False)

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [predict.get_concrete_function()], model
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converted = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=converted, num_threads=1)
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    if (
        tuple(input_info["shape"]) != (1, height, width, 3)
        or tuple(output_info["shape"]) != (1, len(classes))
        or input_info["dtype"] != np.float32
        or output_info["dtype"] != np.float32
    ):
        raise ValueError("The converted model has an unexpected input or output signature.")

    random = np.random.default_rng(42)
    samples = [
        np.zeros((1, height, width, 3), dtype=np.float32),
        random.uniform(0, 255, (1, height, width, 3)).astype(np.float32),
    ]
    for sample in samples:
        interpreter.set_tensor(input_info["index"], sample)
        interpreter.invoke()
        actual = interpreter.get_tensor(output_info["index"])
        np.testing.assert_allclose(actual, predict(sample).numpy(), rtol=1e-3, atol=1e-5)

    with checkpoint.open("rb") as stream:
        checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = {
        "class_names": classes,
        "image_size": [height, width],
        "pixel_range": labels["pixel_range"],
        "checkpoint_sha256": checkpoint_sha256,
        "model_sha256": hashlib.sha256(converted).hexdigest(),
        "source_notebook": "model.ipynb",
        "training_phase": labels.get("selected_phase"),
    }
    evaluation = run_dir / "evaluation.json"
    if evaluation.is_file():
        report = read_json(evaluation)["test_report"]
        metadata["evaluation"] = {
            "source": labels.get("test_source"),
            "images": int(report["macro avg"]["support"]),
            "accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
        }

    assets_dir.mkdir(parents=True, exist_ok=True)
    (assets_dir / "paddy_model.tflite").write_bytes(converted)
    (assets_dir / "paddy_model.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Exported {len(classes)} classes to {assets_dir} ({len(converted):,} model bytes).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Notebook run containing model.keras and labels.json")
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "app/src/main/assets",
    )
    args = parser.parse_args()
    export(args.run_dir, args.assets_dir)
