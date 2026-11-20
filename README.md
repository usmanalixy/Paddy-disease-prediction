# Paddy Disease Prediction

Train and evaluate a ten-class paddy leaf classifier in [model.ipynb](model.ipynb). The default workflow uses a pretrained MobileNetV2 backbone, fixed data splits, and a reproducible evaluation report.

## Android app

The [Android app](android/README.md) runs the trained MobileNetV2 model on-device. Confirm that the photo shows one rice leaf close up to get a **Looks normal**, **Uncertain**, or **Possible disease** result. For field views and grain heads, the app asks for a leaf close-up. It supports Android 9 and later.

[Download the Android APK](https://raw.githubusercontent.com/usmanalixy/Paddy-disease-prediction/android-download/Paddy-Disease-Prediction-debug.apk).

## Run the notebook

1. Install Python 3.12 and the notebook dependencies in a virtual environment.

   ```sh
   python -m pip install jupyterlab "tensorflow==2.21.*" "scikit-learn==1.8.*" numpy pillow matplotlib
   jupyter lab
   ```

2. Put labeled JPEG or PNG images in one folder per class under `data/train_images`, or change `DATA_DIR` in the first code cell. Folder names provide class labels in alphabetical order.

   ```text
   data/train_images/
     bacterial_leaf_blight/
       leaf-001.jpg
     blast/
       leaf-002.jpg
     normal/
       leaf-003.jpg
   ```

3. Run all cells. The ImageNet weights download on the first run. Set `BACKBONE_WEIGHTS = None` to train from scratch without the download, or `FINE_TUNE_EPOCHS = 0` to skip fine-tuning. The default run allows up to 12 classifier-head epochs and eight fine-tuning epochs. Early stopping uses validation loss.

### Use the original Kaggle data

Download or attach the [New Paddy Doctor archive](https://www.kaggle.com/datasets/dasa7753912/new-paddy-doctor-paddy-disease-classification) used by the original project. Set these in the notebook, using the actual location of the extracted `paddy-disease-classification` folder:

```python
DATA_DIR = Path("/path/to/paddy-disease-classification/train_images")
KAGGLE_TRAIN_CSV = DATA_DIR.parent / "train.csv"
```

The CSV selects the 10,407 original labeled photos across ten classes. The mirror packages additional augmented files in `train_images`. The CSV filter excludes those files before splitting, so training can generate its own augmentations after the holdout is fixed. Leave `EXTRA_DATA_DIRS` empty for this run. [Original Paddy Doctor dataset and research](https://paddydoc.github.io/)

When running from the full repository on a CPU, set `CACHE_FROZEN_BACKBONE = True` and `BATCH_SIZE = 64` to reproduce the faster training mode. This mode caches frozen MobileNetV2 activations for each training image and one horizontal flip, then fine-tunes the top layers. The validation and test images are used without augmentation.

## Training and test data

- The notebook hashes decoded RGB pixels before splitting. It removes exact duplicates with the same label and stops if identical images have conflicting labels. Extra sources are opt-in via `EXTRA_DATA_DIRS`.
- The default seeded split targets roughly 80% training, 10% validation, and 10% test with every class represented. Use at least three distinct images per class for the internal split; substantially more are needed to measure accuracy. Only the training dataset reshuffles and receives model-side augmentation.
- Set `HOLDOUT_DIR` to use a separately collected test set with the same class folders. Development data then supplies training and validation. The notebook rejects overlap with the holdout based on exact pixels or declared group IDs.
- Use `GROUP_CSV` for photos from the same plant, capture session, or known near-duplicate group. The CSV columns are `path,group`. Paths are relative to the CSV or absolute and must identify a file in the image inventory. Known groups stay in one split. Exact matching cannot detect crops or differently encoded images whose decoded pixels differ.
- Leave the primary archive’s labeled `test_images` folder out of the evaluation. Its 10,407 entries match training entries on filename, class, CRC32, and byte size. All ten sampled pairs matched byte for byte. A matching secondary Paddy Doctor mirror also shares all 3,656 entries the older notebook selected for its three augmented classes by those metadata checks. Three sampled mirror/primary pairs matched byte for byte. [Matching secondary mirror](https://www.kaggle.com/datasets/imbikramsaha/paddy-doctor)

## Model and outputs

The notebook resizes RGB images to 160 × 160 with aspect-ratio-preserving padding. The model scales pixel values from 0–255 to −1–1 and uses global average pooling and a softmax head. Training weights are derived from class counts. The test set is evaluated only after the checkpoint is selected using validation loss.

For ten classes, the model has 2.27 million parameters, about 95% fewer than the previous v3 model’s 42.76 million. In a local single-thread TensorFlow CPU benchmark, the median warm single-image forward pass took 8.1 ms with the revised model and 27.5 ms with v3. This excludes file loading and resizing, and hardware and batch size affect latency.

Each completed run writes these files in its own directory under `runs/`:

| File | Contents |
| --- | --- |
| `model.keras` | Selected model checkpoint |
| `labels.json` | Class order, image configuration, seed, software versions, and selected training phase |
| `splits.json` | File paths and class IDs for each split, with deduplication and class counts |
| `evaluation.json` | Test accuracy, macro F1, per-class metrics, confusion matrix, and training history |
| `confusion_matrix.png` | Labeled confusion matrix for the test set |

The final cell reloads the exported model and predicts one image. It returns the top class and a softmax score per class. Treat those scores as uncalibrated. Run artifacts include dataset file paths, so review the split manifest before sharing it.

## Verification

The base checkpoint was trained on the original Kaggle photos. After removing 74 exact duplicates from 10,407 originals, the run used 8,265 training photos, 1,034 validation photos, and 1,034 test photos. Its ten-class top prediction got **970 of 1,034 correct (93.8%)** and **0.935 macro F1** on this internal test split. Photos from the same plant or capture session may still be related because the archive does not provide those group IDs. Accuracy on separately collected farm photos is unknown. [Per-class metrics and confusion matrix](reports/kaggle-originals-evaluation.json)

The Android health screen fits a separate normal-versus-symptomatic output on the same training partition. On the held-out test split, it reduced disease alerts on normal photos from **16 of 175 to 7 of 175** compared with the base model's top prediction. It returned **Uncertain** on 8 normal and 53 symptomatic photos, and marked 10 symptomatic photos as **Looks normal**. [Health-screen evaluation](reports/kaggle-health-screen-evaluation.json)

The old notebooks reshuffled their dataset independently for training, validation, and test iteration. Tests reproduced overlap between the splits. The historical v3 score of 95% cannot establish independent real-world accuracy.

The revised notebook was tested on 100 real training-archive JPEGs across all ten classes, using 80 for training, ten for validation, and ten for internal testing. This small run checks the workflow. The duplicate rejection checks also passed, including rejection of sampled training copies as an external holdout.

The earlier CPU pilot used 986 archive photos across ten classes and scored 62 of 99 correct (62.6% accuracy and 0.613 macro F1) on its own internal test split. The pilot and full-data runs used different test splits.
