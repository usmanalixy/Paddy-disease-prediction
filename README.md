# Paddy Disease Prediction

A TensorFlow/Keras image classifier for ten paddy leaf categories, including healthy leaves (`normal`). The project is a notebook. It covers image loading, class distribution, training, sample predictions, a classification report, and a confusion matrix.

Start with [`standard_model_v3/model.ipynb`](../../blob/standard_model_v3/model.ipynb). The default `main` branch contains an earlier Colab experiment with hard-coded Google Drive paths, directory deletion commands, and evaluation cells that reference variables before they are assigned.

## Run the notebook

1. Open the `standard_model_v3` notebook in Kaggle and attach [New Paddy Doctor](https://www.kaggle.com/datasets/dasa7753912/new-paddy-doctor-paddy-disease-classification) and the Paddy Doctor input used for the extra images. The notebook identifies that second input by its mount path. Verify the dataset owner and version before reproducing the run.
2. Confirm both input paths point at a directory containing `paddy-disease-classification/train_images/<class>/`. The expected Kaggle mounts are `/kaggle/input/new-paddy-doctor-paddy-disease-classification` and `/kaggle/input/paddy-doctor`. For local Jupyter, change both input paths and the writable merge directory. Run each merge in a clean directory because an existing filename can cause a second copy with a prefix.
3. Use a GPU for the configured 50-epoch run. For local Jupyter, the notebook imports TensorFlow, NumPy, pandas, Matplotlib, seaborn, and scikit-learn. Install them with `python -m pip install jupyterlab tensorflow numpy pandas matplotlib seaborn scikit-learn`.
4. For a valid evaluation, assign images to fixed, disjoint splits before training. Keep duplicate and near-duplicate images in the same split, shuffle only the training split, and reserve the test split for the final report. See the audit below.

## Training data and model outputs

V3 copies lowercase JPG and PNG images from all primary training classes and adds secondary images for `dead_heart`, `downy_mildew`, and `hispa`. Keras assigns integer labels in alphabetical folder order. The saved notebook reports 19,131 primary images and 3,656 additions, yielding 22,787 images. Counts below come from the saved run.

| Label index | Class | Primary | Added | Total |
| ---: | --- | ---: | ---: | ---: |
| 0 | bacterial_leaf_blight | 1,916 | 0 | 1,916 |
| 1 | bacterial_leaf_streak | 1,900 | 0 | 1,900 |
| 2 | bacterial_panicle_blight | 1,863 | 0 | 1,863 |
| 3 | blast | 1,918 | 0 | 1,918 |
| 4 | brown_spot | 1,930 | 0 | 1,930 |
| 5 | dead_heart | 1,910 | 1,442 | 3,352 |
| 6 | downy_mildew | 1,944 | 620 | 2,564 |
| 7 | hispa | 1,942 | 1,594 | 3,536 |
| 8 | normal | 1,872 | 0 | 1,872 |
| 9 | tungro | 1,936 | 0 | 1,936 |

V3 takes RGB images at 224 × 224 pixels with values in the 0–255 range. The model rescales internally and applies random flips, rotation, and zoom during training. Its convolutional network uses Adam and sparse categorical cross-entropy with batches of 32. Each prediction returns ten softmax scores. The notebook's `predict(model, image_array)` helper returns the highest-scoring class and its score as a percentage.

The V3 notebook takes approximately 80% of the merged labeled batches for training, 15% for validation, and the remainder for testing. Its saved run shows 570, 106, and 37 batches respectively. It never reads the separate `test_images` directory. The repository contains no raw datasets, trained checkpoints, or standalone inference service.

## Review and test results

Each branch's Python cells passed syntax checks after excluding notebook shell commands. With TensorFlow on CPU and 40 generated RGB images arranged across ten class folders, each model loaded its expected input shape, completed one training step with finite metrics, produced normalized ten-class scores, and returned a label and score for one image.

| Branch | RGB input | Saved accuracy | Saved report support |
| --- | --- | ---: | ---: |
| `main` | 120 × 120 | 81% | 1,952 |
| `standard-model` | 224 × 224 | 82% | 1,952 |
| `standard_model_v2` | 224 × 224 | 92% | 992 |
| `standard_model_v3` | 224 × 224 | 95% | 1,184 |

These scores are historical notebook outputs. The 50-epoch training runs, real-image predictions, and independent accuracy were not reproduced. Public Kaggle metadata confirmed the primary dataset listing and some archived image paths. Full image contents, labels, duplicates, and the exact secondary mirror were not independently verified.

### Evaluation issues

- Every branch independently iterates over a reshuffling dataset for its `take` and `skip` splits. A test of the shared function with 100 unique IDs and its default fractions found 10 train/validation overlaps, seven train/test overlaps, and two validation/test overlaps. Published notebook scores cannot establish performance on an independent holdout.
- The earlier notebooks compare filenames in the external `train_images` and `test_images` directories and report at least one common name. The check does not establish whether those files contain the same image. Their final reports use the internal split of `train_images`. The `standard-model` branch also uses its internal test subset in validation during training.
- The split helper reads the global dataset length and ignores the `test_split` argument. The cache, shuffle, and prefetch calls discard the returned datasets. V3's merge handles name collisions but can retain identical image content under different names.
- Several example cells assume a batch contains at least 12 images. Pass the full class index list to scikit-learn when evaluating small subsets so classes without examples still appear in the report and confusion matrix.

## Attribution

The notebooks and saved results come from [Nuwanthapasindu/paddy-disease-prediction-ai-model](https://github.com/Nuwanthapasindu/paddy-disease-prediction-ai-model). The source branch snapshots contain no license file. Consult the dataset pages for their usage terms.
