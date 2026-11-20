# Paddy Disease Prediction for Android

Take a photo of paddy symptoms or choose an existing image. The app displays the photo and the three highest model scores. Inference runs on-device with Google LiteRT. The app declares no internet, camera, or media-storage permission. The system camera and photo picker grant access to the single photo the user chooses.

The bundled MobileNetV2 model was retrained on the [original Kaggle dataset](../README.md#use-the-original-kaggle-data). It achieved **93.8% accuracy** and **0.935 macro F1** on a held-out 1,034-photo split. Its field accuracy is unknown. The scores are uncalibrated, so confirm possible disease with a local crop adviser. [Full evaluation](../reports/kaggle-originals-evaluation.json)

## Build and install

Open this `android/` directory in Android Studio, or use JDK 17+, Android SDK Platform 35 and Build Tools 35.0.0:

```sh
./gradlew :app:assembleDebug :app:testDebugUnitTest
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

The debug APK runs on Android 9 (API 28) and later. Android Studio can also install it directly to a connected phone or emulator. On Windows, use `gradlew.bat`.

## Replace the bundled model

1. Run [model.ipynb](../model.ipynb) on labeled data. Keep the run directory containing `model.keras`, `labels.json`, and optionally `evaluation.json`.
2. In the same Python environment, run this command from the repository root:

   ```sh
   python android/export_model.py runs/<run-directory>
   ```

3. Rebuild the APK. The exporter checks that LiteRT and Keras produce matching scores and updates the model and its class order together.

The app decodes oriented RGB images, reduces very large photos during decode, then applies the notebook’s bilinear resize and black padding. The float model input uses pixel values from 0 to 255. The model applies its own MobileNetV2 scaling. [Bundled model metadata](app/src/main/assets/paddy_model.json) includes class order, test metrics, and artifact hashes.

The bundled LiteRT model matched the Keras checkpoint's top prediction on all 1,034 held-out photos. The Java resize code was checked separately against TensorFlow in the pilot. Both resize implementations produced the same tensors for the same decoded RGB pixels.
