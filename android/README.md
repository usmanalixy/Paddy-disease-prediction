# Paddy Disease Prediction for Android

Take or choose a sharp close-up of one rice leaf. After photo selection, tell the app what the photo shows. Field views, grain heads, and other views get a request for a leaf close-up. For a leaf close-up, the app reports **Looks normal**, **Possible disease**, or **Uncertain**. Inference runs on-device with Google LiteRT. The app declares no internet, camera, or media-storage permission.

The model uses a separate normal-versus-symptomatic output alongside the original ten classes. Thresholds were selected using only validation photos. On the held-out test split, 7 of 175 normal photos were flagged as possibly diseased, compared with 16 using the base classifier's top prediction. Of 859 symptomatic photos, 796 were flagged, 53 were uncertain, and 10 were marked normal. Accuracy on separately collected farm photos and grain heads is unknown. [Health-screen evaluation](../reports/kaggle-health-screen-evaluation.json)

## Build and install

Open this `android/` directory in Android Studio, or use JDK 17+, Android SDK Platform 35 and Build Tools 35.0.0:

```sh
./gradlew :app:assembleDebug :app:testDebugUnitTest
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

The debug APK runs on Android 9 (API 28) and later. Android Studio can also install it directly to a connected phone or emulator. On Windows, use `gradlew.bat`.

## Replace the bundled model

1. Run [model.ipynb](../model.ipynb) on labeled data that includes the `normal` class. Keep the run directory containing `model.keras`, `labels.json`, `splits.json`, and the images referenced in the split manifest.
2. In the same Python environment, run these commands from the repository root:

   ```sh
   python training/fit_health_screen.py runs/<run-directory>
   python android/export_model.py runs/<run-directory>
   ```

3. Rebuild the APK. The exporter checks that LiteRT and Keras produce matching outputs and updates the model, class order, and decision thresholds together.

The app decodes oriented RGB images, reduces very large photos during decode, then applies the notebook’s bilinear resize and black padding. The float model input uses pixel values from 0 to 255. The model applies its own MobileNetV2 scaling. [Bundled model metadata](app/src/main/assets/paddy_model.json) includes class order, test metrics, and artifact hashes.

The Java resize code was checked separately against TensorFlow in the pilot. Both resize implementations produced the same tensors for the same decoded RGB pixels.
