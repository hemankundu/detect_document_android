package org.pytorch.demo.vision;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.ImageFormat;
import android.os.Bundle;
import android.widget.Button;
import android.widget.Toast;

import com.google.common.util.concurrent.ListenableFuture;

import org.pytorch.demo.BaseModuleActivity;
import org.pytorch.demo.StatusBarUtils;

import java.io.File;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.UiThread;
import androidx.annotation.WorkerThread;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.lifecycle.LifecycleOwner;

public abstract class AbstractCameraXActivity<R> extends BaseModuleActivity {
  private static final int REQUEST_CODE_CAMERA_PERMISSION = 200;
  private static final String[] PERMISSIONS = {Manifest.permission.CAMERA};

  protected abstract int getContentViewLayoutId();

  protected abstract PreviewView getCameraPreviewView();

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    StatusBarUtils.setStatusBarOverlay(getWindow(), true);
    setContentView(getContentViewLayoutId());

    startBackgroundThread();

    if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
        != PackageManager.PERMISSION_GRANTED) {
      ActivityCompat.requestPermissions(
          this,
          PERMISSIONS,
          REQUEST_CODE_CAMERA_PERMISSION);
    } else {
      try {
        setupCameraX();
      } catch (ExecutionException | InterruptedException e) {
        e.printStackTrace();
      }
    }
  }

  @Override
  public void onRequestPermissionsResult(
          int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
    if (requestCode == REQUEST_CODE_CAMERA_PERMISSION) {
      if (grantResults[0] == PackageManager.PERMISSION_DENIED) {
        Toast.makeText(
            this,
            "You can't use this without granting CAMERA permission",
            Toast.LENGTH_LONG)
            .show();
        finish();
      } else {
        try {
          setupCameraX();
        } catch (ExecutionException | InterruptedException e) {
          e.printStackTrace();
        }
      }
    }
  }

  private void setupCameraX() throws ExecutionException, InterruptedException {

    ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);
    CameraSelector cameraSelector = new CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build();
      ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
      Preview preview = new Preview.Builder()
              .build();

      final PreviewView previewView = getCameraPreviewView();
      preview.setSurfaceProvider(previewView.createSurfaceProvider());


    final ImageCapture imageCapture =
            new ImageCapture.Builder()
            .setBufferFormat(ImageFormat.YUV_420_888)
            .build();
    cameraProvider.bindToLifecycle((LifecycleOwner)this, cameraSelector, imageCapture, preview);

    Button capture_button = findViewById(org.pytorch.demo.R.id.capture_button);

    capture_button.setOnClickListener(v -> {
      imageCapture.takePicture(Executors.newSingleThreadExecutor(),
        new ImageCapture.OnImageCapturedCallback() {
          @Override
          public void onCaptureSuccess(@NonNull ImageProxy image) {
            final R result = analyzeImage(image);
            if (result != null) {
              runOnUiThread(() -> applyToUiAnalyzeImageResult(result));
            }
            image.close();
          }
        }
      );

      ImageCapture.OutputFileOptions outputFileOptions =
              new ImageCapture.OutputFileOptions.Builder(new File("Image.jpeg")).build();

      imageCapture.takePicture(outputFileOptions, Executors.newSingleThreadExecutor(),
        new ImageCapture.OnImageSavedCallback() {
          @Override
          public void onImageSaved(@NonNull ImageCapture.OutputFileResults outputFileResults) {

          }

          @Override
          public void onError(@NonNull ImageCaptureException exception) {

          }
        }
      );
    });

  }

  @WorkerThread
  @Nullable
  protected abstract R analyzeImage(ImageProxy image);

  @UiThread
  protected abstract void applyToUiAnalyzeImageResult(R result);
}
