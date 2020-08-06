package org.pytorch.demo.vision;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.view.Surface;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import com.google.common.util.concurrent.ListenableFuture;

import org.pytorch.demo.BaseModuleActivity;
import org.pytorch.demo.R;
import org.pytorch.demo.StatusBarUtils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
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
import androidx.core.content.ContextCompat;

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

  private static Bitmap convertImageProxyToBitmap(ImageProxy image) {
    ByteBuffer byteBuffer = image.getPlanes()[0].getBuffer();
    byteBuffer.rewind();
    byte[] bytes = new byte[byteBuffer.capacity()];
    byteBuffer.get(bytes);
    byte[] clonedBytes = bytes.clone();
//    BitmapFactory.Options options = new BitmapFactory.Options();
//    options.outConfig = Bitmap.Config.ARGB_8888;
//    return BitmapFactory.decodeByteArray(clonedBytes, 0, clonedBytes.length, options);
    return BitmapFactory.decodeByteArray(clonedBytes, 0, clonedBytes.length);

  }

  private static Bitmap rotateImage(Bitmap source, float angle) {
    Matrix matrix = new Matrix();
    matrix.postRotate(angle);
    return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(),
            matrix, true);
  }

  private void saveBitmapToFile(String filename, Bitmap bitmap) {

    String state = Environment.getExternalStorageState();
    if (!Environment.MEDIA_MOUNTED.equals(state)) {
      return;
    }

    File file = new File(Environment.getExternalStoragePublicDirectory("DetectDocument/"), filename);
    FileOutputStream outputStream = null;
    try {
      if (!Environment.getExternalStoragePublicDirectory("DetectDocument/").isDirectory()){
        if(!Environment.getExternalStoragePublicDirectory("DetectDocument/").mkdirs()){
          Toast.makeText(this, "Failed to save image", Toast.LENGTH_LONG).show();
        }
      }
      if(file.createNewFile()) {
        outputStream = new FileOutputStream(file);
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, outputStream);
        outputStream.flush();
        outputStream.close();
      } else {
        Toast.makeText(this, "Failed to save image", Toast.LENGTH_LONG).show();
      }
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  private Bitmap drawCircles(Bitmap bitmapImage, DocumentDetectionActivity.AnalysisResult result) {
    //circle canvas setup
    Bitmap overlay = Bitmap.createBitmap(bitmapImage.getWidth(), bitmapImage.getHeight(), bitmapImage.getConfig());
    Canvas canvas = new Canvas(overlay);
    Paint paint = new Paint();
    canvas.drawBitmap(bitmapImage, new Matrix(), null);
    paint.setStrokeWidth(10);

    paint.setColor(Color.RED);
    for (int i = 0; i < 4; i++) {
      canvas.drawCircle(result.firstResults[2*i], result.firstResults[(2*i)+1], 20, paint);
    }

    paint.setColor(Color.BLUE);
    for (int i = 0; i < 4; i++) {
      canvas.drawCircle(result.finalResults[2*i], result.finalResults[(2*i)+1], 20, paint);
    }

    return overlay;
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
            .setTargetRotation(Surface.ROTATION_90)
            .build();
    cameraProvider.bindToLifecycle(this, cameraSelector, imageCapture, preview);

    Button capture_button = findViewById(org.pytorch.demo.R.id.capture_button);

    capture_button.setOnClickListener(v -> {
      imageCapture.takePicture(ContextCompat.getMainExecutor(this),
        new ImageCapture.OnImageCapturedCallback() {
          @Override
          public void onCaptureSuccess(@NonNull ImageProxy image) {

            Bitmap bitmapImage = convertImageProxyToBitmap(image);
            bitmapImage = rotateImage(bitmapImage, 90);
            final R result = analyzeImage(bitmapImage);
            if (result != null) {
              runOnUiThread(() -> applyToUiAnalyzeImageResult(result));

              String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
              saveBitmapToFile("IMG" + timeStamp + ".jpg", bitmapImage);

              Bitmap resultBitmapImage = drawCircles(bitmapImage, (DocumentDetectionActivity.AnalysisResult) result);
              saveBitmapToFile("IMG" + timeStamp + "_out.jpg", resultBitmapImage);

              //show image
              setContentView(org.pytorch.demo.R.layout.show_result_image);
              ImageView resultImageView =  findViewById(org.pytorch.demo.R.id.resultImageView);
              resultImageView.setImageBitmap(resultBitmapImage);
            }
            image.close();
          }
        }
      );

      // Save captured image
//      File file = new File(Environment.getExternalStorageDirectory() + "/" + System.currentTimeMillis() + ".jpg");
//      ImageCapture.OutputFileOptions outputFileOptions =
//              new ImageCapture.OutputFileOptions.Builder(new File("Image.jpeg")).build();
//
//      imageCapture.takePicture(outputFileOptions, Executors.newSingleThreadExecutor(),
//        new ImageCapture.OnImageSavedCallback() {
//          @Override
//          public void onImageSaved(@NonNull ImageCapture.OutputFileResults outputFileResults) {
//
//          }
//
//          @Override
//          public void onError(@NonNull ImageCaptureException exception) {
//
//          }
//        }
//      );
    });

  }

  @WorkerThread
  @Nullable
  protected abstract R analyzeImage(Bitmap bitmapImage);

  @UiThread
  protected abstract void applyToUiAnalyzeImageResult(R result);
}
