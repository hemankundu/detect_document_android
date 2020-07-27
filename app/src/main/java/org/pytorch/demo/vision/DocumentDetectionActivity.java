package org.pytorch.demo.vision;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.os.Bundle;
import android.os.SystemClock;
import android.text.TextUtils;
import android.util.Log;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.demo.Constants;
import org.pytorch.demo.R;
import org.pytorch.demo.Utils;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.util.Locale;
import java.util.Objects;

import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;
import androidx.camera.view.PreviewView;

public class DocumentDetectionActivity extends AbstractCameraXActivity<DocumentDetectionActivity.AnalysisResult> {

  public static final String INTENT_MODULE_DOCUMENT_ASSET_NAME = "INTENT_MODULE_DOCUMENT_ASSET_NAME";
  public static final String INTENT_MODULE_CORNER_ASSET_NAME = "INTENT_MODULE_CORNER_ASSET_NAME";
  public static final String INTENT_INFO_VIEW_TYPE = "INTENT_INFO_VIEW_TYPE";

  private static final int INPUT_TENSOR_DOCUMENT_WIDTH = 32;
  private static final int INPUT_TENSOR_DOCUMENT_HEIGHT = 32;
  private static final int INPUT_TENSOR_CORNER_WIDTH = 32;
  private static final int INPUT_TENSOR_CORNER_HEIGHT = 32;
  public static final String RESULTS_FORMAT = "%d";


  static class AnalysisResult {

    private final float[] documentResults;
    private final int[] firstResults;
    private final int[] finalResults;

    public AnalysisResult(float[] documentResults, int[] firstResults, int[] finalResults) {
      this.documentResults = documentResults;
      this.firstResults = firstResults;
      this.finalResults = finalResults;
    }
  }

  static class CornerRegion {
    public int x1, y1, x2, y2;
    public CornerRegion(int y1, int y2, int x1, int x2) {
      this.x1 = Math.max(x1, 0);
      this.y1 = Math.max(y1, 0);
      this.x2 = Math.max(x2, 0);
      this.y2 = Math.max(y2, 0);
    }
  }

  private boolean mAnalyzeImageErrorState;
  private TextView ResultTextView;
  private Module mDocumentModule;
  private Module mCornerModule;
  private String mDocumentModuleAssetName;
  private String mCornerModuleAssetName;
  private Tensor mDocumentInputTensor;
  private Tensor mCornerInputTensor;

  @Override
  protected int getContentViewLayoutId() {
    return R.layout.activity_detect_document;
  }

  @Override
  protected PreviewView getCameraPreviewView() {
    return findViewById(R.id.detect_document_preview_view);

  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    ResultTextView = findViewById(R.id.detect_document_top1_result_row);
  }

  @Override
  protected void applyToUiAnalyzeImageResult(AnalysisResult result) {
    StringBuilder text = new StringBuilder();
    for (int i = 0; i < 8; i++) {
      text.append(" | ").append(String.format(Locale.US, RESULTS_FORMAT, result.finalResults[i]));
    }
    ResultTextView.setText(text.toString());
  }

  protected String getModuleAssetName(String moduleType) {
    if (moduleType.equals("DOCUMENT")){
      if (!TextUtils.isEmpty(mDocumentModuleAssetName)) {
        return mDocumentModuleAssetName;
      }
      final String moduleAssetNameFromIntent = getIntent().getStringExtra(INTENT_MODULE_DOCUMENT_ASSET_NAME);
      mDocumentModuleAssetName = !TextUtils.isEmpty(moduleAssetNameFromIntent)
          ? moduleAssetNameFromIntent
          : "test1document_shallow_repo.pt";

      return mDocumentModuleAssetName;
    }
    if (!TextUtils.isEmpty(mCornerModuleAssetName)) {
      return mCornerModuleAssetName;
    }
    final String moduleAssetNameFromIntent = getIntent().getStringExtra(INTENT_MODULE_CORNER_ASSET_NAME);
    mCornerModuleAssetName = !TextUtils.isEmpty(moduleAssetNameFromIntent)
            ? moduleAssetNameFromIntent
            : "test1corner_shallow_all.pt";

    return mCornerModuleAssetName;
  }

  @Override
  protected String getInfoViewAdditionalText() {
    return getModuleAssetName("DOCUMENT");
  }

  @Override
  @WorkerThread
  @Nullable
  protected AnalysisResult analyzeImage(Bitmap bitmapImage) {
    if (mAnalyzeImageErrorState) {
      return null;
    }

    try {
      //run document model
      if (mDocumentModule == null) {
        final String moduleFileAbsoluteFilePath = new File(
                Objects.requireNonNull(Utils.assetFilePath(this, getModuleAssetName("DOCUMENT")))).getAbsolutePath();
        mDocumentModule = Module.load(moduleFileAbsoluteFilePath);
      }
      float [] MY_NORM_MEAN_RGB = {0, 0, 0};
      float [] MY_NORM_STD_RGB = {1, 1, 1};
      long startTime = SystemClock.elapsedRealtime();
      Bitmap resizedBitmapImage = Bitmap.createScaledBitmap(bitmapImage, INPUT_TENSOR_DOCUMENT_WIDTH, INPUT_TENSOR_DOCUMENT_HEIGHT, true);
      mDocumentInputTensor = TensorImageUtils.bitmapToFloat32Tensor(resizedBitmapImage, MY_NORM_MEAN_RGB, MY_NORM_STD_RGB);

      final long documentInputPrepareDuration = SystemClock.elapsedRealtime() - startTime;
      final Tensor documentOutputTensor = mDocumentModule.forward(IValue.from(mDocumentInputTensor)).toTensor();
      final long documentModuleForwardDuration = SystemClock.elapsedRealtime() - documentInputPrepareDuration;
      final float[] documentResults = documentOutputTensor.getDataAsFloatArray();

      //test image class
      int imageHeight = bitmapImage.getHeight();
      int imageWidth = bitmapImage.getWidth();

      int [] xCords = new int[4];
      int [] yCords = new int[4];

      for (int i = 0; i < 4; i++) {
        float val = documentResults[i*2];
        xCords[i] = (int)((val>0? val:val) * imageWidth);
      }

      for (int i = 0; i < 4; i++) {
        float val = documentResults[(i*2)+1];
        yCords[i] = (int)((val>0? val:val) * imageHeight);
      }

      final int[] firstResults = new int[8];
      for (int i = 0; i < 4; i++){
        firstResults[2*i] = xCords[i];
        firstResults[(2*i)+1] = yCords[i];
      }

      CornerRegion topLeft = new CornerRegion(
              Math.max(0, (int)(2 * yCords[0] - (yCords[3] + yCords[0]) / 2)),
              (int)((yCords[3] + yCords[0]) / 2),
              Math.max(0, (int)(2 * xCords[0] - (xCords[1] + xCords[0]) / 2)),
              (int)((xCords[1] + xCords[0]) / 2));

      CornerRegion topRight = new CornerRegion(
              Math.max(0, (int)(2 * yCords[1] - (yCords[1] + yCords[2]) / 2)),
              (int)((yCords[1] + yCords[2]) / 2),
              (int)((xCords[1] + xCords[0]) / 2),
              Math.min(imageWidth - 1, (int)(xCords[1] + (xCords[1] - xCords[0]) / 2)));

      CornerRegion bottomRight = new CornerRegion(
              (int)((yCords[1] + yCords[2]) / 2),
              Math.min(imageHeight - 1, (int)(yCords[2] + (yCords[2] - yCords[1]) / 2)),
              (int)((xCords[2] + xCords[3]) / 2),
              Math.min(imageWidth - 1, (int)(xCords[2] + (xCords[2] - xCords[3]) / 2)));

      CornerRegion bottomLeft = new CornerRegion(
              (int)((yCords[0] + yCords[3]) / 2),
              Math.min(imageHeight - 1, (int)(yCords[3] + (yCords[3] - yCords[0]) / 2)),
              Math.max(0, (int)(2 * xCords[3] - (xCords[2] + xCords[3]) / 2)),
              (int)((xCords[3] + xCords[2]) / 2));

      CornerRegion[] cornerRegions= {topLeft, topRight, bottomLeft, bottomRight};

      final int[] finalResults = new int[8];
      int finalResultsIdx = 0;

      //run corner model
      if (mCornerModule == null) {
        final String moduleFileAbsoluteFilePath = new File(
                Objects.requireNonNull(Utils.assetFilePath(this, getModuleAssetName("CORNER")))).getAbsolutePath();
        mCornerModule = Module.load(moduleFileAbsoluteFilePath);
      }

      //circle canvas setup
      Bitmap overlay = Bitmap.createBitmap(bitmapImage.getWidth(), bitmapImage.getHeight(), bitmapImage.getConfig());
      Canvas canvas = new Canvas(overlay);
      Paint paint = new Paint();
      canvas.drawBitmap(bitmapImage, new Matrix(), null);
      paint.setColor(Color.RED);
      paint.setStrokeWidth(10);
      for (int i = 0; i < 4; i++) {
        canvas.drawCircle(xCords[i], yCords[i], 50, paint);
      }
      paint.setColor(Color.BLUE);

      for (CornerRegion cornerRegion : cornerRegions) {
        startTime = SystemClock.elapsedRealtime();
        Bitmap cornerBitmap = Bitmap.createBitmap(bitmapImage, cornerRegion.x1, cornerRegion.y1,
                cornerRegion.x2 - cornerRegion.x1, cornerRegion.y2 - cornerRegion.y1);
        Bitmap resizedCornerBitmap = Bitmap.createScaledBitmap(cornerBitmap, INPUT_TENSOR_CORNER_WIDTH, INPUT_TENSOR_CORNER_HEIGHT, true);

        mCornerInputTensor = TensorImageUtils.bitmapToFloat32Tensor(resizedCornerBitmap, MY_NORM_MEAN_RGB, MY_NORM_STD_RGB);

        final long cornerInputPrepareDuration = SystemClock.elapsedRealtime() - startTime;
        final Tensor cornerOutputTensor = mCornerModule.forward(IValue.from(mCornerInputTensor)).toTensor();
        final long cornerModuleForwardDuration = SystemClock.elapsedRealtime() - cornerInputPrepareDuration;
        final float[] cornerResults = cornerOutputTensor.getDataAsFloatArray();

        int finalCornerX = (int)(cornerResults[0] * (cornerRegion.x2 - cornerRegion.x1)) + cornerRegion.x1;
        int finalCornerY = (int)(cornerResults[1] * (cornerRegion.y2 - cornerRegion.y1)) + cornerRegion.y1;
        finalResults[finalResultsIdx++] = finalCornerX;
        finalResults[finalResultsIdx++] = finalCornerY;

        //draw a circle
        canvas.drawCircle(finalCornerX, finalCornerY, 50, paint);

      }

      final long analysisDuration = SystemClock.elapsedRealtime() - startTime;

      return new AnalysisResult(documentResults, firstResults, finalResults);

    } catch (Exception e) {
      Log.e(Constants.TAG, "Error during image analysis", e);
      mAnalyzeImageErrorState = true;
      runOnUiThread(() -> {
        if (!isFinishing()) {
          showErrorDialog(v -> DocumentDetectionActivity.this.finish());
        }
      });
      return null;
    }
  }

  @Override
  protected int getInfoViewCode() {
    return getIntent().getIntExtra(INTENT_INFO_VIEW_TYPE, -1);
  }

  @Override
  protected void onDestroy() {
    super.onDestroy();
    if (mDocumentModule != null) {
      mDocumentModule.destroy();
    }
    if (mCornerModule != null) {
      mCornerModule.destroy();
    }
  }
}
