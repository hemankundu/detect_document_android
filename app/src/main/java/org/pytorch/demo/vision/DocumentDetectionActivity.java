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
import java.util.Dictionary;
import java.util.Enumeration;
import java.util.Hashtable;
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

  private boolean mAnalyzeImageErrorState;
  private TextView ResultTextView;
  private Module mDocumentModule;
  private Module mCornerModule;
  private String mDocumentModuleAssetName;
  private String mCornerModuleAssetName;

  static class AnalysisResult {
    public final int[] firstResults;
    public final int[] finalResults;
    public final Dictionary<String, Long> durationLog;

    public AnalysisResult(int[] firstResults, int[] finalResults, Dictionary<String, Long> durationLog) {
      this.firstResults = firstResults;
      this.finalResults = finalResults;
      this.durationLog = durationLog;
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

    ResultTextView.setText(String.format(Locale.US, "Time taken | First model: %dms | Second model: %dms",
                            result.durationLog.get("documentModuleForwardDuration"), result.durationLog.get("cornerModuleForwardDuration")));
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

    Dictionary<String, Long> durationLog = new Hashtable<>();

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

      Tensor mDocumentInputTensor = TensorImageUtils.bitmapToFloat32Tensor(resizedBitmapImage, MY_NORM_MEAN_RGB, MY_NORM_STD_RGB);
      Tensor documentOutputTensor = mDocumentModule.forward(IValue.from(mDocumentInputTensor)).toTensor();

      long documentModuleForwardDuration = SystemClock.elapsedRealtime() - startTime;
      durationLog.put("documentModuleForwardDuration" , documentModuleForwardDuration);

      float[] documentResults = documentOutputTensor.getDataAsFloatArray();

      //test image class
      int imageHeight = bitmapImage.getHeight();
      int imageWidth = bitmapImage.getWidth();

      int [] xCords = new int[4];
      int [] yCords = new int[4];

      for (int i = 0; i < 4; i++) {
        xCords[i] = (int)(documentResults[i*2] * imageWidth);
      }
      for (int i = 0; i < 4; i++) {
        yCords[i] = (int)(documentResults[(i*2)+1] * imageHeight);
      }

      int[] firstResults = new int[8];
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

      int[] finalResults = new int[8];
      int finalResultsIdx = 0;

      //run corner model
      if (mCornerModule == null) {
        final String moduleFileAbsoluteFilePath = new File(
                Objects.requireNonNull(Utils.assetFilePath(this, getModuleAssetName("CORNER")))).getAbsolutePath();
        mCornerModule = Module.load(moduleFileAbsoluteFilePath);
      }

      for (CornerRegion cornerRegion : cornerRegions) {
        startTime = SystemClock.elapsedRealtime();
        Bitmap cornerBitmap = Bitmap.createBitmap(bitmapImage, cornerRegion.x1, cornerRegion.y1,
                cornerRegion.x2 - cornerRegion.x1, cornerRegion.y2 - cornerRegion.y1);
        Bitmap resizedCornerBitmap = Bitmap.createScaledBitmap(cornerBitmap, INPUT_TENSOR_CORNER_WIDTH, INPUT_TENSOR_CORNER_HEIGHT, true);

        Tensor mCornerInputTensor = TensorImageUtils.bitmapToFloat32Tensor(resizedCornerBitmap, MY_NORM_MEAN_RGB, MY_NORM_STD_RGB);
        Tensor cornerOutputTensor = mCornerModule.forward(IValue.from(mCornerInputTensor)).toTensor();

        final long cornerModuleForwardDuration = SystemClock.elapsedRealtime() - documentModuleForwardDuration;
        durationLog.put("cornerModuleForwardDuration" , cornerModuleForwardDuration);

        final float[] cornerResults = cornerOutputTensor.getDataAsFloatArray();

        int finalCornerX = (int)(cornerResults[0] * (cornerRegion.x2 - cornerRegion.x1)) + cornerRegion.x1;
        int finalCornerY = (int)(cornerResults[1] * (cornerRegion.y2 - cornerRegion.y1)) + cornerRegion.y1;
        finalResults[finalResultsIdx++] = finalCornerX;
        finalResults[finalResultsIdx++] = finalCornerY;
      }

      return new AnalysisResult(firstResults, finalResults, durationLog);

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
