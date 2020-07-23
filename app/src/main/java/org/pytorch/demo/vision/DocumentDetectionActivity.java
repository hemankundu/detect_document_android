package org.pytorch.demo.vision;

import android.media.Image;
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
import java.nio.FloatBuffer;
import java.util.Locale;

import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;
import androidx.camera.core.ImageProxy;
import androidx.camera.view.PreviewView;

public class DocumentDetectionActivity extends AbstractCameraXActivity<DocumentDetectionActivity.AnalysisResult> {

  public static final String INTENT_MODULE_DOCUMENT_ASSET_NAME = "INTENT_MODULE_DOCUMENT_ASSET_NAME";
  public static final String INTENT_MODULE_CORNER_ASSET_NAME = "INTENT_MODULE_CORNER_ASSET_NAME";
  public static final String INTENT_INFO_VIEW_TYPE = "INTENT_INFO_VIEW_TYPE";

  private static final int INPUT_TENSOR_DOCUMENT_WIDTH = 32;
  private static final int INPUT_TENSOR_DOCUMENT_HEIGHT = 32;
  private static final int INPUT_TENSOR_CORNER_WIDTH = 32;
  private static final int INPUT_TENSOR_CORNER_HEIGHT = 32;
  public static final String RESULTS_FORMAT = "%.2f";

  static class AnalysisResult {

    private final float[] Results;
    private final long analysisDuration;
    private final long moduleForwardDuration;

    public AnalysisResult(float[] Results,
                          long moduleForwardDuration, long analysisDuration) {
      this.Results = Results;
      this.moduleForwardDuration = moduleForwardDuration;
      this.analysisDuration = analysisDuration;
    }
  }

  private boolean mAnalyzeImageErrorState;
  private TextView ResultTextView;
  private Module mDocumentModule;
  private Module mCornerModule;
  private String mDocumentModuleAssetName;
  private String mCornerModuleAssetName;
  private FloatBuffer mDocumentInputTensorBuffer;
  private FloatBuffer mCornerInputTensorBuffer;
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
    String text = "";
    for (int i = 0; i < 8; i++) {
      text += " | " + String.format(Locale.US, RESULTS_FORMAT, result.Results[i]);
    }
    ResultTextView.setText(text);
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
  protected AnalysisResult analyzeImage(ImageProxy image) {
    if (mAnalyzeImageErrorState) {
      return null;
    }

    try {
      //run document model
      if (mDocumentModule == null) {
        final String moduleFileAbsoluteFilePath = new File(
            Utils.assetFilePath(this, getModuleAssetName("DOCUMENT"))).getAbsolutePath();
        mDocumentModule = Module.load(moduleFileAbsoluteFilePath);

        mDocumentInputTensorBuffer =
            Tensor.allocateFloatBuffer(3 * INPUT_TENSOR_DOCUMENT_WIDTH * INPUT_TENSOR_DOCUMENT_HEIGHT);
        mDocumentInputTensor = Tensor.fromBlob(mDocumentInputTensorBuffer, new long[]{1, 3, INPUT_TENSOR_DOCUMENT_HEIGHT, INPUT_TENSOR_DOCUMENT_WIDTH});
      }

      long startTime = SystemClock.elapsedRealtime();
      Image originalImage = image.getImage();
      TensorImageUtils.imageYUV420CenterCropToFloatBuffer(
              originalImage, 90,
          INPUT_TENSOR_DOCUMENT_WIDTH, INPUT_TENSOR_DOCUMENT_HEIGHT,
          TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
          TensorImageUtils.TORCHVISION_NORM_STD_RGB,
          mDocumentInputTensorBuffer, 0);

      final long documentInputPrepareDuration = SystemClock.elapsedRealtime() - startTime;

      final Tensor outputTensor = mDocumentModule.forward(IValue.from(mDocumentInputTensor)).toTensor();

      final long documentModuleForwardDuration = SystemClock.elapsedRealtime() - documentInputPrepareDuration;

      final float[] results = outputTensor.getDataAsFloatArray();

      final float[] Results = new float[8];
      for (int i = 0; i < 8; i++) {
        Results[i] = results[i];
      }

      //test image class
//      originalImage.



//      //run corner model
//      if (mCornerModule == null) {
//        final String moduleFileAbsoluteFilePath = new File(
//                Utils.assetFilePath(this, getModuleAssetName("CORNER"))).getAbsolutePath();
//        mCornerModule = Module.load(moduleFileAbsoluteFilePath);
//
//        mCornerInputTensorBuffer =
//                Tensor.allocateFloatBuffer(3 * INPUT_TENSOR_CORNER_WIDTH * INPUT_TENSOR_CORNER_HEIGHT);
//        mCornerInputTensor = Tensor.fromBlob(mCornerInputTensorBuffer, new long[]{1, 3, INPUT_TENSOR_CORNER_HEIGHT, INPUT_TENSOR_CORNER_WIDTH});
//      }
//
//      startTime = SystemClock.elapsedRealtime();
//
//      TensorImageUtils.imageYUV420CenterCropToFloatBuffer(
//              image.getImage(), rotationDegrees,
//              INPUT_TENSOR_CORNER_WIDTH, INPUT_TENSOR_CORNER_HEIGHT,
//              TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
//              TensorImageUtils.TORCHVISION_NORM_STD_RGB,
//              mCornerInputTensorBuffer, 0);
//
//      final long cornerInputPrepareDuration = SystemClock.elapsedRealtime() - startTime;
//
//      final Tensor outputTensor = mCornerModule.forward(IValue.from(mCornerInputTensor)).toTensor();
//
//      final long cornerModuleForwardDuration = SystemClock.elapsedRealtime() - cornerInputPrepareDuration;
//
//      final float[] results = outputTensor.getDataAsFloatArray();
//
//      final int[] ixs = Utils.topK(results, 8);
//
//      final float[] Results = new float[8];
//      for (int i = 0; i < 8; i++) {
//        final int ix = ixs[i];
//        Results[i] = results[ix];
//      }
//
      final long analysisDuration = SystemClock.elapsedRealtime() - startTime;

      return new AnalysisResult(Results, documentModuleForwardDuration, analysisDuration);

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
