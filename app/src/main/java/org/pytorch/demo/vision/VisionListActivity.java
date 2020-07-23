package org.pytorch.demo.vision;

import android.content.Intent;
import android.os.Bundle;

import org.pytorch.demo.AbstractListActivity;
import org.pytorch.demo.InfoViewFactory;
import org.pytorch.demo.R;

public class VisionListActivity extends AbstractListActivity {

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    findViewById(R.id.vision_card_quantized_click_area).setOnClickListener(v -> {
      final Intent intent = new Intent(VisionListActivity.this, DocumentDetectionActivity.class);
      intent.putExtra(DocumentDetectionActivity.INTENT_MODULE_DOCUMENT_ASSET_NAME, "test1document_shallow.pt");
      intent.putExtra(DocumentDetectionActivity.INTENT_MODULE_CORNER_ASSET_NAME, "test1corner_shallow_all.pt");
      intent.putExtra(DocumentDetectionActivity.INTENT_INFO_VIEW_TYPE,
          InfoViewFactory.INFO_VIEW_TYPE_DETECT_DOCUMENT_QUANTIZED);
      startActivity(intent);
    });
    findViewById(R.id.vision_card_normal_click_area).setOnClickListener(v -> {
      final Intent intent = new Intent(VisionListActivity.this, DocumentDetectionActivity.class);
      intent.putExtra(DocumentDetectionActivity.INTENT_MODULE_DOCUMENT_ASSET_NAME, "test1document_shallow_repo.pt");
      intent.putExtra(DocumentDetectionActivity.INTENT_MODULE_CORNER_ASSET_NAME, "test1corner_shallow_all.pt");
      intent.putExtra(DocumentDetectionActivity.INTENT_INFO_VIEW_TYPE,
          InfoViewFactory.INFO_VIEW_TYPE_DETECT_DOCUMENT_NORMAL);
      startActivity(intent);
    });
  }

  @Override
  protected int getListContentLayoutRes() {
    return R.layout.vision_list_content;
  }
}
