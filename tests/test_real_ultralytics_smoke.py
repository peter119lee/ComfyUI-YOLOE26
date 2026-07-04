"""Real-ultralytics smoke tests for native-resolution mask alignment.

These guard the retina_masks regression with the actual installed ultralytics,
using the plain YOLO26 segmentation model (~6 MB) instead of YOLOE to avoid the
MobileCLIP text-encoder download. Both model families share the same
SegmentationPredictor postprocess path that retina_masks affects.

The tests skip themselves when ultralytics is unavailable or the smoke weights
cannot be downloaded (network-restricted environments).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "nodes.py"
WEIGHTS_CACHE = Path(__file__).resolve().parent / ".cache" / "yolo26n-seg.pt"

spec = importlib.util.spec_from_file_location("comfyui_yoloe26_nodes_smoke", MODULE_PATH)
nodes = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = nodes
spec.loader.exec_module(nodes)

pytest.importorskip("ultralytics")


@pytest.fixture(scope="module")
def smoke_prediction():
    import cv2
    from ultralytics import YOLO
    from ultralytics.utils import ASSETS
    from ultralytics.utils.downloads import attempt_download_asset

    WEIGHTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    try:
        attempt_download_asset(str(WEIGHTS_CACHE))
    except Exception as exc:
        pytest.skip(f"could not download smoke-test weights: {exc}")
    if not WEIGHTS_CACHE.exists():
        pytest.skip("smoke-test weights unavailable")

    # bus.jpg ships inside the ultralytics package and needs letterbox padding
    # (810x1080), which is exactly the case the retina_masks fix addresses.
    image = cv2.imread(str(ASSETS / "bus.jpg"))
    if image is None:
        pytest.skip("ultralytics sample image unavailable")

    model = YOLO(str(WEIGHTS_CACHE))
    predict_kwargs = nodes._build_predict_kwargs("cpu", 0.25, 640, 0.7, 300)
    result = model.predict(image, **predict_kwargs)[0]
    if result.masks is None or len(result.masks.data) == 0:
        pytest.skip("no detections on smoke image")
    return image, result


def test_masks_arrive_at_original_image_resolution(smoke_prediction):
    image, result = smoke_prediction
    height, width = image.shape[:2]
    assert tuple(result.masks.data.shape[-2:]) == (height, width), (
        "masks are not in original-image space; retina_masks regression: "
        f"{tuple(result.masks.data.shape[-2:])} != {(height, width)}"
    )


def test_instance_masks_align_with_detection_boxes(smoke_prediction):
    image, result = smoke_prediction
    height, width = image.shape[:2]
    masks = nodes._build_per_instance_masks(result, height, width)
    boxes = result.boxes.xyxy.detach().cpu().numpy()

    tolerance = 2.5
    checked = 0
    for mask, box in zip(masks, boxes):
        ys, xs = np.where(mask > 0.5)
        if len(xs) == 0:
            continue
        # Correctly aligned native-space masks are contained in their boxes;
        # the padded-resize bug shifted masks outside them.
        assert xs.min() >= box[0] - tolerance and ys.min() >= box[1] - tolerance
        assert xs.max() <= box[2] + tolerance and ys.max() <= box[3] + tolerance
        checked += 1

    assert checked >= 1, "expected at least one non-empty instance mask"
