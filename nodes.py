"""
ComfyUI nodes for YOLOE-26 open-vocabulary prompt segmentation.

See README.md for provenance and attribution notes.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

MAX_PROMPT_LENGTH = 512
MAX_PROMPT_CLASSES = 20
MAX_IMGSZ = 2048
DEFAULT_MASK_THRESHOLD = 0.5
DEFAULT_IOU = 0.7
DEFAULT_MAX_DET = 300
ALLOWED_MODEL_EXTENSIONS = (".pt",)
ALLOWED_AUTO_DOWNLOAD_MODELS = {
    "yoloe-26n-seg.pt": {
        "repo": "ultralytics/assets",
        "release": "v8.4.0",
        "sha256": "f432978449803654a11b386754a7edc96187dca39ea622925b781d3d36a975b8",
    },
    "yoloe-26s-seg.pt": {
        "repo": "ultralytics/assets",
        "release": "v8.4.0",
        "sha256": "6f62bc7ed9f86056112c383e9b85023291a3929086af26b1a8762335fe39a17d",
    },
    "yoloe-26m-seg.pt": {
        "repo": "ultralytics/assets",
        "release": "v8.4.0",
        "sha256": "214afb47524eebd80add0d8aa32f6731b2b540ff0ea42c57a20b9d76069fc756",
    },
    "yoloe-26l-seg.pt": {
        "repo": "ultralytics/assets",
        "release": "v8.4.0",
        "sha256": "a9413bf3f15772c223a03bbedd71b79af5822f830e80aa1b51b1a469e65927b1",
    },
    "yoloe-26x-seg.pt": {
        "repo": "ultralytics/assets",
        "release": "v8.4.0",
        "sha256": "6f55ef8985bf91dbea7918200d187494048d14421723e795ee730980aba0f3c2",
    },
}
REFINE_METHODS = (
    "threshold",
    "open",
    "close",
    "dilate",
    "erode",
    "largest_component",
    "fill_holes",
)
SELECTION_MODES = ("highest_confidence", "largest_area", "confidence_then_area")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _candidate_model_dirs() -> tuple[Path, ...]:
    """Return supported local model directories inside ComfyUI."""
    comfy_base = Path(__file__).resolve().parents[2]
    return (
        comfy_base / "models" / "ultralytics" / "segm",
        comfy_base / "models" / "ultralytics" / "bbox",
        comfy_base / "models" / "ultralytics",
        comfy_base / "models" / "yoloe",
    )


def _candidate_model_choices() -> list[str]:
    available_local_names: set[str] = set()
    for directory in _candidate_model_dirs():
        if not directory.exists():
            continue
        for candidate in directory.glob("*.pt"):
            if candidate.is_file():
                available_local_names.add(candidate.name)

    choices: list[str] = []
    seen: set[str] = set()

    for model_name in ALLOWED_AUTO_DOWNLOAD_MODELS:
        status = "local" if model_name in available_local_names else "downloadable"
        choices.append(f"{model_name} ({status})")
        seen.add(model_name)

    for model_name in sorted(available_local_names):
        if model_name in seen:
            continue
        choices.append(f"{model_name} (local)")

    return choices


def _normalize_model_selection(model_name: str) -> str:
    if not isinstance(model_name, str):
        raise TypeError("model_name must be a string.")

    name = model_name.strip()
    if not name:
        raise ValueError("model_name cannot be empty.")

    if name.endswith(" (local)"):
        return name.removesuffix(" (local)").strip()
    if name.endswith(" (downloadable)"):
        return name.removesuffix(" (downloadable)").strip()

    return name


def _device_choices() -> list[str]:
    choices = ["auto", "cpu"]
    if torch.cuda.is_available():
        choices.append("cuda")
        choices.extend(f"cuda:{index}" for index in range(torch.cuda.device_count()))
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        choices.append("mps")
    seen: set[str] = set()
    unique_choices: list[str] = []
    for choice in choices:
        if choice in seen:
            continue
        seen.add(choice)
        unique_choices.append(choice)
    return unique_choices


def _validate_device(device: str) -> str:
    if not isinstance(device, str):
        raise TypeError("device must be a string.")
    if device not in _device_choices():
        raise ValueError(f"Unsupported device '{device}'.")
    return device


def _runtime_model_path(runtime_model: object, fallback_name: str) -> str:
    ckpt_path = getattr(runtime_model, "ckpt_path", None)
    if isinstance(ckpt_path, str) and ckpt_path:
        return ckpt_path

    inner_model = getattr(runtime_model, "model", None)
    pt_path = getattr(inner_model, "pt_path", None)
    if isinstance(pt_path, str) and pt_path:
        return pt_path

    model_name = getattr(runtime_model, "model_name", None)
    if isinstance(model_name, str) and model_name:
        return model_name

    return fallback_name


def _validate_auto_download_model_name(model_name: str) -> str:
    name = model_name.strip()
    if name not in ALLOWED_AUTO_DOWNLOAD_MODELS:
        raise ValueError(
            f"Auto-download is only supported for official models: {', '.join(ALLOWED_AUTO_DOWNLOAD_MODELS)}."
        )
    return name


def _sha256_file(file_path: str) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_auto_downloaded_model(model_name: str, resolved_path: str) -> None:
    expected = ALLOWED_AUTO_DOWNLOAD_MODELS[model_name].get("sha256")
    if not expected:
        return

    actual = _sha256_file(resolved_path)
    if actual != expected:
        raise RuntimeError(
            f"Downloaded YOLOE-26 model '{model_name}' failed SHA256 verification. "
            f"Expected {expected}, got {actual}."
        )


def _validate_model_bundle(model: dict) -> tuple[object, str, str]:
    if not isinstance(model, dict):
        raise TypeError(
            "model input must be a YOLOE model bundle returned by YOLOE-26 Load Model."
        )

    runtime_model = model.get("model")
    model_path = model.get("model_path")
    device = model.get("device", "auto")

    if runtime_model is None:
        raise ValueError("Invalid model bundle: missing loaded model instance.")

    if not isinstance(model_path, str) or not model_path:
        raise ValueError("Invalid model bundle: missing model_path.")

    if _validate_device(device) != device:
        raise ValueError(f"Unsupported device '{device}'.")

    if not callable(getattr(runtime_model, "set_classes", None)):
        raise ValueError(
            "Invalid model bundle: loaded model instance does not implement set_classes()."
        )

    if not callable(getattr(runtime_model, "predict", None)):
        raise ValueError(
            "Invalid model bundle: loaded model instance does not implement predict()."
        )

    return runtime_model, model_path, device


def _validate_image_batch(image: torch.Tensor, node_name: str) -> None:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"{node_name} expected IMAGE input as torch.Tensor.")

    if image.ndim != 4:
        raise ValueError(
            f"{node_name} expected IMAGE tensor in ComfyUI format (B, H, W, C), "
            f"got shape {tuple(image.shape)}."
        )

    if image.shape[0] == 0:
        raise ValueError(f"{node_name} received an empty image batch.")

    if image.shape[-1] != 3:
        raise ValueError(
            f"{node_name} expected RGB images with 3 channels, got shape {tuple(image.shape)}."
        )


def _validate_mask_batch(mask: torch.Tensor, node_name: str) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{node_name} expected MASK input as torch.Tensor.")

    if mask.ndim == 2:
        mask = mask.unsqueeze(0)

    if mask.ndim != 3:
        raise ValueError(
            f"{node_name} expected MASK tensor with shape (N, H, W) or (H, W), "
            f"got shape {tuple(mask.shape)}."
        )

    if mask.shape[0] == 0:
        raise ValueError(f"{node_name} received an empty mask batch.")

    return mask.detach().cpu().float()


def _parse_classes(prompt: str) -> list[str]:
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string.")

    cleaned_prompt = prompt.strip()
    if not cleaned_prompt:
        raise ValueError("Prompt cannot be empty. Provide at least one class name.")

    if len(cleaned_prompt) > MAX_PROMPT_LENGTH:
        raise ValueError(
            f"Prompt is too long. Limit prompts to {MAX_PROMPT_LENGTH} characters."
        )

    classes = [item.strip() for item in cleaned_prompt.split(",") if item.strip()]
    if not classes:
        raise ValueError("Prompt cannot be empty. Provide at least one class name.")

    if len(classes) > MAX_PROMPT_CLASSES:
        raise ValueError(
            f"Too many prompt classes. Limit to {MAX_PROMPT_CLASSES} comma-separated classes."
        )

    return classes


def _validate_mask_threshold(mask_threshold: float) -> float:
    if not isinstance(mask_threshold, (int, float)):
        raise TypeError("mask_threshold must be a number between 0 and 1.")

    threshold = float(mask_threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("mask_threshold must be between 0 and 1.")

    return threshold


def _build_predict_kwargs(
    device: str, conf: float, imgsz: int, iou: float, max_det: int
) -> dict:
    if not isinstance(conf, (int, float)):
        raise TypeError("conf must be a number between 0 and 1.")

    if not 0.0 <= float(conf) <= 1.0:
        raise ValueError("conf must be between 0 and 1.")

    if not isinstance(iou, (int, float)):
        raise TypeError("iou must be a number between 0 and 1.")

    if not 0.0 <= float(iou) <= 1.0:
        raise ValueError("iou must be between 0 and 1.")

    if not isinstance(imgsz, int):
        raise TypeError("imgsz must be an integer.")

    if imgsz < 64 or imgsz > MAX_IMGSZ:
        raise ValueError(f"imgsz must be between 64 and {MAX_IMGSZ}.")

    if not isinstance(max_det, int):
        raise TypeError("max_det must be an integer.")

    if max_det < 1:
        raise ValueError("max_det must be at least 1.")

    predict_kwargs = {
        "conf": float(conf),
        "iou": float(iou),
        "imgsz": imgsz,
        "max_det": max_det,
        "verbose": False,
    }
    if device != "auto":
        predict_kwargs["device"] = device

    return predict_kwargs


def _prepare_segmentation_runtime(
    model: dict,
    prompt: str,
    conf: float,
    iou: float,
    imgsz: int,
    max_det: int,
):
    classes = _parse_classes(prompt)
    yoloe, model_path, device = _validate_model_bundle(model)
    predict_kwargs = _build_predict_kwargs(device, conf, imgsz, iou, max_det)
    return classes, predict_kwargs, yoloe, model_path


def _run_single_prediction(
    yoloe,
    classes: list[str],
    model_path: str,
    img_bgr: np.ndarray,
    predict_kwargs: dict,
):
    model_name = Path(model_path).name
    try:
        yoloe.set_classes(classes)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to configure YOLOE-26 classes {classes} for model '{model_name}': {exc}"
        ) from exc

    try:
        return yoloe.predict(img_bgr, **predict_kwargs)[0]
    except Exception as exc:
        raise RuntimeError(
            f"YOLOE-26 inference failed for prompt classes {classes}, imgsz={predict_kwargs['imgsz']}: {exc}"
        ) from exc


def _comfy_image_to_bgr(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert a single ComfyUI IMAGE tensor (H, W, C) float32 [0,1] RGB
    to numpy uint8 BGR for OpenCV / Ultralytics."""
    if image_tensor.ndim != 3 or image_tensor.shape[-1] != 3:
        raise ValueError(
            "Expected a single image tensor with shape (H, W, 3) in RGB format, "
            f"got {tuple(image_tensor.shape)}."
        )

    img_np = (
        (image_tensor.detach().cpu().float().numpy() * 255.0)
        .clip(0, 255)
        .astype(np.uint8)
    )
    return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)


def _bgr_to_comfy_image(bgr: np.ndarray) -> torch.Tensor:
    """Convert numpy uint8 BGR back to ComfyUI IMAGE tensor (H, W, C)
    float32 [0,1] RGB."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.astype(np.float32) / 255.0)


def _threshold_mask(mask: np.ndarray, threshold: float) -> np.ndarray:
    return (mask > threshold).astype(np.float32)


def _build_binary_mask(
    result, height: int, width: int, mask_threshold: float = DEFAULT_MASK_THRESHOLD
) -> np.ndarray:
    """Merge all instance masks into a single binary mask (float32 0/1)."""
    mask_threshold = _validate_mask_threshold(mask_threshold)
    merged = np.zeros((height, width), dtype=np.float32)

    if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
        return merged

    for mask_tensor in result.masks.data:
        mask = mask_tensor.detach().cpu().numpy()
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        merged = np.maximum(merged, _threshold_mask(mask, mask_threshold))

    return merged


def _build_per_instance_masks(
    result, height: int, width: int, mask_threshold: float = DEFAULT_MASK_THRESHOLD
) -> list[np.ndarray]:
    """Return a list of individual binary masks (float32 0/1), one per instance."""
    mask_threshold = _validate_mask_threshold(mask_threshold)
    masks: list[np.ndarray] = []

    if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
        return masks

    for mask_tensor in result.masks.data:
        mask = mask_tensor.detach().cpu().numpy()
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        masks.append(_threshold_mask(mask, mask_threshold))

    return masks


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    mask_uint8 = (mask > 0.5).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8
    )
    if num_labels <= 1:
        return mask_uint8.astype(np.float32)

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest_label).astype(np.float32)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    mask_uint8 = (mask > 0.5).astype(np.uint8)
    background = 1 - mask_uint8
    num_labels, labels = cv2.connectedComponents(background, connectivity=8)
    holes = np.zeros_like(mask_uint8, dtype=np.uint8)

    for label in range(1, num_labels):
        component = labels == label
        touches_border = (
            component[0, :].any()
            or component[-1, :].any()
            or component[:, 0].any()
            or component[:, -1].any()
        )
        if not touches_border:
            holes[component] = 1

    return np.maximum(mask_uint8, holes).astype(np.float32)


def _refine_mask(
    mask: np.ndarray, method: str, kernel_size: int, iterations: int, min_area: int
) -> np.ndarray:
    refined = (mask > 0.5).astype(np.float32)
    if method == "threshold":
        pass
    elif method == "largest_component":
        refined = _largest_connected_component(refined)
    elif method == "fill_holes":
        refined = _fill_holes(refined)
    else:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        if method == "dilate":
            refined = cv2.dilate(
                refined.astype(np.uint8), kernel, iterations=iterations
            )
        elif method == "erode":
            refined = cv2.erode(refined.astype(np.uint8), kernel, iterations=iterations)
        elif method == "open":
            refined = cv2.morphologyEx(
                refined.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=iterations
            )
        elif method == "close":
            refined = cv2.morphologyEx(
                refined.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=iterations
            )
        else:
            raise ValueError(f"Unsupported refine method '{method}'.")

    refined = refined.astype(np.float32)
    if min_area > 0 and float(refined.sum()) < float(min_area):
        return np.zeros_like(refined, dtype=np.float32)
    return refined


def _select_best_instance(
    records: list[dict], instance_masks: list[np.ndarray], selection_mode: str
) -> tuple[int, np.ndarray, dict | None]:
    if not records:
        return -1, np.zeros((0, 0), dtype=np.float32), None

    candidates: list[tuple[float, float, int, dict]] = []
    for record in records:
        mask_index = record.get("mask_index")
        if mask_index is None or mask_index >= len(instance_masks):
            continue
        area = float(record.get("mask_area", 0))
        confidence = float(record.get("confidence", 0.0))
        candidates.append(
            (confidence, area, int(record.get("instance_index", 0)), record)
        )

    if not candidates:
        height = int(records[0].get("image_height", 0))
        width = int(records[0].get("image_width", 0))
        return -1, np.zeros((height, width), dtype=np.float32), None

    if selection_mode == "highest_confidence":
        candidates.sort(key=lambda item: (-item[0], item[2]))
    elif selection_mode == "largest_area":
        candidates.sort(key=lambda item: (-item[1], item[2]))
    elif selection_mode == "confidence_then_area":
        candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    else:
        raise ValueError(f"Unsupported selection mode '{selection_mode}'.")

    confidence, area, instance_index, record = candidates[0]
    mask_index = int(record["mask_index"])
    return mask_index, instance_masks[mask_index], record


def _extract_detection_records(
    result,
    classes: list[str],
    height: int,
    width: int,
    batch_index: int,
    instance_masks: list[np.ndarray] | None = None,
) -> list[dict]:
    """Extract structured per-detection metadata from a YOLO result."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = getattr(boxes, "xyxy", None)
    cls_tensor = getattr(boxes, "cls", None)
    conf_tensor = getattr(boxes, "conf", None)
    if xyxy is None or cls_tensor is None or conf_tensor is None:
        return []

    xyxy_np = xyxy.detach().cpu().numpy()
    cls_np = cls_tensor.detach().cpu().numpy().astype(int)
    conf_np = conf_tensor.detach().cpu().numpy().astype(float)
    if instance_masks is None:
        instance_masks = _build_per_instance_masks(result, height, width)

    records: list[dict] = []
    for instance_index in range(len(xyxy_np)):
        class_id = int(cls_np[instance_index]) if instance_index < len(cls_np) else -1
        if 0 <= class_id < len(classes):
            class_name = classes[class_id]
        else:
            class_name = str(class_id)

        box = xyxy_np[instance_index].tolist()
        mask = (
            instance_masks[instance_index]
            if instance_index < len(instance_masks)
            else None
        )
        mask_area = int(float(mask.sum())) if mask is not None else 0

        records.append(
            {
                "batch_index": int(batch_index),
                "instance_index": int(instance_index),
                "class_id": class_id,
                "class_name": class_name,
                "confidence": (
                    float(conf_np[instance_index])
                    if instance_index < len(conf_np)
                    else 0.0
                ),
                "bbox_xyxy": [float(value) for value in box],
                "mask_index": int(instance_index) if mask is not None else None,
                "mask_area": mask_area,
                "image_height": int(height),
                "image_width": int(width),
            }
        )

    return records


def _build_class_masks_from_records(
    records: list[dict],
    instance_masks: list[np.ndarray],
    classes: list[str],
    height: int,
    width: int,
    batch_index: int,
) -> tuple[list[np.ndarray], list[dict]]:
    """Build one merged mask per prompt class for a single image."""
    class_masks: list[np.ndarray] = []
    class_entries: list[dict] = []

    for class_id, class_name in enumerate(classes):
        merged = np.zeros((height, width), dtype=np.float32)
        source_instance_indices: list[int] = []

        for record in records:
            if record["class_id"] != class_id:
                continue
            mask_index = record.get("mask_index")
            if mask_index is None or mask_index >= len(instance_masks):
                continue
            merged = np.maximum(merged, instance_masks[mask_index])
            source_instance_indices.append(int(record["instance_index"]))

        class_masks.append(merged)
        class_entries.append(
            {
                "mask_index": len(class_masks) - 1,
                "batch_index": int(batch_index),
                "class_id": int(class_id),
                "class_name": class_name,
                "source_instance_indices": source_instance_indices,
                "source_instance_count": len(source_instance_indices),
            }
        )

    return class_masks, class_entries


def _serialize_metadata(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Node: Model Loader
# ---------------------------------------------------------------------------


class YOLOE26LoadModel:
    """Validate and prepare a YOLOE-26 model for prompt segmentation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    _candidate_model_choices(),
                    {
                        "default": "yoloe-26s-seg.pt (downloadable)",
                        "tooltip": (
                            "Choose a YOLOE-26 model preset. Labels marked local exist in supported "
                            "ComfyUI model directories; labels marked downloadable can be fetched when "
                            "auto_download is enabled."
                        ),
                    },
                ),
            },
            "optional": {
                "device": (
                    _device_choices(),
                    {
                        "default": "auto",
                        "tooltip": "Inference device. 'auto' lets Ultralytics choose.",
                    },
                ),
                "auto_download": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "False = local-only loading from supported ComfyUI model directories. "
                            "True = if the local weight file is missing, let Ultralytics try downloading "
                            "the requested official model first."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("YOLOE_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "YOLOE26"

    def load_model(
        self, model_name: str, device: str = "auto", auto_download: bool = False
    ):
        if _validate_device(device) != device:
            raise ValueError(f"Unsupported device '{device}'.")

        selected_model_name = _normalize_model_selection(model_name)
        if selected_model_name not in _candidate_model_choices():
            raise ValueError(f"Unsupported model selection '{model_name}'.")

        try:
            resolved = _resolve_model_path(selected_model_name)
        except FileNotFoundError as exc:
            if not auto_download:
                raise FileNotFoundError(
                    f"{exc} Enable auto_download to let Ultralytics try downloading '{selected_model_name}', "
                    "or manually place the .pt file in a supported ComfyUI model directory."
                ) from exc

            download_model_name = _validate_auto_download_model_name(selected_model_name)
            download_config = ALLOWED_AUTO_DOWNLOAD_MODELS[download_model_name]
            try:
                from ultralytics.utils.downloads import attempt_download_asset

                resolved = attempt_download_asset(
                    download_model_name,
                    repo=download_config["repo"],
                    release=download_config["release"],
                )
                if not Path(resolved).exists():
                    raise FileNotFoundError(
                        f"Ultralytics did not return a downloadable path for '{download_model_name}'."
                    )
                _verify_auto_downloaded_model(download_model_name, resolved)
                runtime_model = _create_yoloe(resolved)
            except Exception as download_exc:
                raise RuntimeError(
                    f"Failed to auto-download YOLOE-26 model '{model_name}': {download_exc}. "
                    "Check network access, local write permissions, and Ultralytics upstream availability."
                ) from download_exc

        else:
            try:
                runtime_model = _create_yoloe(resolved)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to validate YOLOE-26 model '{Path(resolved).name}': {exc}"
                ) from exc

        return ({"model": runtime_model, "model_path": resolved, "device": device},)


# ---------------------------------------------------------------------------
# Node: Prompt Segment
# ---------------------------------------------------------------------------


class YOLOE26PromptSegment:
    """Run open-vocabulary segmentation with a text prompt on YOLOE-26.

    Returns an annotated preview image and a merged binary mask.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("YOLOE_MODEL",),
                "image": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {
                        "default": "person",
                        "multiline": False,
                        "tooltip": (
                            "Text prompt for open-vocabulary segmentation. "
                            "Separate multiple classes with commas, e.g. "
                            "'person, car, dog'."
                        ),
                    },
                ),
            },
            "optional": {
                "conf": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Detection confidence threshold.",
                    },
                ),
                "iou": (
                    "FLOAT",
                    {
                        "default": DEFAULT_IOU,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "IoU threshold used by Ultralytics inference.",
                    },
                ),
                "max_det": (
                    "INT",
                    {
                        "default": DEFAULT_MAX_DET,
                        "min": 1,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Maximum number of detections returned per image.",
                    },
                ),
                "mask_threshold": (
                    "FLOAT",
                    {
                        "default": DEFAULT_MASK_THRESHOLD,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Threshold used to binarize instance masks.",
                    },
                ),
                "imgsz": (
                    "INT",
                    {
                        "default": 640,
                        "min": 64,
                        "max": MAX_IMGSZ,
                        "step": 32,
                        "tooltip": "Inference image size.",
                    },
                ),
                "show_boxes": ("BOOLEAN", {"default": True}),
                "show_labels": ("BOOLEAN", {"default": True}),
                "show_conf": ("BOOLEAN", {"default": True}),
                "show_masks": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("annotated_image", "mask", "detection_count")
    FUNCTION = "segment"
    CATEGORY = "YOLOE26"

    def segment(
        self,
        model: dict,
        image: torch.Tensor,
        prompt: str,
        conf: float = 0.1,
        iou: float = DEFAULT_IOU,
        max_det: int = DEFAULT_MAX_DET,
        mask_threshold: float = DEFAULT_MASK_THRESHOLD,
        imgsz: int = 640,
        show_boxes: bool = True,
        show_labels: bool = True,
        show_conf: bool = True,
        show_masks: bool = True,
    ):
        _validate_image_batch(image, "YOLOE-26 Prompt Segment")
        classes, predict_kwargs, yoloe, model_path = _prepare_segmentation_runtime(
            model, prompt, conf, iou, imgsz, max_det
        )

        batch_annotated: list[torch.Tensor] = []
        batch_masks: list[torch.Tensor] = []
        total_detections = 0

        for index in range(image.shape[0]):
            img_bgr = _comfy_image_to_bgr(image[index])
            height, width = img_bgr.shape[:2]
            result = _run_single_prediction(
                yoloe, classes, model_path, img_bgr, predict_kwargs
            )
            instance_masks = _build_per_instance_masks(
                result, height, width, mask_threshold
            )

            annotated_bgr = result.plot(
                conf=show_conf,
                labels=show_labels,
                boxes=show_boxes,
                masks=show_masks,
            )
            batch_annotated.append(_bgr_to_comfy_image(annotated_bgr))

            merged = _build_binary_mask(result, height, width, mask_threshold)
            batch_masks.append(torch.from_numpy(merged))

            records = _extract_detection_records(
                result, classes, height, width, index, instance_masks=instance_masks
            )
            total_detections += len(records)

        annotated_batch = torch.stack(batch_annotated, dim=0)
        mask_batch = torch.stack(batch_masks, dim=0)

        return (annotated_batch, mask_batch, total_detections)


# ---------------------------------------------------------------------------
# Node: Detection Metadata
# ---------------------------------------------------------------------------


class YOLOE26DetectionMetadata:
    """Run YOLOE-26 prompt segmentation and return structured detection metadata as JSON."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("YOLOE_MODEL",),
                "image": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {
                        "default": "person",
                        "multiline": False,
                        "tooltip": (
                            "Text prompt for open-vocabulary segmentation. "
                            "Separate multiple classes with commas."
                        ),
                    },
                ),
            },
            "optional": {
                "conf": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Detection confidence threshold.",
                    },
                ),
                "iou": (
                    "FLOAT",
                    {
                        "default": DEFAULT_IOU,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "IoU threshold used by Ultralytics inference.",
                    },
                ),
                "max_det": (
                    "INT",
                    {
                        "default": DEFAULT_MAX_DET,
                        "min": 1,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Maximum number of detections returned per image.",
                    },
                ),
                "mask_threshold": (
                    "FLOAT",
                    {
                        "default": DEFAULT_MASK_THRESHOLD,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Threshold used when measuring instance mask area.",
                    },
                ),
                "imgsz": (
                    "INT",
                    {
                        "default": 640,
                        "min": 64,
                        "max": MAX_IMGSZ,
                        "step": 32,
                        "tooltip": "Inference image size.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("metadata_json", "detection_count")
    FUNCTION = "detect_metadata"
    CATEGORY = "YOLOE26"

    def detect_metadata(
        self,
        model: dict,
        image: torch.Tensor,
        prompt: str,
        conf: float = 0.1,
        iou: float = DEFAULT_IOU,
        max_det: int = DEFAULT_MAX_DET,
        mask_threshold: float = DEFAULT_MASK_THRESHOLD,
        imgsz: int = 640,
    ):
        _validate_image_batch(image, "YOLOE-26 Detection Metadata")
        classes, predict_kwargs, yoloe, model_path = _prepare_segmentation_runtime(
            model, prompt, conf, iou, imgsz, max_det
        )

        image_entries: list[dict] = []
        total_detections = 0

        for batch_index in range(image.shape[0]):
            img_bgr = _comfy_image_to_bgr(image[batch_index])
            height, width = img_bgr.shape[:2]
            result = _run_single_prediction(
                yoloe, classes, model_path, img_bgr, predict_kwargs
            )
            instance_masks = _build_per_instance_masks(
                result, height, width, mask_threshold
            )
            records = _extract_detection_records(
                result,
                classes,
                height,
                width,
                batch_index,
                instance_masks=instance_masks,
            )
            total_detections += len(records)
            image_entries.append(
                {
                    "batch_index": int(batch_index),
                    "image_height": int(height),
                    "image_width": int(width),
                    "detection_count": len(records),
                    "detections": records,
                }
            )

        metadata_json = _serialize_metadata(
            {
                "version": 1,
                "model_name": Path(model_path).name,
                "prompt_raw": prompt,
                "classes": classes,
                "conf": float(conf),
                "iou": float(iou),
                "max_det": int(max_det),
                "mask_threshold": float(mask_threshold),
                "imgsz": int(imgsz),
                "total_images": int(image.shape[0]),
                "total_detections": int(total_detections),
                "images": image_entries,
            }
        )

        return (metadata_json, total_detections)


class YOLOE26InstanceMasks:
    """Run YOLOE-26 prompt segmentation and output per-instance masks.

    Supports batch input and returns JSON metadata describing which output mask
    belongs to which input image and detection.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("YOLOE_MODEL",),
                "image": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {
                        "default": "person",
                        "multiline": False,
                        "tooltip": (
                            "Text prompt for open-vocabulary segmentation. "
                            "Separate multiple classes with commas."
                        ),
                    },
                ),
            },
            "optional": {
                "conf": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Detection confidence threshold.",
                    },
                ),
                "iou": (
                    "FLOAT",
                    {
                        "default": DEFAULT_IOU,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "IoU threshold used by Ultralytics inference.",
                    },
                ),
                "max_det": (
                    "INT",
                    {
                        "default": DEFAULT_MAX_DET,
                        "min": 1,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Maximum number of detections returned per image.",
                    },
                ),
                "mask_threshold": (
                    "FLOAT",
                    {
                        "default": DEFAULT_MASK_THRESHOLD,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Threshold used when binarizing instance masks.",
                    },
                ),
                "imgsz": (
                    "INT",
                    {
                        "default": 640,
                        "min": 64,
                        "max": MAX_IMGSZ,
                        "step": 32,
                        "tooltip": "Inference image size.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MASK", "STRING", "INT")
    RETURN_NAMES = ("instance_masks", "instance_metadata_json", "count")
    FUNCTION = "segment_instances"
    CATEGORY = "YOLOE26"

    def segment_instances(
        self,
        model: dict,
        image: torch.Tensor,
        prompt: str,
        conf: float = 0.1,
        iou: float = DEFAULT_IOU,
        max_det: int = DEFAULT_MAX_DET,
        mask_threshold: float = DEFAULT_MASK_THRESHOLD,
        imgsz: int = 640,
    ):
        _validate_image_batch(image, "YOLOE-26 Instance Masks")
        classes, predict_kwargs, yoloe, model_path = _prepare_segmentation_runtime(
            model, prompt, conf, iou, imgsz, max_det
        )

        all_masks: list[torch.Tensor] = []
        image_entries: list[dict] = []
        total_instances = 0

        for batch_index in range(image.shape[0]):
            img_bgr = _comfy_image_to_bgr(image[batch_index])
            height, width = img_bgr.shape[:2]
            result = _run_single_prediction(
                yoloe, classes, model_path, img_bgr, predict_kwargs
            )

            instance_masks = _build_per_instance_masks(
                result, height, width, mask_threshold
            )
            records = _extract_detection_records(
                result,
                classes,
                height,
                width,
                batch_index,
                instance_masks=instance_masks,
            )

            output_mask_indices: list[int] = []
            output_detections: list[dict] = []
            for record in records:
                mask_index = record.get("mask_index")
                if mask_index is None or mask_index >= len(instance_masks):
                    continue
                all_masks.append(torch.from_numpy(instance_masks[mask_index]))
                output_index = len(all_masks) - 1
                output_mask_indices.append(output_index)
                output_record = dict(record)
                output_record["output_mask_index"] = output_index
                output_detections.append(output_record)

            total_instances += len(output_mask_indices)
            image_entries.append(
                {
                    "batch_index": int(batch_index),
                    "image_height": int(height),
                    "image_width": int(width),
                    "instance_count": len(output_mask_indices),
                    "output_mask_indices": output_mask_indices,
                    "detections": output_detections,
                    "is_empty_result": bool(len(output_mask_indices) == 0),
                }
            )

        if not all_masks:
            height = int(image.shape[1])
            width = int(image.shape[2])
            all_masks.append(torch.zeros((height, width), dtype=torch.float32))

        mask_batch = torch.stack(all_masks, dim=0)
        metadata_json = _serialize_metadata(
            {
                "version": 1,
                "model_name": Path(model_path).name,
                "prompt_raw": prompt,
                "classes": classes,
                "conf": float(conf),
                "iou": float(iou),
                "max_det": int(max_det),
                "mask_threshold": float(mask_threshold),
                "imgsz": int(imgsz),
                "total_images": int(image.shape[0]),
                "total_instances": int(total_instances),
                "is_empty_result": bool(total_instances == 0),
                "images": image_entries,
            }
        )

        return (mask_batch, metadata_json, total_instances)


class YOLOE26ClassMasks:
    """Run YOLOE-26 prompt segmentation and output one merged mask per prompt class."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("YOLOE_MODEL",),
                "image": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {
                        "default": "person",
                        "multiline": False,
                        "tooltip": (
                            "Text prompt for open-vocabulary segmentation. "
                            "Separate multiple classes with commas."
                        ),
                    },
                ),
            },
            "optional": {
                "conf": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Detection confidence threshold.",
                    },
                ),
                "iou": (
                    "FLOAT",
                    {
                        "default": DEFAULT_IOU,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "IoU threshold used by Ultralytics inference.",
                    },
                ),
                "max_det": (
                    "INT",
                    {
                        "default": DEFAULT_MAX_DET,
                        "min": 1,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Maximum number of detections returned per image.",
                    },
                ),
                "mask_threshold": (
                    "FLOAT",
                    {
                        "default": DEFAULT_MASK_THRESHOLD,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Threshold used when binarizing instance masks.",
                    },
                ),
                "imgsz": (
                    "INT",
                    {
                        "default": 640,
                        "min": 64,
                        "max": MAX_IMGSZ,
                        "step": 32,
                        "tooltip": "Inference image size.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MASK", "STRING", "INT")
    RETURN_NAMES = ("class_masks", "class_metadata_json", "output_mask_count")
    FUNCTION = "segment_class_masks"
    CATEGORY = "YOLOE26"

    def segment_class_masks(
        self,
        model: dict,
        image: torch.Tensor,
        prompt: str,
        conf: float = 0.1,
        iou: float = DEFAULT_IOU,
        max_det: int = DEFAULT_MAX_DET,
        mask_threshold: float = DEFAULT_MASK_THRESHOLD,
        imgsz: int = 640,
    ):
        _validate_image_batch(image, "YOLOE-26 Class Masks")
        classes, predict_kwargs, yoloe, model_path = _prepare_segmentation_runtime(
            model, prompt, conf, iou, imgsz, max_det
        )

        all_class_masks: list[torch.Tensor] = []
        entries: list[dict] = []

        for batch_index in range(image.shape[0]):
            img_bgr = _comfy_image_to_bgr(image[batch_index])
            height, width = img_bgr.shape[:2]
            result = _run_single_prediction(
                yoloe, classes, model_path, img_bgr, predict_kwargs
            )
            instance_masks = _build_per_instance_masks(
                result, height, width, mask_threshold
            )
            records = _extract_detection_records(
                result,
                classes,
                height,
                width,
                batch_index,
                instance_masks=instance_masks,
            )
            class_masks, class_entries = _build_class_masks_from_records(
                records, instance_masks, classes, height, width, batch_index
            )

            for entry, class_mask in zip(class_entries, class_masks):
                all_class_masks.append(torch.from_numpy(class_mask))
                entry["output_mask_index"] = len(all_class_masks) - 1
                entries.append(entry)

        if not all_class_masks:
            height = int(image.shape[1])
            width = int(image.shape[2])
            all_class_masks.append(torch.zeros((height, width), dtype=torch.float32))

        mask_batch = torch.stack(all_class_masks, dim=0)
        metadata_json = _serialize_metadata(
            {
                "version": 1,
                "model_name": Path(model_path).name,
                "prompt_raw": prompt,
                "classes": classes,
                "conf": float(conf),
                "iou": float(iou),
                "max_det": int(max_det),
                "mask_threshold": float(mask_threshold),
                "imgsz": int(imgsz),
                "total_images": int(image.shape[0]),
                "class_count": len(classes),
                "output_mask_count": int(len(all_class_masks)),
                "is_empty_result": bool(
                    len(entries) > 0 and all(entry["source_instance_count"] == 0 for entry in entries)
                ),
                "entries": entries,
            }
        )

        return (mask_batch, metadata_json, len(all_class_masks))


class YOLOE26RefineMask:
    """Apply binary mask refinement operations to a ComfyUI MASK batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
            },
            "optional": {
                "method": (
                    REFINE_METHODS,
                    {
                        "default": "threshold",
                        "tooltip": "Refinement operation applied to each input mask.",
                    },
                ),
                "kernel_size": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 99,
                        "step": 2,
                        "tooltip": "Kernel size used by morphology operations.",
                    },
                ),
                "iterations": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 32,
                        "step": 1,
                        "tooltip": "Number of morphology iterations.",
                    },
                ),
                "min_area": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100000000,
                        "step": 1,
                        "tooltip": "Drop refined masks smaller than this area.",
                    },
                ),
                "metadata_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional upstream metadata JSON to preserve and annotate.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MASK", "STRING", "INT")
    RETURN_NAMES = ("refined_masks", "refined_metadata_json", "count")
    FUNCTION = "refine_mask_batch"
    CATEGORY = "YOLOE26"

    def refine_mask_batch(
        self,
        masks: torch.Tensor,
        method: str = "threshold",
        kernel_size: int = 3,
        iterations: int = 1,
        min_area: int = 0,
        metadata_json: str = "",
    ):
        mask_batch = _validate_mask_batch(masks, "YOLOE-26 Refine Mask")

        if not isinstance(kernel_size, int) or kernel_size < 1:
            raise ValueError(
                "kernel_size must be an integer greater than or equal to 1."
            )
        if not isinstance(iterations, int) or iterations < 1:
            raise ValueError(
                "iterations must be an integer greater than or equal to 1."
            )
        if not isinstance(min_area, int) or min_area < 0:
            raise ValueError("min_area must be an integer greater than or equal to 0.")

        refined_masks = [
            torch.from_numpy(
                _refine_mask(
                    mask_batch[index].numpy(), method, kernel_size, iterations, min_area
                )
            )
            for index in range(mask_batch.shape[0])
        ]
        refined_batch = torch.stack(refined_masks, dim=0)

        if metadata_json.strip():
            try:
                payload = json.loads(metadata_json)
            except json.JSONDecodeError as exc:
                raise ValueError("metadata_json must be valid JSON.") from exc
            if not isinstance(payload, dict):
                raise ValueError("metadata_json must decode to a JSON object.")
            payload = dict(payload)
        else:
            payload = {"version": 1}

        payload["refinement"] = {
            "method": method,
            "kernel_size": int(kernel_size),
            "iterations": int(iterations),
            "min_area": int(min_area),
        }
        if "total_instances" in payload:
            payload["total_instances"] = int(refined_batch.shape[0])
        if "output_mask_count" in payload:
            payload["output_mask_count"] = int(refined_batch.shape[0])

        return (
            refined_batch,
            _serialize_metadata(payload),
            int(refined_batch.shape[0]),
        )


class YOLOE26SelectBestInstance:
    """Select a single best instance mask from YOLOE-26 instance mask outputs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "instance_masks": ("MASK",),
                "instance_metadata_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Metadata JSON produced by YOLOE-26 Instance Masks.",
                    },
                ),
            },
            "optional": {
                "selection_mode": (
                    SELECTION_MODES,
                    {
                        "default": "highest_confidence",
                        "tooltip": "Strategy used when selecting the best instance.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MASK", "STRING", "INT")
    RETURN_NAMES = ("best_mask", "best_instance_metadata_json", "selected_mask_index")
    FUNCTION = "select_best_instance"
    CATEGORY = "YOLOE26"

    def select_best_instance(
        self,
        instance_masks: torch.Tensor,
        instance_metadata_json: str,
        selection_mode: str = "highest_confidence",
    ):
        mask_batch = _validate_mask_batch(
            instance_masks, "YOLOE-26 Select Best Instance"
        )
        if (
            not isinstance(instance_metadata_json, str)
            or not instance_metadata_json.strip()
        ):
            raise ValueError("instance_metadata_json must be a non-empty JSON string.")

        try:
            payload = json.loads(instance_metadata_json)
        except json.JSONDecodeError as exc:
            raise ValueError("instance_metadata_json must be valid JSON.") from exc

        if not isinstance(payload, dict):
            raise ValueError("instance_metadata_json must decode to a JSON object.")

        images = payload.get("images")
        if not isinstance(images, list):
            raise ValueError("instance_metadata_json must include an images list.")

        records: list[dict] = []
        for image_entry in images:
            detections = (
                image_entry.get("detections") if isinstance(image_entry, dict) else None
            )
            if detections is None:
                continue
            if not isinstance(detections, list):
                raise ValueError(
                    "instance_metadata_json image entries must contain a detections list."
                )
            for detection in detections:
                if not isinstance(detection, dict):
                    raise ValueError(
                        "instance_metadata_json detections must be JSON objects."
                    )
                if "output_mask_index" not in detection:
                    raise ValueError(
                        "instance_metadata_json must contain detections with output_mask_index values."
                    )
                output_mask_index = detection["output_mask_index"]
                if not isinstance(output_mask_index, int):
                    raise ValueError("output_mask_index must be an integer.")
                if output_mask_index < 0 or output_mask_index >= int(
                    mask_batch.shape[0]
                ):
                    raise ValueError(
                        "output_mask_index is out of range for the provided instance_masks batch."
                    )
                record = dict(detection)
                record["mask_index"] = output_mask_index
                records.append(record)

        if not records:
            empty_mask = torch.zeros(
                (1, mask_batch.shape[1], mask_batch.shape[2]), dtype=torch.float32
            )
            return (
                empty_mask,
                _serialize_metadata(
                    {
                        "version": int(payload.get("version", 1)),
                        "selection_mode": selection_mode,
                        "selected_mask_index": -1,
                        "candidate_count": 0,
                        "selected_detection": None,
                        "is_empty_result": True,
                    }
                ),
                -1,
            )

        mask_arrays = [
            mask_batch[index].numpy() for index in range(mask_batch.shape[0])
        ]
        selected_mask_index, selected_mask, selected_record = _select_best_instance(
            records, mask_arrays, selection_mode
        )

        if (
            selected_mask_index < 0
            or selected_record is None
            or selected_mask.size == 0
        ):
            empty_mask = torch.zeros(
                (1, mask_batch.shape[1], mask_batch.shape[2]), dtype=torch.float32
            )
            return (
                empty_mask,
                _serialize_metadata(
                    {
                        "version": int(payload.get("version", 1)),
                        "selection_mode": selection_mode,
                        "selected_mask_index": -1,
                        "candidate_count": int(len(records)),
                        "selected_detection": None,
                        "is_empty_result": True,
                    }
                ),
                -1,
            )

        selected_mask_batch = torch.from_numpy(selected_mask).unsqueeze(0)
        selected_output_mask_index = int(
            selected_record.get("output_mask_index", selected_mask_index)
        )
        return (
            selected_mask_batch,
            _serialize_metadata(
                {
                    "version": int(payload.get("version", 1)),
                    "selection_mode": selection_mode,
                    "selected_mask_index": selected_output_mask_index,
                    "candidate_count": int(len(records)),
                    "selected_detection": selected_record,
                    "is_empty_result": False,
                }
            ),
            selected_output_mask_index,
        )


# ---------------------------------------------------------------------------
# Mappings
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "YOLOE26LoadModel": YOLOE26LoadModel,
    "YOLOE26PromptSegment": YOLOE26PromptSegment,
    "YOLOE26DetectionMetadata": YOLOE26DetectionMetadata,
    "YOLOE26InstanceMasks": YOLOE26InstanceMasks,
    "YOLOE26ClassMasks": YOLOE26ClassMasks,
    "YOLOE26RefineMask": YOLOE26RefineMask,
    "YOLOE26SelectBestInstance": YOLOE26SelectBestInstance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YOLOE26LoadModel": "YOLOE-26 Load Model",
    "YOLOE26PromptSegment": "YOLOE-26 Prompt Segment",
    "YOLOE26DetectionMetadata": "YOLOE-26 Detection Metadata",
    "YOLOE26InstanceMasks": "YOLOE-26 Instance Masks",
    "YOLOE26ClassMasks": "YOLOE-26 Class Masks",
    "YOLOE26RefineMask": "YOLOE-26 Refine Mask",
    "YOLOE26SelectBestInstance": "YOLOE-26 Select Best Instance",
}
