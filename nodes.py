"""
ComfyUI nodes for YOLOE-26 open-vocabulary prompt segmentation.

See README.md for provenance and attribution notes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch

try:  # ComfyUI runtime integration (absent in unit tests / standalone usage)
    import folder_paths as _folder_paths
except ImportError:
    _folder_paths = None

try:
    from comfy import model_management as _model_management
    from comfy import utils as _comfy_utils
except ImportError:
    _model_management = None
    _comfy_utils = None

LOGGER = logging.getLogger(__name__)

TEXT_ENCODER_DOWNLOAD_NOTE = (
    "Note: the first text-prompt inference downloads the MobileCLIP text encoder "
    "(~250 MB) through Ultralytics and requires network access."
)

MAX_PROMPT_LENGTH = 512
MAX_PROMPT_CLASSES = 20
MAX_IMGSZ = 2048
DEFAULT_MASK_THRESHOLD = 0.5
DEFAULT_IOU = 0.7
DEFAULT_MAX_DET = 300
ALLOWED_MODEL_EXTENSIONS = (".pt",)
# SHA256 digests verified 2026-07-04 against fresh downloads AND the GitHub
# release-asset digests reported by the API. Ultralytics has replaced assets
# on this release tag in-place before; if verification starts failing for all
# models, the digests below likely need to be refreshed the same way.
ALLOWED_AUTO_DOWNLOAD_MODELS = {
    "yoloe-26n-seg.pt": {
        "repo": "ultralytics/assets",
        "release": "v8.4.0",
        "sha256": "1741c1f8da3cea47e2c01829c334a50dc0b9bbd05e685b90a3ce84fae32c8c1b",
    },
    "yoloe-26s-seg.pt": {
        "repo": "ultralytics/assets",
        "release": "v8.4.0",
        "sha256": "48f24206bc8680d60cbbfa296b0140da849669b9515058b72f5a945142df0654",
    },
    "yoloe-26m-seg.pt": {
        "repo": "ultralytics/assets",
        "release": "v8.4.0",
        "sha256": "585f5ec9028fd358035da8d860c27c56be285a795cba2076fba536a4391c2c83",
    },
    "yoloe-26l-seg.pt": {
        "repo": "ultralytics/assets",
        "release": "v8.4.0",
        "sha256": "a612d2d505f24e14d87ec82d688b823b6cb600646664f16125ce6c84ce360da9",
    },
    "yoloe-26x-seg.pt": {
        "repo": "ultralytics/assets",
        "release": "v8.4.0",
        "sha256": "d08d390a08f98195f7c87807839fe4ff93a5491645fef1bc3bf0700efafdd639",
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

# ComfyUI folder_paths keys and the subdirectories they map to under models/.
# The ultralytics* keys match ComfyUI-Impact-Subpack conventions so both packs
# share the same registered directories.
_MODEL_FOLDER_LAYOUT: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ultralytics_segm", ("ultralytics", "segm")),
    ("ultralytics_bbox", ("ultralytics", "bbox")),
    ("ultralytics", ("ultralytics",)),
    ("yoloe", ("yoloe",)),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fallback_comfy_models_root() -> Path:
    """Best-effort ComfyUI models root for when folder_paths is unavailable."""
    return Path(__file__).resolve().parents[2] / "models"


def _register_model_folders() -> None:
    """Register YOLOE model directories with ComfyUI's folder_paths registry."""
    if _folder_paths is None:
        return
    try:
        models_root = _folder_paths.models_dir
        for folder_key, subdirs in _MODEL_FOLDER_LAYOUT:
            _folder_paths.add_model_folder_path(
                folder_key, os.path.join(models_root, *subdirs)
            )
    except Exception as exc:  # pragma: no cover - defensive against ComfyUI API drift
        LOGGER.warning("ComfyUI-YOLOE26: failed to register model folders: %s", exc)


_register_model_folders()


def _candidate_model_dirs() -> tuple[Path, ...]:
    """Return supported local model directories.

    When running inside ComfyUI this uses the folder_paths registry, which
    includes user-configured extra_model_paths.yaml entries. Outside ComfyUI
    (unit tests, standalone usage) it falls back to directories relative to
    this file assuming a standard custom_nodes layout.
    """
    dirs: list[Path] = []
    seen: set[str] = set()

    def _append(path: Path) -> None:
        key = os.path.normcase(str(path))
        if key not in seen:
            seen.add(key)
            dirs.append(path)

    if _folder_paths is not None:
        for folder_key, _subdirs in _MODEL_FOLDER_LAYOUT:
            try:
                registered = _folder_paths.get_folder_paths(folder_key)
            except Exception as exc:
                LOGGER.debug(
                    "ComfyUI-YOLOE26: folder_paths lookup failed for '%s': %s",
                    folder_key,
                    exc,
                )
                registered = []
            for entry in registered:
                _append(Path(entry))
        models_root = Path(_folder_paths.models_dir)
    else:
        models_root = _fallback_comfy_models_root()

    for _folder_key, subdirs in _MODEL_FOLDER_LAYOUT:
        _append(models_root.joinpath(*subdirs))

    return tuple(dirs)


def _candidate_model_choices() -> list[str]:
    """Return dropdown choices as plain file names.

    Values intentionally carry no availability suffix: suffixed values baked
    into saved workflows break ComfyUI combo validation once a model's local
    status changes.
    """
    available_local_names: set[str] = set()
    for directory in _candidate_model_dirs():
        if not directory.exists():
            continue
        for candidate in directory.glob("*.pt"):
            if candidate.is_file():
                available_local_names.add(candidate.name)

    preferred_order = list(ALLOWED_AUTO_DOWNLOAD_MODELS.keys())
    # Keep yoloe-26s-seg.pt first as the recommended default
    if "yoloe-26s-seg.pt" in preferred_order:
        preferred_order.remove("yoloe-26s-seg.pt")
        preferred_order.insert(0, "yoloe-26s-seg.pt")

    choices = list(preferred_order)
    choices.extend(sorted(available_local_names - set(preferred_order)))
    return choices


def _resolve_model_path(model_name: str) -> str:
    """Resolve a local YOLOE model path from supported ComfyUI model directories."""
    if not isinstance(model_name, str):
        raise TypeError("model_name must be a string.")

    name = model_name.strip()
    if not name:
        raise ValueError("model_name cannot be empty.")

    if Path(name).name != name:
        raise ValueError(
            "model_name must be a file name only. Put the model file inside "
            "ComfyUI/models/ultralytics/... or ComfyUI/models/yoloe/."
        )

    if not name.endswith(ALLOWED_MODEL_EXTENSIONS):
        raise ValueError(
            f"Unsupported model extension for '{name}'. Supported extensions: "
            f"{', '.join(ALLOWED_MODEL_EXTENSIONS)}."
        )

    for directory in _candidate_model_dirs():
        candidate = directory / name
        if candidate.exists() and candidate.is_file():
            return str(candidate)

    raise FileNotFoundError(
        f"Model '{name}' was not found in supported local ComfyUI model directories. "
        "Place the weight file in ComfyUI/models/ultralytics/... or ComfyUI/models/yoloe/."
    )


def _preferred_auto_download_target_dir(model_name: str) -> Path:
    if _folder_paths is not None:
        models_root = Path(_folder_paths.models_dir)
    else:
        models_root = _fallback_comfy_models_root()
    if model_name.endswith("-seg.pt"):
        return models_root / "ultralytics" / "segm"
    return models_root / "ultralytics"


def _download_model_to_target(model_name: str) -> str:
    """Download an allowlisted model directly into the preferred model directory.

    Passing the full target path to attempt_download_asset avoids the previous
    behavior of downloading into the process working directory (the ComfyUI
    root) and leaving a stray copy behind after persisting.
    """
    from ultralytics.utils.downloads import attempt_download_asset

    download_config = ALLOWED_AUTO_DOWNLOAD_MODELS[model_name]
    target_dir = _preferred_auto_download_target_dir(model_name)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / model_name

    downloaded = Path(
        attempt_download_asset(
            str(target_path),
            repo=download_config["repo"],
            release=download_config["release"],
        )
    )
    if not downloaded.exists() or not downloaded.is_file():
        raise FileNotFoundError(
            f"Ultralytics did not produce a downloaded file for '{model_name}' at '{downloaded}'."
        )
    return str(downloaded)


def _create_yoloe(model_path: str):
    from ultralytics import YOLOE

    return YOLOE(model_path)


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


def _candidate_model_base_names() -> set[str]:
    names: set[str] = set(ALLOWED_AUTO_DOWNLOAD_MODELS.keys())
    for directory in _candidate_model_dirs():
        if not directory.exists():
            continue
        for candidate in directory.glob("*.pt"):
            if candidate.is_file():
                names.add(candidate.name)
    return names


def _validate_device(device: str) -> str:
    if not isinstance(device, str):
        raise TypeError("device must be a string.")
    if device not in _device_choices():
        raise ValueError(f"Unsupported device '{device}'.")
    return device


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
        # Remove the unverified file so a later run cannot silently load it
        # as a trusted local model.
        Path(resolved_path).unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded YOLOE-26 model '{model_name}' failed SHA256 verification. "
            f"Expected {expected}, got {actual}. The file was removed. "
            "If this happens for every model, Ultralytics may have republished the "
            "release assets; update this node pack or download the weights manually "
            "into a supported ComfyUI model directory."
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
        # Native-resolution masks. Without this, Ultralytics returns masks in the
        # letterboxed inference space (including padding), and resizing them
        # directly to the original image size shifts and squashes the masks for
        # any aspect ratio that required padding.
        "retina_masks": True,
    }
    if device != "auto":
        predict_kwargs["device"] = device

    return predict_kwargs


def _new_progress_bar(total: int):
    """Create a ComfyUI progress bar when running inside ComfyUI, else None."""
    if _comfy_utils is None or total < 1:
        return None
    try:
        return _comfy_utils.ProgressBar(total)
    except Exception:  # pragma: no cover - progress display must never break inference
        return None


def _raise_if_interrupted() -> None:
    """Honor ComfyUI's cancel button between batch items."""
    if _model_management is not None:
        _model_management.throw_exception_if_processing_interrupted()


def _maybe_offload_model(model: object) -> None:
    """Move the model to CPU after execution when the bundle requests it."""
    if not isinstance(model, dict) or not model.get("offload_to_cpu"):
        return

    runtime_model = model.get("model")
    try:
        if callable(getattr(runtime_model, "to", None)):
            runtime_model.to("cpu")
        elif isinstance(getattr(runtime_model, "model", None), torch.nn.Module):
            runtime_model.model.to("cpu")
        else:
            LOGGER.warning(
                "ComfyUI-YOLOE26: offload_to_cpu is enabled but the loaded model "
                "does not support .to(); skipping offload."
            )
            return
        if _model_management is not None:
            _model_management.soft_empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # pragma: no cover - offload is best-effort
        LOGGER.warning("ComfyUI-YOLOE26: failed to offload model to CPU: %s", exc)


def _configure_prompt_classes(yoloe, classes: list[str], model_path: str) -> None:
    """Configure prompt classes once per node execution.

    set_classes computes text embeddings, which is expensive and, on first use,
    downloads the MobileCLIP text encoder through Ultralytics — so it must not
    run once per image in a batch, and it is skipped entirely when the model is
    already configured with the same classes.
    """
    if getattr(yoloe, "_yoloe26_active_classes", None) == classes:
        return

    model_name = Path(model_path).name
    try:
        yoloe.set_classes(classes)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to configure YOLOE-26 classes {classes} for model '{model_name}': {exc}. "
            f"{TEXT_ENCODER_DOWNLOAD_NOTE}"
        ) from exc
    yoloe._yoloe26_active_classes = list(classes)


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
    _configure_prompt_classes(yoloe, classes, model_path)
    return classes, predict_kwargs, yoloe, model_path


def _run_single_prediction(
    yoloe,
    classes: list[str],
    img_bgr: np.ndarray,
    predict_kwargs: dict,
):
    try:
        return yoloe.predict(img_bgr, **predict_kwargs)[0]
    except Exception as exc:
        oom = isinstance(exc, torch.cuda.OutOfMemoryError) if hasattr(torch.cuda, "OutOfMemoryError") else (
            "out of memory" in str(exc).lower() or "cuda out of memory" in str(exc).lower()
        )
        if oom:
            raise RuntimeError(
                f"YOLOE-26 inference ran out of GPU memory (imgsz={predict_kwargs['imgsz']}). "
                "Try reducing imgsz or batch size."
            ) from exc
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

    arr = image_tensor.detach().cpu().numpy()
    if arr.dtype != np.float32:
        if np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32)
        elif np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
        else:
            raise ValueError(
                f"_comfy_image_to_bgr: unsupported tensor dtype '{arr.dtype}'. "
                "Expected float32 (values in [0, 1]) or an integer dtype."
            )
    img_np = (arr * 255.0).clip(0, 255).astype(np.uint8)
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
        # .float() normalizes the uint8 0/1 masks recent Ultralytics returns.
        mask = mask_tensor.detach().float().cpu().numpy()
        if mask.shape != (height, width):
            # Fallback only: with retina_masks=True masks already arrive at the
            # original image resolution.
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
        # .float() normalizes the uint8 0/1 masks recent Ultralytics returns.
        mask = mask_tensor.detach().float().cpu().numpy()
        if mask.shape != (height, width):
            # Fallback only: with retina_masks=True masks already arrive at the
            # original image resolution.
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
        elif method in REFINE_METHODS:
            raise ValueError(
                f"Refine method '{method}' is listed in REFINE_METHODS but has no implementation "
                "in the kernel-based branch. This is an internal error; please report it."
            )
        else:
            raise ValueError(
                f"Unknown refine method '{method}'. "
                f"Supported methods: {', '.join(REFINE_METHODS)}."
            )

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

    DESCRIPTION = (
        "Load a YOLOE-26 open-vocabulary segmentation model from the ComfyUI model "
        "directories, downloading official weights on demand when auto_download is "
        "enabled. " + TEXT_ENCODER_DOWNLOAD_NOTE
    )

    @classmethod
    def INPUT_TYPES(cls):
        choices = _candidate_model_choices()
        default = "yoloe-26s-seg.pt" if "yoloe-26s-seg.pt" in choices else choices[0]
        return {
            "required": {
                "model_name": (
                    choices,
                    {
                        "default": default,
                        "tooltip": (
                            "YOLOE-26 model file name. Official models are downloaded "
                            "automatically when auto_download is enabled; other entries are "
                            ".pt files found in your ComfyUI model directories (press 'r' in "
                            "ComfyUI to refresh the list after adding files)."
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
                        "default": True,
                        "tooltip": (
                            "True (recommended) = automatically download the model if not found locally. "
                            "False = local-only; the model must already exist in a supported ComfyUI model directory."
                        ),
                    },
                ),
                "offload_to_cpu": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Move the model to CPU after each segmentation node finishes to free "
                            "GPU memory for diffusion models. Slightly slower because weights are "
                            "transferred back to the GPU on the next run."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("YOLOE_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "YOLOE26"

    @classmethod
    def VALIDATE_INPUTS(cls, model_name):
        """Accept legacy suffixed values from previously saved workflows.

        Older releases embedded availability suffixes such as
        'yoloe-26s-seg.pt (downloadable)' in the dropdown values. Defining this
        validator makes ComfyUI skip the strict combo containment check for
        model_name, and execution normalizes the value, so those workflows keep
        running.
        """
        try:
            selected = _normalize_model_selection(model_name)
        except (TypeError, ValueError) as exc:
            return str(exc)
        if selected not in _candidate_model_base_names():
            return (
                f"Unknown YOLOE-26 model '{selected}'. Choose an official model "
                f"({', '.join(ALLOWED_AUTO_DOWNLOAD_MODELS)}) or place the .pt file "
                "in a supported ComfyUI model directory."
            )
        return True

    def load_model(
        self,
        model_name: str,
        device: str = "auto",
        auto_download: bool = True,
        offload_to_cpu: bool = False,
    ):
        if _validate_device(device) != device:
            raise ValueError(f"Unsupported device '{device}'.")

        selected_model_name = _normalize_model_selection(model_name)
        if selected_model_name not in _candidate_model_base_names():
            raise ValueError(f"Unsupported model selection '{model_name}'.")

        try:
            resolved = _resolve_model_path(selected_model_name)
        except FileNotFoundError as exc:
            if not auto_download:
                raise FileNotFoundError(
                    f"{exc} Enable auto_download to let this node download '{selected_model_name}', "
                    "or manually place the .pt file in a supported ComfyUI model directory."
                ) from exc

            download_model_name = _validate_auto_download_model_name(selected_model_name)
            try:
                resolved = _download_model_to_target(download_model_name)
            except Exception as download_exc:
                raise RuntimeError(
                    f"Failed to auto-download YOLOE-26 model '{selected_model_name}': {download_exc}. "
                    "Check network access, local write permissions, and Ultralytics upstream availability."
                ) from download_exc
            _verify_auto_downloaded_model(download_model_name, resolved)

        try:
            runtime_model = _create_yoloe(resolved)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load YOLOE-26 model '{Path(resolved).name}': {exc}"
            ) from exc

        return (
            {
                "model": runtime_model,
                "model_path": resolved,
                "device": device,
                "offload_to_cpu": bool(offload_to_cpu),
            },
        )


# ---------------------------------------------------------------------------
# Node: Prompt Segment
# ---------------------------------------------------------------------------


class YOLOE26PromptSegment:
    """Run open-vocabulary segmentation with a text prompt on YOLOE-26.

    Returns an annotated preview image and a merged binary mask.
    """

    DESCRIPTION = (
        "Segment objects matching a comma-separated text prompt and return an "
        "annotated preview image plus a merged binary mask at the original "
        "image resolution."
    )

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
                        "tooltip": (
                            "Threshold used to binarize instance masks. Recent Ultralytics "
                            "releases already return binary masks, in which case values "
                            "below 1.0 have no additional effect and exactly 1.0 empties "
                            "every mask."
                        ),
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
        progress = _new_progress_bar(int(image.shape[0]))

        try:
            for index in range(image.shape[0]):
                _raise_if_interrupted()
                img_bgr = _comfy_image_to_bgr(image[index])
                height, width = img_bgr.shape[:2]
                result = _run_single_prediction(yoloe, classes, img_bgr, predict_kwargs)
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
                if progress is not None:
                    progress.update(1)
        finally:
            _maybe_offload_model(model)

        annotated_batch = torch.stack(batch_annotated, dim=0)
        mask_batch = torch.stack(batch_masks, dim=0)

        return (annotated_batch, mask_batch, total_detections)


# ---------------------------------------------------------------------------
# Node: Detection Metadata
# ---------------------------------------------------------------------------


class YOLOE26DetectionMetadata:
    """Run YOLOE-26 prompt segmentation and return structured detection metadata as JSON."""

    DESCRIPTION = (
        "Run prompt segmentation and return structured JSON metadata (boxes, "
        "scores, class names, mask areas) without producing image outputs."
    )

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
                        "tooltip": (
                            "Threshold used when measuring instance mask area. Recent "
                            "Ultralytics releases already return binary masks, in which "
                            "case values below 1.0 have no additional effect and exactly "
                            "1.0 empties every mask."
                        ),
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
        progress = _new_progress_bar(int(image.shape[0]))

        try:
            for batch_index in range(image.shape[0]):
                _raise_if_interrupted()
                img_bgr = _comfy_image_to_bgr(image[batch_index])
                height, width = img_bgr.shape[:2]
                result = _run_single_prediction(yoloe, classes, img_bgr, predict_kwargs)
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
                if progress is not None:
                    progress.update(1)
        finally:
            _maybe_offload_model(model)

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

    DESCRIPTION = (
        "Segment with a text prompt and return one mask per detected instance, "
        "with JSON metadata mapping each output mask to its source image and "
        "detection."
    )

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
                        "tooltip": (
                            "Threshold used when binarizing instance masks. Recent "
                            "Ultralytics releases already return binary masks, in which "
                            "case values below 1.0 have no additional effect and exactly "
                            "1.0 empties every mask."
                        ),
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
        progress = _new_progress_bar(int(image.shape[0]))

        try:
            for batch_index in range(image.shape[0]):
                _raise_if_interrupted()
                img_bgr = _comfy_image_to_bgr(image[batch_index])
                height, width = img_bgr.shape[:2]
                result = _run_single_prediction(yoloe, classes, img_bgr, predict_kwargs)

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
                if progress is not None:
                    progress.update(1)
        finally:
            _maybe_offload_model(model)

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

    DESCRIPTION = (
        "Segment with a text prompt and return one merged mask per prompt "
        "class (all-zero when a class has no detections), with JSON metadata."
    )

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
                        "tooltip": (
                            "Threshold used when binarizing instance masks. Recent "
                            "Ultralytics releases already return binary masks, in which "
                            "case values below 1.0 have no additional effect and exactly "
                            "1.0 empties every mask."
                        ),
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
        progress = _new_progress_bar(int(image.shape[0]))

        try:
            for batch_index in range(image.shape[0]):
                _raise_if_interrupted()
                img_bgr = _comfy_image_to_bgr(image[batch_index])
                height, width = img_bgr.shape[:2]
                result = _run_single_prediction(yoloe, classes, img_bgr, predict_kwargs)
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
                if progress is not None:
                    progress.update(1)
        finally:
            _maybe_offload_model(model)

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

    DESCRIPTION = (
        "Post-process masks (morphology, largest component, hole filling, "
        "minimum area) without re-running detection."
    )

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

    DESCRIPTION = (
        "Select a single best mask from YOLOE-26 Instance Masks outputs. With a "
        "batch of input images, selection is global across the whole batch, not "
        "per image."
    )

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
                        "tooltip": (
                            "Strategy used when selecting the best instance. With batched "
                            "input images the best instance is chosen across the entire "
                            "batch, not per image."
                        ),
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
