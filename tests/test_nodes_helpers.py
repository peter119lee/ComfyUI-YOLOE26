import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "nodes.py"
EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
WORKFLOW_PATH = EXAMPLES_DIR / "basic_api_workflow.json"
ALL_NODES_SHOWCASE_PATH = EXAMPLES_DIR / "all_nodes_showcase_api.json"
spec = importlib.util.spec_from_file_location("comfyui_yoloe26_nodes", MODULE_PATH)
nodes = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = nodes
spec.loader.exec_module(nodes)


class FakeMasks:
    def __init__(self, data):
        self.data = data


class FakeBoxes:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = xyxy
        self.cls = cls
        self.conf = conf

    def __len__(self):
        return len(self.cls)


class FakeResult:
    def __init__(self, masks=None, boxes=None, plot_image=None):
        self.masks = masks
        self.boxes = boxes
        self._plot_image = (
            plot_image
            if plot_image is not None
            else np.zeros((2, 2, 3), dtype=np.uint8)
        )

    def plot(self, **kwargs):
        return self._plot_image.copy()


class FakeInnerModel:
    def __init__(self, pt_path=None):
        self.pt_path = pt_path


class FakeModel:
    def __init__(self, results, ckpt_path=None, pt_path=None):
        self.results = list(results)
        self.classes_history = []
        self.predict_history = []
        self.ckpt_path = ckpt_path
        self.model = FakeInnerModel(pt_path=pt_path) if pt_path else None

    def set_classes(self, classes):
        self.classes_history.append(list(classes))

    def predict(self, img_bgr, **kwargs):
        self.predict_history.append(dict(kwargs))
        if not self.results:
            raise RuntimeError("No fake results remaining")
        return [self.results.pop(0)]


class TestNodesHelpers(unittest.TestCase):
    @staticmethod
    def _load_workflow(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _workflow_custom_node_classes(workflow: dict) -> list[str]:
        return [
            node_def["class_type"]
            for node_def in workflow.values()
            if node_def["class_type"].startswith("YOLOE26")
        ]

    @staticmethod
    def _workflow_builtin_node_classes(workflow: dict) -> list[str]:
        return [
            node_def["class_type"]
            for node_def in workflow.values()
            if not node_def["class_type"].startswith("YOLOE26")
        ]

    @staticmethod
    def _find_nodes_by_class_type(workflow: dict, class_type: str) -> list[tuple[str, dict]]:
        return [
            (node_id, node_def)
            for node_id, node_def in workflow.items()
            if node_def["class_type"] == class_type
        ]

    @staticmethod
    def _workflow_has_output_node(workflow: dict) -> bool:
        return any(
            node_def["class_type"] in {"PreviewImage", "SaveImage"}
            for node_def in workflow.values()
        )

    @staticmethod
    def _workflow_node_class(workflow: dict, node_id: str) -> str:
        return workflow[node_id]["class_type"]

    @staticmethod
    def _node_references_output(node_def: dict, upstream_node_id: str, output_index: int) -> bool:
        target_reference = [upstream_node_id, output_index]
        return any(value == target_reference for value in node_def["inputs"].values())

    def test_load_model_input_types_exposes_model_choice_list(self):
        input_types = nodes.YOLOE26LoadModel.INPUT_TYPES()
        model_type, model_config = input_types["required"]["model_name"]
        self.assertIsInstance(model_type, list)
        self.assertIn("yoloe-26s-seg.pt (downloadable)", model_type)
        self.assertIn("yoloe-26m-seg.pt (downloadable)", model_type)
        self.assertIn("yoloe-26l-seg.pt (downloadable)", model_type)
        self.assertIn("yoloe-26x-seg.pt (downloadable)", model_type)
        self.assertEqual(model_config["default"], "yoloe-26s-seg.pt (downloadable)")
        self.assertIn("tooltip", model_config)

    def test_load_model_raises_file_not_found_when_model_missing_and_auto_download_disabled(
        self,
    ):
        with patch.object(
            nodes, "_resolve_model_path", side_effect=FileNotFoundError("missing model")
        ) as mock_resolve:
            with patch.object(nodes, "_create_yoloe") as mock_create:
                with self.assertRaises(FileNotFoundError):
                    nodes.YOLOE26LoadModel().load_model(
                        "yoloe-26s-seg.pt", auto_download=False
                    )

        mock_resolve.assert_called_once_with("yoloe-26s-seg.pt")
        mock_create.assert_not_called()

    def test_load_model_downloads_missing_model_when_auto_download_enabled(self):
        runtime_model = FakeModel([], ckpt_path="L:/downloaded/yoloe-26s-seg.pt")

        with patch.object(
            nodes, "_resolve_model_path", side_effect=FileNotFoundError("missing model")
        ) as mock_resolve:
            with patch.object(
                nodes, "_validate_auto_download_model_name", return_value="yoloe-26s-seg.pt"
            ) as mock_validate:
                with patch.object(
                    nodes, "_verify_auto_downloaded_model"
                ) as mock_verify:
                    with patch.object(
                        nodes, "_create_yoloe", return_value=runtime_model
                    ) as mock_create:
                        with patch(
                            "pathlib.Path.exists", return_value=True
                        ) as mock_exists:
                            with patch(
                                "ultralytics.utils.downloads.attempt_download_asset",
                                return_value="L:/downloaded/yoloe-26s-seg.pt",
                            ) as mock_attempt:
                                (bundle,) = nodes.YOLOE26LoadModel().load_model(
                                    "yoloe-26s-seg.pt (downloadable)", auto_download=True
                                )

        mock_resolve.assert_called_once_with("yoloe-26s-seg.pt")
        mock_validate.assert_called_once_with("yoloe-26s-seg.pt")
        mock_attempt.assert_called_once_with(
            "yoloe-26s-seg.pt", repo="ultralytics/assets", release="v8.4.0"
        )
        mock_create.assert_called_once_with("L:/downloaded/yoloe-26s-seg.pt")
        mock_verify.assert_called_once_with(
            "yoloe-26s-seg.pt", "L:/downloaded/yoloe-26s-seg.pt"
        )
        self.assertIs(bundle["model"], runtime_model)
        self.assertEqual(bundle["model_path"], "L:/downloaded/yoloe-26s-seg.pt")
        self.assertEqual(bundle["device"], "auto")

    def test_load_model_does_not_download_when_model_already_exists_even_if_auto_download_enabled(
        self,
    ):
        runtime_model = FakeModel([])

        with patch.object(
            nodes, "_resolve_model_path", return_value="L:/models/yoloe-26s-seg.pt"
        ) as mock_resolve:
            with patch.object(
                nodes, "_create_yoloe", return_value=runtime_model
            ) as mock_create:
                (bundle,) = nodes.YOLOE26LoadModel().load_model(
                    "yoloe-26s-seg.pt (local)", auto_download=True
                )

        mock_resolve.assert_called_once_with("yoloe-26s-seg.pt")
        mock_create.assert_called_once_with("L:/models/yoloe-26s-seg.pt")
        self.assertEqual(bundle["model_path"], "L:/models/yoloe-26s-seg.pt")

    def test_load_model_raises_runtime_error_when_auto_download_fails(self):
        with patch.object(
            nodes, "_resolve_model_path", side_effect=FileNotFoundError("missing model")
        ) as mock_resolve:
            with patch.object(
                nodes,
                "_validate_auto_download_model_name",
                return_value="yoloe-26s-seg.pt",
            ):
                with patch.object(
                    nodes,
                    "_create_yoloe",
                    side_effect=RuntimeError("network disabled"),
                ) as mock_create:
                    with self.assertRaises(RuntimeError) as exc:
                        nodes.YOLOE26LoadModel().load_model(
                            "yoloe-26s-seg.pt (downloadable)", auto_download=True
                        )

        mock_resolve.assert_called_once_with("yoloe-26s-seg.pt")
        mock_create.assert_called_once_with("yoloe-26s-seg.pt")
        self.assertIn("auto-download", str(exc.exception))
        self.assertIn("yoloe-26s-seg.pt", str(exc.exception))

    def test_load_model_runtime_model_path_prefers_inner_pt_path_fallback(self):
        runtime_model = FakeModel([], pt_path="L:/cache/yoloe-26s-seg.pt")

        with patch.object(
            nodes, "_resolve_model_path", side_effect=FileNotFoundError("missing model")
        ):
            with patch.object(
                nodes, "_validate_auto_download_model_name", return_value="yoloe-26s-seg.pt"
            ):
                with patch.object(
                    nodes, "_verify_auto_downloaded_model"
                ) as mock_verify:
                    with patch.object(nodes, "_create_yoloe", return_value=runtime_model):
                        with patch(
                            "pathlib.Path.exists", return_value=True
                        ) as mock_exists:
                            with patch(
                                "ultralytics.utils.downloads.attempt_download_asset",
                                return_value="L:/cache/yoloe-26s-seg.pt",
                            ) as mock_attempt:
                                (bundle,) = nodes.YOLOE26LoadModel().load_model(
                                    "yoloe-26s-seg.pt", auto_download=True
                                )

        mock_attempt.assert_called_once_with(
            "yoloe-26s-seg.pt", repo="ultralytics/assets", release="v8.4.0"
        )
        mock_verify.assert_called_once_with(
            "yoloe-26s-seg.pt", "L:/cache/yoloe-26s-seg.pt"
        )
        self.assertEqual(bundle["model_path"], "L:/cache/yoloe-26s-seg.pt")

    def test_load_model_runtime_model_path_falls_back_to_model_name(self):
        runtime_model = FakeModel([])

        with patch.object(
            nodes, "_resolve_model_path", side_effect=FileNotFoundError("missing model")
        ):
            with patch.object(
                nodes, "_validate_auto_download_model_name", return_value="yoloe-26s-seg.pt"
            ):
                with patch.object(nodes, "_create_yoloe", return_value=runtime_model):
                    with patch.object(nodes, "_verify_auto_downloaded_model") as mock_verify:
                        with patch(
                            "ultralytics.utils.downloads.attempt_download_asset",
                            return_value="yoloe-26s-seg.pt",
                        ) as mock_attempt:
                            (bundle,) = nodes.YOLOE26LoadModel().load_model(
                                "yoloe-26s-seg.pt", auto_download=True
                            )

        mock_attempt.assert_called_once_with(
            "yoloe-26s-seg.pt", repo="ultralytics/assets", release="v8.4.0"
        )
        mock_verify.assert_called_once_with(
            "yoloe-26s-seg.pt", "yoloe-26s-seg.pt"
        )
        self.assertEqual(bundle["model_path"], "yoloe-26s-seg.pt")

    def test_load_model_raises_runtime_error_when_local_model_validation_fails(
        self,
    ):
        with patch.object(
            nodes, "_resolve_model_path", return_value="L:/models/yoloe-26s-seg.pt"
        ) as mock_resolve:
            with patch.object(
                nodes,
                "_create_yoloe",
                side_effect=FileNotFoundError("weights corrupted"),
            ) as mock_create:
                with self.assertRaises(RuntimeError) as exc:
                    nodes.YOLOE26LoadModel().load_model(
                        "yoloe-26s-seg.pt", auto_download=True
                    )

        mock_resolve.assert_called_once_with("yoloe-26s-seg.pt")
        mock_create.assert_called_once_with("L:/models/yoloe-26s-seg.pt")
        self.assertIn("Failed to validate YOLOE-26 model", str(exc.exception))

    def test_all_example_api_workflows_parse_and_reference_known_nodes_smoke(self):
        custom_nodes = nodes.NODE_CLASS_MAPPINGS
        builtin_nodes = {"LoadImage", "PreviewImage", "SaveImage", "MaskToImage", "PreviewAny"}

        for workflow_path in EXAMPLES_DIR.glob("*_api*.json"):
            workflow = self._load_workflow(workflow_path)
            self.assertIsInstance(workflow, dict, workflow_path.name)
            self.assertTrue(workflow, workflow_path.name)

            for node_def in workflow.values():
                class_type = node_def["class_type"]
                self.assertTrue(
                    class_type in custom_nodes or class_type in builtin_nodes,
                    f"{workflow_path.name}: unknown class_type {class_type}",
                )

    def test_all_nodes_showcase_workflow_covers_all_custom_nodes_smoke(self):
        workflow = self._load_workflow(ALL_NODES_SHOWCASE_PATH)
        custom_classes = set(self._workflow_custom_node_classes(workflow))
        self.assertEqual(custom_classes, set(nodes.NODE_CLASS_MAPPINGS.keys()))

    def test_practical_workflows_include_expected_primary_nodes_smoke(self):
        expected_nodes_by_file = {
            "practical_prompt_segment_api.json": {"YOLOE26PromptSegment"},
            "practical_best_instance_api.json": {
                "YOLOE26InstanceMasks",
                "YOLOE26SelectBestInstance",
            },
            "practical_class_masks_api.json": {"YOLOE26ClassMasks"},
            "practical_refine_mask_api.json": {
                "YOLOE26PromptSegment",
                "YOLOE26RefineMask",
            },
            "practical_detection_metadata_api.json": {"YOLOE26DetectionMetadata"},
            "practical_batch_multi_class_api.json": {
                "YOLOE26ClassMasks",
                "YOLOE26DetectionMetadata",
            },
        }

        for file_name, expected_nodes in expected_nodes_by_file.items():
            workflow = self._load_workflow(EXAMPLES_DIR / file_name)
            custom_classes = set(self._workflow_custom_node_classes(workflow))
            self.assertTrue(
                expected_nodes.issubset(custom_classes),
                f"{file_name}: missing expected nodes {expected_nodes - custom_classes}",
            )

    def test_example_workflows_include_required_node_inputs_for_high_risk_nodes(self):
        required_inputs = {
            "YOLOE26LoadModel": {"model_name"},
            "YOLOE26SelectBestInstance": {
                "instance_masks",
                "instance_metadata_json",
            },
            "PreviewImage": {"images"},
            "SaveImage": {"images", "filename_prefix"},
        }

        for workflow_path in EXAMPLES_DIR.glob("*_api*.json"):
            workflow = self._load_workflow(workflow_path)
            for node_def in workflow.values():
                class_type = node_def["class_type"]
                if class_type not in required_inputs:
                    continue
                self.assertTrue(
                    required_inputs[class_type].issubset(set(node_def["inputs"].keys())),
                    f"{workflow_path.name}: missing required inputs for {class_type}",
                )

    def test_example_workflow_matches_custom_node_interfaces(self):
        workflow = self._load_workflow(ALL_NODES_SHOWCASE_PATH)
        custom_nodes = nodes.NODE_CLASS_MAPPINGS
        builtin_nodes = {"LoadImage", "PreviewImage", "SaveImage", "MaskToImage", "PreviewAny"}

        expected_workflow_inputs = {
            "YOLOE26LoadModel": {"model_name", "device", "auto_download"},
            "YOLOE26PromptSegment": {
                "model",
                "image",
                "prompt",
                "conf",
                "iou",
                "max_det",
                "mask_threshold",
                "imgsz",
                "show_boxes",
                "show_labels",
                "show_conf",
                "show_masks",
            },
            "YOLOE26DetectionMetadata": {
                "model",
                "image",
                "prompt",
                "conf",
                "iou",
                "max_det",
                "mask_threshold",
                "imgsz",
            },
            "YOLOE26InstanceMasks": {
                "model",
                "image",
                "prompt",
                "conf",
                "iou",
                "max_det",
                "mask_threshold",
                "imgsz",
            },
            "YOLOE26ClassMasks": {
                "model",
                "image",
                "prompt",
                "conf",
                "iou",
                "max_det",
                "mask_threshold",
                "imgsz",
            },
            "YOLOE26RefineMask": {
                "masks",
                "method",
                "kernel_size",
                "iterations",
                "min_area",
                "metadata_json",
            },
            "YOLOE26SelectBestInstance": {
                "instance_masks",
                "instance_metadata_json",
                "selection_mode",
            },
        }

        for node_id, node_def in workflow.items():
            class_type = node_def["class_type"]
            self.assertTrue(class_type in custom_nodes or class_type in builtin_nodes)

            if class_type in builtin_nodes:
                continue

            node_cls = custom_nodes[class_type]
            input_types = node_cls.INPUT_TYPES()
            allowed_keys = set(input_types.get("required", {}).keys()) | set(
                input_types.get("optional", {}).keys()
            )
            self.assertTrue(set(node_def["inputs"].keys()).issubset(allowed_keys))
            if class_type in expected_workflow_inputs:
                self.assertTrue(
                    expected_workflow_inputs[class_type].issubset(
                        set(node_def["inputs"].keys())
                    )
                )

        self.assertEqual(
            nodes.YOLOE26RefineMask.RETURN_NAMES,
            ("refined_masks", "refined_metadata_json", "count"),
        )
        self.assertEqual(
            nodes.YOLOE26SelectBestInstance.RETURN_NAMES,
            ("best_mask", "best_instance_metadata_json", "selected_mask_index"),
        )

    def test_build_predict_kwargs_includes_device_only_when_not_auto(self):
        auto_kwargs = nodes._build_predict_kwargs("auto", 0.25, 640, 0.7, 300)
        self.assertEqual(
            auto_kwargs,
            {
                "conf": 0.25,
                "iou": 0.7,
                "imgsz": 640,
                "max_det": 300,
                "verbose": False,
            },
        )
        self.assertNotIn("device", auto_kwargs)

        cuda_kwargs = nodes._build_predict_kwargs("cuda:0", 0.25, 640, 0.7, 300)
        self.assertEqual(cuda_kwargs["device"], "cuda:0")

    def test_build_predict_kwargs_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            nodes._build_predict_kwargs("auto", -0.1, 640, 0.7, 300)

        with self.assertRaises(ValueError):
            nodes._build_predict_kwargs("auto", 0.25, 32, 0.7, 300)

        with self.assertRaises(ValueError):
            nodes._build_predict_kwargs("auto", 0.25, 640, 0.7, 0)

    def test_validate_mask_threshold_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            nodes._validate_mask_threshold(1.5)

        with self.assertRaises(ValueError):
            nodes._validate_mask_threshold(-0.1)

        with self.assertRaises(ValueError):
            nodes._validate_mask_threshold(float("nan"))

    def test_validate_model_bundle_requires_runtime_model_interface(self):
        class PredictOnlyModel:
            def predict(self, img_bgr, **kwargs):
                return []

        with self.assertRaises(ValueError) as exc:
            nodes._validate_model_bundle(
                {
                    "model": PredictOnlyModel(),
                    "model_path": "L:/models/yoloe-26s-seg.pt",
                    "device": "auto",
                }
            )
        self.assertIn("set_classes", str(exc.exception))

        class SetClassesOnlyModel:
            def set_classes(self, classes):
                return None

        with self.assertRaises(ValueError) as exc:
            nodes._validate_model_bundle(
                {
                    "model": SetClassesOnlyModel(),
                    "model_path": "L:/models/yoloe-26s-seg.pt",
                    "device": "auto",
                }
            )
        self.assertIn("predict", str(exc.exception))

    def test_validate_image_batch_rejects_invalid_shapes(self):
        with self.assertRaises(ValueError):
            nodes._validate_image_batch(
                torch.zeros((2, 2, 3), dtype=torch.float32), "Test Node"
            )

        with self.assertRaises(ValueError):
            nodes._validate_image_batch(
                torch.zeros((1, 2, 2, 1), dtype=torch.float32), "Test Node"
            )

    def test_extract_detection_records_returns_empty_for_missing_boxes(self):
        records = nodes._extract_detection_records(
            FakeResult(masks=None, boxes=None), ["person"], 2, 2, 0
        )
        self.assertEqual(records, [])

    def test_build_binary_mask_merges_instances(self):
        result = FakeResult(
            masks=FakeMasks(
                [
                    torch.tensor([[0.0, 0.8], [0.7, 0.1]], dtype=torch.float32),
                    torch.tensor([[0.9, 0.2], [0.0, 0.6]], dtype=torch.float32),
                ]
            )
        )

        merged = nodes._build_binary_mask(result, 2, 2)
        expected = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        self.assertTrue(np.array_equal(merged, expected))

    def test_build_binary_mask_rejects_invalid_mask_threshold(self):
        result = FakeResult(
            masks=FakeMasks(
                [
                    torch.tensor([[0.9, 0.1], [0.0, 0.0]], dtype=torch.float32),
                ]
            )
        )

        with self.assertRaises(ValueError):
            nodes._build_binary_mask(result, 2, 2, mask_threshold=1.5)

    def test_build_per_instance_masks_rejects_invalid_mask_threshold(self):
        result = FakeResult(
            masks=FakeMasks(
                [
                    torch.tensor([[0.9, 0.1], [0.0, 0.0]], dtype=torch.float32),
                ]
            )
        )

        with self.assertRaises(ValueError):
            nodes._build_per_instance_masks(result, 2, 2, mask_threshold=-0.1)

    def test_prompt_segment_node_uses_threshold_and_predict_kwargs(self):
        fake_model = FakeModel(
            [
                FakeResult(
                    masks=FakeMasks(
                        [
                            torch.tensor([[0.8, 0.6], [0.2, 0.9]], dtype=torch.float32),
                        ]
                    ),
                    boxes=FakeBoxes(
                        xyxy=torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32),
                        cls=torch.tensor([0], dtype=torch.float32),
                        conf=torch.tensor([0.9], dtype=torch.float32),
                    ),
                    plot_image=np.full((2, 2, 3), 255, dtype=np.uint8),
                )
            ]
        )
        bundle = {
            "model": fake_model,
            "model_path": "L:/models/yoloe-26s-seg.pt",
            "device": "auto",
        }
        image = torch.zeros((1, 2, 2, 3), dtype=torch.float32)

        (
            annotated_batch,
            mask_batch,
            detection_count,
        ) = nodes.YOLOE26PromptSegment().segment(
            bundle,
            image,
            "person",
            conf=0.2,
            iou=0.4,
            max_det=7,
            mask_threshold=0.75,
            imgsz=640,
        )

        self.assertEqual(detection_count, 1)
        self.assertEqual(tuple(annotated_batch.shape), (1, 2, 2, 3))
        self.assertEqual(tuple(mask_batch.shape), (1, 2, 2))
        self.assertTrue(
            np.array_equal(
                mask_batch[0].numpy(),
                np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            )
        )
        self.assertTrue(
            torch.allclose(
                annotated_batch[0], torch.ones((2, 2, 3), dtype=torch.float32)
            )
        )
        self.assertEqual(fake_model.predict_history[0]["iou"], 0.4)
        self.assertEqual(fake_model.predict_history[0]["max_det"], 7)
        self.assertNotIn("device", fake_model.predict_history[0])

    def test_extract_detection_records_returns_structured_metadata(self):
        result = FakeResult(
            masks=FakeMasks(
                [
                    torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float32),
                    torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
                ]
            ),
            boxes=FakeBoxes(
                xyxy=torch.tensor(
                    [[1.0, 2.0, 11.0, 12.0], [3.0, 4.0, 13.0, 14.0]],
                    dtype=torch.float32,
                ),
                cls=torch.tensor([0, 1], dtype=torch.float32),
                conf=torch.tensor([0.9, 0.75], dtype=torch.float32),
            ),
        )

        records = nodes._extract_detection_records(result, ["person", "car"], 2, 2, 3)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["batch_index"], 3)
        self.assertEqual(records[0]["class_name"], "person")
        self.assertEqual(records[1]["class_name"], "car")
        self.assertEqual(records[0]["mask_area"], 2)
        self.assertEqual(records[1]["bbox_xyxy"], [3.0, 4.0, 13.0, 14.0])

    def test_build_class_masks_from_records_groups_by_class(self):
        instance_masks = [
            np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
            np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32),
            np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        ]
        records = [
            {"class_id": 0, "instance_index": 0, "mask_index": 0},
            {"class_id": 0, "instance_index": 1, "mask_index": 1},
            {"class_id": 1, "instance_index": 2, "mask_index": 2},
        ]

        class_masks, entries = nodes._build_class_masks_from_records(
            records,
            instance_masks,
            ["person", "car"],
            2,
            2,
            0,
        )

        self.assertEqual(len(class_masks), 2)
        self.assertTrue(
            np.array_equal(
                class_masks[0],
                np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32),
            )
        )
        self.assertTrue(
            np.array_equal(
                class_masks[1],
                np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            )
        )
        self.assertEqual(entries[0]["source_instance_count"], 2)
        self.assertEqual(entries[1]["source_instance_count"], 1)

    def test_serialize_metadata_emits_json_string(self):
        payload = {"version": 1, "model_name": "yoloe-26s-seg.pt", "images": []}
        metadata_json = nodes._serialize_metadata(payload)
        self.assertIn('"version": 1', metadata_json)
        self.assertIn('"model_name": "yoloe-26s-seg.pt"', metadata_json)

    def test_detection_metadata_node_returns_expected_schema(self):
        fake_model = FakeModel(
            [
                FakeResult(
                    masks=FakeMasks(
                        [
                            torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32),
                        ]
                    ),
                    boxes=FakeBoxes(
                        xyxy=torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32),
                        cls=torch.tensor([0], dtype=torch.float32),
                        conf=torch.tensor([0.85], dtype=torch.float32),
                    ),
                )
            ]
        )
        bundle = {
            "model": fake_model,
            "model_path": "L:/models/yoloe-26s-seg.pt",
            "device": "auto",
        }
        image = torch.zeros((1, 2, 2, 3), dtype=torch.float32)

        metadata_json, detection_count = (
            nodes.YOLOE26DetectionMetadata().detect_metadata(bundle, image, "person")
        )

        payload = json.loads(metadata_json)
        self.assertEqual(detection_count, 1)
        self.assertEqual(payload["model_name"], "yoloe-26s-seg.pt")
        self.assertEqual(payload["total_detections"], 1)
        self.assertEqual(payload["iou"], 0.7)
        self.assertEqual(payload["max_det"], 300)
        self.assertEqual(payload["mask_threshold"], 0.5)
        self.assertEqual(payload["images"][0]["detections"][0]["class_name"], "person")

    def test_instance_masks_node_returns_zero_placeholder_when_empty(self):
        fake_model = FakeModel([FakeResult(masks=FakeMasks([]), boxes=None)])
        bundle = {
            "model": fake_model,
            "model_path": "L:/models/yoloe-26s-seg.pt",
            "device": "auto",
        }
        image = torch.zeros((1, 2, 2, 3), dtype=torch.float32)

        mask_batch, metadata_json, count = (
            nodes.YOLOE26InstanceMasks().segment_instances(bundle, image, "person")
        )

        payload = json.loads(metadata_json)
        self.assertEqual(count, 0)
        self.assertEqual(tuple(mask_batch.shape), (1, 2, 2))
        self.assertEqual(payload["images"][0]["instance_count"], 0)
        self.assertEqual(payload["images"][0]["detections"], [])
        self.assertTrue(payload["images"][0]["is_empty_result"])

    def test_instance_masks_node_returns_all_instance_masks_and_metadata_indices(self):
        fake_model = FakeModel(
            [
                FakeResult(
                    masks=FakeMasks(
                        [
                            torch.tensor([[0.9, 0.0], [0.0, 0.0]], dtype=torch.float32),
                            torch.tensor([[0.0, 0.9], [0.0, 0.0]], dtype=torch.float32),
                        ]
                    ),
                    boxes=FakeBoxes(
                        xyxy=torch.tensor(
                            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                            dtype=torch.float32,
                        ),
                        cls=torch.tensor([0, 1], dtype=torch.float32),
                        conf=torch.tensor([0.9, 0.8], dtype=torch.float32),
                    ),
                ),
                FakeResult(
                    masks=FakeMasks(
                        [
                            torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
                        ]
                    ),
                    boxes=FakeBoxes(
                        xyxy=torch.tensor(
                            [[9.0, 10.0, 11.0, 12.0]], dtype=torch.float32
                        ),
                        cls=torch.tensor([0], dtype=torch.float32),
                        conf=torch.tensor([0.7], dtype=torch.float32),
                    ),
                ),
            ]
        )
        bundle = {
            "model": fake_model,
            "model_path": "L:/models/yoloe-26s-seg.pt",
            "device": "auto",
        }
        image = torch.zeros((2, 2, 2, 3), dtype=torch.float32)

        (
            mask_batch,
            metadata_json,
            count,
        ) = nodes.YOLOE26InstanceMasks().segment_instances(
            bundle,
            image,
            "person,car",
            iou=0.6,
            max_det=5,
            mask_threshold=0.5,
            imgsz=640,
        )

        payload = json.loads(metadata_json)
        self.assertEqual(count, 3)
        self.assertEqual(tuple(mask_batch.shape), (3, 2, 2))
        self.assertEqual(payload["total_images"], 2)
        self.assertEqual(payload["total_instances"], 3)
        self.assertEqual(payload["images"][0]["instance_count"], 2)
        self.assertEqual(payload["images"][0]["output_mask_indices"], [0, 1])
        self.assertEqual(payload["images"][1]["instance_count"], 1)
        self.assertEqual(payload["images"][1]["output_mask_indices"], [2])
        self.assertEqual(payload["images"][0]["detections"][0]["output_mask_index"], 0)
        self.assertEqual(payload["images"][0]["detections"][1]["output_mask_index"], 1)
        self.assertEqual(payload["images"][1]["detections"][0]["output_mask_index"], 2)
        self.assertEqual(payload["images"][0]["detections"][0]["class_name"], "person")
        self.assertEqual(payload["images"][0]["detections"][1]["class_name"], "car")
        self.assertEqual(fake_model.predict_history[0]["iou"], 0.6)
        self.assertEqual(fake_model.predict_history[0]["max_det"], 5)
        self.assertNotIn("device", fake_model.predict_history[0])

    def test_class_masks_node_returns_zero_masks_for_each_class_when_no_detections(
        self,
    ):
        fake_model = FakeModel([FakeResult(masks=FakeMasks([]), boxes=None)])
        bundle = {
            "model": fake_model,
            "model_path": "L:/models/yoloe-26s-seg.pt",
            "device": "auto",
        }
        image = torch.zeros((1, 2, 2, 3), dtype=torch.float32)

        mask_batch, metadata_json, output_mask_count = (
            nodes.YOLOE26ClassMasks().segment_class_masks(bundle, image, "person,car")
        )

        payload = json.loads(metadata_json)
        self.assertEqual(output_mask_count, 2)
        self.assertEqual(tuple(mask_batch.shape), (2, 2, 2))
        self.assertEqual(payload["class_count"], 2)
        self.assertEqual(payload["output_mask_count"], 2)
        self.assertEqual(len(payload["entries"]), 2)
        self.assertEqual(payload["entries"][0]["source_instance_count"], 0)
        self.assertEqual(payload["entries"][1]["source_instance_count"], 0)
        self.assertEqual(payload["entries"][0]["source_instance_indices"], [])
        self.assertEqual(payload["entries"][1]["source_instance_indices"], [])
        self.assertTrue(
            torch.equal(mask_batch[0], torch.zeros((2, 2), dtype=torch.float32))
        )
        self.assertTrue(
            torch.equal(mask_batch[1], torch.zeros((2, 2), dtype=torch.float32))
        )
        self.assertNotIn("device", fake_model.predict_history[0])
        self.assertTrue(payload["is_empty_result"])

    def test_class_masks_node_merges_instances_per_class(self):
        fake_model = FakeModel(
            [
                FakeResult(
                    masks=FakeMasks(
                        [
                            torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32),
                            torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float32),
                            torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
                        ]
                    ),
                    boxes=FakeBoxes(
                        xyxy=torch.tensor(
                            [
                                [1.0, 2.0, 3.0, 4.0],
                                [5.0, 6.0, 7.0, 8.0],
                                [9.0, 10.0, 11.0, 12.0],
                            ],
                            dtype=torch.float32,
                        ),
                        cls=torch.tensor([0, 0, 1], dtype=torch.float32),
                        conf=torch.tensor([0.9, 0.85, 0.8], dtype=torch.float32),
                    ),
                )
            ]
        )
        bundle = {
            "model": fake_model,
            "model_path": "L:/models/yoloe-26s-seg.pt",
            "device": "auto",
        }
        image = torch.zeros((1, 2, 2, 3), dtype=torch.float32)

        (
            mask_batch,
            metadata_json,
            output_mask_count,
        ) = nodes.YOLOE26ClassMasks().segment_class_masks(
            bundle,
            image,
            "person,car",
            iou=0.65,
            max_det=9,
            mask_threshold=0.5,
            imgsz=640,
        )

        payload = json.loads(metadata_json)
        self.assertEqual(output_mask_count, 2)
        self.assertEqual(tuple(mask_batch.shape), (2, 2, 2))
        self.assertEqual(payload["class_count"], 2)
        self.assertEqual(payload["output_mask_count"], 2)
        self.assertEqual(len(payload["entries"]), 2)
        self.assertEqual(payload["entries"][0]["class_name"], "person")
        self.assertEqual(payload["entries"][0]["source_instance_indices"], [0, 1])
        self.assertEqual(payload["entries"][0]["source_instance_count"], 2)
        self.assertEqual(payload["entries"][1]["class_name"], "car")
        self.assertEqual(payload["entries"][1]["source_instance_indices"], [2])
        self.assertEqual(payload["entries"][1]["source_instance_count"], 1)
        self.assertTrue(
            torch.equal(
                mask_batch[0],
                torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(
                mask_batch[1],
                torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
            )
        )
        self.assertNotIn("device", fake_model.predict_history[0])
        self.assertEqual(fake_model.predict_history[0]["iou"], 0.65)
        self.assertEqual(fake_model.predict_history[0]["max_det"], 9)

    def test_fill_holes_preserves_border_touching_foreground(self):
        mask = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32
        )
        filled = nodes._fill_holes(mask)
        self.assertTrue(np.array_equal(filled, mask))

    def test_refine_mask_node_applies_largest_component_and_appends_metadata(self):
        masks = torch.tensor(
            [
                [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        metadata_json = json.dumps({"version": 1, "total_instances": 2})

        (
            refined_masks,
            refined_metadata_json,
            count,
        ) = nodes.YOLOE26RefineMask().refine_mask_batch(
            masks,
            method="largest_component",
            kernel_size=3,
            iterations=1,
            min_area=0,
            metadata_json=metadata_json,
        )

        payload = json.loads(refined_metadata_json)
        self.assertEqual(count, 2)
        self.assertEqual(tuple(refined_masks.shape), (2, 3, 3))
        self.assertTrue(
            torch.equal(
                refined_masks[0],
                torch.tensor(
                    [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    dtype=torch.float32,
                ),
            )
        )
        self.assertEqual(payload["version"], 1)
        self.assertEqual(payload["total_instances"], 2)
        self.assertEqual(payload["refinement"]["method"], "largest_component")

    def test_refine_mask_node_applies_min_area_to_threshold_method(self):
        masks = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]], dtype=torch.float32)

        (
            refined_masks,
            refined_metadata_json,
            count,
        ) = nodes.YOLOE26RefineMask().refine_mask_batch(
            masks,
            method="threshold",
            min_area=2,
        )

        payload = json.loads(refined_metadata_json)
        self.assertEqual(count, 1)
        self.assertEqual(tuple(refined_masks.shape), (1, 2, 2))
        self.assertTrue(
            torch.equal(refined_masks[0], torch.zeros((2, 2), dtype=torch.float32))
        )
        self.assertEqual(payload["refinement"]["min_area"], 2)

    def test_select_best_instance_node_returns_highest_confidence_mask(self):
        instance_masks = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 1.0], [0.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        metadata_json = json.dumps(
            {
                "version": 1,
                "images": [
                    {
                        "detections": [
                            {
                                "output_mask_index": 0,
                                "instance_index": 0,
                                "confidence": 0.7,
                                "mask_area": 1,
                                "image_height": 2,
                                "image_width": 2,
                            },
                            {
                                "output_mask_index": 1,
                                "instance_index": 1,
                                "confidence": 0.9,
                                "mask_area": 1,
                                "image_height": 2,
                                "image_width": 2,
                            },
                        ]
                    }
                ],
            }
        )

        (
            best_mask,
            best_metadata_json,
            selected_mask_index,
        ) = nodes.YOLOE26SelectBestInstance().select_best_instance(
            instance_masks,
            metadata_json,
            selection_mode="highest_confidence",
        )

        payload = json.loads(best_metadata_json)
        self.assertEqual(selected_mask_index, 1)
        self.assertEqual(tuple(best_mask.shape), (1, 2, 2))
        self.assertTrue(torch.equal(best_mask[0], instance_masks[1]))
        self.assertEqual(payload["selected_mask_index"], 1)
        self.assertEqual(payload["selection_mode"], "highest_confidence")
        self.assertEqual(payload["selected_detection"]["instance_index"], 1)

    def test_select_best_instance_node_supports_largest_area_mode(self):
        instance_masks = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[1.0, 1.0], [0.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        metadata_json = json.dumps(
            {
                "version": 1,
                "images": [
                    {
                        "detections": [
                            {
                                "output_mask_index": 0,
                                "instance_index": 0,
                                "confidence": 0.95,
                                "mask_area": 1,
                                "image_height": 2,
                                "image_width": 2,
                            },
                            {
                                "output_mask_index": 1,
                                "instance_index": 1,
                                "confidence": 0.5,
                                "mask_area": 2,
                                "image_height": 2,
                                "image_width": 2,
                            },
                        ]
                    }
                ],
            }
        )

        (
            best_mask,
            best_metadata_json,
            selected_mask_index,
        ) = nodes.YOLOE26SelectBestInstance().select_best_instance(
            instance_masks,
            metadata_json,
            selection_mode="largest_area",
        )

        payload = json.loads(best_metadata_json)
        self.assertEqual(selected_mask_index, 1)
        self.assertTrue(torch.equal(best_mask[0], instance_masks[1]))
        self.assertEqual(payload["selection_mode"], "largest_area")

    def test_select_best_instance_node_requires_instance_metadata_schema(self):
        with self.assertRaises(ValueError):
            nodes.YOLOE26SelectBestInstance().select_best_instance(
                torch.zeros((1, 2, 2), dtype=torch.float32),
                json.dumps(
                    {"version": 1, "images": [{"detections": [{"confidence": 0.9}]}]}
                ),
            )

    def test_select_best_instance_node_returns_zero_mask_when_no_detections(self):
        instance_masks = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]], dtype=torch.float32)
        metadata_json = json.dumps({"version": 1, "images": [{"detections": []}]})

        (
            best_mask,
            best_metadata_json,
            selected_mask_index,
        ) = nodes.YOLOE26SelectBestInstance().select_best_instance(
            instance_masks,
            metadata_json,
            selection_mode="highest_confidence",
        )

        payload = json.loads(best_metadata_json)
        self.assertEqual(selected_mask_index, -1)
        self.assertEqual(tuple(best_mask.shape), (1, 2, 2))
        self.assertTrue(
            torch.equal(best_mask[0], torch.zeros((2, 2), dtype=torch.float32))
        )
        self.assertEqual(payload["candidate_count"], 0)
        self.assertTrue(payload["is_empty_result"])

    def test_all_example_api_workflows_parse_and_reference_known_nodes_reference(self):
        custom_nodes = nodes.NODE_CLASS_MAPPINGS
        builtin_nodes = {"LoadImage", "PreviewImage", "SaveImage", "MaskToImage", "PreviewAny"}

        for workflow_path in EXAMPLES_DIR.glob("*_api*.json"):
            workflow = self._load_workflow(workflow_path)
            self.assertIsInstance(workflow, dict, workflow_path.name)
            self.assertTrue(workflow, workflow_path.name)

            for node_def in workflow.values():
                class_type = node_def["class_type"]
                self.assertTrue(
                    class_type in custom_nodes or class_type in builtin_nodes,
                    f"{workflow_path.name}: unknown class_type {class_type}",
                )

    def test_runnable_example_workflows_include_explicit_output_nodes(self):
        runnable_workflows = {
            "basic_api_workflow.json",
            "practical_prompt_segment_api.json",
            "practical_detection_metadata_api.json",
            "practical_refine_mask_api.json",
        }

        for file_name in runnable_workflows:
            workflow = self._load_workflow(EXAMPLES_DIR / file_name)
            self.assertTrue(
                self._workflow_has_output_node(workflow),
                f"{file_name}: missing explicit output node",
            )

    def test_reference_workflows_do_not_route_mask_outputs_to_image_sinks(self):
        mask_only_workflows = {
            "practical_best_instance_api.json": {"YOLOE26SelectBestInstance"},
            "practical_class_masks_api.json": {"YOLOE26ClassMasks"},
            "practical_batch_multi_class_api.json": {"YOLOE26ClassMasks"},
        }
        mask_output_nodes = {
            "YOLOE26ClassMasks",
            "YOLOE26DetectionMetadata",
            "YOLOE26InstanceMasks",
            "YOLOE26RefineMask",
            "YOLOE26SelectBestInstance",
        }

        for file_name, expected_nodes in mask_only_workflows.items():
            workflow = self._load_workflow(EXAMPLES_DIR / file_name)
            custom_classes = set(self._workflow_custom_node_classes(workflow))
            self.assertTrue(
                expected_nodes.issubset(custom_classes),
                f"{file_name}: missing expected mask-oriented nodes",
            )

            for node_def in workflow.values():
                if node_def["class_type"] not in {"PreviewImage", "SaveImage"}:
                    continue
                for value in node_def["inputs"].values():
                    if not (isinstance(value, list) and len(value) == 2):
                        continue
                    upstream_node_id, output_index = value
                    upstream_node = workflow[str(upstream_node_id)]
                    self.assertNotIn(
                        upstream_node["class_type"],
                        mask_output_nodes,
                        f"{file_name}: mask output routed to {node_def['class_type']}",
                    )

    def test_metadata_workflow_includes_detection_metadata_and_visualization_branch(self):
        workflow = self._load_workflow(
            EXAMPLES_DIR / "practical_detection_metadata_api.json"
        )

        metadata_nodes = self._find_nodes_by_class_type(
            workflow, "YOLOE26DetectionMetadata"
        )
        segment_nodes = self._find_nodes_by_class_type(workflow, "YOLOE26PromptSegment")
        preview_nodes = self._find_nodes_by_class_type(workflow, "PreviewImage")

        self.assertTrue(metadata_nodes)
        self.assertTrue(segment_nodes)
        self.assertTrue(preview_nodes)

        segment_node_id, _ = segment_nodes[0]
        self.assertTrue(
            any(
                self._node_references_output(node_def, segment_node_id, 0)
                for _, node_def in preview_nodes
            ),
            "practical_detection_metadata_api.json: missing visualization branch from annotated_image",
        )

    def test_all_nodes_showcase_workflow_covers_all_custom_nodes_and_uses_only_image_sinks(
        self,
    ):
        workflow = self._load_workflow(ALL_NODES_SHOWCASE_PATH)
        custom_classes = set(self._workflow_custom_node_classes(workflow))
        self.assertEqual(custom_classes, set(nodes.NODE_CLASS_MAPPINGS.keys()))

        preview_nodes = self._find_nodes_by_class_type(workflow, "PreviewImage")
        save_nodes = self._find_nodes_by_class_type(workflow, "SaveImage")
        self.assertTrue(preview_nodes)
        self.assertTrue(save_nodes)

        for _, node_def in preview_nodes + save_nodes:
            images_input = node_def["inputs"]["images"]
            self.assertIsInstance(images_input, list)
            self.assertEqual(len(images_input), 2)

            upstream_node_id = str(images_input[0])
            upstream_class = self._workflow_node_class(workflow, upstream_node_id)
            if upstream_class == "MaskToImage":
                mask_input = workflow[upstream_node_id]["inputs"]["mask"]
                self.assertIsInstance(mask_input, list)
                self.assertEqual(len(mask_input), 2)
                mask_source_class = self._workflow_node_class(
                    workflow, str(mask_input[0])
                )
                self.assertIn(
                    mask_source_class,
                    {
                        "YOLOE26ClassMasks",
                        "YOLOE26InstanceMasks",
                        "YOLOE26PromptSegment",
                        "YOLOE26RefineMask",
                        "YOLOE26SelectBestInstance",
                    },
                    "all_nodes_showcase_api.json: mask preview branch must originate from a YOLOE-26 node",
                )
                continue

            self.assertEqual(
                upstream_class,
                "YOLOE26PromptSegment",
                "all_nodes_showcase_api.json: image sinks must reference the annotated preview branch or a mask preview branch",
            )

    def test_practical_workflows_include_expected_primary_nodes_reference(self):
        expected_nodes_by_file = {
            "practical_prompt_segment_api.json": {"YOLOE26PromptSegment"},
            "practical_best_instance_api.json": {
                "YOLOE26InstanceMasks",
                "YOLOE26SelectBestInstance",
            },
            "practical_class_masks_api.json": {"YOLOE26ClassMasks"},
            "practical_refine_mask_api.json": {
                "YOLOE26PromptSegment",
                "YOLOE26RefineMask",
            },
            "practical_detection_metadata_api.json": {"YOLOE26DetectionMetadata"},
            "practical_batch_multi_class_api.json": {
                "YOLOE26ClassMasks",
                "YOLOE26DetectionMetadata",
            },
        }

        for file_name, expected_nodes in expected_nodes_by_file.items():
            workflow = self._load_workflow(EXAMPLES_DIR / file_name)
            custom_classes = set(self._workflow_custom_node_classes(workflow))
            self.assertTrue(
                expected_nodes.issubset(custom_classes),
                f"{file_name}: missing expected nodes {expected_nodes - custom_classes}",
            )

    def test_example_workflow_matches_custom_node_interfaces_reference(self):
        workflow = self._load_workflow(ALL_NODES_SHOWCASE_PATH)
        custom_nodes = nodes.NODE_CLASS_MAPPINGS
        builtin_nodes = {"LoadImage", "PreviewImage", "SaveImage", "MaskToImage", "PreviewAny"}

        expected_workflow_inputs = {
            "YOLOE26LoadModel": {"model_name", "device", "auto_download"},
            "YOLOE26PromptSegment": {
                "model",
                "image",
                "prompt",
                "conf",
                "iou",
                "max_det",
                "mask_threshold",
                "imgsz",
                "show_boxes",
                "show_labels",
                "show_conf",
                "show_masks",
            },
            "YOLOE26DetectionMetadata": {
                "model",
                "image",
                "prompt",
                "conf",
                "iou",
                "max_det",
                "mask_threshold",
                "imgsz",
            },
            "YOLOE26InstanceMasks": {
                "model",
                "image",
                "prompt",
                "conf",
                "iou",
                "max_det",
                "mask_threshold",
                "imgsz",
            },
            "YOLOE26ClassMasks": {
                "model",
                "image",
                "prompt",
                "conf",
                "iou",
                "max_det",
                "mask_threshold",
                "imgsz",
            },
            "YOLOE26RefineMask": {
                "masks",
                "method",
                "kernel_size",
                "iterations",
                "min_area",
                "metadata_json",
            },
            "YOLOE26SelectBestInstance": {
                "instance_masks",
                "instance_metadata_json",
                "selection_mode",
            },
        }

        for node_id, node_def in workflow.items():
            class_type = node_def["class_type"]
            self.assertTrue(class_type in custom_nodes or class_type in builtin_nodes)

            if class_type in builtin_nodes:
                continue

            node_cls = custom_nodes[class_type]
            input_types = node_cls.INPUT_TYPES()
            allowed_keys = set(input_types.get("required", {}).keys()) | set(
                input_types.get("optional", {}).keys()
            )
            self.assertTrue(set(node_def["inputs"].keys()).issubset(allowed_keys))
            if class_type in expected_workflow_inputs:
                self.assertTrue(
                    expected_workflow_inputs[class_type].issubset(
                        set(node_def["inputs"].keys())
                    )
                )

        self.assertEqual(
            nodes.YOLOE26RefineMask.RETURN_NAMES,
            ("refined_masks", "refined_metadata_json", "count"),
        )
        self.assertEqual(
            nodes.YOLOE26SelectBestInstance.RETURN_NAMES,
            ("best_mask", "best_instance_metadata_json", "selected_mask_index"),
        )


if __name__ == "__main__":
    unittest.main()
