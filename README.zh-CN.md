# ComfyUI-YOLOE26

[English](README.md) | [简体中文](README.zh-CN.md)

由 Ultralytics YOLOE-26 驱动、可用于 ComfyUI 的 open-vocabulary prompt segmentation 自定义节点包。

這是一個 ComfyUI custom node pack，不是獨立應用程式。

## 狀態

- 可用的 **beta / release candidate**
- helper / node-level tests 已通過
- 在更廣泛公開發布前，仍建議先做真實 ComfyUI smoke / integration 驗證
- **Local-first model loading**；automatic download 為可選，且預設關閉。啟用後，loader 會依照 Ultralytics 上游的 model-loading 行為，嘗試解析並下載指定的官方權重。

## 這個 node pack 提供什麼

這個 repository 為 ComfyUI 增加了一組精簡的 YOLOE-26 節點，讓你可以：

- 載入本地 YOLOE-26 模型一次，並在 workflow 中重複使用
- 透過 `person`、`car`、`red apple` 這類文字 prompt 做 segmentation
- 取得 merged mask、per-instance masks、或 per-class merged masks
- 保留結構化 JSON metadata，供下游 routing 或 automation 使用
- 在不重新執行 detection 的情況下，精修既有的 `MASK` batch
- 從 instance-mask 結果中選出單一最佳 instance mask

## 內含節點

| Node | 用途 | 主要輸出 |
| --- | --- | --- |
| `YOLOE-26 Load Model` | 載入並驗證本地 YOLOE-26 模型檔 | `YOLOE_MODEL` |
| `YOLOE-26 Prompt Segment` | 執行 prompt-based segmentation 並預覽結果 | `IMAGE`, `MASK`, `INT` |
| `YOLOE-26 Detection Metadata` | 以 JSON metadata 形式輸出 detections | `STRING`, `INT` |
| `YOLOE-26 Instance Masks` | 每個 detected instance 輸出一個 mask | `MASK`, `STRING`, `INT` |
| `YOLOE-26 Class Masks` | 每個 prompt class 輸出一個 merged mask | `MASK`, `STRING`, `INT` |
| `YOLOE-26 Refine Mask` | 對既有 `MASK` batch 做後處理 | `MASK`, `STRING`, `INT` |
| `YOLOE-26 Select Best Instance` | 從 instance-mask 輸出中挑出最佳單一 mask | `MASK`, `STRING`, `INT` |

## 快速開始

### 1. 安裝到 ComfyUI

把這個 repository clone 到你的 ComfyUI `custom_nodes` 目錄：

```bash
git clone https://github.com/peter119lee/ComfyUI-YOLOE26 ComfyUI-YOLOE26
```

請在 **與 ComfyUI 相同的 Python 環境** 安裝 dependencies：

```bash
pip install -r requirements.txt
```

這個 node pack 預期 host 端的 ComfyUI 環境已經提供相容的 `torch`、`numpy`、`cv2`。
為了降低與 host 環境衝突的機率，`requirements.txt` 目前只列出 `ultralytics`。

### 2. 放置本地 YOLOE-26 模型

loader 採用 local-first 設計。請把 `.pt` 檔放到以下支援路徑之一；如果你希望在本地模型缺失時由 Ultralytics 嘗試自動下載官方權重，也可以在 loader node 啟用 `auto_download`。

把 `.pt` 檔放到以下其中一個支援位置：

- `ComfyUI/models/ultralytics/segm/`
- `ComfyUI/models/ultralytics/bbox/`
- `ComfyUI/models/ultralytics/`
- `ComfyUI/models/yoloe/`

範例模型名稱：

```text
yoloe-26s-seg.pt
```

### 3. 重新啟動 ComfyUI

重啟後，請確認 node list 中能看到這 7 個 `YOLOE-26 ...` 節點。

### 4. 建立最小可用 workflow

```text
Load Image --------------------\
Prompt String -------------------> YOLOE-26 Prompt Segment -> annotated_image -> PreviewImage
YOLOE-26 Load Model ----------/                             \-> mask -> use downstream
```

建議第一輪測試參數：

- `model_name`: `yoloe-26s-seg.pt`
- `device`: `auto`
- `prompt`: `person`
- `conf`: `0.1`
- `iou`: `0.7`
- `max_det`: `300`
- `mask_threshold`: `0.5`
- `imgsz`: `640`

## 常見 workflow 模式

### 簡單 prompt segmentation

```text
Load Image --------------------\
Prompt String -------------------> YOLOE-26 Prompt Segment -> annotated_image -> PreviewImage
YOLOE-26 Load Model ----------/                             \-> mask -> use merged mask for inpainting / compositing / cropping
```

### 保留每個物件各自的 masks

```text
Load Image --------------------\
Prompt String -------------------> YOLOE-26 Instance Masks -> use instance_masks + instance_metadata_json downstream
YOLOE-26 Load Model ----------/
```

### 只保留最佳單一物件

```text
YOLOE-26 Instance Masks
  ├-> instance_masks -----------\
  └-> instance_metadata_json ----> YOLOE-26 Select Best Instance -> best_mask -> PreviewImage / SaveImage
```

### 精修既有 mask batch

```text
Any MASK-producing node
  -> YOLOE-26 Refine Mask
  -> refined_masks -> PreviewImage / SaveImage
```

### 每個 prompt class 建立一個 mask

```text
Load Image --------------------\
Prompt String -------------------> YOLOE-26 Class Masks -> class_masks -> PreviewImage / SaveImage
YOLOE-26 Load Model ----------/
```

## Prompt 格式

使用逗號分隔的文字 classes：

```text
person
person, car, dog
red apple, green bottle
```

## Node 參考

### 1. YOLOE-26 Load Model

載入並驗證本地 YOLOE-26 模型檔。
只應載入你信任的 `.pt` 檔。

**Inputs**
- `model_name`: 本地權重檔名，例如 `yoloe-26s-seg.pt`
- `device`: `auto`、`cpu`、`cuda`、`cuda:N`，以及支援時的 `mps`
- `auto_download`: `false` = 僅從支援的 ComfyUI model 目錄做本地載入；`true` = 若本地模型不存在，讓 Ultralytics 先嘗試下載指定官方模型

**Outputs**
- `YOLOE_MODEL`

### 2. YOLOE-26 Prompt Segment

執行 text-prompt segmentation，並回傳：
- annotated preview image
- merged binary mask
- detection count

**Required inputs**
- `model`
- `image`
- `prompt`

**Optional controls**
- `conf`
- `iou`
- `max_det`
- `mask_threshold`
- `imgsz`
- `show_boxes`
- `show_labels`
- `show_conf`
- `show_masks`

### 3. YOLOE-26 Detection Metadata

執行 text-prompt segmentation，並回傳：
- `metadata_json`，包含 per-image detections、boxes、scores、mask areas、class names
- `detection_count`

**Required inputs**
- `model`
- `image`
- `prompt`

**Optional controls**
- `conf`
- `iou`
- `max_det`
- `mask_threshold`
- `imgsz`

### 4. YOLOE-26 Instance Masks

執行 text-prompt segmentation，並針對 input batch 中每個 detected instance 回傳一個 mask。

**Required inputs**
- `model`
- `image`
- `prompt`

**Returns**
- `instance_masks`
- `instance_metadata_json`，只描述實際產生輸出 mask 的 detections，包含 `batch_index`、class、score、box、以及 `output_mask_index`
- `count`

如果沒有偵測到任何目標，這個 node 目前會回傳 placeholder zero mask，並回報 `count = 0`。

**Optional controls**
- `conf`
- `iou`
- `max_det`
- `mask_threshold`
- `imgsz`

### 5. YOLOE-26 Class Masks

執行 text-prompt segmentation，並為每張輸入圖片的每個 prompt class 回傳一個 merged mask。

**Required inputs**
- `model`
- `image`
- `prompt`

**Returns**
- `class_masks`
- `class_metadata_json`，描述輸出 mask 順序、來源 instances、以及 `output_mask_count`
- `output_mask_count`，代表回傳 `MASK` batch 中實際輸出的 class mask 數量

這個 node 會對每張輸入圖片的每個 prompt class 都回傳一個 mask。若某個 class 沒有被偵測到，對應 mask 會是全零。

**Optional controls**
- `conf`
- `iou`
- `max_det`
- `mask_threshold`
- `imgsz`

### 6. YOLOE-26 Refine Mask

對輸入的 `MASK` batch 做 binary post-processing，而不重新執行 detection。

**Required inputs**
- `masks`

**Returns**
- `refined_masks`
- `refined_metadata_json`，附加 `refinement` 設定
- `count`

**Optional controls**
- `method`
- `kernel_size`
- `iterations`
- `min_area`
- `metadata_json`

支援的 refinement methods：
- `threshold`
- `open`
- `close`
- `dilate`
- `erode`
- `largest_component`
- `fill_holes`

### 7. YOLOE-26 Select Best Instance

使用 instance metadata，從 `YOLOE-26 Instance Masks` 的輸出中選出單一最佳 mask。

**Required inputs**
- `instance_masks`
- `instance_metadata_json`

**Returns**
- `best_mask`
- `best_instance_metadata_json`，包含 `selection_mode`、`candidate_count`、`selected_detection`
- `selected_mask_index`

**Optional controls**
- `selection_mode`

支援的 selection modes：
- `highest_confidence`
- `largest_area`
- `confidence_then_area`

## Outputs 與 metadata

這個 node pack 使用標準 ComfyUI types：

- `IMAGE`: 帶有 YOLOE 標註的 ComfyUI image tensor
- `MASK`: ComfyUI 格式的 float mask tensor
- `INT`: 依節點不同，代表 detection count、output mask count、或 selected mask index
- `STRING`: JSON metadata，可用於 detections、instance-mask indexing、class-mask indexing、refinement settings、或 best-instance selection

metadata 輸出是 plain JSON strings，因此容易檢視、保存、或在下游 workflow 中解析。
如果你要在外部 UI 或 web view 中渲染它們，請把它們視為 untrusted data。

注意：

- 對 `YOLOE-26 Class Masks` 而言，metadata 裡的 `class_count` 是 prompt classes 的數量，而 `output_mask_count` 是 `MASK` batch 中實際回傳的 masks 數量。
- 對 `YOLOE-26 Select Best Instance` 而言，這個 node 預期接收由 `YOLOE-26 Instance Masks` 產生的 `instance_metadata_json`，以確保 `output_mask_index` 與 `instance_masks` batch 對齊。

## Example workflows

`examples/` 中的所有 example 檔案，都是 **ComfyUI API-format workflows**，使用精簡的 `class_type` + `inputs` prompt 風格。
它們不是完整的 UI-exported graph JSON 檔。
如果你是透過 ComfyUI API 測試，請把它們當成 payload references 使用。
如果你只是在 UI 中測試，請手動重建相同的 nodes 與參數。

Example workflows 分成兩類：

- **Runnable examples**：目前已確認有合法 `IMAGE` sink，可以直接做 smoke test
- **Reference/API examples**：展示 node wiring 與 API payload；不保證每個 `MASK` 輸出都有 repo 內已確認的直接 preview/save sink

### 1. 最小入門範例

建議先從：

- `examples/basic_api_workflow.json`

這是 first-run smoke test 的最小成功路徑：

```text
LoadImage -> YOLOE26LoadModel -> YOLOE26PromptSegment -> PreviewImage
```

當你只想確認 model loading、prompt segmentation、annotated-image preview、以及 merged-mask output 能完整串起來時，這是建議的 Quick Start workflow。

### 2. 完整 node-pack showcase

如果你想看涵蓋整個 custom node pack 的 wiring reference，請使用：

- `examples/all_nodes_showcase_api.json`

這個 workflow 包含全部 7 個 custom nodes：

- `YOLOE26LoadModel`
- `YOLOE26PromptSegment`
- `YOLOE26DetectionMetadata`
- `YOLOE26InstanceMasks`
- `YOLOE26ClassMasks`
- `YOLOE26RefineMask`
- `YOLOE26SelectBestInstance`

它也包含代表性的 built-in output sinks，但只保留 annotated image preview 的支線。
如果你想要一份能示範完整 node set 與典型 downstream wiring 的 API workflow，這份最適合。

### 3. 實用應用 workflows

這些 practical examples 聚焦在常見的 downstream use cases：

| Example | 用途 | 主要節點 | 預期輸出 |
| --- | --- | --- | --- |
| `examples/basic_api_workflow.json` | Quick Start 與首次 smoke test | `YOLOE26LoadModel`, `YOLOE26PromptSegment`, `PreviewImage` | 來自 `annotated_image` 的 annotated preview image |
| `examples/all_nodes_showcase_api.json` | 整個 node pack 的完整 wiring reference | 全部 7 個 custom nodes，加上代表性的 `PreviewImage` / `SaveImage` sinks，但只用於 annotated image 分支 | Prompt-segment preview，加上完整 node coverage |
| `examples/practical_prompt_segment_api.json` | 適合 inpainting、compositing、cropping 的基本 prompt segmentation | `YOLOE26PromptSegment`, `PreviewImage`, `SaveImage` | 立即可見的 annotated preview 與已保存的 prompt-segment 結果 |
| `examples/practical_best_instance_api.json` | API/reference workflow：從 per-instance 結果中保留最佳單一目標 | `YOLOE26InstanceMasks`, `YOLOE26SelectBestInstance` | `best_mask`、`best_instance_metadata_json`、`selected_mask_index` |
| `examples/practical_class_masks_api.json` | API/reference workflow：依 prompt class 做分流 | `YOLOE26ClassMasks` | `class_masks`、`class_metadata_json`、`output_mask_count` |
| `examples/practical_refine_mask_api.json` | 在不重跑 detection 的情況下後處理既有 mask batch | `YOLOE26PromptSegment`, `YOLOE26RefineMask`, `PreviewImage` | annotated preview，以及 `refined_masks` / `refined_metadata_json` |
| `examples/practical_detection_metadata_api.json` | 以 metadata 驅動 automation，並保留可視 smoke-test branch | `YOLOE26DetectionMetadata`, `YOLOE26PromptSegment`, `PreviewImage` | `metadata_json` 加上 annotated preview output |
| `examples/practical_batch_multi_class_api.json` | API/reference workflow：多類別 prompt routing 與 metadata 對齊檢查 | `YOLOE26ClassMasks`, `YOLOE26DetectionMetadata` | class-mask batch 與 detection metadata 的對照 |

`basic_api_workflow.json` 是建議的 Quick Start 入口。
如果你想在同一處檢查所有 custom nodes，請看 `all_nodes_showcase_api.json`。

## 驗證與發布文件

如需實際 setup、smoke testing、與 release checks，請參考：

- `SMOKE_TEST_GUIDE.md`
- `RELEASE_CHECKLIST_SHORT.md`
- `TODO_RELEASE_AND_USAGE.md`

## 目前限制

- loader 目前仍只接受 `.pt` 模型名稱，並以 local-first 方式從支援的 ComfyUI model 目錄解析
- automatic model download 依賴 Ultralytics 上游模型可用性、網路連線、當地寫入權限，以及上游對官方 YOLOE-26 權重的當前解析行為
- metadata outputs 目前是 JSON strings，而不是自訂的 ComfyUI structured type
- `YOLOE-26 Instance Masks` 在無 detections 時會回傳 placeholder zero mask，並回報 `count = 0`
- `YOLOE-26 Class Masks` 會對每張輸入圖片的每個 prompt class 都回傳一個 mask；若 class 未被偵測到，mask 會是全零
- 在更廣泛公開發布前，repo 仍建議先在真實 ComfyUI workflows 中完成驗證

## 疑難排解

- 如果 model loading 失敗，請先確認 `.pt` 檔是否放在支援的 ComfyUI model 目錄之一。
- 如果已啟用 `auto_download` 但仍載入失敗，請確認 Ultralytics 是否能連到其模型來源、環境是否能把下載的權重寫入 cache 目錄，以及你指定的模型名稱是否受目前安裝的 Ultralytics 版本支援。
- 如果升級 Ultralytics 後 inference 失敗，請確認目前安裝版本仍提供 `from ultralytics import YOLOE`。
- 這個 node pack 目前目標是 `ultralytics>=8.3.200` API family；若上游 YOLOE API 改變，可能需要同步更新。
- 如果你只需要 mask，可以忽略 JSON outputs。
- 對 batch inputs，請檢查回傳的 JSON metadata，以便把每個 output mask 對應回來源圖片。

## Attribution status

這個 repository 參考了與 `prompt_segment.py` script 相關的實作想法，該 script 歸因於 [spawner1145](https://github.com/spawner1145)。
目前尚未從這個 repo 獨立驗證其確切上游 repository、檔案 URL、與 license。
在 provenance 確認前，這個 repository **不**提供 project-level `LICENSE` 檔。
[Ultralytics](https://github.com/ultralytics/ultralytics) 仍是獨立的 third-party dependency，並有其自身 license terms。
