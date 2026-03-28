# ComfyUI-YOLOE26

[English](README.md) | [简体中文](README.zh-CN.md)

由 Ultralytics YOLOE-26 驱动、可用于 ComfyUI 的 open-vocabulary prompt segmentation 自定义节点包。

這是一個 ComfyUI custom node pack，不是獨立應用程式。

## 状态

- 公開發布
- 所有 node-level tests 已通過
- 已在真實 ComfyUI 環境中完成端對端 smoke 驗證
- **測試基準：** ComfyUI 0.18.1 · Python 3.12.7 · PyTorch 2.7.1+cu118 · ultralytics 8.3.207
- **Local-first model loading**；automatic download 為可選，且默認關閉。啟用後，目前只支持 allowlisted 的官方 YOLOE-26 segmentation 權重，實際解析與下載仍受 Ultralytics 當前上游 asset 命名與下載行為影響。
- 模型選擇器是下拉框：只有當對應 `.pt` 文件已經存在於受支持的 ComfyUI model 目錄中時，選項才會顯示為 `(local)`；以 `(downloadable)` 結尾的選項表示這是只有在啟用 `auto_download` 後才會嘗試抓取的官方 segmentation 預設。example workflow JSON 現在不再強制寫死 `(downloadable)` 標籤。

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

这個 node pack 预期 host 端的 ComfyUI 环境已经提供相容的 `torch`、`numpy`、`cv2`。
为了降低与 host 环境冲突的概率，`requirements.txt` 目前只列出 `ultralytics`。

### 2. 放置本地 YOLOE-26 模型

loader 采用 local-first 设计。请使用 `YOLOE-26 Load Model` 的下拉选单：只有当对应 `.pt` 文件已经存在于下列受支持的 ComfyUI model 目录中时，选项才会显示为 `(local)`；以 `(downloadable)` 结尾的选项表示这是官方 segmentation 预设，启用 `auto_download` 后才会尝试下载。

> **首次自动下载说明：** 当你第一次使用 `(downloadable)` 模型执行 workflow 时，模型会自动下载并且 workflow 可以成功运行。但下拉选单的标签在重启 ComfyUI 之前仍会显示 `(downloadable)`。重启后（按 **F5** 或重新启动服务器），相同的模型会显示为 `(local)`。**第一次运行不需要重启**，workflow 可以直接跑通。

当前 auto-download 限制：

- 目前只支持 allowlisted 的官方 segmentation 权重：`yoloe-26n-seg.pt`、`yoloe-26s-seg.pt`、`yoloe-26m-seg.pt`、`yoloe-26l-seg.pt`、`yoloe-26x-seg.pt`
- 非 allowlisted 名称、非 segmentation 预设、或重新命名后的文件，不属于当前 auto-download contract
- 是否能成功下载，也取决于 Ultralytics 上游是否继续以相同方式解析这些官方 asset 名称

把 `.pt` 文件放到以下其中一个支持位置：

- `ComfyUI/models/ultralytics/segm/`
- `ComfyUI/models/ultralytics/bbox/`
- `ComfyUI/models/ultralytics/`
- `ComfyUI/models/yoloe/`

示例模型名称：

```text
yoloe-26s-seg.pt
```

### 3. 重新启动 ComfyUI

重啟後，請確認 node list 中能看到這 7 個 `YOLOE-26 ...` 節點。

### 4. 建立最小可用 workflow

```text
Load Image --------------------\
Prompt String -------------------> YOLOE-26 Prompt Segment -> annotated_image -> PreviewImage
YOLOE-26 Load Model ----------/                             \-> mask -> use downstream
```

这个 `mask` 输出是给其他支持 MASK 的节点使用的。如果你要接 image sink，请先用 mask-to-image 节点做转换。

建议第一轮测试参数：

- `model_name`: `yoloe-26s-seg.pt`
- `device`: `auto`
- `prompt`: `person`
- `conf`: `0.1`
- `iou`: `0.7`
- `max_det`: `300`
- `mask_threshold`: `0.5`
- `imgsz`: `640`

## 常见 workflow 模式

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
  └-> instance_metadata_json ----> YOLOE-26 Select Best Instance -> best_mask -> MaskToImage -> PreviewImage / SaveImage
```

### 精修既有 mask batch

```text
Any MASK-producing node
  -> YOLOE-26 Refine Mask
  -> refined_masks -> MaskToImage -> PreviewImage / SaveImage
```

### 每個 prompt class 建立一個 mask

```text
Load Image --------------------\
Prompt String -------------------> YOLOE-26 Class Masks -> class_masks -> MaskToImage -> PreviewImage / SaveImage
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

载入并验证本地 YOLOE-26 模型文件。
只应载入你信任的 `.pt` 文件。
模型下拉选单会把官方预设显示为 `(downloadable)`，把已经存在于受支持 ComfyUI model 目录中的文件显示为 `(local)`。如果你明明有本地模型，却仍只看到 `(downloadable)`，请先确认该权重是不是放在支持的 ComfyUI model 目录，而不是 workspace 其他位置。

**Inputs**
- `model_name`: 本地权重文件名，例如 `yoloe-26s-seg.pt`
- `device`: `auto`、`cpu`、`cuda`、`cuda:N`，以及支持时的 `mps`
- `auto_download`: `false` = 仅从受支持的 ComfyUI model 目录做本地载入；`true` = 如果本地模型不存在，让 Ultralytics 先尝试下载请求的 allowlisted 官方 segmentation 模型

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

`examples/` 里的 example 文件都是 **ComfyUI API-format workflows**，使用精简的 `class_type` + `inputs` prompt 风格。
它们不是完整的 UI-exported graph JSON 文件。
如果你是通过 ComfyUI API 测试，请把它们当作 payload reference 使用。
如果你只是在 UI 中测试，请手动重建相同的 nodes 与参数。

以当前证据等级来看，建议这样理解这些 examples：

- **Smoke-target examples**：graph 形状符合 repo 当前预期的 smoke path 或 visualization path
- **Reference examples**：用于说明 node wiring 与 downstream usage pattern；在真实 ComfyUI 里跑出 release-grade 证据之前，不应把它们当成已验证完成的公开发布凭据

### 1. 最小入门范例

建议先从：

- `examples/basic_api_workflow.json`

这是 first-run smoke attempt 的最小文档化路径：

```text
LoadImage -> YOLOE26LoadModel -> YOLOE26PromptSegment -> PreviewImage
```

当你只想确认 model loading、prompt segmentation、annotated-image preview、以及 merged-mask output 的 wiring 是否完整时，这是建议的 Quick Start workflow。
但它目前仍应视为要实际执行的 smoke target，而不是已经留档完成的 public-release 证据。

### 2. 完整 node-pack showcase

如果你想看覆盖整个 custom node pack 的 wiring reference，请使用：

- `examples/all_nodes_showcase_api.json`

这个 workflow 包含全部 7 个 custom nodes：

- `YOLOE26LoadModel`
- `YOLOE26PromptSegment`
- `YOLOE26DetectionMetadata`
- `YOLOE26InstanceMasks`
- `YOLOE26ClassMasks`
- `YOLOE26RefineMask`
- `YOLOE26SelectBestInstance`

它也包含代表性的 built-in sinks 与 preview helpers，其中包括用 `MaskToImage` 做 mask 可视化的分支。
请把它当作面向 coverage 的 wiring reference，而不是证明所有分支都已在真实 ComfyUI session 中 smoke-validated 的证据。

### 3. 实用 workflows

这些 practical examples 聚焦在常见的 downstream use cases：

| Example | 用途 | 主要节点 | 当前证据等级 |
| --- | --- | --- | --- |
| `examples/basic_api_workflow.json` | Quick Start 与首次 smoke target | `YOLOE26LoadModel`, `YOLOE26PromptSegment`, `PreviewImage` | 已定义 smoke target；release docs 中仍待补真实 ComfyUI evidence |
| `examples/all_nodes_showcase_api.json` | 整个 node pack 的完整 wiring reference | 全部 7 个 custom nodes，加上代表性的 preview/save helpers 与 `MaskToImage` 可视化分支 | wiring/reference example，不是 release-proof smoke evidence |
| `examples/practical_prompt_segment_api.json` | 适合 inpainting、compositing、cropping 的基础 prompt segmentation | `YOLOE26PromptSegment`, `PreviewImage`, `SaveImage` | 面向 annotated-image output 的 smoke-target example |
| `examples/practical_best_instance_api.json` | 通过 `MaskToImage` 可视化最佳单一实例选择 | `YOLOE26InstanceMasks`, `YOLOE26SelectBestInstance`, `MaskToImage` | reference workflow，仍待真实 ComfyUI smoke 确认 |
| `examples/practical_class_masks_api.json` | 通过 `MaskToImage` 可视化 class mask 路由 | `YOLOE26ClassMasks`, `MaskToImage` | reference workflow，仍待真实 ComfyUI smoke 确认 |
| `examples/practical_refine_mask_api.json` | 在不重跑 detection 的情况下后处理现有 mask batch | `YOLOE26PromptSegment`, `YOLOE26RefineMask`, `MaskToImage`, `PreviewImage` | smoke-target/reference hybrid；仍待补已留档的真实 ComfyUI evidence |
| `examples/practical_detection_metadata_api.json` | metadata-driven automation，并保留可见的 annotated preview 分支 | `YOLOE26DetectionMetadata`, `YOLOE26PromptSegment`, `PreviewImage` | 面向 metadata + preview 的 smoke-target example |
| `examples/practical_batch_multi_class_api.json` | 多类别 prompt routing 与 metadata alignment 检查，并通过 `MaskToImage` 可视化 | `YOLOE26ClassMasks`, `YOLOE26DetectionMetadata`, `MaskToImage` | reference workflow，仍待真实 ComfyUI smoke 确认 |

`basic_api_workflow.json` 仍是建议的 Quick Start 入口。
如果你想做最快的 sanity smoke，请先跑它。
如果你想用尽量少的 workflow 覆盖整个 custom node pack，请再跑 `all_nodes_showcase_api.json`。

## 驗證與發布文件

如需實際 setup、smoke testing、與 release checks，請參考：

- `SMOKE_TEST_GUIDE.md`
- `RELEASE_CHECKLIST_SHORT.md`
- `TODO_RELEASE_AND_USAGE.md`

Workflow 匯入提示：
- example workflows 使用的是 ComfyUI 的 `LoadImage` 節點，所以匯入後仍可能需要你在 ComfyUI UI 內重新指定或選擇輸入圖片。
- `basic_api_workflow.json` 現在已移除明確的 `upload` flag，但 `LoadImage` 節點本質上仍代表使用者提供的圖片輸入，而不是 repo 內已打包好的固定資產。

## 目前限制

- loader 目前仍只接受 `.pt` 模型名稱，並以 local-first 方式從支援的 ComfyUI model 目錄解析
- automatic model download 目前只支援 allowlisted 的官方 YOLOE-26 segmentation 權重，並依賴 Ultralytics 上游模型可用性、網路連線、本地寫入權限，以及這些官方 asset 當前的解析行為
- metadata outputs 目前是 JSON strings，而不是自訂的 ComfyUI structured type
- `YOLOE-26 Instance Masks` 在無 detections 時會回傳 placeholder zero mask，並回報 `count = 0`
- `YOLOE-26 Class Masks` 會對每張輸入圖片的每個 prompt class 都回傳一個 mask；若 class 未被偵測到，mask 會是全零
- 已在真實 ComfyUI 環境（ComfyUI 0.18.1 · Python 3.12.7 · PyTorch 2.7.1+cu118 · ultralytics 8.3.207）完成 smoke 驗證

## 疑難排解

- 如果 model loading 失敗，請先確認 `.pt` 檔是否放在支援的 ComfyUI model 目錄之一。
- 如果已啟用 `auto_download` 但仍載入失敗，請確認 Ultralytics 是否能連到其模型來源、環境是否能把下載的權重寫入 cache 目錄，以及你指定的模型名稱是否受目前安裝的 Ultralytics 版本支援。
- 如果升級 Ultralytics 後 inference 失敗，請確認目前安裝版本仍提供 `from ultralytics import YOLOE`。
- 這個 node pack 目前目標是 `ultralytics>=8.3.200` API family；若上游 YOLOE API 改變，可能需要同步更新。
- 如果你只需要 mask，可以忽略 JSON outputs。
- 對 batch inputs，請檢查回傳的 JSON metadata，以便把每個 output mask 對應回來源圖片。

## Attribution and licensing status

這個 repository 參考了與 `prompt_segment.py` script 相關的實作想法，該 script 由 [spawner1145](https://github.com/spawner1145) 撰寫，並已獲得其明確授權使用。

這個 repository 現在已提供 project-level `LICENSE`，用於本 repo 內的原創 source code 與 documentation。
該 MIT license **只**適用於本 repository 內的原始程式碼與文件；它**不會**重新授權 third-party dependencies、model weights、downloaded assets、或任何 upstream project。
[Ultralytics](https://github.com/ultralytics/ultralytics) 仍是獨立的 third-party dependency，並有其自身 license terms；官方 YOLOE / Ultralytics model weights 也仍受其上游條款約束。
