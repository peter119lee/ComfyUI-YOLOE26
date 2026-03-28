# ComfyUI-YOLOE26 Smoke Test Guide

這份文件是給 maintainer 或測試者在真機 ComfyUI 環境做快速驗證用的操作指南。
它描述的是目前仍需補齊的 real ComfyUI smoke / integration 驗證，**不代表** repo 已經具備對外發布所需的 smoke evidence。
目前較可靠的 evidence 仍以 code-level tests 與 workflow-contract checks 為主；在有人完成並記錄真機 ComfyUI 執行結果前，請把 repo 視為可評估 beta，而不是 public-release-ready。
目標不是做完整 benchmark，而是確認這個 node pack 在實際工作流中可正常載入、推理、輸出 mask 與 metadata。

---

## 1. 測試前準備

在開始前，先確認下面幾件事：

- [ ] `ComfyUI-YOLOE26` 已放在 `ComfyUI/custom_nodes/`
- [ ] 已在 ComfyUI 相同的 Python 環境執行：
  ```bash
  pip install -r requirements.txt
  ```
- [ ] 本機環境已有可用的 `torch`
- [ ] 本機環境已有可用的 `numpy`
- [ ] 本機環境已有可用的 `cv2`
- [ ] `yoloe-26s-seg.pt` 或其他 YOLOE-26 權重已放在支援目錄，或你明確要驗證 allowlisted auto-download
- [ ] ComfyUI input 目錄中已有至少一張測試圖，例如 `example.png`

支援模型路徑：

- `ComfyUI/models/ultralytics/segm/`
- `ComfyUI/models/ultralytics/bbox/`
- `ComfyUI/models/ultralytics/`
- `ComfyUI/models/yoloe/`

auto-download 限制也要先確認：

- 目前只支援 allowlisted 的官方 segmentation 權重：`yoloe-26n-seg.pt`、`yoloe-26s-seg.pt`、`yoloe-26m-seg.pt`、`yoloe-26l-seg.pt`、`yoloe-26x-seg.pt`
- 其他名稱、非 segmentation 權重、或重新命名後的檔案，不屬於目前的 auto-download contract
- example JSON 中若出現 `(downloadable)` 標籤，只代表下拉選單選項可顯示；不代表已完成真機下載 smoke 驗證

---

## 2. 啟動驗證

重新啟動 ComfyUI 後，先做載入檢查：

- [ ] 啟動時沒有 import error
- [ ] custom node 成功載入
- [ ] 節點清單中可看到以下節點：
  - [ ] `YOLOE-26 Load Model`
  - [ ] `YOLOE-26 Prompt Segment`
  - [ ] `YOLOE-26 Detection Metadata`
  - [ ] `YOLOE-26 Instance Masks`
  - [ ] `YOLOE-26 Class Masks`
  - [ ] `YOLOE-26 Refine Mask`
  - [ ] `YOLOE-26 Select Best Instance`

如果這一步失敗，先不要往下測推理。
先回頭確認 dependency、模型、以及 ComfyUI 啟動 log。

---

## 3. 最小 UI smoke test

先做最小成功路徑。

### 節點流程

```text
Load Image --------------------\
                                -> YOLOE-26 Prompt Segment -> PreviewImage
YOLOE-26 Load Model ----------/                           \-> mask -> downstream or MaskToImage
```

### 建議參數

#### YOLOE-26 Load Model
- `model_name`: `yoloe-26s-seg.pt`
- `device`: `auto`
- `auto_download`: `false`（除非你這次是刻意驗證 allowlisted auto-download）

#### YOLOE-26 Prompt Segment
- `prompt`: `person`
- `conf`: `0.1`
- `iou`: `0.7`
- `max_det`: `300`
- `mask_threshold`: `0.5`
- `imgsz`: `640`
- `show_boxes`: `true`
- `show_labels`: `true`
- `show_conf`: `true`
- `show_masks`: `true`

### 預期結果

- [ ] `YOLOE-26 Load Model` 成功載入模型
- [ ] `YOLOE-26 Prompt Segment` 成功執行
- [ ] 有輸出 annotated `IMAGE`
- [ ] 有輸出 merged `MASK`
- [ ] 有輸出 detection `INT`
- [ ] 畫面上的標註與 prompt 大致相符

### 變更參數再驗一次

至少再改下面幾個值各跑一次：

- [ ] 把 `conf` 提高，確認 detection 數量可能下降
- [ ] 把 `mask_threshold` 提高，確認 mask 面積可能縮小
- [ ] 改 `prompt` 為多類別，例如 `person, car, dog`

### 建議留證

本輪 smoke 建議至少保存：

- [ ] ComfyUI 啟動成功截圖或 log 片段
- [ ] 最小 workflow 執行成功截圖
- [ ] 使用的 ComfyUI / Python / torch / ultralytics 版本資訊
- [ ] 使用的是本地模型還是 allowlisted auto-download

---

## 4. Example workflow smoke plan

repo 內附的 examples 都是 **API-format workflows**，不是完整的 UI-exported graph。
在目前 release docs 的證據等級下，請把它們分成「待執行的 smoke target」與「wiring/reference example」，不要把任何單一 example 直接當成已完成的 public-release proof。

### 4.1 Minimal smoke target

- `examples/basic_api_workflow.json`

這是最小 smoke target，覆蓋 4 個節點：

```text
LoadImage -> YOLOE26LoadModel -> YOLOE26PromptSegment -> PreviewImage
```

執行後確認：

- [ ] workflow 可跑完
- [ ] `Prompt Segment` 有輸出 annotated `IMAGE`
- [ ] `Prompt Segment` 有輸出 merged `MASK`
- [ ] `iou` / `max_det` / `mask_threshold` 參數有被正確接受

### 4.2 All-nodes wiring reference

- `examples/all_nodes_showcase_api.json`

這是完整 node-pack wiring reference，應覆蓋全部 7 個 custom nodes：

- [ ] `YOLOE26LoadModel`
- [ ] `YOLOE26PromptSegment`
- [ ] `YOLOE26DetectionMetadata`
- [ ] `YOLOE26InstanceMasks`
- [ ] `YOLOE26ClassMasks`
- [ ] `YOLOE26RefineMask`
- [ ] `YOLOE26SelectBestInstance`

這份 workflow 的用途是核對 node coverage 與合法 wiring。
它實際上也包含 `MaskToImage` 視覺化分支，所以不需要再聲稱所有 `MASK` 都不能在 repo 內接到 image sink；比較準確的說法是，這份範例的重點不是證明每個分支都已完成 release-grade smoke，而是提供 coverage-oriented wiring reference。
執行後確認：

- [ ] `Prompt Segment` 的 annotated `IMAGE` branch 可用
- [ ] `Detection Metadata` 有輸出 `metadata_json`
- [ ] `Instance Masks` 有輸出 `instance_masks`
- [ ] `Class Masks` 有輸出 `class_masks`
- [ ] `Refine Mask` 有輸出 `refined_masks`
- [ ] `Select Best Instance` 有輸出 `best_mask`
- [ ] `MaskToImage` 視覺化分支能正確把 mask 轉成可預覽的 `IMAGE`

### 4.3 Practical workflow set

下列 practical examples 建議至少抽測 2 份；在真實證據補齊前，請把它們視為 smoke targets 或 wiring references，而不是已完成的對外 showcase：

| file name | scenario focus | current doc role |
| --- | --- | --- |
| `practical_prompt_segment_api.json` | merged mask for inpainting / cropping / compositing | smoke target for annotated-image path |
| `practical_best_instance_api.json` | best-subject selection with mask visualization via `MaskToImage` | reference workflow pending real ComfyUI smoke evidence |
| `practical_class_masks_api.json` | class-wise routing with mask visualization via `MaskToImage` | reference workflow pending real ComfyUI smoke evidence |
| `practical_refine_mask_api.json` | post-process masks while keeping visible preview branches | smoke-target/reference hybrid |
| `practical_detection_metadata_api.json` | metadata-driven automation plus annotated preview | smoke target for metadata + preview |
| `practical_batch_multi_class_api.json` | multi-class prompt + metadata alignment checks with `MaskToImage` | reference workflow pending real ComfyUI smoke evidence |

每份 workflow 至少確認其主節點輸出可用，且用途與檔名相符。
若你需要可視化 `MASK`，請確認 workflow 經由 `MaskToImage` 或你環境中已驗證相容的 downstream converter，而不是直接接 `PreviewImage` / `SaveImage`。

### 執行前確認

- [ ] `example.png` 已放到 ComfyUI input 目錄
- [ ] `yoloe-26s-seg.pt` 已放到支援模型路徑，或你這次是刻意驗證 allowlisted auto-download
- [ ] ComfyUI 可正常讀取這些檔案
- [ ] 理解這些 JSON 是 API payload 參考，不是 UI graph 匯入檔

---

## 5. 七個節點快速驗證重點

### 5.1 YOLOE-26 Load Model

- [ ] 存在的模型名稱可以載入
- [ ] 不存在的模型名稱會報清楚錯誤
- [ ] `device=auto` 可用
- [ ] 如果有 GPU，`cuda` 或 `cuda:0` 可用
- [ ] 如果是 Apple Silicon，`mps` 可用
- [ ] 若測 `auto_download=true`，只有 allowlisted 官方 segmentation 權重會被接受

### 5.2 YOLOE-26 Prompt Segment

- [ ] 單一 prompt 可用
- [ ] 多個 prompt 可用
- [ ] `show_*` 開關能影響預覽輸出
- [ ] 空 prompt 會報清楚錯誤

### 5.3 YOLOE-26 Detection Metadata

- [ ] `metadata_json` 可輸出
- [ ] JSON 內可看到實際欄位，例如 `class_name` / `confidence` / `bbox_xyxy` / `mask_area` / `batch_index`
- [ ] JSON 內有 inference controls：`conf` / `iou` / `max_det` / `mask_threshold` / `imgsz`

### 5.4 YOLOE-26 Instance Masks

- [ ] 每個 detection 都能對應到輸出 mask
- [ ] `output_mask_index` 與輸出順序一致
- [ ] 無 detection 時目前行為可接受
  - 目前設計：placeholder zero mask，`count = 0`

### 5.5 YOLOE-26 Class Masks

- [ ] 每個 prompt class 都有對應輸出 mask
- [ ] 同一 class 的多個 instance 會 merge
- [ ] 未偵測到 class 時仍回傳零 mask
- [ ] 理解 `output_mask_count` 才是輸出 mask batch 的數量

### 5.6 YOLOE-26 Refine Mask

- [ ] 能正確接收既有 `MASK` batch
- [ ] `threshold` 可用
- [ ] `open` / `close` / `dilate` / `erode` 可用
- [ ] `largest_component` 可用
- [ ] `fill_holes` 可用
- [ ] `refined_metadata_json` 會附加 `refinement`

### 5.7 YOLOE-26 Select Best Instance

- [ ] 使用來自 `YOLOE-26 Instance Masks` 的 `instance_metadata_json`
- [ ] `highest_confidence` 可用
- [ ] `largest_area` 可用
- [ ] `confidence_then_area` 可用
- [ ] `selected_mask_index` 與 metadata 一致

---

## 6. 建議再補的真實場景驗證

如果最小 smoke test 過了，再做下面這些：

- [ ] 單張圖片
- [ ] batch 圖片
- [ ] 高解析度圖片
- [ ] 沒有任何可偵測目標的圖片
- [ ] 多類別 prompt
- [ ] 接 downstream 節點：
  - [ ] inpainting
  - [ ] compositing
  - [ ] cropping
  - [ ] region control
- [ ] 驗證 JSON metadata 能被保存或下游解析

---

## 7. 常見失敗情境

如果 smoke test 失敗，優先檢查：

- [ ] 模型檔是否真的存在
- [ ] 模型檔名是否輸入正確
- [ ] `ultralytics` 版本是否符合 `>=8.3.200,<8.5.0`
- [ ] `from ultralytics import YOLOE` 是否仍可用
- [ ] ComfyUI 使用的 Python 環境是否正確
- [ ] prompt 是否為空字串或只有逗號
- [ ] 是否選了不可用的 `device`
- [ ] GPU 雖可見，但實際推理是否失敗
- [ ] 若測 `auto_download`，是否使用 allowlisted 官方 segmentation 權重
- [ ] 若測 `auto_download`，Ultralytics 是否仍能解析到相同上游 asset

---

## 8. 通過標準

如果下面項目都成立，可以視為這一輪 smoke evidence 已補齊到可交付狀態：

- [ ] 7 個節點都能載入
- [ ] 最小 UI 流程可跑
- [ ] `basic_api_workflow.json` 可跑
- [ ] `practical_prompt_segment_api.json` 或 `practical_detection_metadata_api.json` 至少一份可跑
- [ ] `all_nodes_showcase_api.json` 已核對 wiring，且 mask preview branch 經由 `MaskToImage`
- [ ] `Prompt Segment` 可輸出 image / mask / count
- [ ] `Detection Metadata` 可輸出可解析 JSON
- [ ] `Instance Masks` / `Class Masks` 行為符合預期
- [ ] `Refine Mask` 與 `Select Best Instance` 的 contract 可在 workflow 內驗證
- [ ] 已保存足夠的截圖 / log / version info 作為真實 ComfyUI 證據

如果這些條目尚未全部打勾，就不應在 release 文件中把 smoke 描述成已完成。

下一步請看：

- `RELEASE_CHECKLIST_SHORT.md`
- `TODO_RELEASE_AND_USAGE.md`
