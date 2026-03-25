# ComfyUI-YOLOE26 Smoke Test Guide

這份文件是給 maintainer 或測試者在真機 ComfyUI 環境做快速驗證用的操作指南。
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
- [ ] `yoloe-26s-seg.pt` 或其他 YOLOE-26 權重已放在支援目錄
- [ ] ComfyUI input 目錄中已有至少一張測試圖，例如 `example.png`

支援模型路徑：

- `ComfyUI/models/ultralytics/segm/`
- `ComfyUI/models/ultralytics/bbox/`
- `ComfyUI/models/ultralytics/`
- `ComfyUI/models/yoloe/`

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
YOLOE-26 Load Model ----------/                           \-> SaveImage（可選）
```

### 建議參數

#### YOLOE-26 Load Model
- `model_name`: `yoloe-26s-seg.pt`
- `device`: `auto`

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

---

## 4. Example workflow smoke test

repo 內附的 examples 都是 **API-format workflows**，不是完整的 UI-exported graph：

### 4.1 Minimal smoke test

- `examples/basic_api_workflow.json`

這是最小成功路徑，只覆蓋：

- `YOLOE26LoadModel`
- `YOLOE26PromptSegment`

執行後確認：

- [ ] workflow 可跑完
- [ ] `Prompt Segment` 有輸出 annotated `IMAGE`
- [ ] `Prompt Segment` 有輸出 merged `MASK`
- [ ] `iou` / `max_det` / `mask_threshold` 參數有被正確接受

### 4.2 All-nodes showcase reference

- `examples/all_nodes_showcase_api.json`

這是完整 node-pack wiring reference，應覆蓋全部 7 個 custom nodes：

- [ ] `YOLOE26LoadModel`
- [ ] `YOLOE26PromptSegment`
- [ ] `YOLOE26DetectionMetadata`
- [ ] `YOLOE26InstanceMasks`
- [ ] `YOLOE26ClassMasks`
- [ ] `YOLOE26RefineMask`
- [ ] `YOLOE26SelectBestInstance`

這份 workflow 的用途是核對 node coverage 與合法 wiring，不要求所有 `MASK` 輸出都能在 repo 內直接接到 built-in image sink。
執行後確認：

- [ ] `Prompt Segment` 的 annotated `IMAGE` branch 可用
- [ ] `Detection Metadata` 有輸出 `metadata_json`
- [ ] `Instance Masks` 有輸出 `instance_masks`
- [ ] `Class Masks` 有輸出 `class_masks`
- [ ] `Refine Mask` 有輸出 `refined_masks`
- [ ] `Select Best Instance` 有輸出 `best_mask`
- [ ] 沒有把 `MASK` 輸出直接接到 `PreviewImage` / `SaveImage`

### 4.3 Real application smoke set

下列 practical examples 建議至少抽測 2 份，公開發布前可全測：

| file name | scenario focus | main nodes |
| --- | --- | --- |
| `practical_prompt_segment_api.json` | merged mask for inpainting / cropping / compositing | `YOLOE26PromptSegment` |
| `practical_best_instance_api.json` | API/reference workflow for selecting one best subject | `YOLOE26InstanceMasks`, `YOLOE26SelectBestInstance` |
| `practical_class_masks_api.json` | API/reference workflow for class-wise routing | `YOLOE26ClassMasks` |
| `practical_refine_mask_api.json` | post-process masks while keeping a visible annotated branch | `YOLOE26PromptSegment`, `YOLOE26RefineMask` |
| `practical_detection_metadata_api.json` | metadata-driven automation | `YOLOE26DetectionMetadata`, `YOLOE26PromptSegment` |
| `practical_batch_multi_class_api.json` | multi-class prompt + metadata alignment checks | `YOLOE26ClassMasks`, `YOLOE26DetectionMetadata` |

每份 practical workflow 至少確認其主節點輸出可用，且用途與檔名相符。
若你需要可視化 `MASK`，請接使用者環境中已確認相容的 downstream node；repo 內目前不假設任何特定 built-in `MASK -> IMAGE` converter。

### 執行前確認

- [ ] `example.png` 已放到 ComfyUI input 目錄
- [ ] `yoloe-26s-seg.pt` 已放到支援模型路徑
- [ ] ComfyUI 可正常讀取這兩個檔案
- [ ] 理解這些 JSON 是 API payload 參考，不是 UI graph 匯入檔

---

## 5. 七個節點快速驗證重點

### 5.1 YOLOE-26 Load Model

- [ ] 存在的模型名稱可以載入
- [ ] 不存在的模型名稱會報清楚錯誤
- [ ] `device=auto` 可用
- [ ] 如果有 GPU，`cuda` 或 `cuda:0` 可用
- [ ] 如果是 Apple Silicon，`mps` 可用

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
- [ ] `ultralytics` 版本是否符合 `>=8.3.200`
- [ ] `from ultralytics import YOLOE` 是否仍可用
- [ ] ComfyUI 使用的 Python 環境是否正確
- [ ] prompt 是否為空字串或只有逗號
- [ ] 是否選了不可用的 `device`
- [ ] GPU 雖可見，但實際推理是否失敗

---

## 8. 通過標準

如果下面項目都成立，可以視為 smoke test 通過：

- [ ] 7 個節點都能載入
- [ ] 最小 UI 流程可跑
- [ ] `basic_api_workflow.json` 可跑
- [ ] `practical_prompt_segment_api.json` 可跑
- [ ] `practical_detection_metadata_api.json` 可跑
- [ ] `all_nodes_showcase_api.json` 可作為 reference 並符合合法 wiring
- [ ] `Prompt Segment` 可輸出 image / mask / count
- [ ] `Detection Metadata` 可輸出可解析 JSON
- [ ] `Instance Masks` / `Class Masks` 行為符合預期
- [ ] `Refine Mask` 與 `Select Best Instance` 的 contract 可在 workflow 內驗證
- [ ] 至少 2 份 practical examples 驗證成功

如果 smoke test 通過，但你還要公開發布，下一步請看：

- `RELEASE_CHECKLIST_SHORT.md`
- `TODO_RELEASE_AND_USAGE.md`
