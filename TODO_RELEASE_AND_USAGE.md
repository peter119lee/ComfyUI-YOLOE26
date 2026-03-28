# ComfyUI-YOLOE26 使用與發布清單

這份文件是把目前這個 node pack 從「可評估的 beta」推進到「可 defensibly 對外發布」所需要做的事情整理成 checklist。
在 provenance clarification、release-grade compatibility matrix、以及真機 ComfyUI smoke evidence 補齊前，這個 repo 仍不應描述成 public-release-ready。

---

## 1. 先完成本機安裝

- [ ] 把 `ComfyUI-YOLOE26` 放進 ComfyUI 的 `custom_nodes` 目錄
- [ ] 在 **ComfyUI 相同的 Python 環境** 安裝 dependency
  ```bash
  pip install -r requirements.txt
  ```
- [ ] 確認 `ultralytics` 版本符合目前要求
  - 目前需求：`ultralytics>=8.3.200,<8.5.0`
- [ ] 確認宿主 ComfyUI 環境已經提供可用的 `torch`
- [ ] 確認宿主 ComfyUI 環境已經提供可用的 `numpy`
- [ ] 確認宿主 ComfyUI 環境已經提供可用的 `cv2`

> 目前 `requirements.txt` 只列出 `ultralytics`，其餘套件預期由 ComfyUI 本身環境提供。

---

## 2. 放好模型檔

- [ ] 準備本機 YOLOE-26 `.pt` 權重，或決定要使用 allowlisted auto-download
- [ ] 如果使用本機模型，把模型放到任一支援目錄：
  - `ComfyUI/models/ultralytics/segm/`
  - `ComfyUI/models/ultralytics/bbox/`
  - `ComfyUI/models/ultralytics/`
  - `ComfyUI/models/yoloe/`
- [ ] 如果使用 auto-download，先用 allowlisted 官方 segmentation 權重驗證一次：
  - `yoloe-26n-seg.pt`
  - `yoloe-26s-seg.pt`
  - `yoloe-26m-seg.pt`
  - `yoloe-26l-seg.pt`
  - `yoloe-26x-seg.pt`
- [ ] 確認 auto-download 使用前有可用網路、可寫入權限，且下載後會通過 SHA256 驗證
- [ ] 理解 auto-download 目前只涵蓋 allowlisted 官方 segmentation 權重，並依賴目前 Ultralytics upstream asset 命名 / 解析行為

---

## 3. 啟動前檢查

- [ ] 重新啟動 ComfyUI
- [ ] 確認 ComfyUI 啟動時沒有 import error
- [ ] 確認 custom node 有被成功載入
- [ ] 在節點清單中確認以下 7 個節點都出現：
  - [ ] `YOLOE-26 Load Model`
  - [ ] `YOLOE-26 Prompt Segment`
  - [ ] `YOLOE-26 Detection Metadata`
  - [ ] `YOLOE-26 Instance Masks`
  - [ ] `YOLOE-26 Class Masks`
  - [ ] `YOLOE-26 Refine Mask`
  - [ ] `YOLOE-26 Select Best Instance`

---

## 4. 先跑最小 smoke test

建議先做最小驗證，不要一開始就測很多組 workflow。

- [ ] 準備一張測試圖片放進 ComfyUI input 目錄
- [ ] 準備一個可辨識目標，例如 `person`
- [ ] 建立最小流程：
  - `Load Image`
  - `YOLOE-26 Load Model`
  - `YOLOE-26 Prompt Segment`
- [ ] 驗證 `YOLOE-26 Load Model` 能正常載入模型
- [ ] 驗證 `YOLOE-26 Prompt Segment` 有回傳：
  - [ ] annotated `IMAGE`
  - [ ] merged `MASK`
  - [ ] detection `INT`
- [ ] 驗證調整 `conf` 時，偵測數量會改變
- [ ] 驗證調整 `mask_threshold` 時，輸出的 mask 面積會改變
- [ ] 驗證最小 smoke test 的截圖 / log / version info 已保存，可作為真機 ComfyUI evidence

---

## 5. 逐一驗證每個節點契約

### 5.1 YOLOE-26 Load Model

- [ ] 測試 `model_name` 指向存在的本機模型
- [ ] 測試不存在的模型名稱，確認錯誤訊息可讀
- [ ] 測試 `auto_download=false` 時，本機缺模型會正確失敗
- [ ] 測試 `auto_download=true` 時，allowlisted 官方模型可下載、驗證並載入
- [ ] 測試 `auto_download=true` 時，非 allowlisted 模型名稱會被拒絕
- [ ] 測試 digest mismatch / cache 汙染時，模型不會在驗證前載入
- [ ] 測試 `device=auto`
- [ ] 如果有 GPU，再測：
  - [ ] `device=cuda`
  - [ ] `device=cuda:0`
- [ ] 如果是 Apple Silicon，再測：
  - [ ] `device=mps`

### 5.2 YOLOE-26 Prompt Segment

- [ ] 測試單一 prompt：`person`
- [ ] 測試多個 prompt：`person, car, dog`
- [ ] 驗證 `show_boxes` / `show_labels` / `show_conf` / `show_masks` 開關都能作用
- [ ] 驗證空 prompt 或格式異常時，節點有明確錯誤
- [ ] 驗證 batch image 輸入時不會 shape 爆掉

### 5.3 YOLOE-26 Detection Metadata

- [ ] 驗證有輸出 `metadata_json`
- [ ] 驗證 JSON 內包含：
  - [ ] class name
  - [ ] score
  - [ ] box
  - [ ] mask area
  - [ ] batch index
- [ ] 驗證 metadata 內有記錄 inference controls：
  - [ ] `conf`
  - [ ] `iou`
  - [ ] `max_det`
  - [ ] `mask_threshold`
  - [ ] `imgsz`
- [ ] 把 JSON 餵給你的下游流程，確認格式足夠穩定

### 5.4 YOLOE-26 Instance Masks

- [ ] 驗證每個 detection 都會對應一張 instance mask
- [ ] 驗證 `instance_metadata_json` 裡的 `output_mask_index` 與輸出 mask 順序一致
- [ ] 驗證多張輸入圖時 `batch_index` 是正確的
- [ ] 驗證無 detection 時目前的行為是否符合你的 workflow 預期
  - 目前設計：回傳 placeholder zero mask，`count = 0`

### 5.5 YOLOE-26 Class Masks

- [ ] 驗證每個 prompt class 都會得到一張輸出 mask
- [ ] 驗證同一 class 多個 instance 會被 merge 成單一 mask
- [ ] 驗證未偵測到某 class 時，仍回傳全零 mask
- [ ] 驗證你理解 `class_count` 與 `output_mask_count` 的差別
  - `class_count`：prompt class 的數量
  - `output_mask_count`：實際回傳在 `MASK` batch 裡的 mask 數量

### 5.6 YOLOE-26 Refine Mask

- [ ] 用 `Instance Masks` 或 `Class Masks` 的輸出接進來測
- [ ] 驗證以下方法都能跑：
  - [ ] `threshold`
  - [ ] `open`
  - [ ] `close`
  - [ ] `dilate`
  - [ ] `erode`
  - [ ] `largest_component`
  - [ ] `fill_holes`
- [ ] 測試 `kernel_size`
- [ ] 測試 `iterations`
- [ ] 測試 `min_area`
- [ ] 驗證 `refined_metadata_json` 會附加 `refinement` 資訊

### 5.7 YOLOE-26 Select Best Instance

- [ ] 只把 `YOLOE-26 Instance Masks` 的輸出接進來
- [ ] 驗證 `instance_metadata_json` 來源正確
- [ ] 測試以下選擇策略：
  - [ ] `highest_confidence`
  - [ ] `largest_area`
  - [ ] `confidence_then_area`
- [ ] 驗證輸出的 `selected_mask_index` 與 metadata 一致
- [ ] 驗證沒有候選 instance 時，你的下游流程能接受目前回傳格式

---

## 6. 跑一次範例 workflow

- [ ] 使用 `examples/basic_api_workflow.json`
- [ ] 理解這是一個 **API-format workflow example**，不是完整的 UI-exported workflow graph
- [ ] 把 `example.png` 放到 ComfyUI input 目錄
- [ ] 確認 `yoloe-26s-seg.pt` 已放到支援路徑
- [ ] 把 `examples/basic_api_workflow.json` 視為 smoke target，而不是已完成的 release evidence
- [ ] 把其餘 examples 區分成 smoke-target 與 wiring/reference 用途，不要把 showcase/reference JSON 寫成已完成的真機 smoke 證據
- [ ] 執行 workflow，確認 4 個節點都能串起來（LoadImage → YOLOE26LoadModel → YOLOE26PromptSegment → PreviewImage）：
  - [ ] `LoadImage`
  - [ ] `YOLOE26LoadModel`
  - [ ] `YOLOE26PromptSegment`
  - [ ] `PreviewImage`
- [ ] 確認 workflow 中的這些控制參數都可用：
  - [ ] `iou`
  - [ ] `max_det`
  - [ ] `mask_threshold`

---

## 7. 做真實 ComfyUI 場景驗證

單元測試過了，不代表在真實 ComfyUI pipeline 內完全沒問題，所以這一段很重要。

- [ ] 在 ComfyUI UI 裡手動拉一次節點流程
- [ ] 測試單張圖片
- [ ] 測試 batch 圖片
- [ ] 測試高解析度圖片
- [ ] 測試沒有任何目標的圖片
- [ ] 測試多類別 prompt
- [ ] 測試很接近的類別詞，例如：
  - `person`
  - `man`
  - `woman`
  - `red apple`
  - `green bottle`
- [ ] 驗證輸出 mask 可直接接到你常用的 downstream 節點
  - inpainting
  - compositing
  - cropping
  - region control
- [ ] 驗證 JSON metadata 在你的實際 workflow 中可被保存 / 顯示 / 解析

---

## 8. 做錯誤情境測試

你要確認這個 node pack 失敗時也是「可用」的，而不是只有成功路徑可用。

- [ ] 模型不存在
- [ ] 模型檔損壞
- [ ] auto-download 網路失敗
- [ ] auto-download 下載成功但 SHA256 驗證失敗
- [ ] `ultralytics` 版本不相容
- [ ] `from ultralytics import YOLOE` 無法使用
- [ ] prompt 為空字串
- [ ] prompt 只有逗號或空白
- [ ] 輸入 image tensor shape 不正確
- [ ] 輸入 mask tensor shape 不正確
- [ ] metadata JSON 壞掉或格式錯誤
- [ ] `Select Best Instance` 收到不是來自 `Instance Masks` 的 metadata
- [ ] 使用不存在的 device
- [ ] `cuda` 可見但實際推理失敗

對每個情境都要確認：

- [ ] 錯誤訊息是否足夠讓使用者知道怎麼修
- [ ] 不會讓 ComfyUI 整個流程進入難以理解的壞狀態

---

## 9. 發布前整理 README

README 已經涵蓋目前主要功能，但正式分享前還可以再補一些實用內容。

- [ ] 加一張節點總覽圖或 workflow screenshot
- [ ] 每個節點補一組最小示例
- [ ] 補充 `metadata_json` 的實際 JSON 範例
- [ ] 補充 `instance_metadata_json` 範例
- [ ] 補充 `class_metadata_json` 範例
- [ ] 補充 `refined_metadata_json` 範例
- [ ] 補充 `best_instance_metadata_json` 範例
- [ ] 補「常見錯誤與處理方式」段落
- [ ] 補「哪些輸出適合接哪些下游節點」建議
- [ ] 如果你要對外發佈，補上版本號與 changelog 習慣

---

## 10. 發布前技術驗證

- [ ] 在乾淨環境重新安裝一次
- [ ] 在另一台機器或另一個 Python 環境再測一次
- [ ] 如果有 GPU 與 CPU 兩種環境，兩邊都測
- [ ] 確認 README 安裝步驟從零到可用是可重現的
- [ ] 確認範例 workflow 不是過期內容
- [ ] 確認沒有殘留 debug 輸出
- [ ] 確認沒有明顯 dead code
- [ ] 確認沒有把暫存測試資料一起發出去

---

## 11. 如果你要對外發布，建議再補的項目

這些不是「現在不能用」的 release blocker。
其中一部分是發布品質補強，另一部分是發布後也可以持續做的 roadmap 項目。

### 高優先

- [ ] 做至少一次真機 ComfyUI end-to-end 驗證
- [ ] 收集 3 到 5 組真實使用案例
- [ ] 補充更多 workflow example
- [ ] 確認不同 prompt 類型的穩定度
- [ ] 確認沒有因 upstream API 變動而脆弱的地方

### 中優先

- [ ] 新增一個專門做 metadata pretty-print / parse 的輔助節點
- [ ] 新增可選輸出：每類別 detection count
- [ ] 新增更明確的 no-detection policy 設定
- [ ] 新增更多 refinement presets
- [ ] 新增更完整的 batch mapping 範例

### 低優先

- [ ] 補 demo 圖片或 demo GIF
- [ ] 補 benchmark 資訊
- [ ] 補版本相容表（ComfyUI / ultralytics / torch）

---

## 12. 目前可以怎麼判定「已經可用了」

如果下面這些你都能打勾，我會把它視為「可正常使用的 node pack」：

- [ ] 7 個節點都能在 ComfyUI 正常出現
- [ ] 模型能正常載入
- [ ] `Prompt Segment` 可穩定輸出 image / mask / count
- [ ] `Detection Metadata` 的 JSON 可被實際使用
- [ ] `Instance Masks` 與 `output_mask_index` 對得上
- [ ] `Class Masks` 在未偵測到目標時也能穩定回傳零 mask
- [ ] `Refine Mask` 能正確處理常見後處理需求
- [ ] `Select Best Instance` 能正確選到你預期的 mask
- [ ] example workflow 能跑
- [ ] 至少一個真實工作流已驗證成功

如果下面這些也都完成，才比較接近「可放心對外分享」：

- [ ] 乾淨環境驗證通過
- [ ] CPU / GPU 至少測過其中你目標使用者會用的環境
- [ ] README 足夠讓陌生使用者自己裝起來
- [ ] 常見錯誤情境都有測過
- [ ] 發布內容沒有殘留暫存或不必要檔案

---

## 13. 建議你的實際執行順序

如果你現在要把事情一次做完，建議照這個順序：

1. [ ] 安裝 dependency
2. [ ] 放模型
3. [ ] 啟動 ComfyUI 確認節點有出現
4. [ ] 跑最小 smoke test
5. [ ] 跑 `examples/basic_api_workflow.json`
6. [ ] 逐一驗證 7 個節點契約
7. [ ] 做真實 ComfyUI 場景測試
8. [ ] 做錯誤情境測試
9. [ ] 補 README 範例與 screenshot
10. [ ] 在乾淨環境重裝驗證
11. [ ] 再決定是否公開發布

---

## 14. 目前狀態總結

目前這個 repo 的狀態，比較準確的描述是：

- 已經不是 prototype
- 已經具備基本可用性
- 目前 helper / node-level 單元測試通過
- README 與 example workflow 已完成本輪 release-surface 對齊，但目前描述的是「預期 public surface」與「待補證據的 smoke target / reference 分類」，不是已完成的公開發布證據
- 目前更接近 **可評估的 beta**，而不是 public-release-ready 或完全打磨完成的正式版
- 正式對外發布前，仍卡在 provenance clarification、release-grade compatibility matrix、以及真機 ComfyUI smoke / integration evidence

也就是說：

**現在可以開始用，也可以持續做驗證。**

但如果你要把它當成一個對外公開、別人拿去就能順利用的 node pack，請先把上述 release blockers 補齊，再決定是否公開發布。
