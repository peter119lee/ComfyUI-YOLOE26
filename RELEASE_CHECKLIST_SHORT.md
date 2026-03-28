# ComfyUI-YOLOE26 Short Release Checklist

這是一份精簡版發布清單。
如果你不想先看完整文件，可以先跑這份。
完整版請看：`TODO_RELEASE_AND_USAGE.md`

在 provenance clarification、release-grade compatibility matrix、以及已留檔的真機 ComfyUI smoke evidence 補齊前，這個 repo 仍不應描述成 public-release-ready。

---

## 必做：開始使用前

不完成這些前置準備，就不適合開始測試或分享。

- [ ] 專案已放進 `ComfyUI/custom_nodes/`
- [ ] 已在正確 Python 環境執行 `pip install -r requirements.txt`
- [ ] 本機已有可用 `torch` / `numpy` / `cv2`
- [ ] YOLOE-26 `.pt` 模型已放進支援目錄，或你明確只打算測 allowlisted auto-download
- [ ] 如果要測 `auto_download`，只使用 allowlisted 官方 segmentation 權重：`yoloe-26n-seg.pt`、`yoloe-26s-seg.pt`、`yoloe-26m-seg.pt`、`yoloe-26l-seg.pt`、`yoloe-26x-seg.pt`
- [ ] 如果要測 `auto_download`，你接受它仍依賴目前 Ultralytics upstream asset 命名 / 解析假設

## 必做：安裝後最小驗證

這些是安裝完成後，至少要成功一次的基本驗證。

- [ ] ComfyUI 重啟後可看到 7 個節點
- [ ] 最小流程 `Load Image + YOLOE-26 Load Model -> YOLOE-26 Prompt Segment` 可跑
- [ ] 已保存最小 smoke 的截圖 / log / version info，作為真機 ComfyUI evidence

---

## 必做：公開發布前

不完成這些項目，就不建議對外分享，也不應宣稱 public-release-ready。

- [ ] `examples/basic_api_workflow.json` 可跑
- [ ] 7 個節點都至少手動驗證過一次
- [ ] 至少做一次真機 ComfyUI smoke test，且把結果留檔
- [ ] 至少驗證一個真實 downstream workflow
- [ ] 常見錯誤情境測過：模型不存在、空 prompt、壞 metadata、錯誤 device
- [ ] 未偵測到目標時的 workflow 測試已做過
- [ ] provenance 狀態已明確、可接受，且與 repo-level MIT `LICENSE` 不衝突
- [ ] release-grade compatibility matrix 已補齊
- [ ] README 安裝步驟可重現
- [ ] README 已補足必要說明，且沒有把 example/reference workflow 寫成已完成 smoke 證據
- [ ] auto-download 限制已明寫清楚：僅 allowlisted 官方 segmentation 權重，且依賴目前 upstream asset 假設
- [ ] 發布內容沒有暫存檔或無關檔案

---

## 可選：建議補強

這些不阻擋基本發布，但能明顯提高穩定性與可理解性。

- [ ] CPU / GPU 都測
- [ ] batch image 測試
- [ ] 高解析度圖片測試
- [ ] 多類別 prompt 測試
- [ ] README 補 screenshot
- [ ] README 補 metadata JSON 範例
- [ ] README 補常見錯誤處理方式

---

## 之後再做：非 release blocker

- [ ] 更多 workflow examples
- [ ] metadata pretty-print / parse 輔助節點
- [ ] 每類別 detection count 額外輸出
- [ ] 更明確的 no-detection policy
- [ ] 更多 refinement presets
- [ ] benchmark 與版本相容表

---

## 快速判定

### 可以開始用

如果下面都成立，表示這個 node pack 已經進入可用狀態：

- [ ] 7 個節點都能載入
- [ ] 模型能載入
- [ ] `Prompt Segment` 能輸出結果
- [ ] example workflow 能跑
- [ ] 至少一個真實 workflow 可用

### 可以對外分享

如果下面都成立，才比較接近可公開分享的狀態：

- [ ] smoke test 通過，且 evidence 已留檔
- [ ] README 足夠讓其他人自行安裝
- [ ] 常見錯誤情境有測過
- [ ] provenance 與 compatibility matrix 已處理，且 README 已清楚區分 repo MIT license 與第三方依賴 / 權重授權
- [ ] 發布內容整理乾淨
- [ ] 目前狀態不再需要依賴降級 wording 才能避免誇大 release readiness
