# ComfyUI-YOLOE26 Short Release Checklist

這是一份精簡版發布清單。
如果你不想先看完整文件，可以先跑這份。
完整版請看：`TODO_RELEASE_AND_USAGE.md`

---

## 必做：開始使用前

不完成這些前置準備，就不適合開始測試或分享。

- [ ] 專案已放進 `ComfyUI/custom_nodes/`
- [ ] 已在正確 Python 環境執行 `pip install -r requirements.txt`
- [ ] 本機已有可用 `torch` / `numpy` / `cv2`
- [ ] YOLOE-26 `.pt` 模型已放進支援目錄

## 必做：安裝後最小驗證

這些是安裝完成後，至少要成功一次的基本驗證。

- [ ] ComfyUI 重啟後可看到 7 個節點
- [ ] 最小流程 `Load Image + YOLOE-26 Load Model -> YOLOE-26 Prompt Segment` 可跑

---

## 必做：公開發布前

不完成這些項目，就不建議對外分享。

- [ ] `examples/basic_api_workflow.json` 可跑
- [ ] 7 個節點都至少手動驗證過一次
- [ ] 至少做一次真機 ComfyUI smoke test
- [ ] 至少驗證一個真實 downstream workflow
- [ ] 常見錯誤情境測過：模型不存在、空 prompt、壞 metadata、錯誤 device
- [ ] 未偵測到目標時的 workflow 測試已做過
- [ ] README 安裝步驟可重現
- [ ] README 已補足必要說明
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

如果下面都成立，表示比較接近可公開分享的狀態：

- [ ] smoke test 通過
- [ ] README 足夠讓其他人自行安裝
- [ ] 常見錯誤情境有測過
- [ ] 發布內容整理乾淨
- [ ] 你接受目前狀態仍屬於 **beta / release candidate**
