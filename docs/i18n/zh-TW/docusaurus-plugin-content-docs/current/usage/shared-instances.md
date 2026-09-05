---
sidebar_position: 7.7
title: 共用的伺服器
description: 一台 CodefyUI 伺服器會讓所有連得上的人共用什麼 -- 環境層級的憑證、帳單算在誰頭上，以及每張圖各自存了什麼。
---

# 共用的伺服器

CodefyUI 是透過 HTTP 運作的桌面工具。單人安裝使用環境層級憑證是預期行為；當其他人也能連線時，同一組憑證會由所有人共用。本頁說明這項差異。

若尚未閱讀[發佈](./publish)，請先查看其中第 6 節，了解綁定區域網路會暴露哪些功能。本頁接著說明憑證的影響。

## 一台伺服器只有一個身分

沒有使用者帳號。工作階段權杖（session token）只能證明請求端能讀取機器上的檔案，不能識別發出請求的人。因此，伺服器持有的所有憑證都屬於整個實例，不屬於目前開啟瀏覽器分頁的個人。

**在共用機器上，設定憑證的人會支付所有人的用量，而且系統不記錄各人的支出。**

有三種憑證是這樣運作的。

### ChatGPT 登入

`POST /api/llm/codex/login` 完成 OAuth 流程後，會把存取權杖與更新權杖寫入使用者資料目錄下的 `llm/codex_auth.json`：由 `cdui start` 或 `cdui dev` 啟動的伺服器使用 `<install dir>/.codefyui_dev/llm/`（預設安裝目錄為 `~/CodefyUI`）；事先匯出 `CODEFYUI_USER_DATA_DIR` 時使用 `<dir>/llm/`；平台目錄（Windows 的 `%LOCALAPPDATA%\codefyui\llm\`、Linux 的 `~/.local/share/codefyui/llm/`、macOS 的 `~/Library/Application Support/codefyui/llm/`）只供手動啟動的 uvicorn 使用。詳見[專案目錄](./project-directories#6-建立-api-keyinvoke-需要)。支援 Unix 權限的平台會將檔案設為 chmod 0600；Windows 則由資料夾的個人帳號 ACL 保護。

代理層只檢查「有人登入了」，不檢查「登入的是不是你」：

- 只要有一個人登入，這台伺服器上**每一張**用 ChatGPT 供應商的圖，帳都算在那個人的個人 ChatGPT 帳號上。
- `POST /api/llm/codex/logout` 除了工作階段權杖以外不需要任何參數。任何連得上編輯器的人都可以把你登出。

### 來自環境變數的 LLM API 金鑰

當 `LLMChat` 節點的金鑰參數是空的，節點會依序退回去讀行程的環境變數：

- OpenAI：先 `CODEFYUI_OPENAI_API_KEY`，再 `OPENAI_API_KEY`
- Anthropic：先 `CODEFYUI_ANTHROPIC_API_KEY`，再 `ANTHROPIC_API_KEY`

伺服器以 `--project` 啟動時，`cdui` 會在啟動階段從專案的 `.env` 載入這些變數（見[專案目錄](./project-directories)）。因此，該檔案中的金鑰可由實例上的每個 graph 使用。這項 fallback 刻意不顯示提示：金鑰參數留空的 graph 不會指出它使用了實例的金鑰。**請假設任何人能執行的 graph 都能消耗組織的 LLM 額度。**

這項 fallback 有一個刻意的限制：`custom` 供應商永遠不會收到金鑰，因此把 `base_url` 指向攻擊者的 graph 無法將金鑰傳出機器。

### Kaggle

`KaggleDataset` 節點使用 `KAGGLE_USERNAME` 加 `KAGGLE_KEY`，或者是這個服務帳號的 `~/.kaggle/kaggle.json`。下載會算在那個 Kaggle 帳號頭上，包括你以那個帳號同意過的競賽規則。

## 那什麼才是每張圖各自的

型別標示為 SECRET 的參數，例如 `LLMChat` 節點的 `openai_api_key`，屬於輸入該值的人，處理方式不同。伺服器會從寫出的每份副本中清除這些值，包括已儲存的 graph、匯出檔案、已發佈應用程式版本、預設組合（preset）、產生的 Python 程式碼與執行紀錄。

這會產生以下結果：

- SECRET 參數**不會**儲存在任何位置。重新載入編輯器、重新匯入 graph，或在伺服器重新啟動後執行原本排隊的工作時，欄位都會是空白，節點會以原有的「需要 api key」錯誤失敗。這是刻意的取捨，不是 bug。
- **新版不會儲存之後輸入的金鑰。** 值不會寫入資料庫，而且刪除的資料庫頁面會清零，不會保留內容後再重複使用。因此，執行紀錄被清除後不會留下可讀副本，不需要輪替金鑰或執行清理步驟。
- **如果你曾在舊版執行含有 SECRET 參數的 graph，請將該金鑰視為已外洩並立即輪替。** 舊版會把執行時的 graph 原樣寫入 `exec_runs.graph_snapshot` 欄位，而執行紀錄依**數量**保留最新 200 筆，不是依時間清除，因此低用量實例可能長期保留該值。升級會清除仍存在的執行紀錄中的值，並記錄清除數量；但無法處理升級前已被清除的執行。這些資料列已不存在，而舊版釋放的資料庫頁面仍可能保留內容，直到執行 `VACUUM`。輪替金鑰是處理此空窗期的唯一完整方法。包含此修正的版本請見 CHANGELOG。

## 如果你需要分辨是誰用的

目前產品不支援個別使用者歸屬。請為每個人執行獨立實例，並分別設定 `.env`、`CODEFYUI_USER_DATA_DIR` 與埠號，讓每個人提供自己的憑證。其他部署方式都會共用同一個身分，而且編輯器不會顯示這項共用狀態。
