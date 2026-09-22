---
sidebar_position: 7.7
title: 共用的伺服器
description: 一台 CodefyUI 伺服器會讓所有連得上的人共用什麼 -- 環境層級的憑證、帳單算在誰頭上，以及每張圖各自存了什麼。
---

# 共用的伺服器

CodefyUI 是透過 HTTP 運作的桌面工具。單人安裝使用環境層級憑證是預期行為；當其他人也能連線時，同一組憑證會由所有人共用。本頁說明這項差異。

若尚未閱讀[發佈](./publish)，請先查看其中第 6 節，了解綁定區域網路會暴露哪些功能。本頁接著說明憑證的影響。

## 一台伺服器只有一個身分 {/* #an-instance-has-one-identity */}

沒有使用者帳號。工作階段權杖（session token）只能證明請求端能讀取機器上的檔案，不能識別發出請求的人。因此，伺服器持有的所有憑證都屬於整個實例，不屬於目前開啟瀏覽器分頁的個人。

**在共用機器上，設定憑證的人會支付所有人的用量，而且系統不記錄各人的支出。**

有五種憑證是這樣運作的。

### ChatGPT 登入 {/* #chatgpt-sign-in */}

`POST /api/llm/codex/login` 完成 OAuth 流程後，會把存取權杖與更新權杖寫入使用者資料目錄下的 `llm/codex_auth.json`：由 `cdui start` 或 `cdui dev` 啟動的伺服器使用 `<install dir>/.codefyui_dev/llm/`（預設安裝目錄為 `~/CodefyUI`）；事先匯出 `CODEFYUI_USER_DATA_DIR` 時使用 `<dir>/llm/`；平台目錄（Windows 的 `%LOCALAPPDATA%\codefyui\llm\`、Linux 的 `~/.local/share/codefyui/llm/`、macOS 的 `~/Library/Application Support/codefyui/llm/`）只供手動啟動的 uvicorn 使用。詳見[把 graph 當成函式呼叫](./graph-as-a-function#2-getting-the-token-for-external-scripts)。支援 Unix 權限的平台會將檔案設為 chmod 0600；Windows 則由資料夾的個人帳號 ACL 保護。

代理層只檢查「有人登入了」，不檢查「登入的是不是你」：

- 只要有一個人登入，這台伺服器上**每一張** `LLMChat` 使用 **Codex** 供應商的圖，帳都算在那個人的 ChatGPT 帳號上；呼叫 `/api/llm/chat` 的 `openai-codex` 供應商的外掛也一樣，例如 [Graph Copilot](/advanced/graph-copilot)。**ChatGPT API** 供應商則改用 OpenAI API 金鑰（見下一節）。
- `POST /api/llm/codex/logout` 除了工作階段權杖以外不需要任何參數。任何連得上編輯器的人都可以把你登出。

### 來自環境變數的 LLM API 金鑰 {/* #llm-api-keys-from-the-environment */}

當 `LLMChat` 節點的金鑰參數是空的，節點會依序退回去讀行程的環境變數：

- OpenAI：先 `CODEFYUI_OPENAI_API_KEY`，再 `OPENAI_API_KEY`
- Anthropic：先 `CODEFYUI_ANTHROPIC_API_KEY`，再 `ANTHROPIC_API_KEY`

伺服器以 `--project` 啟動時，`cdui` 會在啟動階段從專案的 `.env` 載入這些變數（見[專案目錄](./project-directories)）。因此，該檔案中的金鑰可由實例上的每個 graph 使用。這項 fallback 刻意不顯示提示：金鑰參數留空的 graph 不會指出它使用了實例的金鑰。**請假設任何人能執行的 graph 都能消耗組織的 LLM 額度。**

這項 fallback 有一個刻意的限制：**Ollama** 供應商永遠不會收到金鑰，因此把 `ollama_base_url` 指向攻擊者伺服器的 graph 無法將金鑰傳出機器。

### Kaggle

`KaggleDataset` 節點使用 `KAGGLE_USERNAME` 加 `KAGGLE_KEY`，或者是這個服務帳號的 `~/.kaggle/kaggle.json`。下載會算在那個 Kaggle 帳號頭上，包括你以那個帳號同意過的競賽規則。

### Hugging Face {/* #hugging-face */}

`HuggingFaceDataset`、`TextCorpusDataset`（`source` 設為 `huggingface` 時）與套件中心的模型下載，會使用伺服器環境變數中的 `HF_TOKEN` 驗證身分；沒有設定時，改用 `hf auth login` 存放在服務帳號家目錄中的權杖檔（`~/.cache/huggingface/token`）。受限（gated）的資料集與模型會以該 Hugging Face 帳號下載，包括以該帳號同意過的使用條款。

### Git {/* #git */}

以 `--project` 啟動的伺服器中，**版本控制**分頁會在專案目錄執行伺服器帳號自己的 `git`。fetch、pull 與 push 會使用該帳號既有的憑證（憑證輔助程式、SSH 金鑰），每個 commit 的作者也都是伺服器上設定的同一個 git 身分，因此任何連得上編輯器的人都會以該帳號推送到專案的遠端。請見[版本控制](./source-control)。

## 那什麼才是每張圖各自的 {/* #what-is-per-graph-instead */}

型別標示為 SECRET 的參數，例如 `LLMChat` 節點的 `openai_api_key`，屬於輸入該值的人，處理方式不同。伺服器會從寫出的每份副本中清除這些值，包括已儲存的 graph、匯出檔案、已發佈應用程式版本、預設組合（preset）、產生的 Python 程式碼與執行紀錄。

這會產生以下結果：

- SECRET 參數**不會**儲存在任何位置。重新載入編輯器或重新匯入匯出的 graph 後，欄位會是空白；`LLMChat` 節點會改用實例環境變數中的金鑰（若有設定），否則會以缺少金鑰的錯誤失敗。排隊中的執行只會把輸入的值保存在伺服器記憶體中，直到該執行結束為止；如果伺服器在它開始執行前停止，該執行會被標為 `interrupted`，值也隨之消失。這是刻意的取捨，不是 bug。
- **新版不會儲存之後輸入的金鑰。** 值不會寫入資料庫，而且刪除的資料庫頁面會清零，不會保留內容後再重複使用。因此，執行紀錄被清除後不會留下可讀副本，不需要輪替金鑰或執行清理步驟。
- **如果你曾在舊版執行含有 SECRET 參數的 graph，請將該金鑰視為已外洩並立即輪替。** 舊版會把執行時的 graph 原樣寫入 `exec_runs.graph_snapshot` 欄位，而執行紀錄依**數量**保留最新 200 筆，不是依時間清除，因此低用量實例可能長期保留該值。升級會清除仍存在的執行紀錄中的值，並記錄清除數量；但無法處理升級前已被清除的執行。這些資料列已不存在，而舊版釋放的資料庫頁面仍可能保留內容，直到執行 `VACUUM`。輪替金鑰是處理此空窗期的唯一完整方法。包含此修正的版本請見 CHANGELOG。

## 如果你需要分辨是誰用的 {/* #if-you-need-per-person-attribution */}

目前產品不支援個別使用者歸屬。請為每個人執行獨立實例，並讓每個人提供自己的憑證：

- 每個實例使用自己的安裝目錄（安裝程式的 `CODEFYUI_DIR`）、環境變數檔與埠號。從同一個安裝啟動的實例會共用 SQLite 資料庫（執行紀錄、已發佈應用程式、API key）、已儲存的 graph、模型、圖片、媒體與上傳的資料檔，以及套件包與外掛安裝時加入套件的 Python 環境；`cdui start` 每個安裝只會執行一個背景伺服器，而 `cdui stop` 會停止從該安裝啟動的所有伺服器。不同的 `CODEFYUI_USER_DATA_DIR` 只會移動工作階段權杖、ChatGPT 登入、下載的外掛與其 lockfile、下載快取，以及套件包的控制檔。
- 如果有人使用存放在家目錄中的憑證，請讓每個實例以各自的作業系統帳號執行：`~/.kaggle/kaggle.json`、Hugging Face 權杖檔、git 的憑證輔助程式與 SSH 金鑰。

其他部署方式都會共用同一個身分，而且編輯器不會顯示這項共用狀態。
