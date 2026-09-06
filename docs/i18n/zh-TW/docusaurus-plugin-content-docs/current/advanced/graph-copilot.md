---
sidebar_position: 5
title: Graph Copilot
description: 使用 AI 助理，透過外掛前端擴充 API 與 LLM 串流代理產生及編輯節點圖。
---

# Graph Copilot

Graph Copilot 是一個 CodefyUI 外掛，會在編輯器中新增聊天面板。使用自然語言描述所需的圖，AI 就會產生新增節點、連接連接埠及設定參數的操作。這些操作會一次套用，所以每次 AI 編輯只會建立一個復原步驟。你可以停止串流 response、重試失敗的 request，並跨工作階段繼續對話。

:::note 可用性
Graph Copilot 建構於兩項 CodefyUI 功能之上：[外掛前端擴充 API](/advanced/plugin-frontend-extensions) 與統一的 LLM 代理端點（`/api/llm/chat`）。兩者皆自 CodefyUI **1.3.0** 起內建。若 `cdui --version` 顯示更舊的版本，請先執行 `cdui update` 再安裝。
:::

## 安裝

```bash
cdui plugin install graph-copilot
```

你也可以在編輯器中開啟[外掛中心](/advanced/plugins#plugin-center)，從目錄安裝 `graph-copilot`。透過外掛中心安裝後，面板會立即載入。從終端機安裝後，請按 F5 重新載入編輯器。Graph Copilot 面板會以浮動小工具的形式出現在編輯器中。

外掛原始碼與問題回報：[github.com/CodefyUI/CodefyUI-Plugin-Graph-Copilot](https://github.com/CodefyUI/CodefyUI-Plugin-Graph-Copilot)

## 快速上手

1. 依上述步驟安裝外掛。
2. 點擊畫布右下角的圓形 **Graph Copilot** 按鈕，開啟聊天面板。
3. 點擊 **Settings**（齒輪）圖示，選擇提供者並貼上你的 API 金鑰。若要使用 **OpenAI Codex**，請點擊 **Sign in**，並在開啟的分頁中授權。接著選擇模型；使用 **Refresh** 可取得提供者的模型清單。
4. 在輸入框輸入需求，例如 `建立一個小型 MLP 分類器`，按 **Enter**。
5. AI 串流輸出計畫時，會在畫布上新增並連接節點。按一次 **Ctrl+Z** 即可復原整批編輯，或再送出訊息來調整結果。

瀏覽器會儲存提供者與金鑰，所以只需設定一次。以下章節說明各項功能。

## 選擇 LLM 提供者 {/* #選擇-llm-提供商 */}

點擊 Graph Copilot 面板中的 **Settings**（齒輪）圖示，以設定提供者與金鑰。

| 提供者 | 說明 |
|--------|------|
| **OpenAI API** | 使用標準 `https://api.openai.com/v1` 端點，需要 OpenAI API 金鑰，按 token 計費。 |
| **OpenAI Codex（ChatGPT 登入）** | 透過 Codex CLI 的 PKCE flow 與 client ID，使用 ChatGPT 帳號進行 OAuth 登入。它會使用 ChatGPT 訂閱配額而非 API credit，並受 ChatGPT 使用上限與 OpenAI 條款約束。token 儲存在**伺服器上**，位於 user-data 目錄下的 `llm/codex_auth.json`。所有使用該伺服器的人共用一個登入帳號，而 **Sign out** 會清除所有人的 token。OAuth callback 會在伺服器行程中監聽 `localhost:1455`（或 `1457`）。請在 5 分鐘內，使用執行伺服器那台機器上的瀏覽器完成登入。 |
| **OpenRouter** | 在單一金鑰下彙整多個提供者。代理會將請求送到 `https://openrouter.ai/api/v1`；請選擇你偏好的模型。 |
| **Claude API** | 透過 CodefyUI 代理存取 Anthropic API。代理會轉換 OpenAI 相容格式的 request。需要 Anthropic API 金鑰。 |
| **自訂（OpenAI 相容）** | 任何實作 OpenAI `/chat/completions` endpoint 的伺服器。例如，你可以使用位於 `http://localhost:11434/v1` 的本機 Ollama instance。請設定 base URL，並在需要時填入金鑰。 |

代理也提供 `POST /api/llm/models`，供 **Refresh** 列出提供者的模型。`POST /api/llm/codex/login`、`GET /api/llm/codex/status` 與 `POST /api/llm/codex/logout` 用於 ChatGPT 登入。相同控制項也位於 **設定 → LLM 提供者**。只有 **OpenAI API** 與 **OpenAI Codex** 會使用 `reasoning_effort`；代理只對 **OpenAI Codex** 拒絕 `ultra`（`400`），對 **OpenAI API** 則原樣轉送任何值。編輯器會提供這些 `POST` routes 所需的 session token。

## 金鑰處理

API 金鑰會以 Graph Copilot 專用的 namespace 儲存在 `localStorage`。每次 request 都會將選擇的金鑰送到本機 CodefyUI 後端。`/api/llm/chat` 會把金鑰與訊息轉送給設定的提供者，再將 response 串流回來。它不會記錄或持久化金鑰與訊息。每個提供者都有固定的上游主機；只有**自訂**會使用你提供的 base URL。這些金鑰與 `CODEFYUI_OPENAI_API_KEY` 和 `CODEFYUI_ANTHROPIC_API_KEY` 分開。只有 `LLMChat` 節點會讀取這兩個環境變數；代理不會讀取它們。

## 使用方式

### 發送請求

在聊天輸入框中輸入需求，按 Enter（或點擊 **Send**）。範例：

- "新增一個含 ReLU 活化函數的兩層 MLP"
- "將 CrossEntropy 節點連接到最後一個 Linear 的輸出"
- "將 Linear-1 的 hidden size 設為 512"

AI 會先回傳計畫，再回傳操作清單。套用操作時，每項操作會顯示一個 chip，例如「add Linear」、「add ReLU」或「connect」。

### 對話記錄

目前圖的聊天記錄會儲存在 `localStorage`。重新開啟或重新載入編輯器時，Graph Copilot 會還原該對話。

### 中止與重試

在串流過程中點擊 **Stop** 可取消進行中的請求，部分回應將被捨棄。點擊任意 AI 訊息上的 **Retry**，可在相同上下文中重新送出該輪對話。

### 復原 AI 編輯 {/* #撤銷-ai-編輯 */}

每次 AI 編輯都是一個復原快照。按一次 **Ctrl+Z**（macOS 上為 Cmd+Z）即可復原最後一次 AI 回應的整批操作。

## 使用技巧

- 提供你正在建構的內容背景：「我正在建構一個採用 ResNet 骨幹網路的視覺分類器」有助於 AI 做出更好的選擇。
- 若 AI 新增了不在你節點面板中的節點類型，該操作會被跳過並回報——請先使用 `cdui plugin install` 安裝所需的外掛包。
- Graph Copilot 在每次請求前會讀取目前的圖表狀態與完整節點面板，因此它知道有哪些類型可用，以及畫布上已有什麼。

## 另請參閱

- [外掛前端擴充](/advanced/plugin-frontend-extensions) — Graph Copilot 所基於的 JS API。
- [外掛](/advanced/plugins) — 外掛包系統。
- [API 參考](/advanced/api-reference) — `/api/llm/chat` 串流端點。
