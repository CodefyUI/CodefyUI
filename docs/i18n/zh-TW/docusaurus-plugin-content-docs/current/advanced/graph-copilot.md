---
sidebar_position: 5
title: Graph Copilot
description: 以對話方式讓 AI 助理生成、調整並改善你的節點圖——基於外掛前端擴充 API 與統一 LLM 串流代理。
---

# Graph Copilot

Graph Copilot 是一個 CodefyUI 外掛，在編輯器中新增一個聊天面板。你用自然語言描述需求，AI 就會產生一系列圖表操作（新增節點、連接連接埠、設定參數），並以原子方式套用——每次 AI 編輯只佔用一個撤銷步驟。你可以在串流過程中中止、重試失敗的請求，並在不同 session 中繼續對話。

:::note 可用性
Graph Copilot 建構於兩項 CodefyUI 功能之上：[外掛前端擴充 API](/advanced/plugin-frontend-extensions) 與統一的 LLM 代理端點（`/api/llm/chat`）。兩者皆自 **1.2.1** 之後的首個版本起內建於 CodefyUI。若 `cdui --version` 顯示 1.2.1 或更早，請先更新到最新版本（或直接從 `main` 執行）再安裝。
:::

## 安裝

```bash
cdui plugin install treeleaves30760/CodefyUI-Plugin-Graph-Copilot
```

接著重新載入編輯器（按 F5 或關閉後重新開啟分頁）。Graph Copilot 面板將以浮動小工具的形式出現在編輯器中。

外掛原始碼與問題回報：[github.com/treeleaves30760/CodefyUI-Plugin-Graph-Copilot](https://github.com/treeleaves30760/CodefyUI-Plugin-Graph-Copilot)

## 快速上手

1. 依上述步驟安裝外掛並重新載入編輯器。
2. 點擊畫布右下角的圓形 **Graph Copilot** 按鈕，開啟聊天面板。
3. 點擊**設定**（齒輪）圖示，選擇提供商並貼上你的 API 金鑰——若使用 **OpenAI Codex**，則點擊**登入**並在開啟的分頁中授權。接著挑選模型（可用 **Load list** 抓取該提供商的模型清單）。
4. 在輸入框輸入需求，例如 `建立一個小型 MLP 分類器`，按 **Enter**。
5. 隨著 AI 串流輸出計畫，節點會在畫布上自動出現並連線。按一次 **Ctrl+Z** 即可撤銷整批編輯，或繼續對話來微調。

提供商與金鑰只需設定一次——它們會保存在你的瀏覽器中。本頁其餘章節將詳細說明各部分。

## 選擇 LLM 提供商

點擊 Graph Copilot 面板中的設定圖示，即可設定你的提供商與金鑰。

| 提供商 | 說明 |
|--------|------|
| **OpenAI API** | 使用標準 `https://api.openai.com/v1` 端點，需要 OpenAI API 金鑰，按 token 計費。 |
| **OpenAI Codex（ChatGPT 登入）** | 使用 ChatGPT 網頁 session，無需另外申請 API 金鑰，但受 ChatGPT 使用配額限制，且自動化使用內部 session API 屬 OpenAI 服務條款的灰色地帶，並非官方認可的用途。 |
| **OpenRouter** | 在單一金鑰下彙整多個提供商。將 base URL 設為 `https://openrouter.ai/api/v1` 並選擇你偏好的模型。 |
| **Claude API** | Anthropic 的 API，透過 CodefyUI 代理存取，代理會將 OpenAI 相容格式的請求轉換後送出。需要 Anthropic API 金鑰。 |
| **自訂（OpenAI 相容）** | 任何實作 OpenAI `/chat/completions` 端點的伺服器——例如本機的 Ollama 實例：`http://localhost:11434/v1`。設定 base URL，並視需要填入金鑰。 |

## 金鑰處理

API 金鑰以 Graph Copilot 私有的命名空間儲存於 `localStorage`，絕不傳送至 CodefyUI 後端或任何第三方——只會傳送至你所設定的提供商。本機 CodefyUI 後端（`/api/llm/chat`）作為串流代理，將請求轉發至設定的提供商並串流回應；它不會記錄或持久化金鑰或訊息內容。

## 使用方式

### 發送請求

在聊天輸入框中輸入需求，按 Enter（或點擊送出）。範例：

- "新增一個含 ReLU 激活函式的兩層 MLP"
- "將 CrossEntropy 節點連接到最後一個 Linear 的輸出"
- "將 Linear-1 的 hidden size 設為 512"

AI 會回傳一份計畫，接著列出一系列操作。你可以看到每個操作以 chip 標籤的形式（例如「add Linear」、「add ReLU」、「connect」）在套用時逐一出現。

### 對話記錄

目前圖表的聊天記錄會儲存於 `localStorage`。重新開啟編輯器或重新載入頁面時，Graph Copilot 會從上次中斷的地方繼續對話。

### 中止與重試

在串流過程中點擊**停止**可取消進行中的請求，部分回應將被捨棄。點擊任意 AI 訊息上的**重試**，可在相同上下文中重新送出該輪對話。

### 撤銷 AI 編輯

每次 AI 編輯都是一個單一撤銷快照。按一次 **Ctrl+Z**（macOS 上為 Cmd+Z）即可撤銷最後一次 AI 回應的整批操作。

## 使用技巧

- 提供你正在構建的內容背景：「我正在構建一個採用 ResNet 骨幹網路的視覺分類器」有助於 AI 做出更好的選擇。
- 若 AI 新增了不在你面板中的節點類型，該操作會被跳過並回報——請先使用 `cdui plugin install` 安裝所需的外掛包。
- Graph Copilot 在每次請求前會讀取目前的圖表狀態與完整節點面板，因此它知道有哪些類型可用，以及畫布上已有什麼。

## 另請參閱

- [外掛前端擴充](/advanced/plugin-frontend-extensions) — Graph Copilot 所基於的 JS API。
- [外掛](/advanced/plugins) — 外掛包系統。
- [API 參考](/advanced/api-reference) — `/api/llm/chat` 串流端點。
