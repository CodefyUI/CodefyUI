---
sidebar_position: 4
title: 外掛前端擴充
description: 隨外掛包附上一個 JavaScript bundle，讓外掛能新增 UI 小工具、檢視圖表並驅動編輯器——Graph Copilot 等工具的基礎。
---

# 外掛前端擴充

外掛包可以在 Python 節點之外，附上一個 JavaScript bundle。CodefyUI 編輯器載入時，會探索並以 ES 模組形式匯入該 bundle，讓外掛取得一個穩定的 JavaScript API，用於操作 UI、圖表及代理 HTTP 請求。

:::note 可用性
前端擴充功能自 CodefyUI **1.3.0** 起內建。請執行 `cdui --version` 確認；若顯示更舊的版本，請執行 `cdui update`。
:::

## 宣告前端進入點

在 `cdui.plugin.toml` 中加入 `[frontend]` 區段：

```toml
[plugin]
id = "my-plugin"
name = "My Plugin"
version = "0.1.0"
requires_codefyui = ">=1.3.0"

[frontend]
entry = "frontend/index.js"
```

`requires_codefyui` 為提示性中繼資料（會被記錄，但目前安裝時並不強制檢查）；請將它設為首個內建你外掛所需功能的 CodefyUI 版本——前端擴充功能於 1.3.0 登場。

`entry` 路徑必須**相對於外掛根目錄**，且必須位於 `frontend/` 之下。該檔案必須是合法的 ES 模組，並包含一個預設匯出（參見下方的[activate 合約](#activate-合約)）。

## 編輯器如何提供並探索 bundle

後端啟動時，會將每個已安裝外掛的 `frontend/` 目錄掛載於：

```
/plugins/<plugin-id>/frontend/<file>
```

外掛列表端點會揭露進入點，讓編輯器得以載入：

```
GET /api/plugins
```

回應範例（節錄）：

```json
{
  "id": "my-plugin",
  "name": "My Plugin",
  "version": "0.1.0",
  "frontend_entry": "/plugins/my-plugin/frontend/index.js"
}
```

若 `frontend_entry` 為 `null`，代表該外掛沒有前端 bundle。只有當 `frontend_entry` 非 null 時，編輯器才會載入該模組。

## activate 合約

你的 bundle 必須匯出一個名為 `activate` 的單一預設函式。編輯器在所有外掛載入完成後，於啟動時呼叫一次該函式，並傳入 `CodefyUIPluginAPI` 物件：

```js
// frontend/index.js
export default function activate(api) {
  // api 是一個 CodefyUIPluginAPI 實例
}
```

`activate` 可以是 `async`。編輯器會等待它完成後，才將外掛標記為就緒。在 `activate` 內部拋出的錯誤會被捕獲並記錄至瀏覽器主控台，不會使其他外掛崩潰。

## CodefyUIPluginAPI v1 參考

### `api.ui` — 編輯器 UI

| 方法 | 簽名 | 說明 |
|------|------|------|
| `addFloatingWidget` | `(id, element, options?) => void` | 將任意 DOM 元素掛載為可拖曳的浮動面板。`id` 必須唯一。`options.title` 設定面板標頭。 |
| `toast` | `(message, level?) => void` | 顯示一個暫時性通知。`level` 為 `"info"`（預設）、`"warning"` 或 `"error"`。 |

### `api.graph` — 圖表讀寫

| 方法 | 簽名 | 說明 |
|------|------|------|
| `getGraph` | `() => GraphSnapshot` | 回傳目前圖表狀態（節點、邊、參數）的深層副本。 |
| `getNodeDefinitions` | `() => NodeDefinition[]` | 回傳完整的節點面板：型別、連接埠 schema、參數 schema。 |
| `applyOperations` | `(ops: GraphOp[]) => Promise<ApplyResult>` | 套用一批圖表操作。整個批次以**單一撤銷快照**的形式提交。 |
| `onGraphChanged` | `(callback: (snapshot: GraphSnapshot) => void) => () => void` | 訂閱圖表變更事件。回傳一個取消訂閱函式。 |

#### GraphOp 表

所有七種操作類型都共用屬性 `op`（判別字串）。

| `op` | 必填欄位 | 說明 |
|------|----------|------|
| `"add_node"` | `type: string`、`id?: string`、`x?: number`、`y?: number` | 新增指定類型的節點。若省略 `id`，則自動產生。 |
| `"remove_node"` | `id: string` | 移除節點及所有與其相連的邊。 |
| `"add_edge"` | `from_node: string`、`from_port: string`、`to_node: string`、`to_port: string` | 連接兩個相容的連接埠。 |
| `"remove_edge"` | `from_node: string`、`from_port: string`、`to_node: string`、`to_port: string` | 中斷指定的邊。 |
| `"set_param"` | `node_id: string`、`param: string`、`value: unknown` | 設定節點參數值。 |
| `"move_node"` | `id: string`、`x: number`、`y: number` | 在畫布上重新定位節點。 |
| `"clear_graph"` | *（無）* | 移除所有節點與邊。 |

#### ApplyResult 形狀

```ts
interface ApplyResult {
  ok: boolean;           // 若所有操作均成功則為 true
  applied: string[];     // 已套用的操作 id
  failed: { op: GraphOp; reason: string }[];  // 被跳過的操作
}
```

**批次語義：** 單次 `applyOperations` 呼叫中的所有操作形成一個撤銷快照——在 AI 編輯後按 Ctrl+Z 會一次撤銷整個批次。操作依序套用；失敗的操作會被跳過並回報於 `failed`，但其餘操作仍會繼續。同一批次中先前 `add_node` 建立的節點 `id`，可供該批次後續操作引用。

### `api.http` — 具 session 意識的 fetch

| 方法 | 簽名 | 說明 |
|------|------|------|
| `fetch` | `(path: string, init?: RequestInit) => Promise<Response>` | 與瀏覽器的 `fetch` API 完全相同，但會自動附加 CodefyUI session token 標頭。`path` 必須是相對路徑（例如 `/api/llm/chat`）。所有對 CodefyUI 後端的呼叫都應使用此方法。 |

### `api.storage` — 命名空間鍵值儲存

儲存以 `localStorage` 為後端，並自動以你的外掛 id 進行命名空間隔離，因此不同外掛之間不會發生衝突。

| 方法 | 簽名 | 說明 |
|------|------|------|
| `get` | `(key: string) => string \| null` | 取回已儲存的值。 |
| `set` | `(key: string, value: string) => void` | 儲存一個值。 |
| `remove` | `(key: string) => void` | 刪除一個鍵。 |

## 信任模型

外掛 JavaScript 在編輯器頁面內執行，對**編輯器 DOM、圖表狀態和 session token 擁有完整存取權**。請只安裝來自你信任來源的外掛。每當外掛宣告前端進入點時，`cdui plugin install` CLI 都會列印警告。

後端的 AST 安全閘門適用於外掛 Python；外掛 JavaScript 並無沙盒機制——它以與編輯器本身相同的信任層級執行。

## 最小可運作範例

以下片段是官方 Graph Copilot demo 所使用的模式。它新增一個工具列按鈕，插入兩個相容節點並將它們連接起來。

```js
// frontend/index.js
export default function activate(api) {
  const btn = document.createElement("button");
  btn.textContent = "Insert Linear + ReLU";
  btn.style.cssText =
    "padding:6px 12px;background:#0d9488;color:#fff;border:none;border-radius:4px;cursor:pointer";

  btn.addEventListener("click", async () => {
    const result = await api.graph.applyOperations([
      { op: "add_node", type: "Linear", id: "lin1", x: 200, y: 200 },
      { op: "add_node", type: "ReLU",   id: "relu1", x: 440, y: 200 },
      { op: "add_edge",
        from_node: "lin1", from_port: "output",
        to_node: "relu1", to_port: "input" },
    ]);
    if (!result.ok) {
      api.ui.toast(`部分操作失敗：${result.failed.map(f => f.reason).join(", ")}`, "warning");
    }
  });

  api.ui.addFloatingWidget("demo-insert-panel", btn, { title: "Demo" });
}
```

## 另請參閱

- [外掛](/advanced/plugins) — 安裝外掛包、資訊清單格式與 `cdui plugin` CLI。
- [Graph Copilot](/advanced/graph-copilot) — 前端擴充 API 的首個正式消費者。
- [API 參考](/advanced/api-reference) — 後端 REST 端點，包括 `/api/llm/chat`。
