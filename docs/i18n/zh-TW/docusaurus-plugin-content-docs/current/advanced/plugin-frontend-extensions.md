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

編輯器每次頁面載入時呼叫 `activate` 一次，且**不會** await 它的回傳值——請以同步方式完成設定（你仍可啟動非同步工作，編輯器只是不會等待）。在 `activate` 內同步拋出的錯誤會被逐一外掛捕獲、記錄至瀏覽器主控台並以 toast 呈現；它們無法使編輯器或其他外掛崩潰。匯入另有 10 秒逾時限制。（只要求*預設匯出是一個函式*；名稱 `activate` 只是慣例。）

## CodefyUIPluginAPI v1 參考

### `api.ui` — 編輯器 UI

| 方法 | 簽名 | 說明 |
|------|------|------|
| `addFloatingWidget` | `({ id }) => HTMLElement` | 在編輯器的浮動元件堆疊中建立（或重用）一個容器 `<div>` 並回傳。`id` 在同一外掛內必須唯一。回傳的元素歸你所有——填入你自己的 DOM，或在其上掛載一個 React root。 |
| `toast` | `(message, level?) => void` | 顯示一個暫時性通知。`level` 為 `"info"`（預設）、`"warning"` 或 `"error"`。 |

### `api.graph` — 圖表讀寫

| 方法 | 簽名 | 說明 |
|------|------|------|
| `getGraph` | `() => GraphSnapshot` | 回傳目前圖表狀態（節點、邊、參數）的深層副本。 |
| `getNodeDefinitions` | `() => NodeDefinition[]` | 回傳完整的節點面板：型別、連接埠 schema、參數 schema。 |
| `applyOperations` | `(ops: GraphOp[]) => ApplyResult` | **同步**套用一批圖表操作（直接回傳結果，非 Promise）。整個批次以**單一撤銷快照**的形式提交。 |
| `onGraphChanged` | `(callback: (snapshot: GraphSnapshot) => void) => () => void` | 訂閱圖表變更事件。回傳一個取消訂閱函式。 |

#### GraphOp 表

所有操作類型都共用屬性 `op`（判別字串）。以下欄位名稱為精確值。

| `op` | 欄位 | 說明 |
|------|------|------|
| `"add_node"` | `node_type: string`、`ref?: string`、`params?: Record<string, unknown>`、`position?: { x: number; y: number }` | 新增指定類型的節點。`ref` 是呼叫端自選的別名，同一批次中後續操作可用它代替產生的節點 id。`position` 預設為錯落排列。 |
| `"connect"` | `source: string`、`source_handle: string`、`target: string`、`target_handle: string` | 連接一個輸出 handle 到一個輸入 handle。`source`/`target` 接受節點 id 或先前 `add_node` 的 `ref`。觸發邊請用 `source_handle: "trigger"`。 |
| `"set_params"` | `node_id: string`、`params: Record<string, unknown>` | 將參數值合併進節點。 |
| `"remove_node"` | `node_id: string` | 移除節點及所有與其相連的邊。 |
| `"remove_edge"` | `source: string`、`target: string`、`source_handle?: string`、`target_handle?: string` | 中斷兩節點間相符的邊。 |
| `"clear_graph"` | *（無）* | 移除所有節點與邊。 |
| `"auto_layout"` | *（無）* | 重新執行自動圖表佈局。 |

#### ApplyResult 形狀

```ts
interface OpResult {
  index: number;      // 操作在批次中的位置
  ok: boolean;        // 此操作是否套用成功
  error?: string;     // ok 為 false 時的失敗原因
  node_id?: string;   // 解析出的節點 id（add_node / set_params）
}

interface ApplyResult {
  results: OpResult[];            // 每個操作一筆，依輸入順序
  refs: Record<string, string>;  // ref 別名 -> 產生的節點 id
  node_count: number;            // 批次後的節點數
  edge_count: number;            // 批次後的邊數
}
```

**批次語義：** 單次 `applyOperations` 呼叫中的所有操作形成一個撤銷快照——在 AI 編輯後按 Ctrl+Z 會一次撤銷整個批次。操作依序套用；失敗的操作會被跳過並回報於其 `results` 條目（`ok: false` 加上 `error`），其餘操作仍會繼續。同一批次中先前 `add_node` 建立的 `ref` 別名可供後續操作使用，並會回傳於 `refs`。

### `api.nodes` — 自訂 node 渲染

需要 `api.apiVersion >= 2`。

| 方法 | 簽名 | 說明 |
|------|------|------|
| `registerRenderer` | `(nodeType, renderer) => () => void` | 用你自己的 UI 繪製某個外掛 node 型別的卡片內容。回傳一個取消註冊函式。 |

`nodeType` 是該 node 在 `getNodeDefinitions()` 中的**命名空間化**型別。注意命名空間是你外掛 id 的 snake_case 形式——外掛 `my-plugin` 對應 node 型別 `my_plugin:MyNode`。renderer 採命令式介面，讓宿主與框架無關：

```ts
interface NodeRenderContext {
  node: { id: string; type: string; params: Record<string, unknown> };
}
interface PluginNodeRenderer {
  mount(container: HTMLElement, ctx: NodeRenderContext): void;
  update?(container: HTMLElement, ctx: NodeRenderContext): void; // 參數變更時
  unmount?(container: HTMLElement): void;
}
```

編輯器仍會渲染標準的 node 卡片（標題、連接埠、參數列），並把一個 `<div>` 交給你的 renderer 當作**內容區**——位於連接埠與參數之間。沒有註冊 renderer 的 node 型別，渲染結果與預設 node 完全相同。

```js
api.nodes.registerRenderer('my_plugin:MyNode', {
  mount(el, ctx) { el.textContent = `value: ${ctx.node.params.value}`; },
  update(el, ctx) { el.textContent = `value: ${ctx.node.params.value}`; },
});
```

[外掛模板](https://github.com/treeleaves30760/CodefyUI-Plugin-Official)的 SDK 會用 `createRoot` 包裝它，讓你能以 React 元件撰寫內容區。

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

以下片段僅使用原始 API——不需建置步驟、不需框架：一個按鈕，插入兩個節點並將它們連接起來。（真正的 React 面板請參考 Graph Copilot 外掛原始碼。）

```js
// frontend/index.js
export default function activate(api) {
  const btn = document.createElement("button");
  btn.textContent = "Insert Linear + ReLU";
  btn.style.cssText =
    "padding:6px 12px;background:#0d9488;color:#fff;border:none;border-radius:4px;cursor:pointer";

  btn.addEventListener("click", () => {
    // applyOperations 是同步的——不需 await。
    const result = api.graph.applyOperations([
      { op: "add_node", node_type: "Linear", ref: "lin1", position: { x: 200, y: 200 } },
      { op: "add_node", node_type: "ReLU",   ref: "relu1", position: { x: 440, y: 200 } },
      // handle 名稱（此處的 "output"/"input"）來自各節點的連接埠 schema——
      // 呼叫 api.graph.getNodeDefinitions() 來查詢。
      { op: "connect",
        source: "lin1", source_handle: "output",
        target: "relu1", target_handle: "input" },
    ]);
    const failed = result.results.filter((r) => !r.ok);
    if (failed.length > 0) {
      api.ui.toast(`部分操作失敗：${failed.map((r) => r.error).join(", ")}`, "warning");
    }
  });

  // addFloatingWidget 回傳一個容器 <div>，由你自行填入內容。
  const panel = api.ui.addFloatingWidget({ id: "demo-insert-panel" });
  panel.appendChild(btn);
}
```

## 另請參閱

- [外掛](/advanced/plugins) — 安裝外掛包、資訊清單格式與 `cdui plugin` CLI。
- [Graph Copilot](/advanced/graph-copilot) — 前端擴充 API 的首個正式消費者。
- [API 參考](/advanced/api-reference) — 後端 REST 端點，包括 `/api/llm/chat`。
