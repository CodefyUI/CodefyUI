---
sidebar_position: 4
title: 外掛前端擴充
description: 隨外掛包附上一個 JavaScript bundle，讓外掛能新增 UI 小工具、檢視圖表並驅動編輯器——Graph Copilot 等工具的基礎。
---

# 外掛前端擴充

外掛包可以在 Python 節點之外，附上一個 JavaScript bundle。CodefyUI 編輯器載入時，會探索並以 ES 模組形式匯入該 bundle，讓外掛取得一個穩定的 JavaScript API，用於操作 UI、圖表及代理 HTTP 請求。

:::note 可用性
前端擴充功能自 CodefyUI **1.3.0** 起內建。請執行 `cdui --version` 確認；若顯示更舊的版本，請執行 `cdui update`。

停靠面板、工具列按鈕、執行事件與執行歷史介面（`api.runs`）需要 **apiVersion 3**（CodefyUI 2.0.0 起）；`graph.getView` 需要 **apiVersion 4**（CodefyUI 2.3.0 起）；`api.workspace` 與六個代理畫布操作需要 **apiVersion 5**（CodefyUI 2.5.0 起）。使用前請先檢查功能版本，詳見 [API 版本](#api-版本)。
:::

## API 版本

`api.apiVersion` 只會增加；到目前為止，每個版本的 API **都只新增介面**：舊版支援的功能未被移除，方法簽名也未變更。為 apiVersion 2 撰寫的外掛可直接在 apiVersion 5 編輯器上運作。apiVersion 5 只有一項例外，詳見下方的[唯讀分頁現在會拒絕寫入](#apiversion-5-之前的外掛要注意的一件事)。

| `apiVersion` | CodefyUI | 新增內容 |
|--------------|----------|----------|
| 1 | 1.3.0 | `ui.addFloatingWidget`、`ui.toast`、`graph.*`、`http.fetch`、`storage.*` |
| 2 | 1.3.0 | `nodes.registerRenderer` |
| 3 | 2.0.0 | `ui.addPanel` / `removePanel`、`ui.addToolbarButton` / `removeToolbarButton`、`events.onExecution`、`runs.*` |
| 4 | 2.3.0 | `graph.getView`——使用者正在看圖表的哪一層 |
| 5 | 2.5.0 | `workspace.*`——分頁、快照與比較後寫入；`move_node`、`set_segment` / `remove_segment`、`add_note` / `update_note`、`set_node_meta` |

使用超過外掛最低需求版本的新功能前，請先檢查目前的 API 版本，並在版本不足時降級處理，不要直接拋出錯誤：

```js
export default function activate(api) {
  if (api.apiVersion >= 3) {
    mountDashboard(api.ui.addPanel({ id: "dash", title: "Dashboard" }));
  } else {
    mountDashboard(api.ui.addFloatingWidget({ id: "dash" }));
  }
}
```

若有破壞性變更，會提升 `apiVersion` 並附上遷移說明，不會在既有版本中無預警變更。

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

`requires_codefyui` 是提示性中繼資料：系統會記錄，但目前不會在安裝時強制檢查。請將它設為第一個包含外掛所需功能的 CodefyUI 版本；前端擴充自 1.3.0 起提供。

`entry` 路徑必須**相對於外掛根目錄**，且必須位於 `frontend/` 之下。該檔案必須是合法的 ES 模組，並包含一個預設匯出（參見下方的[activate 合約](#activate-合約)）。

## 編輯器如何提供並探索 bundle

後端會在以下路徑提供已啟用外掛的 `frontend/` 目錄：

```
/plugins/<plugin-id>/frontend/<file>
```

這些目錄不會在啟動時掛載。每個 request 都會由 route 從 lockfile 解析外掛目錄。因此，bundle 會在安裝或重載後立即可用，並在外掛停用或解除安裝後立即回傳 `404`；兩種變更都不需要重新啟動。只有已啟用且 manifest 宣告 `[frontend].entry` 的外掛，才會提供 frontend 檔案；只有 `frontend/` 目錄並不足夠。response 使用 `Cache-Control: no-cache`，讓瀏覽器重新驗證檔案，並在下次載入時取得更新。後端會透過 `/plugins/<plugin-id>/assets/<file>`，依相同的啟用規則提供 `assets/`，使用 GET 與 HEAD 且不要求 manifest entry。

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

manifest 沒有 `[frontend]` entry、宣告的檔案不存在，或外掛已停用時，`frontend_entry` 會是 `null`。停用的外掛不提供 UI、frontend 檔案或 assets。只有當 `frontend_entry` 非 null 時，編輯器才會載入該模組。

## activate 合約

你的 bundle 必須匯出一個名為 `activate` 的單一預設函式。編輯器在所有外掛載入完成後，於啟動時呼叫一次該函式，並傳入 `CodefyUIPluginAPI` 物件：

```js
// frontend/index.js
export default function activate(api) {
  // api 是一個 CodefyUIPluginAPI 實例
}
```

編輯器每次載入頁面時會呼叫 `activate` 一次，但**不會** await 回傳值。請同步完成初始化；函式仍可啟動非同步工作，但編輯器不會等待。`activate` 內同步拋出的錯誤會按外掛個別捕捉，記錄至瀏覽器主控台，並顯示 toast，不會使編輯器或其他外掛崩潰。模組匯入另有 10 秒逾時限制。（唯一要求是*預設匯出必須為函式*；`activate` 只是慣用名稱。）

## CodefyUIPluginAPI 參考

### `api.ui` — 編輯器 UI

| 方法 | 簽名 | 說明 |
|------|------|------|
| `addFloatingWidget` | `({ id }) => HTMLElement` | 在編輯器的浮動元件堆疊中建立或重用容器 `<div>`，並回傳該元素。`id` 在同一外掛內必須唯一。外掛可自行填入 DOM，或在元素上掛載 React root。 |
| `toast` | `(message, level?) => void` | 顯示一個暫時性通知。`level` 為 `"info"`（預設）、`"success"`、`"warning"` 或 `"error"`。 |
| `addPanel` | `(opts) => HTMLElement` | **apiVersion 3。** 註冊一個停靠面板，回傳它的容器元素。 |
| `removePanel` | `(id: string) => void` | **apiVersion 3。** 移除你自己的某個面板。 |
| `addToolbarButton` | `(opts) => () => void` | **apiVersion 3。** 新增一個工具列按鈕；回傳移除函式。 |
| `removeToolbarButton` | `(id: string) => void` | **apiVersion 3。** 依 id 移除你自己的某個按鈕。 |

#### 停靠面板

需要 `api.apiVersion >= 3`。

```ts
interface PluginPanelOptions {
  id: string;                 // 同一外掛內唯一
  title: string;              // 分頁標籤，或右側區塊標題
  icon?: string;              // 顯示在標題前的短符號
  dock?: "bottom" | "right";  // 預設為 "bottom"
  onShow?: () => void;        // 元素已掛入文件
  onHide?: () => void;        // ……又被卸下
}
```

`"bottom"` 面板會成為編輯器底部 dock 的分頁，排在**執行紀錄**、**訓練**與**執行任務**之後。`"right"` 面板會成為右側欄的區塊，與**節點設定**及檢視器面板並列。分頁外框、順序與位置由主程式管理；外掛則負責元素內的所有內容。

**面板存續期間會重用同一個元素。** 元素本身不會更換，因此只需掛載一次：

```js
const el = api.ui.addPanel({ id: "runs", title: "My Runs", icon: "~" });
createRoot(el).render(<MyPanel />);   // 只做一次，不是每次切分頁
```

編輯器只會掛載目前作用中的 dock 分頁。使用者切換分頁時，包含面板的主程式容器會拆除並重建，但面板元素只會卸離再重新附加，其子元素與狀態都會保留。使用相同 `id` 再次呼叫 `addPanel` 會回傳同一個元素，並更新標題、圖示與 dock。

**需要注意面板隱藏時的執行狀態。** 面板不在畫面上時，你的程式仍會執行，並渲染到目前不在 document 中的元素。若面板包含成本較高的工作，例如圖表、輪詢或動畫，請依顯示狀態啟動與停止：

```js
api.ui.addPanel({
  id: "runs", title: "My Runs",
  onShow: () => chart.start(),
  onHide: () => chart.stop(),
});
```

請把這兩個 callback 寫成連續呼叫兩次也無害：React 的開發模式會重播掛載 effect，所以單次切換分頁有可能多產生一組 `onHide`/`onShow`。

外掛被卸載或熱重載時，面板會自動移除；`removePanel` 是給你想更早移除的面板用的。

#### 工具列按鈕

需要 `api.apiVersion >= 3`。

```ts
interface PluginToolbarButtonOptions {
  id: string;        // 同一外掛內唯一
  icon: string;      // 短符號——工具列的空間只容得下一個符號
  tooltip: string;   // 滑鼠提示與無障礙文字；圖示本身不算標籤
  onClick: () => void;
}
```

```js
const remove = api.ui.addToolbarButton({
  id: "sweep", icon: "~", tooltip: "Start a sweep",
  onClick: () => startSweep(),
});
```

按鈕會依註冊順序顯示在工具列右側的同一組。外掛無法指定位置，顯示數量也由編輯器決定：寬視窗最多直接顯示三個，窄視窗則收合至單一 overflow 選單。這可避免五個已安裝外掛把**執行**移出工具列。請將 `tooltip` 寫成完整標籤，因為選單會用它作為標籤。

若 `onClick` 拋出錯誤，編輯器會記錄下來並繼續運作；工具列不受影響。

再次註冊相同 id 會取代原有按鈕。回傳的移除函式只對應該次註冊；若按鈕之後已被取代，呼叫舊的移除函式不會移除新按鈕。若要移除該 id 目前對應的按鈕，請呼叫 `removeToolbarButton(id)`。外掛卸載或熱重載時，按鈕會自動移除。

### `api.graph` — 圖表讀寫

| 方法 | 簽名 | 說明 |
|------|------|------|
| `getGraph` | `() => SerializedGraph` | 回傳**完整**圖表狀態的深層副本，包括節點、邊、參數及 `subgraphs` 中的區塊定義。不論使用者目前開啟哪一層，這個方法一律回傳頂層圖表。 |
| `getNodeDefinitions` | `() => NodeDefinition[]` | 回傳完整的節點面板：型別、連接埠 schema、參數 schema。 |
| `applyOperations` | `(ops: GraphOp[]) => ApplyResult` | **同步**套用一批圖表操作（直接回傳結果，非 Promise）。整個批次會建立**單一復原快照**，並套用至使用者目前開啟的畫布——參見[使用者正在看哪一層](#使用者正在看哪一層)。 |
| `onGraphChanged` | `(callback: () => void) => () => void` | 訂閱圖表變更事件，包括使用者進入或離開區塊。callback 不帶參數；請在其中呼叫 `getGraph()` 取得內容。回傳取消訂閱函式。 |
| `getView` | `() => GraphView` | **apiVersion 4。** 唯讀：使用者正在看圖表的哪一層。 |

#### GraphOp 表

十三種操作類型都共用屬性 `op`（判別字串）。以下欄位名稱為精確值。

| `op` | 欄位 | 說明 |
|------|------|------|
| `"add_node"` | `node_type: string`、`ref?: string`、`params?: Record<string, unknown>`、`position?: { x: number; y: number }` | 新增指定類型的節點。`ref` 是呼叫端自選的別名，同一批次中後續操作可用它代替產生的節點 id。`position` 預設為錯落排列。 |
| `"connect"` | `source: string`、`source_handle: string`、`target: string`、`target_handle: string` | 連接一個輸出 handle 到一個輸入 handle。`source`/`target` 接受節點 id 或先前 `add_node` 的 `ref`。觸發邊請用 `source_handle: "trigger"`。 |
| `"set_params"` | `node_id: string`、`params: Record<string, unknown>` | 將參數值合併進節點。 |
| `"remove_node"` | `node_id: string` | 移除節點及所有與其相連的邊。 |
| `"remove_edge"` | `source: string`、`target: string`、`source_handle?: string`、`target_handle?: string` | 中斷兩節點間相符的邊。 |
| `"clear_graph"` | *（無）* | 移除所有節點與邊。 |
| `"auto_layout"` | *（無）* | 重新執行自動圖表佈局。 |
| `"move_node"` | `node_id: string`、`position: { x: number; y: number }` | **apiVersion 5。** 將節點放到指定位置。綁定至該節點的註記會隨之移動，與使用者拖曳節點時相同。 |
| `"set_segment"` | `segment_id?: string`、`head_node_id: string`、`tail_node_id: string` | **apiVersion 5。** 建立或取代段落標示；編輯器會以外框包住從起點到終點之資料路徑上的所有節點。省略 `segment_id` 會建立標示；傳入既有 id 會移動標示。兩種結果都會在 `segment_id` 回傳 id。若起點與終點之間沒有資料邊路徑，或任一端是註記，操作就會失敗。 |
| `"remove_segment"` | `segment_id: string` | **apiVersion 5。** 移除段落標示。 |
| `"add_note"` | `ref?: string`、`text: string`、`position?: { x: number; y: number }`、`color?: string`、`bind_to?: string` | **apiVersion 5。** 新增文字註記。`text` 長度為 1 到 4000 個字元，可含換行與 tab，但不可含其他控制字元；`color` 為 `#rrggbb`；`bind_to` 會將註記綁定至節點，使其隨節點移動，預設位置則在節點旁。註記不會被執行、匯出或驗證。 |
| `"update_note"` | `node_id: string`、`text?: string`、`color?: string` | **apiVersion 5。** 更新既有註記。`text` 與 `color` 至少要提供一項；`text` 只適用於文字註記，圖片註記的內容是 data URL。 |
| `"set_node_meta"` | `node_id: string`、`label: string` | **apiVersion 5。** 為節點命名。標籤長度為 1 到 120 個字元，僅限單行，並會移除前後空白。標籤與 `params` 並列儲存，不會置於其中，而且可在儲存與重新載入後保留。 |

#### ApplyResult 形狀

```ts
interface OpResult {
  index: number;      // 操作在批次中的位置
  ok: boolean;        // 此操作是否套用成功
  error?: string;     // ok 為 false 時的失敗原因
  node_id?: string;   // 解析出的節點 id，凡是建立或修改節點的操作都會帶；remove_node 不帶
  segment_id?: string; // apiVersion 5：set_segment 建立或取代的那個 id
}

interface ApplyResult {
  results: OpResult[];            // 每個操作一筆，依輸入順序
  refs: Record<string, string>;  // ref 別名 -> 產生的節點 id
  node_count: number;            // 批次後的節點數
  edge_count: number;            // 批次後的邊數
}
```

**批次語義：** 單次 `applyOperations` 呼叫中的所有操作會形成一個復原快照。AI 編輯完成後按 Ctrl+Z，會一次復原整個批次。操作依序套用；失敗的操作會略過，並在對應的 `results` 項目回報（`ok: false` 與 `error`），其餘操作仍會繼續。同一批次中，先前由 `add_node` 建立的 `ref` 別名可供後續操作使用，也會回傳於 `refs`。

#### 使用者正在看哪一層

需要 `api.apiVersion >= 4`。

CodefyUI 圖表支援巢狀結構。每個**區塊**（subgraph）都有自己的畫布，使用者可以進入區塊；此時畫布上方的路徑列會顯示 `主圖 > Encoder`。編輯器只有一張畫布，進入區塊時會將區塊內容換到該畫布上，因此編輯工具在區塊內外的行為相同。

這會決定外掛編輯套用的位置：

- **`getGraph()` 一律回傳完整圖表。** 編輯器會在序列化前把目前開啟的區塊內容合併回完整圖表，與**儲存**及**執行**採用相同流程，因此內容和使用者儲存的檔案相同。
- **`applyOperations()` 會寫入使用者目前開啟的畫布。** 在區塊內，`add_node` 會將節點加入*該區塊*，`clear_graph` 也只會清空*該區塊*，不會清空完整圖表。從 `getGraph()` 取得的頂層節點 id 不存在於區塊畫布中，因此指定這些 id 的操作會回傳 `ok: false` 與錯誤訊息。

因此，外掛可能根據完整圖表得出正確結果，卻將結果寫入使用者目前未查看的層級。請先使用 `getView()` 判斷目前層級：

```ts
interface GraphViewLevel {
  subgraphId: string;  // 區塊定義的 id，與 getGraph() 中指涉它的方式相同
  name: string;        // 區塊的名稱，與畫布上方麵包屑顯示的一致
}

interface GraphView {
  depth: number;           // 最上層是 0，在一個區塊裡是 1，在區塊裡的區塊裡是 2
  path: GraphViewLevel[];  // 已打開的區塊，由外而內；在最上層時為空陣列
  atTopLevel: boolean;     // depth === 0，也就是你通常真正想做的那個判斷
}
```

```js
const view = api.graph.getView();
if (!view.atTopLevel) {
  const inside = view.path[view.path.length - 1].name;
  api.ui.toast(`請先離開「${inside}」——現在編輯會寫進那個區塊裡面。`, "warning");
  return;
}
api.graph.applyOperations(ops);
```

外掛不一定要拒絕寫入，也可以等待使用者離開區塊，或將編輯限制在適合該區塊的範圍。這項檢查讓外掛能明確選擇處理方式，不必依賴目前畫布的偶然狀態。

這個 view 是**唯讀**且即時的：每次呼叫都會取得目前狀態。API 刻意不提供讓外掛替使用者切換圖表層級的方法。使用者進入或離開區塊時，`onGraphChanged` 會因畫布變更而觸發；顯示寫入目標的面板可在該 callback 中重新呼叫 `getView()`。

寫入目標層級是編輯器既有的行為，本次只將其明確記錄。未來版本可能讓操作明確指定目標層級；該功能會以新增介面的方式提供，不會改變既有外掛的寫入目標。

### `api.workspace` — 分頁與比較後寫入

需要 `api.apiVersion >= 5`。舊版編輯器中的 `api.workspace` 為 `undefined`，不會提供一組只會拋出錯誤的 stub 方法，因此可用 `typeof api.workspace?.openGraphs === "function"` 正確檢查功能是否存在。

`api.graph` 存取使用者目前開啟的圖表；`api.workspace` 則存取整個分頁列。外掛可將提案開在另一個分頁，讓使用者檢查而不修改原本的工作，並只在原圖未於期間變更時寫回。

| 方法 | 簽名 | 說明 |
|------|------|------|
| `openGraphs` | `(entries, options?) => WorkspaceOpenResult[]` | 將一張或多張圖表開啟為編輯器分頁。結果會**依位置對應**：`result[i]` 描述 `entries[i]`，單一無效項目不會影響其他項目。 |
| `tabs` | `() => WorkspaceTabInfo[]` | 依分頁列順序列出每個分頁，使用者正在看的那一個帶有 `active`。 |
| `snapshot` | `(tabId?) => WorkspaceSnapshot` | 回傳分頁識別資訊與完整圖表。省略 id 時使用作用中分頁；id 不存在時回傳 `{ error: "unknown_tab" }`，不會拋出錯誤。 |
| `applyOperations` | `(request) => WorkspaceApplyResult` | 對指定分頁套用一個批次，可選擇只在版本號仍相符時才寫入，也可選擇全有全無。 |
| `onChanged` | `(callback) => () => void` | 訂閱所有分頁的分頁與文件變更。回傳一個取消訂閱函式。 |

#### 版本號

每個分頁都有一個 `revision`，初始值為 1。分頁文件每次變更時，該值會加一。拖曳節點、復原與重做都會變更文件；復原與重做雖然還原舊內容，仍屬於變更。重新命名或切換分頁、選取節點、平移畫布及標示段落不會增加版本號。執行圖表也不會增加版本號，因為節點上的執行狀態、錯誤與進度只顯示於畫布，不會寫入存檔。因此，模型訓練期間不會讓比較後寫入的版本號每秒失效多次。

版本號只會增加，並與分頁一起儲存，因此重新載入前保存的版本號在載入後仍有效。外掛可保存版本號，進行較長時間的處理，再將該版本號連同寫入要求一起提交。

```ts
interface WorkspaceTabInfo {
  tabId: string;
  title: string;
  revision: number;
  readOnly: boolean;
  transient: boolean;               // gone after a reload
  source: WorkspaceSource | null;   // who opened it
  active: boolean;
}

interface WorkspaceSource {
  kind: string;         // your own label, e.g. "agent-variant"
  pluginId: string;
  jobId?: string;
  variantId?: string;
  [key: string]: unknown;   // opaque to the editor, handed back verbatim
}
```

#### 開啟圖表

```ts
const opened = api.workspace.openGraphs(
  candidates.map((c) => ({
    title: `${hypothesis} — ${c.label}`,
    graph: c.graph,                 // the shape getGraph() answers with
    readOnly: true,
    source: { kind: "agent-variant", pluginId: api.pluginId, variantId: c.id },
  })),
  { activate: "first" },
);

for (const [i, result] of opened.entries()) {
  if ("error" in result) {
    api.ui.toast(`Candidate ${i + 1} not opened: ${result.error}`, "error");
    continue;
  }
  remember(result.tabId, result.revision);
}
```

每一筆項目依此順序驗證，失敗時產生一筆帶有 `error` 句子與 `code` 的結果：

1. `title` 必須是非空字串——`invalid_graph`。
2. `graph` 必須能通過 `JSON.stringify`——`invalid_graph`。
3. 該 JSON 最多 8 MiB——`too_large`。
4. 圖表必須能由編輯器的文件讀取器讀取，也就是開啟範例圖庫中的範例時使用的同一個讀取器。節點的 `params` 會完全保留項目提供的值，不會從節點定義或 preset 補入內容；省略的參數會維持省略。未知的頂層 key 會忽略，`subgraphs`、`segmentGroups` 與 `presets` 會保留；若 `format_version` 高於編輯器支援的版本，分頁會和一般檔案一樣以唯讀模式開啟。讀取失敗時回傳 `invalid_graph` 與讀取器的原始訊息。
5. 開啟後不得讓編輯器超過 32 個分頁——`too_many_tabs`。

分頁數會最後檢查，並使用檢查當下的分頁數量。因此，如果只剩一個分頁額度，同一次呼叫中的兩個項目不會同時通過。由於分頁數最後才檢查，以 `too_many_tabs` 拒絕的項目已經由讀取器處理；即使未開啟分頁，其中包含且伺服器尚未見過的 preset 仍會合併至節點面板。

`options.activate` 可設為 `"first"`（預設）、`"last"` 或 `"none"`；最後一個選項不會切換使用者目前的分頁。

開啟後即為一般分頁。使用者可以重新命名、關閉，或透過**另存新檔...**存成圖表檔案；重新開啟該檔案時會依一般流程建立可編輯分頁。`readOnly` 屬於分頁本身，直到分頁關閉為止。

外掛開啟的分頁預設為**暫時分頁**：不會寫入編輯器自動存檔，重新載入後會消失。若候選方案需要在重新載入後保留，請傳入 `persist: true`。API 刻意不提供 `closeTab`；是否關閉使用者正在查看的分頁，由使用者決定。

#### 在比較後寫入

```ts
const before = api.workspace.snapshot();          // the active tab
const armed = { tabId: before.tabId, revision: before.revision };

// ...minutes pass, experiments run...

const result = api.workspace.applyOperations({
  tabId: armed.tabId,
  expectedRevision: armed.revision,
  operations: winner.operations,
  atomic: true,
});

if (result.conflict === "revision_mismatch") {
  // result.revision is the CURRENT one, so you can re-arm without re-reading.
  api.ui.toast("The graph changed while the study was running.", "warning");
} else if (result.conflict === "read_only") {
  api.ui.toast("That tab is read-only — promote into an editable one.", "warning");
} else if (result.conflict === "editing_subgraph") {
  api.ui.toast("Step out of the block first — the write is waiting.", "warning");
} else if (!result.committed) {
  const failed = result.results.filter((r) => !r.ok);
  api.ui.toast(`Nothing applied: ${failed.map((r) => r.error).join("; ")}`, "error");
} else {
  armed.revision = result.revision;                // continue the chain
}
```

檢查依下列順序進行；任一檢查失敗都會直接回傳，不會變更內容：

1. `tabId`（或作用中分頁）必須存在，否則回傳 `conflict: "unknown_tab"`、`results: []`、`committed: false`、`revision: 0`。
2. 分頁不能是唯讀，否則回傳 `conflict: "read_only"`、`results: []`、`committed: false` 與分頁目前的 `revision`。
3. 分頁不能正在顯示區塊內部，否則回傳 `conflict: "editing_subgraph"`。區塊開啟時，畫布包含區塊內容，而不是 `snapshot()` 描述的文件；此時寫入會套用至外掛未讀取的內容。請在使用者離開區塊後重試。
4. 若有傳入 `expectedRevision`，其值必須等於分頁的 `revision`；否則回傳 `conflict: "revision_mismatch"` 與**目前**版本號，讓外掛不需再次讀取即可更新預期版本。
5. 批次會套用至副本。
6. 使用 `atomic: true` 時，只要任何操作失敗，就不會寫入：`committed: false`、`revision` 不變，並回傳**完整長度**的 `results`，以指出失敗的操作。
7. 否則，只要內容有變更，就建立一個復原快照並執行一次寫入，回傳 `committed: true` 與新的 `revision`。未變更內容的批次不會寫入，也不會建立復原步驟。

每當呼叫未提交內容——包括拒絕、`atomic` 前置檢查失敗或批次未變更內容——`node_count` 與 `edge_count` 都描述分頁目前的狀態，不會回傳已捨棄副本的計數。`unknown_tab` 是例外，因為沒有可計數的分頁。

衝突會**以結果回傳，不會拋出錯誤**。每個批次仍只建立一個復原步驟，每項操作的語義也與 `api.graph.applyOperations` 相同：未使用 `atomic` 時，失敗的操作會略過並回報，其餘操作仍會套用。若提交後的圖表已移除詳細資料視窗或畫布選取所指向的節點，兩者都會清除；之後復原該節點時，不會自動重新開啟詳細資料視窗。

```ts
type WorkspaceConflict =
  "revision_mismatch" | "read_only" | "unknown_tab" | "editing_subgraph";

interface WorkspaceApplyResult extends ApplyResult {
  tabId: string;
  revision: number;        // AFTER this call; unchanged on a conflict or a preflight failure
  committed: boolean;
  conflict?: WorkspaceConflict;
}
```

#### 監看所有分頁

```ts
const off = api.workspace.onChanged((event) => {
  if (event.type === "graph" && event.origin?.pluginId === api.pluginId) return;  // your own write
  if (event.type === "tabs" && event.removed) forget(event.tabId);
});
```

```ts
type WorkspaceEvent =
  | { type: "graph"; tabId: string; revision: number; origin?: { pluginId: string } }
  | { type: "tabs"; tabId: string; revision: number; removed: boolean }
  | { type: "active-tab"; tabId: string; revision: number };
```

事件會在變更後同步送達，順序固定：新增分頁、文件變更、作用中分頁變更、移除分頁。`origin` 只會在變更來自外掛寫入時出現在 `graph` 事件上。`workspace.applyOperations` 與舊有的 `graph.applyOperations` 都會加入此欄位，外掛可據此區分自己的寫入與使用者的寫入。

如果 callback 拋出錯誤，編輯器會記錄錯誤並取消該訂閱，避免外掛影響分頁儲存。`api.graph.onGraphChanged` 維持不變：仍只追蹤作用中分頁，且不帶 payload。

#### apiVersion 5 之前的外掛要注意的一件事

唯讀分頁現在也會拒絕 `api.graph.applyOperations`。每個操作都會回傳 `{ ok: false, error: "tab is read-only" }`，且不會寫入內容。在 apiVersion 5 之前，這類寫入仍會套用，因為 `readOnly` 只控制 UI，外掛寫入路徑不會檢查它。若外掛會寫入使用者目前開啟的分頁，請先檢查 `api.workspace.snapshot().readOnly`，或檢查原本就會回傳的 `ok` 旗標。

另有一項全新的拒絕規則，所涉及的兩個操作也都是 apiVersion 5 新增的：使用者位於區塊內時，舊版寫入路徑中只要批次包含 `set_segment` 或 `remove_segment`，整個批次都會被拒絕，每個操作都回報 `set_segment and remove_segment cannot apply while a block is open`。段落屬於頂層狀態；若從區塊內提交段落，儲存檔會包含一個引用區塊內部節點 id 的段落標示。該標示不會渲染，使用者也無法在畫布上刪除。其他操作仍會和以往一樣，在區塊內寫入目前開啟的畫布。

### `api.nodes` — 自訂 node 渲染

需要 `api.apiVersion >= 2`。

| 方法 | 簽名 | 說明 |
|------|------|------|
| `registerRenderer` | `(nodeType, renderer) => () => void` | 用你自己的 UI 繪製某個外掛 node 型別的卡片內容。回傳一個取消註冊函式。 |

`nodeType` 必須符合 `getNodeDefinitions()` 中節點的**命名空間型別**：`<plugin-id>:<NODE_NAME>`。外掛 id 會完全照 manifest 複製，包括連字號，因此外掛 `my-plugin` 會提供 `my-plugin:MyNode`。只有 Python import path 會將連字號轉為底線（`cdui_plugins.my_plugin`），所以註冊 `my_plugin:MyNode` 不會有作用。renderer 使用命令式 API，主程式與外掛都不必使用特定 UI framework：

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

編輯器仍會渲染標準節點卡片，包括標題、連接埠與參數列，並將一個位於連接埠和參數之間的 `<div>` 作為**內容區**交給 renderer。未註冊 renderer 的節點型別會使用預設節點樣式。

```js
api.nodes.registerRenderer('my-plugin:MyNode', {
  mount(el, ctx) { el.textContent = `value: ${ctx.node.params.value}`; },
  update(el, ctx) { el.textContent = `value: ${ctx.node.params.value}`; },
});
```

[外掛模板](https://github.com/CodefyUI/CodefyUI-Plugin-Official)的 SDK 會用 `createRoot` 包裝此介面，讓你能以 React 元件撰寫內容區。

### `api.events` — 即時執行事件

需要 `api.apiVersion >= 3`。

| 方法 | 簽名 | 說明 |
|------|------|------|
| `onExecution` | `(cb: (event: ExecutionEvent) => void) => () => void` | 訂閱執行事件串流。回傳一個取消訂閱函式。 |

```ts
type ExecutionEvent =
  | { type: "run_started";  run_id: string; cursor: number; seq: number }
  | { type: "node_status";  run_id: string; cursor: number; seq: number;
      node_id: string; status: string; error?: string }
  | { type: "metric";       run_id: string; cursor: number; seq: number;
      points: readonly RunMetricPoint[] }
  | { type: "run_finished"; run_id: string; cursor: number; seq: number;
      status: "succeeded" | "failed" | "cancelled" | "interrupted";
      error?: string };
```

```js
const off = api.events.onExecution((event) => {
  if (event.type === "metric") {
    for (const p of event.points) record(p.name, p.step, p.value);
  }
  if (event.type === "run_finished") summarise(event.run_id, event.status);
});
```

使用此介面前，請先了解下列行為：

- **一則 `metric` 事件的 `points` 包含記錄時的完整批次。** 其中的項目是 [`RunMetricPoint`](#apiruns--執行歷史唯讀)，與 `api.runs.metrics()` 回傳的型別相同，因此同一個 fold 函式可處理即時串流與 REST 回填。兩邊遇到非有限數字時，`value` 都是 `null`：發散的 loss 代表**曲線上的缺口，不是零**，且事件仍會送出。（唯一差異是 `api.runs.metrics()` 回傳的點有 `ts`，即時事件則沒有。）
- **事件會凍結。** 所有訂閱者共用同一個事件物件，因此事件及其 `points` 都經過 `Object.freeze`。外掛無法修改其他外掛收到的內容；轉換前請先複製。
- **事件會依動畫影格批次派送。** 若一次執行每秒產生數百筆指標，訂閱者會在每個影格收到一批 callback 呼叫，而不是每則訊息各呼叫一次；編輯器更新節點徽章也使用相同機制。編輯器視窗位於背景或遭遮蔽時不會繪製，因此事件會等到視窗恢復顯示後才送達。緩衝上限依儲存成本計算：每個事件加上其中的每個指標點，各計一個單位，總上限約兩萬個。超過上限後會捨棄最舊的指標與節點狀態，但不會捨棄 `run_started` 或 `run_finished`。由於一個 `metric` 事件可包含整批指標，大批次與單點批次得到相同的記憶體額度，而非相同事件數。若需要所有指標點，請從 `api.runs.metrics()` 重新讀取。
- **串流提供即時尾端，不是完整記錄。** 編輯器每次附加至執行時，都會重播完整的執行紀錄，例如頁面重新載入時仍在進行的執行，或使用者從**執行任務**面板選擇觀看的執行。重播項目會經過相同串流，主程式會濾除已送達的項目，因此重新附加不會產生重複項目或重複計數。例外情形請見[當編輯器附掛到一次你沒看過的執行](#當編輯器附掛到一次你沒看過的執行)。
- **完成後請取消訂閱。** 取消會立即生效，即使正在派送批次也一樣。外掛卸載或熱重載時，編輯器也會自動取消訂閱。
- **串流只涵蓋編輯器已附加的執行**：從畫布分頁啟動的執行，以及使用者從**執行任務**面板選擇觀看的執行。無人觀看且由 `cdui run` 提交的執行，可透過 `api.runs` 取得，不會出現在此串流。
- **若 callback 拋出錯誤，**編輯器會記錄錯誤並繼續。其他訂閱者不受影響，但該 callback 會遺漏該事件。

#### `cursor` 與 `seq`

每則事件都帶有這兩個數值。兩者用途不同，混用會使儀表板顯示錯誤結果。

**`cursor` 表示事件在該次執行持久紀錄中的位置**，與 `GET /api/runs/{id}/events` 分頁使用的 cursor，以及 `api.runs.get(id).last_cursor` 回報的值相同。請用它將事件與 REST 資料對齊。

它在同一次執行中嚴格遞增，但**不保證連續，跳號也不表示資料遺失**。執行紀錄還包含不會發布至此串流的項目，每個項目都會使用一個 cursor：

- `artifact`——每次儲存執行檢查點時寫入一筆；
- `run_warning`；
- 被拒絕的提交，以及沒有執行可取消時的取消要求；
- payload 過大而由伺服器折疊的指標項目。

因此，即使訓練正常，每個 epoch 儲存檢查點仍會在每個 epoch 產生 cursor 缺口。請勿將此情況判定為資料遺失。

**`seq` 是串流自身的計數器，也是判斷事件遺失的依據。** 它會連續計算該次執行送達的事件：下一個事件的 `seq` 應比前一個增加 1。唯一例外是主程式因上述緩衝上限捨棄事件，此時這個數值才會出現缺口。

```js
// 每次執行各記一份：記住上一個 seq，看到洞就反應。
const lastSeq = new Map();
api.events.onExecution((event) => {
  const previous = lastSeq.get(event.run_id);
  lastSeq.set(event.run_id, event.seq);
  if (previous !== undefined && event.seq > previous + 1) {
    // 唯一的成因是緩衝溢位。從 REST 把它補回來。
    void api.runs.metrics(event.run_id).then(backfill);
  }
  apply(event);
});
```

你為某次執行看到的第一個 `seq` 是你的基準，不一定是 `1`：它從*編輯器*開始串流那次執行時算起，而那可能早於你的外掛訂閱的時間。

#### 當編輯器附掛到一次你沒看過的執行

上述去重狀態由編輯器針對每次執行維護，整個頁面共用一份，不會為每個外掛分別維護。因此有兩種情況：

- 若編輯器附加至**尚未串流任何事件**的執行，例如使用者在**執行任務**面板選擇某次執行，伺服器會從頭重播該次執行的記錄。訂閱者會先依 cursor 順序收到重播項目，再收到即時事件。每個項目仍只送達一次，但最先收到的事件可能描述過去狀態。
- 若編輯器附加至**已經串流過事件**的執行，所有訂閱者都不會再次收到已重播的項目。若外掛比其他外掛晚訂閱，即使該外掛從未看過這次執行，也可能完全收不到重播內容。請使用 `api.runs` 初始化資料，不要依賴重播。

若需判斷事件是否描述過去狀態，`api.runs.get(run_id)` 會回報 `last_cursor`。對仍在執行的項目，不能直接在 promise 回傳後分類：重播已經開始，這個數值也可能持續變更。請先緩衝事件，等 promise 完成後再分類。

編輯器只會記住最近串流的 **1024** 次執行。若同一個頁面工作階段附加至超過 1024 次不同執行，再回到最早的執行，該執行會再次重播。一般工作階段通常不會達到此上限，但外掛應將它視為已知限制。

### `api.runs` — 執行歷史（唯讀）

需要 `api.apiVersion >= 3`。

| 方法 | 簽名 | 說明 |
|------|------|------|
| `list` | `(opts?) => Promise<RunListPage>` | 由新到舊的一頁執行紀錄。`opts` 為 `{ status?, limit?, offset? }`。 |
| `get` | `(id: string) => Promise<RunInfo \| null>` | 取得單次執行；指定 id 不存在時回傳 `null`。 |
| `metrics` | `(id: string, name?: string) => Promise<RunMetrics>` | 已記錄的純量序列，依 `(name, step)` 排序。 |

```js
const page = await api.runs.list({ status: ["running"], limit: 1 });
const active = page.runs[0];
if (active) {
  const recorded = await api.runs.metrics(active.id);
  for (const point of recorded.metrics) {
    if (point.value !== null) record(point.name, point.step, point.value);
  }
}
```

```ts
interface RunListPage { runs: RunSummary[]; total: number; limit: number; offset: number }
interface RunInfo extends RunSummary { last_cursor: number }
interface RunMetrics { run_id: string; names: string[]; metrics: RunMetricPoint[] }
interface RunMetricPoint {
  node_id: string | null; name: string; step: number;
  value: number | null;   // null 代表發散（非有限）的值——是缺口，不是零
  ts?: string;            // ISO-8601 UTC；這裡有，即時 metric 事件上沒有
}
```

`RunMetricPoint` 與即時 `metric` 事件在 `points` 中使用相同型別，因此儀表板可用同一個 fold 函式處理兩個來源。唯一差異是 `ts`：`api.runs.metrics()` 會記錄每個點的寫入時間，即時串流只提供圖表依 `step` 繪製所需的欄位。忽略 `ts` 的 fold 函式可直接用於兩者。

`RunSummary` 對應執行歷史的一列：`id`、`name`、`status`、`error`、`options`、`queue_key`、`created_at`、`started_at`、`finished_at`、`git_commit`、`git_dirty`、`plugin_pins`、`queue_position`、`final_metrics` 與 `active`。完整型別在隨附的 SDK types 中，背後的端點則記載於 [API 參考](/advanced/api-reference)。

這組介面讓常見情境不需自行實作 fetch。編輯器會透過自己的 API client 發送 request，並自動附上所需的驗證資訊。外掛不必組合 URL，session token 也不會傳入外掛程式碼或出現在 `api.runs` 的回傳值中。這只提供便利，並非沙箱（參見[信任模型](#信任模型)）。

此版本的介面刻意設為**唯讀**，不提供 `submit` 或 `cancel`。在使用者的機器上啟動或停止工作，應由使用者開啟的 UI 操作觸發，不應由外掛自行呼叫。若需提供此功能，請讓使用者按下按鈕後，再透過 `api.http.fetch` 執行。

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

後端的 AST 安全閘門適用於外掛 Python；外掛 JavaScript 並無沙箱機制——它以與編輯器本身相同的信任層級執行。

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

## 即時執行指標面板

下方範例示範 apiVersion 3 的主要用途：建立一個底部停靠分頁，即時列出執行指標。`events.onExecution` 提供即時事件，`runs` 提供面板開啟前的歷史資料。範例會先訂閱，再回填歷史資料，避免遺漏兩者之間產生的事件。

```js
// frontend/index.js
export default function activate(api) {
  if (api.apiVersion < 3) return;

  const series = new Map();   // name -> { last, points }

  // 兩半共用同一個 fold：`event.points` 與 `runs.metrics().metrics`
  // 是同一種 RunMetricPoint[]。
  const fold = (points) => {
    for (const p of points) {
      const previous = series.get(p.name);
      series.set(p.name, {
        // null 是發散的值——是缺口，所以保留上一個有限值。
        last: p.value ?? previous?.last ?? null,
        points: (previous?.points ?? 0) + 1,
      });
    }
  };

  const render = () => {
    if (!el.isConnected) return;   // 分頁沒開，不必畫
    el.textContent = [...series.entries()]
      .map(([name, s]) =>
        `${name}  last=${s.last === null ? "--" : s.last.toFixed(4)}  n=${s.points}`)
      .join("\n");
  };

  const el = api.ui.addPanel({
    id: "run-metrics", title: "Run Metrics", icon: "~",
    onShow: render,   // 進來時先畫一次，分頁才不會是空白
  });

  // 1. 即時尾巴
  api.events.onExecution((event) => {
    if (event.type === "run_started") series.clear();
    if (event.type === "metric") fold(event.points);
    render();
  });

  // 2. 回填：面板出現之前就已經在跑的執行
  api.runs.list({ status: ["running"], limit: 1 }).then(async (page) => {
    const active = page.runs[0];
    if (!active) return;
    const recorded = await api.runs.metrics(active.id);
    // 同一個 fold，只挑即時尾巴還沒涵蓋到的序列。
    fold(recorded.metrics.filter((p) => !series.has(p.name)));
    render();
  });
}
```

[外掛骨架](/advanced/plugins)在 `ui/src/examples/run-metrics-panel.tsx` 附了同一個範例的 React 版本，使用 SDK 的 `mountPanel`、`useExecutionEvents` 與 `useRuns`。

## 另請參閱

- [外掛](/advanced/plugins) — 安裝外掛包、manifest 格式與 `cdui plugin` CLI。
- [Graph Copilot](/advanced/graph-copilot) — 第一個在正式環境使用前端擴充 API 的外掛。
- [API 參考](/advanced/api-reference) — 後端 REST 端點，包括 `/api/llm/chat`。
