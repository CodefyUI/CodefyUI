---
sidebar_position: 4
title: 外掛前端擴充
description: 隨外掛包附上一個 JavaScript bundle，讓外掛能新增 UI 小工具、檢視圖表並驅動編輯器——Graph Copilot 等工具的基礎。
---

# 外掛前端擴充

外掛包可以在 Python 節點之外，附上一個 JavaScript bundle。CodefyUI 編輯器載入時，會探索並以 ES 模組形式匯入該 bundle，讓外掛取得一個穩定的 JavaScript API，用於操作 UI、圖表及代理 HTTP 請求。

:::note 可用性
前端擴充功能自 CodefyUI **1.3.0** 起內建。請執行 `cdui --version` 確認；若顯示更舊的版本，請執行 `cdui update`。

停靠面板、工具列按鈕、執行事件與 runs 門面需要 **apiVersion 3**（CodefyUI 1.5.0 起）；`graph.getView` 需要 **apiVersion 4**（CodefyUI 2.3.0 起）。使用前請先檢查版本——參見 [API 版本](#api-版本)。
:::

## API 版本

`api.apiVersion` 是一個只增不減的數字，而目前每一次改版都是**純粹新增**：舊版能用的東西，沒有被移除過，也沒有被改過形狀。為 apiVersion 2 撰寫的外掛，在 apiVersion 4 的編輯器上不必改任何一行就能繼續運作。

| `apiVersion` | CodefyUI | 新增內容 |
|--------------|----------|----------|
| 1 | 1.3.0 | `ui.addFloatingWidget`、`ui.toast`、`graph.*`、`http.fetch`、`storage.*` |
| 2 | 1.3.0 | `nodes.registerRenderer` |
| 3 | 1.5.0 | `ui.addPanel` / `removePanel`、`ui.addToolbarButton` / `removeToolbarButton`、`events.onExecution`、`runs.*` |
| 4 | 2.3.0 | `graph.getView`——使用者正在看圖表的哪一層 |

在使用比你所要求的版本更新的功能之前先檢查它，而且是降級處理，不是直接拋錯：

```js
export default function activate(api) {
  if (api.apiVersion >= 3) {
    mountDashboard(api.ui.addPanel({ id: "dash", title: "Dashboard" }));
  } else {
    mountDashboard(api.ui.addFloatingWidget({ id: "dash" }));
  }
}
```

因為改版都是純新增，破壞性變更一定會伴隨 `apiVersion` 提升與遷移說明——不會靜悄悄發生。

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

## CodefyUIPluginAPI 參考

### `api.ui` — 編輯器 UI

| 方法 | 簽名 | 說明 |
|------|------|------|
| `addFloatingWidget` | `({ id }) => HTMLElement` | 在編輯器的浮動元件堆疊中建立（或重用）一個容器 `<div>` 並回傳。`id` 在同一外掛內必須唯一。回傳的元素歸你所有——填入你自己的 DOM，或在其上掛載一個 React root。 |
| `toast` | `(message, level?) => void` | 顯示一個暫時性通知。`level` 為 `"info"`（預設）、`"warning"` 或 `"error"`。 |
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

`"bottom"` 面板會成為編輯器底部面板的一個分頁，排在「執行記錄」、「訓練」與「執行任務」之後。`"right"` 面板則成為右側欄的一個區塊，與節點設定和檢視器面板並列。分頁外框、排序與位置由主程式決定；元素裡面的一切則歸你。

**在面板存在的期間，那個元素都是你的。** 它的身分永遠不變，所以只需掛載一次：

```js
const el = api.ui.addPanel({ id: "runs", title: "My Runs", icon: "~" });
createRoot(el).render(<MyPanel />);   // 只做一次，不是每次切分頁
```

編輯器只會掛載*當前*的底部分頁，所以裝著你面板的那個容器會隨著使用者切換分頁而被拆掉重建。你的元素不會：編輯器只是把它卸下再掛回去，子節點與其狀態都完整保留。用同一個 `id` 再呼叫一次 `addPanel`，回傳的是同一個元素，只是更新標題、圖示與停靠位置。

代價只有一件事要留意：面板不在畫面上時，你的程式仍在*執行*，而且畫的是一個不在文件中的元素。如果面板做的事情不便宜——圖表、輪詢、動畫——請把它關掉：

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

按鈕會依註冊順序，集中在工具列右側的同一組裡。你無法指定位置，顯示幾個也由編輯器決定：視窗夠寬時最多三個直接排在列上，視窗窄時全部收進一個溢位選單。正是這條規則讓五個已安裝外掛不會把「執行」擠出工具列，所以請把 `tooltip` 當成標籤來寫——在選單裡，它就是標籤。

若 `onClick` 拋出錯誤，編輯器會記錄下來並繼續運作；工具列不受影響。

用同一個 id 再次註冊會取代原本的按鈕。回傳的移除函式只屬於當次那一筆註冊，所以如果按鈕後來被你自己取代掉了，呼叫舊的移除函式不會有任何作用，而不會把新的那顆一起移除。若你的意思是「不管現在掛在這個 id 底下的是哪一顆，都移除掉」，請改用 `removeToolbarButton(id)`。外掛被卸載或熱重載時，按鈕會自動移除。

### `api.graph` — 圖表讀寫

| 方法 | 簽名 | 說明 |
|------|------|------|
| `getGraph` | `() => GraphSnapshot` | 回傳**整張**圖表狀態（節點、邊、參數，以及 `subgraphs` 底下的區塊定義）的深層副本——不論使用者當下打開哪一層，讀到的永遠是最上層。 |
| `getNodeDefinitions` | `() => NodeDefinition[]` | 回傳完整的節點面板：型別、連接埠 schema、參數 schema。 |
| `applyOperations` | `(ops: GraphOp[]) => ApplyResult` | **同步**套用一批圖表操作（直接回傳結果，非 Promise）。整個批次以**單一撤銷快照**的形式提交，而且會寫進使用者當下打開的那張畫布——參見[使用者正在看哪一層](#使用者正在看哪一層)。 |
| `onGraphChanged` | `(callback: () => void) => () => void` | 訂閱圖表變更事件，包含使用者走進或走出一個區塊。回呼不帶任何參數；需要內容時請在回呼裡呼叫 `getGraph()`。回傳一個取消訂閱函式。 |
| `getView` | `() => GraphView` | **apiVersion 4。** 唯讀：使用者正在看圖表的哪一層。 |

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

#### 使用者正在看哪一層

需要 `api.apiVersion >= 4`。

CodefyUI 的圖表是可以嵌套的。一個**區塊**（subgraph）有它自己的畫布，使用者可以走進去——畫布上方那條列就會顯示成 `Main > Encoder`。而且從頭到尾只有一張畫布：走進區塊是把區塊的內容**換**到那張畫布上，這也正是為什麼每個編輯工具在區塊裡和在區塊外的行為完全一樣。

對外掛來說，這件事有一個後果，而且它決定了你的編輯會落在哪裡：

- **`getGraph()` 讀到的永遠是整張圖。** 編輯器會先把打開的層折回去再序列化，跟存檔與執行走的是同一條路，所以你讀到的位元組，和使用者按存檔會得到的檔案一致。
- **`applyOperations()` 寫進的是使用者當下打開的那張畫布。** 在區塊裡面，`add_node` 是把節點加進*那個區塊*，`clear_graph` 清空的是*那個區塊*而不是整張圖。你從 `getGraph()` 讀到的節點 id 在那裡並不存在，所以指名它們的操作會回傳 `ok: false` 與一段錯誤說明。

也就是說，一個先讀、再推理、然後寫的外掛，可以對整張圖的判斷完全正確，卻把結果寫到使用者根本沒在看的地方。`getView()` 就是讓你在寫之前先分辨這兩種處境：

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

拒絕並不是唯一誠實的答案——等使用者走出來，或是把編輯縮小到在區塊裡也說得通的範圍，都同樣可以。重點是這個選擇現在由你來做，而不是丟一次硬幣。

這個 view 是**唯讀**的，而且是即時讀取：每次呼叫都是當下的新答案；而且我們刻意沒有提供任何讓外掛帶著使用者跳到別層的方法。使用者走進或走出區塊時 `onGraphChanged` 會觸發（畢竟畫布真的換了），所以想顯示「我會寫到哪裡」的面板，可以在那個回呼裡重新讀一次 `getView()`。

寫入會落在哪一層，是編輯器一直以來的行為，這次是把它寫下來，而不是把它改掉。未來的改版可能會讓操作自己指名目標層級；那也會是用「新增一個東西」的方式做到，而不是悄悄改寫既有外掛已經在做的那些寫入。

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

[外掛模板](https://github.com/CodefyUI/CodefyUI-Plugin-Official)的 SDK 會用 `createRoot` 包裝它，讓你能以 React 元件撰寫內容區。

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

在你拿它蓋東西之前，值得先知道的幾件事：

- **一則 `metric` 事件會帶著它被記錄下來時的整批點**，放在 `points` 裡。那些是 [`RunMetricPoint`](#apiruns--執行歷史唯讀)——和 `api.runs.metrics()` 回傳的*同一個*型別——所以同一個 fold 函式可以同時服務即時尾巴與 REST 回填。特別是兩邊的 `value` 對非有限數字都是 `null`：發散的 loss 是**曲線上的缺口，不是零**，而且會被送出，不會被跳過。（唯一的差別：`ts` 只有 `api.runs.metrics()` 回來的點才有，即時事件上沒有。）
- **事件是凍結的。** 同一個事件物件會交給每一個訂閱者，所以它和它的 `points` 都經過 `Object.freeze`——你改不到別的外掛收到的內容，要轉換請先複製。
- **事件會批次對齊到動畫影格。** 一次每秒推送數百筆指標的執行，到你手上是每影格一叢呼叫，而不是每則訊息一次呼叫——和編輯器自己更新節點徽章用的是同一套批次機制。被切到背景或被遮住的編輯器視窗不會重繪，所以在它回來之前不會派送任何東西；期間累積的量以「保留成本」計算——一個事件本身，加上它攜帶的每一個指標點，上限約兩萬個——超過上限後，最舊的指標與節點狀態會被丟棄（`run_started` 與 `run_finished` 永遠不會）。計算單位很重要，因為一個 `metric` 事件裝的是一整批指標：批次很大的執行與每次只寫一個點的執行，拿到的是同樣的記憶體預算，而不是同樣的事件數量。若你需要每一個點，請改用 `api.runs.metrics()` 重新讀取。
- **它是尾巴，不是逐字稿。** 編輯器每次附掛到一次執行時，都會重播該次執行完整的記錄——重新整理頁面時仍在跑的執行，或使用者在「執行任務」面板挑一次執行來看的時候。那些重播的項目會經過同一條串流，而主程式會濾掉每一筆已經派送過的，所以重新附掛不可能塞給你重複的資料讓你重複計算。例外情形寫在[當編輯器附掛到一次你沒看過的執行](#當編輯器附掛到一次你沒看過的執行)。
- **記得取消訂閱**；它會立即生效，包含在一批事件派送到一半的時候。外掛卸載或熱重載時，編輯器也會替你取消。
- **串流涵蓋的是編輯器已附掛的執行**——從畫布分頁啟動的那些，加上使用者在「執行任務」面板選擇觀看的執行。由 `cdui run` 送出、沒人在看的執行，要從 `api.runs` 看，不在這裡。
- **若你的 callback 拋出錯誤**，編輯器會記錄下來並繼續。其他訂閱者不受影響，但你會漏掉那一則事件。

#### `cursor` 與 `seq`

每則事件都帶兩個數字，把它們搞混，是做出一個會騙使用者的儀表板最快的方法。

**`cursor` 是這則事件在該次執行持久記錄中的位置**——和 `GET /api/runs/{id}/events` 分頁、`api.runs.get(id).last_cursor` 講的是同一個 cursor。用它把事件對回 REST 那一側。

它在同一次執行內嚴格遞增，但**不連續，而且跳號本身沒有任何意義**。那份記錄裡還有一些項目不會經由這條串流發布，而每一個都佔掉一個 cursor：

- `artifact`——執行每存一次檢查點就寫一筆；
- `run_warning`；
- 被拒絕的送出，以及沒有東西可取消的取消；
- 因為 payload 過大而被伺服器摺疊掉的指標項目。

也就是說，一次每個 epoch 存檢查點、健康得不得了的訓練，每個 epoch 都會產生一次 cursor 缺口。不要把那當成資料遺失。

**`seq` 是串流自己的計數器，它才是遺失的訊號。** 它逐一計算某次執行實際派送出去的事件，而且是連續的：你為某次執行收到的下一個事件，`seq` 一定剛好比上一個大 1——除非主程式在上面說的緩衝上限下丟了事件，而那是唯一能在它上面打洞的東西。

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

上面說的去重，是編輯器**以執行為單位、整個頁面共用一份**的帳，不是每個外掛各記一份。這有兩個後果：

- 當編輯器附掛到一次**還沒有任何東西串流過**的執行時——使用者在「執行任務」面板點了某次執行——伺服器會從頭重播那次執行的記錄，而你會收到它，依 cursor 順序，然後才接上即時尾巴。每個項目仍然只到達一次，但你為那次執行看到的最初幾個事件描述的是過去。
- 當編輯器附掛到一次**已經串流過東西**的執行時，重播會對所有人一起被濾掉。如果你的外掛比另一個晚訂閱，你會繼承那份濾除，所以對於一次你個人從沒看過的執行，你可能從重播裡**什麼都收不到**。不要靠重播來填滿自己；那是 `api.runs` 的工作。

若你需要知道一則事件描述的是不是過去，`api.runs.get(run_id)` 會回報 `last_cursor`。但要注意，對還在跑的執行來說它不是一行就解決的事：你讀到的是一個會動的目標，而且是在重播已經開始*之後*才讀到，所以誠實的作法是先把事件緩衝起來，等 promise 回來之後再分類。

還有一個上限，與其讓你自己撞到，不如先講：編輯器會記住最近 **1024** 次它串流過的執行。在同一次頁面工作階段中附掛超過 1024 次不同的執行，然後再回頭去看那個工作階段最早的那一次，它會被重播給你第二次。一般使用根本碰不到，寫在這裡是為了讓這個限制是一條寫明的條件，而不是一個意外。

### `api.runs` — 執行歷史（唯讀）

需要 `api.apiVersion >= 3`。

| 方法 | 簽名 | 說明 |
|------|------|------|
| `list` | `(opts?) => Promise<RunListPage>` | 由新到舊的一頁執行紀錄。`opts` 為 `{ status?, limit?, offset? }`。 |
| `get` | `(id: string) => Promise<RunInfo \| null>` | 單筆執行；伺服器沒聽過這個 id 時回傳 `null`。 |
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

`RunMetricPoint` 和即時 `metric` 事件放在 `points` 裡的是同一個型別，所以儀表板可以用同一個函式 fold 兩邊。`ts` 是兩個來源之間唯一有差的欄位：`api.runs.metrics()` 會記錄每個點寫入的時間，即時串流則只帶圖表拿來對 `step` 畫的東西。忽略 `ts` 的 fold 在兩邊都能原封不動地用。

`RunSummary` 對應執行歷史的一列：`id`、`name`、`status`、`error`、`options`、`queue_key`、`created_at`、`started_at`、`finished_at`、`git_commit`、`git_dirty`、`plugin_pins`、`queue_position`、`final_metrics` 與 `active`。完整型別在隨附的 SDK types 中，背後的端點則記載於 [API 參考](/advanced/api-reference)。

這個門面的存在，是為了讓常見情境不必自己拼 fetch：請求由編輯器透過它自己的 API 客戶端送出，需要的驗證資訊都已附上。你不必自己組 URL，token 也不會傳進你的程式碼、不會出現在 `api.runs` 的任何回傳值裡——但這是便利，不是沙箱（參見[信任模型](#信任模型)）。

它在這一版是刻意**唯讀**的。沒有 `submit`，也沒有 `cancel`：在別人的機器上啟動或停止工作，應該發生在使用者自己打開的介面背後，而不是一次外掛呼叫背後。若你確實需要，請由使用者按下的按鈕出發，透過 `api.http.fetch` 進行。

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

## 即時執行指標面板

下面這個範例，就是 apiVersion 3 這組介面被加進來的目的：一個底部分頁，把一次執行的指標邊跑邊列出來。它把兩半湊在一起——`events.onExecution` 負責即時尾巴，`runs` 負責面板打開之前已經發生的事——而且*先*訂閱再回填，這樣兩者之間就不會有東西漏掉。

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

- [外掛](/advanced/plugins) — 安裝外掛包、資訊清單格式與 `cdui plugin` CLI。
- [Graph Copilot](/advanced/graph-copilot) — 前端擴充 API 的首個正式消費者。
- [API 參考](/advanced/api-reference) — 後端 REST 端點，包括 `/api/llm/chat`。
