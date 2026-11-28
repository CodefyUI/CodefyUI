---
sidebar_position: 7.8
title: 專案目錄
description: 用 cdui project 把服務變成獨立完整的 git 儲存庫：logic/layout 分離、專案內 assets 與機密、可上 CI 的驗證，以及發佈時記錄 git 溯源。
---

# 專案目錄

**專案目錄**是一個獨立完整的 git 儲存庫，它本身就是你服務的儲存空間。編輯器
直接讀寫其中的檔案：每個 graph 一份乾淨的 logic 檔、節點位置放在旁邊的
layout 檔、專案內的 assets 與機密、可上 CI 的驗證，以及每次發佈都記錄下
git commit。

```
my-service/
  codefyui.project.toml   manifest: name, plugin pins, default publish target
  graphs/    <name>.graph.json    logic (nodes/edges/params/presets/subgraphs, optional settings.device / settings.seed)
             <name>/              or the same graph as a folder (see "Graphs stored as folders")
  layout/    <name>.layout.json   positions, note geometry, segments (reviewable, generated)
  assets/images/   assets/models/   assets/data/    scaffolded empty
  assets/output/                                    created on demand (e.g. ImageWriter)
  assets/media/                                     created on demand (run-produced video, served by /api/media)
  .env.example     committed template of required secret keys
  .env             your secrets (gitignored, never committed)
```

## 為什麼要分離？ {/* #why-the-split */}

`graphs/<name>.graph.json` 只儲存會改變 graph *行為* 的內容（節點、邊、參數、內嵌 presets、子圖定義，以及選填的 `settings` 區塊，例如 `{"device": "cuda"}`，用來指定這張 graph 執行的裝置）。節點**位置**（包括子圖定義內部節點的位置）、便利貼幾何資訊與比較用的段落標記（`segmentGroups`）則存放在 `layout/<name>.layout.json`。因此，拖動節點只會變更 `layout/`，修改參數只會變更 `graphs/`；code review 可以聚焦於邏輯變更，不會受到節點位移的 diff 干擾。（已知例外：`SequentialModel` 子圖的層位置存放在 `params.layers`，因此仍位於 logic 檔。）

缺少 layout 檔（或某個節點沒有已儲存的位置）時，編輯器會在載入時自動排版，
並於下次存檔時寫回結果；便利貼若只缺幾何資訊（尺寸/綁定）則直接使用預設值 --
「只缺幾何資訊」刻意不視為缺少 layout。

## 完整流程 {/* #a-complete-walkthrough */}

### 1. 建立專案 {/* #1-create-the-project */}

```bash
cdui project init my-service
cd my-service
```

`init` 會建立 `graphs/`、`layout/` 與 `assets/{images,models,data}/`（空目錄，以 `.gitkeep` 追蹤），寫入 `codefyui.project.toml` / `.gitignore` / `.gitattributes` / `.env.example` / `README.md`，並執行 `git init`。它不會建立 commit，而會印出後續步驟。`--force` 允許寫入非空目錄，但這五個檔案若已存在都不會被覆寫。`assets/output/` 不會預先建立；節點（例如 ImageWriter）第一次寫入時才會出現。

`.gitattributes` 有四行與換行及 diff 有關，另有三行 `merge=codefyui`（這些與 `*.jsonl`、`graphs/**/nodes/*` 兩行見[以資料夾儲存的圖](#graphs-stored-as-folders)）。`*.json text eol=lf` 讓 graph 與 layout 檔在 checkout 時維持 LF：伺服器以 LF 寫出這些檔案，若 Windows 上設定了 `core.autocrlf=true`，git 會把它們以 CRLF 交回，每次存檔都會改寫每一行。`layout/*.layout.json linguist-generated=true` 會讓 layout 檔在 GitHub diff 中收合。2.6.0 之前建立的專案會保留舊的 `.gitattributes`，其中沒有第一行；請自行加入 `*.json text eol=lf`。

### 2. 加入一個 graph {/* #2-add-a-graph */}

可以在編輯器裡建（`cdui start --project .`，放一個 **Start**、一個名為 `x`
的 **GraphInput**、一個名為 `y` 的 **GraphOutput**，把 Start 的 trigger 接到
GraphInput、GraphInput 的 value 接到 GraphOutput，然後按 **Ctrl/Cmd+S** 命名
為 `echo`），或直接把這個檔案放到 `graphs/echo.graph.json`：

```json
{
  "format_version": 1,
  "name": "echo",
  "description": "Echo the input string",
  "nodes": [
    {"id": "start", "type": "Start", "data": {"params": {}}},
    {"id": "gi", "type": "GraphInput", "data": {"params": {"name": "x", "type": "string", "required": true, "default": "", "description": "text to echo"}}},
    {"id": "out", "type": "GraphOutput", "data": {"params": {"name": "y", "description": "the echoed text"}}}
  ],
  "edges": [
    {"id": "t1", "source": "start", "target": "gi", "sourceHandle": "trigger", "targetHandle": "", "type": "trigger"},
    {"id": "d1", "source": "gi", "target": "out", "sourceHandle": "value", "targetHandle": "value", "type": "data"}
  ],
  "presets": [],
  "settings": {"device": "cuda"}
}
```

`settings` 是選填的。沒有它的 graph 會跑在客戶端選的裝置上：編輯器裡是設定的裝置，
不帶 `--device` 的 `cdui run` 則是 `cpu`。`settings.device` 接受的值，以及存檔何時寫入這個區塊，請見[裝置後端](../advanced/device-backends.md#the-graph-settings-object)。

### 3. 提交 {/* #3-commit */}

```bash
git config user.name  "You"
git config user.email "you@example.com"
git add -A
git commit -m "echo service"
```

`.env` 已被 gitignore；`.env.example` 要提交。大型資料請提交一個小的下載
腳本，永遠不要提交資料或權重本身。伺服器在專案上跑起來之後（第 5 步），
編輯器的[版本控制](./source-control)分頁就能在側邊欄做同樣的暫存與提交。

### 4. 驗證（CI 關卡） {/* #4-validate-the-ci-gate */}

```bash
cdui project validate .
```

`validate` 會初始化**完整**的 registry（builtin + custom + plugin 的節點與
presets，跟伺服器完全一樣），並依相同順序對每個 graph 執行發佈會做的檢查：graph 內含機密
檢查、contract、進入點、接線，以及節點／preset 有效性。它與發佈並不完全相同：

- 它另外檢查 `settings.device`（無效的值會回報 `invalid_settings` 錯誤）；發佈不做這項檢查。
- 它不會把 graph 的子圖定義交給節點檢查，因此使用子圖的 graph 會以 `Unknown subgraph`
  失敗，但發佈可以接受。
- 它的機密檢查只看頂層節點：子圖或 preset 定義裡的 SECRET 值可以通過 `validate`，
  發佈則會以 409 `secret_in_graph` 拒絕。

後兩項是已知問題。`validate` 也會在 `.env` 被 git 追蹤時報錯，並對缺少的 plugin pins
提出警告（加 `--strict` 則變成錯誤）。在 CI 上，請先 **restore 再 validate**：

```bash
cdui project restore .   # install the manifest's plugin pins by exact SHA
cdui project validate .
```

`validate` 會檢查 `graphs/` 底下的**每一個** graph，並印出檢查數量。空的 `graphs/` 會回報 `Validation passed (0 graphs checked)`，明確表示未檢查任何 graph。**canvas-only** graph（例如沒有宣告任何 **GraphOutput** 的訓練 graph）會在 contract 關卡失敗，因為每個可發佈的 graph 至少要宣告一個輸出。你可以加入合理的輸出；MNIST 範例專案會把 checkpoint 路徑作為 `weights_path` 輸出發佈。也可以只驗證要發佈的目標：

```bash
cdui project validate . --graph serve   # repeatable: --graph a --graph b
```

`--graph` 指定的名稱若不存在於 `graphs/`，驗證會報錯，因此拼寫錯誤不會讓 CI 在未檢查目標 graph 的情況下通過。

Pins 來自 `cdui project freeze .`：它讀取你本機安裝的 plugins，把每一個的
確切 commit SHA 寫進 `codefyui.project.toml` 的 `[plugins]` 表（以本機開發
連結安裝的 plugin 會被略過 -- 機器限定的路徑沒有 SHA 可釘選）。安裝或更新
plugin 之後執行它，並在下次 push 前提交 manifest 的變更：

```bash
cdui project freeze .
```

Freeze 會直接改寫 manifest：你自行加入的 key（位於 `[project]`、`[publish]` 或自訂 table 中）都會保留，但註解不會，而 `[plugins]` table 會完全依目前已安裝的內容重新產生。

### 5. 在專案上啟動伺服器 {/* #5-start-the-server-on-the-project */}

```bash
cdui start --project .
```

log 會印出 `Project: <abs> (git <short-sha>)`，若有釘選的 plugin 未安裝，
會警告一次並指名 `cdui project restore`。（`cdui dev --project .` 也會如此，
並提供 hot reload。）

使用 `--project` 啟動，也會載入 `<project>/.env`：一般的 `KEY=VALUE` 行（允許開頭有 `export `，也允許值外圍有引號），會在探索 node 與 plugin 前以 `os.environ.setdefault` 語意套用，因此 shell 中已設定的變數優先。只有執行期間的機密適合放在這裡 —— LLM API key，以及 node 在執行時從環境讀取的任何值。檔案中的 `CODEFYUI_*` 設定不會生效，因為伺服器設定在讀取檔案前就已固定；請改在 shell 或 systemd unit 中設定。log 只會記錄載入的數量，絕不記錄值；不帶 `--project` 時完全不會讀取 `.env`。

### 6. 建立 API key（invoke 需要） {/* #6-create-an-api-key-invoke-needs-one */}

Session token 位於 `<install dir>/.codefyui_dev/session.token`（由 `cdui start` 或 `cdui dev` 啟動的伺服器；預設安裝目錄為 `~/CodefyUI`，Windows 上也就是 `$HOME\CodefyUI`）；若在啟動器執行前已匯出 `CODEFYUI_USER_DATA_DIR`，則位於 `<CODEFYUI_USER_DATA_DIR>/session.token`。該目錄裡的其他檔案、手動啟動的 `uvicorn app.main:app` 所用的平台目錄，以及 token 為何每次重啟都會輪換，請見[把 graph 當成函式呼叫](./graph-as-a-function.md#2-getting-the-token-for-external-scripts)。

PowerShell：

```powershell
# payload.json: {"name": "demo"}
$token = Get-Content "$HOME\CodefyUI\.codefyui_dev\session.token"
curl.exe -s -X POST "http://127.0.0.1:8000/api/keys" `
  -H "X-CodefyUI-Token: $token" -H "Content-Type: application/json" `
  --data "@payload.json"
```

bash：

```bash
TOKEN=$(cat ~/CodefyUI/.codefyui_dev/session.token)
curl -s -X POST http://127.0.0.1:8000/api/keys \
  -H "X-CodefyUI-Token: $TOKEN" -H "Content-Type: application/json" \
  --data '{"name": "demo"}'
```

`# -> {"id": 1, "name": "demo", "prefix": "cdui_xxxxxxx", "token": "cdui_..."}`（完整金鑰只顯示這一次，在 "token" 欄位）

### 7. 發佈（記錄 git commit） {/* #7-publish-records-the-git-commit */}

`cdui project publish` 包裝的是同一個 [publish](./publish.md) 端點
（`POST /api/apps/{slug}/publish`），外加專案模式防護與自動 git 溯源。先在
`codefyui.project.toml` 設定一次預設目標：

```toml
[publish]
graph = "echo"
slug = "echo-svc"
```

`graph`（以及 `--graph`）是 graph 的[位址](./graph-as-a-function.md#the-graphs-address)：`graphs/` 底下的檔名去掉 `.graph.json`，可能與**圖表**分頁顯示的標題不同。建立專案時產生的 manifest 另有一行註解掉的 `record_io = true`。設定 `record_io` 後，每次 `cdui project publish` 都會送出它，因此每次發佈都會重新套用這個值，包括覆蓋以 `PATCH /api/apps/{slug}` 做的變更；若要保留 app 目前的設定，請不要設定它（見[發佈](./publish.md#1-publish-lifecycle)）。

先提交 manifest；未提交的 manifest 變更會使下一步顯示 dirty 工作樹警告。接著發佈：

```bash
git add -A && git commit -m "set publish target"
cdui project publish .
# -> Published echo-svc v1 (git 1a2b3c4)
cdui project publish . --note "first cut"   # --note attaches an immutable note to the version
```

v1 的發佈**僅限本機**：它會確認 `GET /api/health` 回報目前開啟的是此專案，避免把其他專案的內容記到這個 commit；接著計算 `git rev-parse HEAD` 與 `git status --porcelain`，工作樹有未提交變更時會顯示醒目警告。從 git 儲存庫發佈時，每次都會在 commit 旁記錄 `git_dirty` 的 `true` 或 `false`；工作樹不乾淨時也會顯示上述警告橫幅。若 commit 已解析成功，但 `git status` 本身失敗，`git_dirty` 會記為 `null`（未知），不會錯誤地記為 `false`。

它會連線到 `http://127.0.0.1:8000`；若執行指令的 shell 設定了 `CODEFYUI_PORT`，則改用該連接埠。它不像 `cdui run` 那樣使用 `cdui start` 記錄的位址。若伺服器以 `--port 9000` 啟動，請執行 `CODEFYUI_PORT=9000 cdui project publish .`（PowerShell：先設定 `$env:CODEFYUI_PORT = "9000"`）。只綁定單一區域網路位址（`--host 192.168.1.20`）的伺服器不會監聽 127.0.0.1，因此這個指令無法連上它。

第一次發佈時，只有 manifest 中已提交的 `[publish].slug` 目標會自動建立 app。命令列明確傳入的 `--slug` 若指向伺服器未知的 app，會以 404 `app_not_found` 失敗，避免拼寫錯誤建立第二個 app。CLI 會提示使用 `--create`，明確建立新的命令列 slug：

```bash
cdui project publish . --graph echo --slug echo-svc --create
```

> **遠端 / CI 部署不在 v1 範圍內。**`cdui project validate` 可以在 CI 上
> 跑，但發佈需要一台開著該專案的本機伺服器。已排定的後續項目是管理範圍、
> 用 API key 的發佈（`--url` / `--key`）。

### 8. 呼叫（invoke） {/* #8-invoke */}

PowerShell：

```powershell
# payload.json: {"inputs": {"x": "hello"}}
curl.exe -s -X POST "http://127.0.0.1:8000/api/apps/echo-svc/invoke" `
  -H "Authorization: Bearer cdui_YOUR_KEY" -H "Content-Type: application/json" `
  --data "@payload.json"
```

bash：

```bash
curl -s -X POST http://127.0.0.1:8000/api/apps/echo-svc/invoke \
  -H "Authorization: Bearer cdui_YOUR_KEY" -H "Content-Type: application/json" \
  --data '{"inputs": {"x": "hello"}}'
```

`# -> {"status": "ok", "outputs": {"y": "hello"}, ...}`

### 9. 查「這是哪個 commit 發佈的」 {/* #9-see-which-commit-built-this */}

PowerShell：

```powershell
curl.exe -s "http://127.0.0.1:8000/api/apps/echo-svc/versions" -H "X-CodefyUI-Token: $token"
```

bash：

```bash
curl -s http://127.0.0.1:8000/api/apps/echo-svc/versions \
  -H "X-CodefyUI-Token: $TOKEN"
```

`# -> [{"version": 1, "git_commit": "1a2b...", "git_dirty": false, "active": true, ...}]`

作用中版本的 `GET /api/apps/echo-svc/openapi.json` 的 `info` 區塊也帶有
`x-codefyui-git-commit` 與 `x-codefyui-git-dirty`。

## 以資料夾儲存的圖 {/* #graphs-stored-as-folders */}

manifest 裡寫了

```toml
[storage]
graph_format = "dir"
```

的專案，會把每張**新**圖存成一個資料夾，而不是單一檔案。伺服器建立的專案與 `cdui project init` 都會寫這個設定；沒有這個鍵的專案繼續存成單一檔案。既有的圖維持它在磁碟上的格式，直到執行 `cdui project migrate`。

```
graphs/<name>/
  index.json                  storage_version 2、format_version、name、description、settings、requires
  nodes/<node>.json           每個節點一個檔：{id, type, o, data}
  nodes/<node>.code.py        長文字參數（含換行且超過 400 字元）
  edges.jsonl                 每行一條線，依 (source, sourceHandle, target, targetHandle) 排序
  subgraphs/<block>/          區塊定義，結構相同，介面寫在它的 index.json
  presets/<preset>.json       每個內嵌 preset 一個檔
layout/<name>.layout.json     不變：位置、note 幾何、segment
```

- **改一處，寫一個檔。** 改一個參數只會改寫那個節點的檔；新增節點會新增它的檔並更新 layout 檔；沒有變更的存檔不寫任何檔。1,000 個節點的圖存檔遠低於一秒。
- **順序。** 每個節點檔與每行線都有整數 `o`。新項目取最大值加 1024，移到兩者之間的項目取中點，只有在間隔用完時（同一位置連續插入十次之後）才全部重新編號。手寫、沒有 `o` 的節點檔排在最後。
- **檔名只是提示。** 檔名由節點 id 清理成各檔案系統都接受的名稱；需要清理、會與另一個只差大小寫的名稱相撞、或是 Windows 保留名稱時，會加上 id 的短雜湊。以檔案內的 `id` 為準。
- **長文字**移到自己的檔（參數名為 `code`、`script` 或以 `_code` 結尾用 `.py`，其餘 `.txt`），程式碼的修改就會顯示成逐行差異。文字跨過 400 字元時會在檔內與檔外之間移動。含 Windows 換行的文字留在節點檔裡。
- `index.json` 裡的 **`requires`** 列出這張圖用到的插件節點型別（依插件 id）與自訂節點型別，不含版本；釘版本仍是 manifest `[plugins]` 的事。這台機器不認得的型別會保留上次存檔寫下的那一筆。
- **存檔不會遺失資料。** 寫入前，伺服器會確認要寫的檔讀回來與正在存的圖完全相同；不相同就什麼都不寫（HTTP 500）。寫入順序是新的與有變的檔、`edges.jsonl`、`index.json`，最後才刪除多餘的檔。你放進資料夾的其他檔不會被動到。
- **修復不會悄悄存回去。** 資料夾需要修復才能讀時（線接到檔案已不見的節點、重複的線、兩個檔宣告同一個 id、文字檔不見、節點檔不是合法 JSON、不是 UTF-8 的位元組、symbolic link、沒有完成的存檔），編輯器會顯示一次「修復了 N 處」，分頁標示為未存檔；只有存檔時修復後的圖才會寫回。`cdui project validate` 會失敗並列出每一項修復。`index.json` 不存在或讀不了是錯誤，不是修復。
- 由較新版 CodefyUI 寫的資料夾（較高的 `storage_version`）會列在 Graphs 面板，但不能開啟、改名或覆寫。同名的資料夾與單一檔案並存（例如 checkout 了舊 commit）會被拒絕，訊息會列出兩者；刪掉不要的那一個。無法開啟的圖仍會留在 Graphs 面板，顯示為灰色，原因在 tooltip。
- **存檔進行中**，資料夾裡會有 `.saving` 標記（已列入 gitignore）；沒有完成的存檔（當機、斷電）會留下它，下次開啟時會說明資料夾可能混合了兩個版本。
- **不跟著 symbolic link 走。** 圖資料夾裡的 link 一律略過並回報；圖檔、圖資料夾或 layout 檔本身是指到專案外的 link 時拒絕開啟。
- **大小。** 圖資料夾裡單一檔案最多 16 MB；超過時整張圖拒絕開啟，而不是略過那個檔。
- **名稱。** 圖名最多 100 字元。Windows 保留名稱（`CON`、`aux`、`LPT1`……）存檔時會加上結尾的 `_`，讓專案能在 Windows 上 checkout；先前以這種名稱存的圖維持原名。

新專案的 `.gitattributes` 也讓 `*.jsonl` 與 `graphs/**/nodes/*` 維持 LF。

### 轉換格式 {/* #migrating */}

```bash
cdui project migrate my-service            # 所有圖轉成資料夾
cdui project migrate my-service --to file  # 轉回單一檔案
cdui project migrate my-service --graph echo
```

每張圖以新格式寫出後再讀回；只有讀回的圖（logic 與 layout）相同時才刪除舊格式，任何失敗都會讓舊格式原封不動。轉換整個專案時也會設定 `[storage] graph_format`。需要修復的資料夾，或含有資料夾無法保留的線（接到不存在節點的線、同一對連接埠之間的第二條線）的圖會被拒絕：在編輯器中開啟並存檔一次即可。git 會把轉換看成刪檔加新檔，所以 `git log --follow` 看不到轉換前的歷史；請改用舊路徑查（`git log -- graphs/<name>.graph.json`），並把轉換單獨 commit。

### 合併 {/* #merging */}

新專案的 `.gitattributes` 把 `edges.jsonl`、`index.json` 與 layout 檔標為 `merge=codefyui`：它們**依鍵**合併，而不是逐行。兩個分支各自新增節點與線，或一個刪線、另一個在旁邊加線，都能乾淨合併；只有兩邊對同一條線、同一個位置或同一個索引項目做了不同的修改才會留下衝突，此時寫入的是 git 一般的衝突標記。資料夾形式的圖，layout 依節點 id 排序，所以即使一般的逐行合併衝突也比較少。

從 Source Control 執行的合併（pull、stash pop、帶著變更切換分支）會自動使用這個合併方式。在終端機合併時，每個 clone 設定一次：

```bash
cdui project git-setup my-service
```

它會把合併 driver 寫進這個 repository 的 `.git/config`（較舊的專案也會在 `.gitattributes` 加上 `merge=codefyui` 那幾行）。沒有設定時，git 對這些檔使用一般的逐行合併：衝突會多一些，但不會合併錯。

在 Source Control 裡，資料夾形式的圖是一列，附上這張圖改了什麼的摘要；展開可看各個檔。對這一列暫存、取消暫存與捨棄，會作用在整個資料夾與它的 layout 檔。

## 預設專案與切換專案 {/* #the-default-project-and-switching-projects */}

### 伺服器開哪個專案 {/* #which-project-the-server-opens */}

啟動時依序決定：

1. 用 `--project` 或 `CODEFYUI_PROJECT_DIR` 指定的專案。
2. 設了 `CODEFYUI_GRAPHS_DIR` 時：平面模式。
3. 上次開著的專案（`<user data>/projects.json`）。
4. `backend/data/graphs` 裡還有專案功能推出前的圖時：平面模式。什麼都不搬；專案 chip 會提供複製到專案的選項（見下方）。
5. 以上皆非：開**預設專案** `~/CodefyUI-Projects/workspace`（Windows 是 `%USERPROFILE%\CodefyUI-Projects\workspace`），
   第一次會建立成 git 儲存庫（分支與第一個 commit 見下一節）。`CODEFYUI_PROJECTS_ROOT` 可改變 `CodefyUI-Projects` 的位置；`CODEFYUI_DEFAULT_PROJECT=0` 關掉這一步。
   如果 `workspace` 已存在、不是空的、又不是專案，就不去動它：伺服器維持平面模式並記一行警告。

### 新專案的儲存庫 {/* #a-new-projects-repository */}

預設專案、**新增專案…** 與匯入到新專案時，資料夾會成為 git 儲存庫：分支用你設定的 `init.defaultBranch`，沒設就是 `main`，
並有一個 `Initial project` commit，內容是骨架檔。Source Control 一開始是乾淨的，存第一張圖就只看到那張圖。
沒有設定 git 身分（`user.name`、`user.email`）時，這一個 commit 以 `CodefyUI <codefyui@localhost>` 建立，只用在這一個 commit，不寫進任何設定。
`cdui project init` 照舊不建立 commit。

### 專案 chip {/* #the-project-chip */}

工具列左側的 chip 顯示目前的專案、git 分支，有未提交的變更時多一個點。視窗寬度不到 1600px 時只顯示專案名，**重新載入節點**也改放到**檔案**選單裡，讓工具列維持一行；
分支與是否有未提交的變更放在 tooltip 與選單第一列。它的選單可以開啟最近的專案、開啟資料夾（**開啟資料夾…**）、
新增專案（**新增專案…**，建在專案根目錄，名稱邊打邊檢查）或 clone（**從 git URL clone…**，只接受 https，clone 到專案根目錄下與 repository 同名的資料夾）。
平面模式下顯示**沒有專案**；如果有專案功能推出前的圖，會提供把它們連同用到的 models、images、media 檔案一起複製到專案，對話框列出每個檔案的大小。
全部是複製，不會搬移。

切換會重新載入頁面。每個專案有自己的分頁（見[分頁與持久化](./tabs-persistence)）；切換前會列出未存檔的分頁，它們會留在你離開的那個專案。

**以下情況會拒絕切換**（訊息會說明是什麼在忙）：有執行中、排隊中或正在送出的 run；有 study 在跑；正在安裝外掛或套件包；
Source Control 有 git 操作進行中；另一個切換正在進行；或有已連線的 worker 用自己的資料夾宣告了這台伺服器的 mount label（`--mount data=...`）——
先停掉那個 worker、切換，再用新專案的資料夾重新啟動它。

**其他視窗。** 還停在舊專案的視窗，在你回到它時（或它要存檔、執行時）會發現專案已經換了，並提供重新載入。

**run 與 app 屬於它們的專案。** 執行清單預設列出這個專案的 run；**所有專案**列出全部。其他專案的 run 保留紀錄，
但它的檔案（checkpoint、TensorBoard 紀錄、worker 上傳的檔案）只有在那個專案開著時才拿得到，在那之前也不會被保留政策清掉。
從某個專案發佈的 app 版本，只有在那個專案開著時才會執行（否則回 409 `app_project_not_open`）。studies、sweeps 與 apps 的清單目前還沒有依專案篩選。

### 專案的 `.env` {/* #the-projects-env */}

這個安裝自己建立的專案（預設專案、**新增專案**、匯入）以及用 `--project` 指定的專案，照舊載入 `.env`；
但任何專案的 `.env` 都不能設定 `PYTHON*`、`LD_*` 或 `DYLD_*`：它們決定伺服器的子程序執行哪些程式碼，所以會略過，
並在伺服器記錄中列出名稱。真的需要的話，請在啟動 `cdui start` 前自己在 shell 裡 export。
其他你開啟的資料夾和 clone 下來的專案，編輯器會先列出變數**名稱**（從不顯示值）並詢問。答案會依檔案內容記在 `<user data>/trust.json`；
檔案改了就會再問一次。拒絕時專案照常開啟，只是沒有那些變數。

有些名稱在執行期開啟專案時永遠不載入，不論大小寫（`https_proxy` 和 `HTTPS_PROXY` 一樣會被擋下）：

- 程式怎麼啟動、程式碼從哪裡載入：`PATH`、`PYTHON*`、`LD_*`、`DYLD_*`、`CODEFYUI_*`、`NODE_OPTIONS`、`NODE_PATH`、`PERL5*`、`RUBY*`、
  `JAVA_TOOL_OPTIONS`、`_JAVA_OPTIONS`；
- 套件從哪裡安裝：`PIP_*`、`UV_*`、`CONDA_*`；
- git、ssh 與 shell 怎麼執行：`GIT_*`、`SSH_*`、`HOME`、`USERPROFILE`、`SHELL`、`COMSPEC`、`BASH_ENV`、`ENV`、`IFS`、`XDG_*`、
  `EDITOR`、`VISUAL`、`PAGER`、`BROWSER`、`TMPDIR`、`TEMP`、`TMP`；
- 網路流量往哪裡送、信任哪些憑證：`*_PROXY`（`HTTPS_PROXY`、`ALL_PROXY`、`NO_PROXY` 等）、`SSL_CERT_*`、`REQUESTS_CA_BUNDLE`、`CURL_CA_BUNDLE`。

名稱以 `_URL`、`_BASE`、`_ENDPOINT`、`_HOST` 或 `_API_BASE` 結尾的會載入，但提示裡會特別標出來：這類變數可以把請求（以及隨請求送出的金鑰）
導到別的伺服器（例如指向你不認識的主機的 `OPENAI_BASE_URL`）。同意前請先打開 `.env` 確認它們的值。

另外有些函式庫只在啟動時讀一次設定（`CUDA_VISIBLE_DEVICES`、`HF_HOME` 與其他 `HF_*`、
`TORCH_HOME`、`OMP_NUM_THREADS` 等）：切換後編輯器會說明哪些要重新啟動伺服器才會生效。

## 專案 mods {/* #project-mods */}

專案可以在 `mods/` 裡帶著自己的節點，和圖檔放在一起：

```
mods/
  nodes/
    focal_loss.py      # BaseNode 子類別，一個檔案可放多個
    _helpers.py        # 給節點檔匯入用：from ._helpers import ...
  presets/
    strong_aug.json    # 預設組合卡片，格式同匯出的預設組合
```

`mods/nodes/` 裡的節點在節點面板和存下的圖裡一律叫做 `project:<NODE_NAME>`（例如 `project:FocalLoss`）。
沒有前綴的名稱永遠不會指到專案節點：專案裡叫 `Conv2d` 的節點是 `project:Conv2d`，圖裡寫 `Conv2d` 的仍是內建的那個。
節點面板會標出每個節點的來源：這個專案的 `mods/`、安裝在這台電腦上的自訂節點，或外掛。

### 信任 {/* #project-mods-trust */}

這些檔案是 Python。信任它們，就會以你的權限在這個 CodefyUI 伺服器中執行它們。掃描結果只是提示，不是沙箱。

在你同意之前，`mods/` 裡的東西都不會執行。從資料夾開啟或 clone 下來的專案會先詢問：提示會列出每個檔案的大小、行數、
外掛閘門的掃描結果、它定義的節點名稱（從原始碼讀出，列出時不匯入任何東西），以及上次同意後改過的是哪些檔。
你的答案會依那些位元組（每個檔案和它的可執行位元）記在 `<user data>/trust.json`；任何變更都會在下次載入時
（重新載入節點、切換專案、重新啟動）再問一次。

- **信任並載入** 會執行這些檔案。掃描有發現的檔案不會載入，除非你勾選 **完全信任這些檔案**。
- **暫時不要** 會開啟專案但不載入 mods。同一個瀏覽器工作階段內，同樣的檔案不會再問；專案選單的 **檢視 mods/**、
  圖上的橫幅，以及節點上的 **未載入** 標記都能再打開提示。
- 用到未載入專案節點的圖仍可開啟，存檔時這些節點會原樣保留（類型、參數、連線）。執行、驗證和發布會拒絕並說明原因。

這個安裝自己建立的專案（預設專案、**新增專案**、匯入）會記一個基準：建立時以及之後每次載入時的 git `HEAD` 與 `mods/` 檔案
（一開始可以是沒有任何檔案）。`HEAD` 還停在基準的位置時，你自己的修改會直接載入。`HEAD` 移動且 `mods/` 跟著改變時
（pull、merge、checkout），就和其他專案一樣會詢問；從來沒有 mods 的專案第一次拉到 mods 也一樣。不是 git repo 的專案不會詢問。
這個版本之前建立的專案還沒有基準，裡面已經有的 mods 會詢問一次。

`mods/` 載入之後在磁碟上又變了（在 Source Control 或終端機裡 pull），編輯器會告訴你：目前執行的仍是先前載入的版本，
按 **重新載入並檢查** 就會走上面的檢查。不會自動重新載入。

只會讀取直接放在 `mods/nodes/` 的 `.py` 檔（不含 `__init__.py`）以及 `mods/presets/*.json`。提示列出的其他檔案，
例如 `mods/__init__.py`、`conftest.py` 或 `.pth` 檔，會標為 **不會載入**，排在會載入的檔案之後。
掃描結果以檔案的角度說明，例如「第 2 行匯入 socket：需要 network」。

`mods/` 裡有些東西永遠不會載入；只要有任何一個，`mods/` 整個都不載入，提示會列出它們：符號連結（或 `mods` 本身就是連結）、
不是一般檔案的東西、編譯過的檔案（`.pyc`、`.so`、`.pyd` 等）、`__pycache__` 資料夾，以及 `nodes/` 底下是套件
（有 `__init__.py`）的資料夾。只有直接放在 `mods/nodes/` 的檔案才是節點檔；每個檔案只讀一次、比對你同意過的內容，
並照讀到的內容執行。兩個檔案定義同一個 `NODE_NAME` 時兩者都不載入。內部節點沒有載入、或名稱和內建預設組合重複的預設組合會略過並列出。

切換到其他專案會卸載它的節點與預設組合，但卸載不等於隔離：執行過的程式碼可能留下東西。另一個專案的 mods 在這個伺服器中執行過之後，
專案 chip 會顯示 **建議重新啟動**；要完全移除，請重新啟動伺服器。

用到 `project:` 節點的圖只能在 **This server** 上執行：送到 worker 或 `any` 的執行會在開始前被拒絕，
因為 worker 沒有這個專案的 mods（也沒有你同意執行它們）。

### 在命令列 {/* #project-mods-cli */}

```bash
cdui project mods .                 # 掃描結果與 sha256，不執行任何東西
cdui project mods . --trust         # 同上，然後詢問
cdui project mods . --trust --yes   # 不詢問直接記錄（CI 用）
cdui project mods . --write         # 改寫 manifest 的 [mods]
```

`--yes` 就是同意執行這些程式碼，和在提示裡按同意一樣。`--full`（搭配 `--trust`）也會執行掃描有發現的檔案。
`cdui project validate` 只在信任紀錄和檔案完全相符時載入專案的 mods，否則會說明用到 `project:` 節點的圖會失敗與原因。
伺服器以外不會替擁有者自動信任任何東西。

manifest 的 `[mods]` 區塊列出每個檔案的 sha256 以及它定義的節點或預設組合，讓 review 看得出哪個 mod 改了。
它由 `cdui project freeze` 與 `cdui project mods --write` 寫入（存檔時不會改），就地修改並保留你的註解；
`cdui project validate` 在它和 `mods/` 不符時警告。它永遠不用來判斷信任：儲存庫不能替自己擔保。

`cdui project freeze --used-only` 只釘選專案裡的圖用到的外掛。存成資料夾的圖在 `index.json` 列出它需要的東西：
`requires.plugins`、`requires.mods`（它的 `project:` 節點）與 `requires.custom`（安裝在作者電腦上的自訂節點）。

`cdui project restore` 安裝 lockfile 裡沒有的釘選前，會先顯示來源、它要求匯入的模組與要求的能力，並詢問；在腳本裡請加 `--yes`。

## 遷移既有的扁平 graphs 目錄 {/* #migrating-an-existing-flat-graphs-dir */}

如果你用的是舊的「[版本控管你的 graphs](/usage/version-control-graphs)」做法
（`CODEFYUI_GRAPHS_DIR` 指向扁平的 `*.json` 目錄），可用一個指令匯入：

```bash
cdui project init my-service --adopt /path/to/old-graphs
```

每個 `*.json` 都會複製進 `graphs/` 並拆分成 logic/layout 一對檔案。

## 注意事項與限制（v1） {/* #notes-and-limits-v1 */}

- 一台伺服器同時只開一個專案；用專案 chip 切換（見上方）。
- `DB_PATH`、custom nodes 與[匯出的預設組合](/advanced/presets#creating-your-own)仍由整個安裝共用；
  [plugins](/advanced/plugins) 才是可攜的機制（在 manifest 中以 SHA 釘選），專案自己的節點則放在 [`mods/`](#project-mods)。
- `assets/data/` 是相對的 `Dataset` 或 `FileReader` 路徑所解析到的位置。
  `CSVReader`、`DocumentLoader` 與 `TextCorpusDataset` 則會把輸入的相對路徑
  解析到專案根目錄，所以請寫 `assets/data/table.csv`，而不是 `table.csv`；
  離開專案的路徑會被拒絕。單純的檔名會先在透過它們的 DATA_FILE 下拉選單上傳
  的檔案中尋找，這些檔案屬於安裝全域 —— 位於 `backend/data/files`，或由
  `CODEFYUI_DATA_FILES_DIR` 指定 —— 就像 `DB_PATH` 與 custom nodes 一樣。
  隨附的範例仍然解析到安裝目錄：`CSVReader` 的 `data/samples/iris.csv`，
  以及 `DocumentLoader` 中 `data/samples/` 底下的路徑。
- `CODEFYUI_MODELS_DIR`、`CODEFYUI_IMAGES_DIR` 與 `CODEFYUI_MEDIA_DIR` 可搬移模型、
  圖片與執行媒體的存放位置；專案模式下預設為 `<project>/assets/models`、
  `assets/images` 與 `assets/media`，除非明確設定。
- 編輯器與手動改檔之間採「後寫者勝」。只有透過[版本控制](./source-control)
  分頁做的寫入（拉取、切換分支、取出儲藏、捨棄）會跳出「磁碟上已變更」的提示與
  重新載入按鈕；用其他工具改的檔案不會被偵測到。請把專案目錄排除在
  OneDrive/Dropbox 同步之外 -- 同步軟體會弄壞 `.git` 並跟原子改名互相競爭；
  請改用真正的 git remote。
- 由較新版 CodefyUI 寫出的 graph 會以**唯讀**開啟（可檢視／執行，儲存
  停用），讓舊版永遠不會弄丟它不認識的欄位。另存新檔... 也被同一道防護擋住，
  這是刻意的：graph 載入的那一刻，記憶體中的副本就已經丟失那些未知欄位，
  另存新檔... 只會把這份有損的副本換個名字寫出去。
