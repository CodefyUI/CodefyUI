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
  graphs/    <name>.graph.json    logic (nodes/edges/params/presets)
  layout/    <name>.layout.json   positions (reviewable, generated)
  assets/images/   assets/models/   assets/data/    scaffolded empty
  assets/output/                                    created on demand (e.g. ImageWriter)
  assets/media/                                     created on demand (run-produced video, served by /api/media)
  .env.example     committed template of required secret keys
  .env             your secrets (gitignored, never committed)
```

## 為什麼要分離？

`graphs/<name>.graph.json` 只儲存會改變 graph *行為* 的內容（節點、邊、參數與內嵌 presets）。節點**位置**與便利貼幾何資訊則存放在 `layout/<name>.layout.json`。因此，拖動節點只會變更 `layout/`，修改參數只會變更 `graphs/`；code review 可以聚焦於邏輯變更，不會受到節點位移的 diff 干擾。（已知例外：`SequentialModel` 子圖的層位置存放在 `params.layers`，因此仍位於 logic 檔。）

缺少 layout 檔（或某個節點沒有已儲存的位置）時，編輯器會在載入時自動排版，
並於下次存檔時寫回結果；便利貼若只缺幾何資訊（尺寸/綁定）則直接使用預設值 --
「只缺幾何資訊」刻意不視為缺少 layout。

## 完整流程

### 1. 建立專案

```bash
cdui project init my-service
cd my-service
```

`init` 會建立 `graphs/`、`layout/` 與 `assets/{images,models,data}/`（空目錄，以 `.gitkeep` 追蹤），寫入 `.gitignore` / `.gitattributes` / `.env.example` / `README.md`，並執行 `git init`。它不會建立 commit，而會印出後續步驟。`--force` 允許寫入非空目錄，但不會覆寫既有的 manifest 或 `README.md`。`assets/output/` 不會預先建立；節點（例如 ImageWriter）第一次寫入時才會出現。

### 2. 加入一個 graph

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
  "presets": []
}
```

### 3. 提交

```bash
git config user.name  "You"
git config user.email "you@example.com"
git add -A
git commit -m "echo service"
```

`.env` 已被 gitignore；`.env.example` 要提交。大型資料請提交一個小的下載
腳本，永遠不要提交資料或權重本身。

### 4. 驗證（CI 關卡）

```bash
cdui project validate .
```

`validate` 會初始化**完整**的 registry（builtin + custom + plugin 的節點與
presets，跟伺服器完全一樣），並對每個 graph 執行發佈前檢查：graph 內含機密
檢查、contract、進入點、接線，以及節點／preset 有效性。它也會在 `.env` 被
git 追蹤時報錯，並對缺少的 plugin pins 提出警告（加 `--strict` 則變成錯誤）。
在 CI 上，請先 **restore 再 validate**：

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

### 5. 在專案上啟動伺服器

```bash
cdui start --project .
```

log 會印出 `Project: <abs> (git <short-sha>)`，若有釘選的 plugin 未安裝，
會警告一次並指名 `cdui project restore`。（`cdui dev --project .` 也會如此，
並提供 hot reload。）

使用 `--project` 啟動，也會載入 `<project>/.env`：一般的 `KEY=VALUE` 行（允許開頭有 `export `，也允許值外圍有引號），會在探索 node 與 plugin 前以 `os.environ.setdefault` 語意套用，因此 shell 中已設定的變數優先。只有執行期間的機密適合放在這裡 —— LLM API key，以及 node 在執行時從環境讀取的任何值。檔案中的 `CODEFYUI_*` 設定不會生效，因為伺服器設定在讀取檔案前就已固定；請改在 shell 或 systemd unit 中設定。log 只會記錄載入的數量，絕不記錄值；不帶 `--project` 時完全不會讀取 `.env`。

### 6. 建立 API key（invoke 需要）

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

`# -> {"id": 1, "name": "demo", "prefix": "cdui_xxxxxxxx", "token": "cdui_..."}`（完整金鑰只顯示這一次，在 "token" 欄位）

### 7. 發佈（記錄 git commit）

`cdui project publish` 包裝的是同一個 [publish](./publish.md) 端點
（`POST /api/apps/{slug}/publish`），外加專案模式防護與自動 git 溯源。先在
`codefyui.project.toml` 設定一次預設目標：

```toml
[publish]
graph = "echo"
slug = "echo-svc"
```

先提交 manifest；未提交的 manifest 變更會使下一步顯示 dirty 工作樹警告。接著發佈：

```bash
git add -A && git commit -m "set publish target"
cdui project publish .
# -> Published echo-svc v1 (git 1a2b3c4)
cdui project publish . --note "first cut"   # --note attaches an immutable note to the version
```

v1 的發佈**僅限本機**：它會確認 `GET /api/health` 回報目前開啟的是此專案，避免把其他專案的內容記到這個 commit；接著計算 `git rev-parse HEAD` 與 `git status --porcelain`，工作樹有未提交變更時會顯示醒目警告。從 git 儲存庫發佈時，每次都會在 commit 旁記錄 `git_dirty` 的 `true` 或 `false`；工作樹不乾淨時也會顯示上述警告橫幅。若 commit 已解析成功，但 `git status` 本身失敗，`git_dirty` 會記為 `null`（未知），不會錯誤地記為 `false`。

第一次發佈時，只有 manifest 中已提交的 `[publish].slug` 目標會自動建立 app。命令列明確傳入的 `--slug` 若指向伺服器未知的 app，會以 404 `app_not_found` 失敗，避免拼寫錯誤建立第二個 app。CLI 會提示使用 `--create`，明確建立新的命令列 slug：

```bash
cdui project publish . --graph echo --slug echo-svc --create
```

> **遠端 / CI 部署不在 v1 範圍內。**`cdui project validate` 可以在 CI 上
> 跑，但發佈需要一台開著該專案的本機伺服器。已排定的後續項目是管理範圍、
> 用 API key 的發佈（`--url` / `--key`）。

### 8. 呼叫（invoke）

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

### 9. 查「這是哪個 commit 發佈的」

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

## 遷移既有的扁平 graphs 目錄

如果你用的是舊的「[版本控管你的 graphs](/usage/version-control-graphs)」做法
（`CODEFYUI_GRAPHS_DIR` 指向扁平的 `*.json` 目錄），可用一個指令匯入：

```bash
cdui project init my-service --adopt /path/to/old-graphs
```

每個 `*.json` 都會複製進 `graphs/` 並拆分成 logic/layout 一對檔案。

## 注意事項與限制（v1）

- 每個伺服器實例一個專案（編輯器內還沒有專案切換器）。
- `DB_PATH` 與 custom nodes 仍是安裝層級的全域設定；[plugins](/advanced/plugins)
  才是可攜的機制（在 manifest 中以 SHA 釘選）。
- `assets/data/` 是相對的 `Dataset` 或 `FileReader` 路徑所解析到的位置。
  透過 DATA_FILE 下拉選單（`CSVReader`、`DocumentLoader`、
  `TextCorpusDataset`）上傳的檔案也同樣屬於安裝全域 —— 位於
  `backend/data/files`，或由 `CODEFYUI_DATA_FILES_DIR` 指定 —— 就像
  `DB_PATH` 與 custom nodes 一樣。
- `CODEFYUI_MODELS_DIR`、`CODEFYUI_IMAGES_DIR` 與 `CODEFYUI_MEDIA_DIR` 可搬移模型、
  圖片與執行媒體的存放位置；專案模式下預設為 `<project>/assets/models`、
  `assets/images` 與 `assets/media`，除非明確設定。
- 編輯器與手動改檔之間採「後寫者勝」（「磁碟上已變更」警告是後續項目）。
  請把專案目錄排除在 OneDrive/Dropbox 同步之外 -- 同步軟體會弄壞 `.git`
  並跟原子改名互相競爭；請改用真正的 git remote。
- 由較新版 CodefyUI 寫出的 graph 會以**唯讀**開啟（可檢視／執行，儲存
  停用），讓舊版永遠不會弄丟它不認識的欄位。另存新檔... 也被同一道防護擋住，
  這是刻意的：graph 載入的那一刻，記憶體中的副本就已經丟失那些未知欄位，
  另存新檔... 只會把這份有損的副本換個名字寫出去。
