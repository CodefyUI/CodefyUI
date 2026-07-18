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
  .env.example     committed template of required secret keys
  .env             your secrets (gitignored, never committed)
```

## 為什麼要分離？

`graphs/<name>.graph.json` 只存會改變 graph *行為* 的內容（節點、邊、參數、
內嵌 presets）。節點**位置**與便利貼的幾何資訊放在
`layout/<name>.layout.json`。因此拖動節點只會在 `layout/` 產生 diff，改參數
只會在 `graphs/` 產生 diff -- code review 看到的是邏輯變更，而不是一整面
「像素移動」的雜訊。（已知例外：`SequentialModel` 子圖的層位置存在
`params.layers` 內，留在 logic 檔中。）

缺少 layout 檔（或某個節點沒有已儲存的位置）時，編輯器會在載入時自動排版，
並於下次存檔時寫回結果；便利貼若只缺幾何資訊（尺寸/綁定）則直接使用預設值 --
「只缺幾何資訊」刻意不視為缺少 layout。

## 完整流程

### 1. 建立專案

```bash
cdui project init my-service
cd my-service
```

`init` 會鋪好 `graphs/`、`layout/` 與 `assets/{images,models,data}/`（空目錄，
以 `.gitkeep` 追蹤），寫入 `.gitignore` / `.gitattributes` / `.env.example` /
`README.md`，並執行 `git init`（不建立 commit -- 它會印出後續步驟）。
`assets/output/` 不會預先建立；第一次有節點（例如 ImageWriter）寫入時才會
出現。

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

`validate` 會檢查 `graphs/` 底下的**每一個** graph，並印出檢查數量 -- 空的
`graphs/` 會回報 `Validation passed (0 graphs checked)`，而不是一個空洞的
綠燈。**canvas-only** 的 graph（例如沒有宣告任何 **GraphOutput** 的訓練
graph）會在 contract 關卡失敗，因為每個可發佈的 graph 至少要宣告一個輸出。
你可以給它一個真正的輸出（MNIST 範例專案就把 checkpoint 路徑以
`weights_path` 輸出發佈出來），或只驗證你的發佈目標：

```bash
cdui project validate . --graph serve   # repeatable: --graph a --graph b
```

`--graph` 指到 `graphs/` 裡不存在的名字會直接報錯，所以打錯字絕不會讓 CI
關卡變成空洞的通過。

Pins 來自 `cdui project freeze .`：它讀取你本機安裝的 plugins，把每一個的
確切 commit SHA 寫進 `codefyui.project.toml` 的 `[plugins]` 表（以本機開發
連結安裝的 plugin 會被略過 -- 機器限定的路徑沒有 SHA 可釘選）。安裝或更新
plugin 之後執行它，並在下次 push 前提交 manifest 的變更：

```bash
cdui project freeze .
```

### 5. 在專案上啟動伺服器

```bash
cdui start --project .
```

log 會印出 `Project: <abs> (git <short-sha>)`，若有釘選的 plugin 未安裝，
會警告一次並指名 `cdui project restore`。

### 6. 建立 API key（invoke 需要）

Session token 位於 `<user_data_dir>/codefyui/session.token` -- Windows 在
`%LOCALAPPDATA%\codefyui\session.token`，macOS 在 `~/Library/Application
Support/codefyui/session.token`，Linux 在 `~/.local/share/codefyui/session.token`
（完整說明見 [Graph as a Function](/usage/graph-as-a-function)）。

PowerShell：

```powershell
# payload.json: {"name": "demo"}
$token = Get-Content "$env:LOCALAPPDATA\codefyui\session.token"
curl.exe -s -X POST "http://127.0.0.1:8000/api/keys" `
  -H "X-CodefyUI-Token: $token" -H "Content-Type: application/json" `
  --data "@payload.json"
```

bash：

```bash
TOKEN=$(cat ~/.local/share/codefyui/session.token)   # macOS: ~/Library/Application Support/codefyui/session.token
curl -s -X POST http://127.0.0.1:8000/api/keys \
  -H "X-CodefyUI-Token: $TOKEN" -H "Content-Type: application/json" \
  --data '{"name": "demo"}'
```

`# -> {"id": 1, "name": "demo", "prefix": "cdui_xxxxxxxx", "token": "cdui_..."}`（完整金鑰只顯示這一次，在 "token" 欄位）

### 7. 發佈（記錄 git commit）

`cdui project publish` 包裝的是同一個 [publish](/usage/publish) 端點
（`POST /api/apps/{slug}/publish`），外加專案模式防護與自動 git 溯源。先在
`codefyui.project.toml` 設定一次預設目標：

```toml
[publish]
graph = "echo"
slug = "echo-svc"
```

提交它 -- 未提交的 manifest 變更正是下一步會警告的那種 dirty 工作樹 --
然後發佈：

```bash
git add -A && git commit -m "set publish target"
cdui project publish .
# -> Published echo-svc v1 (git 1a2b3c4)
```

v1 的發佈是**僅限本機**：它會確認 `GET /api/health` 回報開啟的是「這個」
專案（所以絕不會把錯的 commit 記在別人的位元組上），計算
`git rev-parse HEAD` + `git status --porcelain`，並在工作樹 dirty 時大聲
警告。從 git 儲存庫發佈時，每次都會把 `git_dirty` 以 `true` 或 `false` 記
在 commit 旁邊 -- dirty 的工作樹還會額外印出上面說的警告橫幅。若 commit
已解析出來、`git status` 本身卻失敗了，`git_dirty` 會記錄為 `null`（= 未知），
絕不捏造成 `false`。

第一次發佈時自動建立 app，**只**適用於 manifest 裡已提交的
`[publish].slug` 目標。在命令列明確傳入的 `--slug` 若指到伺服器不認識的
app，會以 404 `app_not_found` 失敗 -- 打錯字不會再默默生出第二個 app --
CLI 會提示你用 `--create` 來刻意首發一個新的命令列 slug：

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
（`CODEFYUI_GRAPHS_DIR` 指向一個扁平的 `*.json` 目錄），一個指令就能收編：

```bash
cdui project init my-service --adopt /path/to/old-graphs
```

每個 `*.json` 都會複製進 `graphs/` 並拆分成 logic/layout 一對檔案。

## 注意事項與限制（v1）

- 每個伺服器實例一個專案（編輯器內還沒有專案切換器）。
- `DB_PATH` 與 custom nodes 仍是安裝層級的全域設定；[plugins](/advanced/plugins)
  才是可攜的機制（在 manifest 中以 SHA 釘選）。
- 編輯器與手動改檔之間採「後寫者勝」（「磁碟上已變更」警告是後續項目）。
  請把專案目錄排除在 OneDrive/Dropbox 同步之外 -- 同步軟體會弄壞 `.git`
  並跟原子改名互相競爭；請改用真正的 git remote。
- 由較新版 CodefyUI 寫出的 graph 會以**唯讀**開啟（可檢視／執行，Save
  停用），讓舊版永遠不會弄丟它不認識的欄位。Save As 也被同一道防護擋住，
  這是刻意的：graph 載入的那一刻，記憶體中的副本就已經丟失那些未知欄位，
  Save As 只會把這份有損的副本換個名字寫出去。
