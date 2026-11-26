---
sidebar_position: 5.5
title: 代理技能
description: 讓 Graph Copilot 等代理外掛使用標準的 Agent Skills：技能從哪裡來、怎麼安裝、同名時用哪一個，以及為什麼技能裡的東西永遠不會被執行。
---

# 代理技能

**代理技能**（Agent Skill）是一個裝著指令的資料夾，AI 代理在任務需要時才去讀它。CodefyUI 透過 `/api/skills` 把技能提供給 [Graph Copilot](./graph-copilot) 等代理外掛，所以任何外掛都能使用，為其他代理寫的技能在這裡也能用。技能遵循公開的 [Agent Skills 規格](https://agentskills.io/specification)。

## 技能是什麼 {/* #what-a-skill-is */}

一個技能是一個以技能名稱命名的資料夾，裡面有 `SKILL.md`：

```markdown
---
name: lr-finder
description: Find a learning rate for a new model with a short range test. Use when the user does not know what learning rate to start from.
license: MIT
---

# Learning-rate range test

1. ...
```

- YAML frontmatter 必須有 `name`（1 到 64 個小寫英文字母、數字與單一連字號，並與資料夾同名）與 `description`（最多 1024 字元，說明技能做什麼、何時使用）。`license`、`compatibility`、`metadata` 與 `allowed-tools` 為選填。
- Markdown 本文就是指令。請保持在 500 行以內，細節放在其他檔案。
- 選用的子資料夾：`references/`（更多文件）、`assets/`（範本、資料）與 `scripts/`。

代理分三步看到技能，所以很長的技能在用到之前不佔任何成本：所有使用中技能的 `name` 與 `description` 一直在代理的指令裡；任務符合描述時，代理才載入技能本文；本文指向 `references/` 裡的檔案時，代理才去讀那個檔案。

## 技能從哪裡來 {/* #where-skills-come-from */}

CodefyUI 會在五個地方尋找。兩個技能同名時，使用下表中順位較前的那一個，其他的標記為被蓋掉：

| 來源 | 資料夾 |
| --- | --- |
| `project` | `<專案>/skills/<name>/`，開啟[專案目錄](/usage/project-directories)時才有 |
| `user` | CodefyUI 使用者資料夾裡的 `skills/<name>/`：你安裝的或手動放進去的技能 |
| `plugin:<id>` | 已啟用外掛內的 `skills/<name>/` |
| `builtin` | CodefyUI 安裝目錄裡的 `skills/<name>/` |
| `extra` | `CODEFYUI_SKILLS_EXTRA_DIRS` 列出的每個資料夾（以 `:` 分隔，Windows 為 `;`）。預設為空。 |

技能的 id 是 `<source>/<name>`，例如 `builtin/build-graph`。無效或已停用的技能不會蓋掉其他技能，所以停用專案裡的某個技能後，內建的同名技能就會恢復使用。

若要讓 CodefyUI 的代理使用你為其他代理準備的技能，請在啟動伺服器時列出那些資料夾，例如 `CODEFYUI_SKILLS_EXTRA_DIRS=~/.claude/skills:~/.agents/skills cdui start`。你沒有設定時，CodefyUI 不會讀取它們。

### 內建技能 {/* #the-built-in-skills */}

| 技能 | 用途 |
| --- | --- |
| `build-graph` | 建立或修改圖：節點與 port 型別、`Start` 觸發、驗證錯誤與修法、自動排版 |
| `train-and-evaluate` | 訓練與評估模型：標準的 `TrainingLoop` 圖、seed、閱讀訓練分頁、checkpoint、語言模型與 PushWorld VLA |
| `run-experiments` | 參數搜尋、消融與多 seed 比較：實驗設計、重複次數、怎麼解讀區間，以及證據支持不了的結論 |

## 安裝技能 {/* #installing-a-skill */}

帶著 session token 把來源送到 `POST /api/skills/install`（見 [API 參考](./api-reference#agent-skills-and-context)）。來源可以是：

- 伺服器磁碟上的資料夾，例如 `/home/me/skills/lr-finder`；
- 伺服器磁碟上的 `.zip`、`.tar.gz` 或 `.tgz`，內容是技能資料夾或資料夾裡的檔案；
- HTTPS git URL，可在結尾加上 `@<分支、標籤或 commit>`，例如 `https://github.com/me/lr-finder@v1`。repository 的根目錄必須就是技能。

技能會以 frontmatter 裡的名稱複製到 `user` 來源。以下情況會拒絕安裝，且不留下任何檔案：來源沒有含合法 `name` 與 `description` 的 `SKILL.md`、超過 20 MB 或 500 個檔案、含有符號連結，或壓縮檔中有會寫到技能資料夾外的成員。已經安裝同名技能時不會覆蓋（409），請先刪除。

要移除時使用 `DELETE /api/skills/user/<name>`，只能刪除已安裝的技能。`PUT /api/skills/enabled` 可停用或啟用任何技能，設定存在使用者資料夾的 `skills/state.json`。

## 專案指令：AGENTS.md {/* #project-instructions-agentsmd */}

在專案模式下，專案目錄根部的 `AGENTS.md` 存放給所有在這個專案工作的代理的長期指令，例如「每次執行都要設定 seed」或「評估用的圖是 `serve`」。`GET /api/agent/context` 會把它連同專案名稱、圖的清單與使用中的技能一起交給代理外掛。最多讀取 32 KB。

## 安全性 {/* #safety */}

技能是文字。CodefyUI 從不執行技能裡的任何東西：不執行 `scripts/` 裡的檔案，不執行安裝步驟，不安裝依賴。複製 git 來源時不會執行任何 hook。伺服器只列出、驗證並回傳檔案內容，並拒絕讀取技能資料夾以外的檔案（不論是透過 `..`、絕對路徑或符號連結），也拒絕二進位檔。

代理如何使用技能的指令，取決於代理與你：代理外掛在修改或執行你的圖之前仍會先詢問。請像安裝外掛一樣，只安裝來源可信的技能，因為技能的文字會影響代理提出的建議。

## 技能沒有被使用時 {/* #when-a-skill-is-not-used */}

`GET /api/skills` 會列出找到的每個技能，包含沒有使用的，並說明原因：

- `valid: false` 且 `issues` 中有等級為 `error` 的項目：沒有 frontmatter、frontmatter 不是合法的 YAML，或缺少 `description`。這個技能不會被載入。
- `issues` 中有等級為 `warning` 的項目：規格要求的某件事不符，例如 `name` 與資料夾名稱不同。這個技能仍會以資料夾名稱載入。
- `shadowed_by`：改用了較高順位來源的同名技能。
- `enabled: false`：已被停用。
