# CodefyUI 專案分析報告

> 撰寫日期：2026-03-21（最後更新：2026-05-17）
> 目標：評估 CodefyUI 現況、分析競品最新表現、給出 SWOT 與可延伸的改進方向
>
> **本次更新（2026-05-17）重點：**
> - 過去三週（2026-04-26 → 2026-05-17）共 35 次 commit，重大產物：**Plugin 系統上線**（PR #21，6 個教學章節包 + 12 個 Edu 節點）、**安全架構** session token + Host 白名單 + AST 驗證 + codegen 白名單、**Python 匯出可執行化**（PR #20/#22）、**no-Node 發佈管線**（end users 只需 uv + Python 即可裝）、新增 22 個節點（Diffusion ×6、Classical ×5、LLM ×3、Data ×4、RNN ×1、Transformer MoE ×1、RL RLHF ×2）。
> - 節點數由 69 → **91**（13 → **15** 類別），後端測試由 24 → **128 個 test files**（新增 60 個節點專屬測試）。
> - **競品大洗牌**：n8n GitHub stars 從 40k → **188k**（成為視覺化 AI 工作流第一）、Dify 從 55k → **90k+**、ComfyUI ≈ 100k、Langflow 仍 100k+；**新進場玩家**：Adobe Project Graph（2025-11，Creative Cloud 節點式工作流，含 Capsule 打包概念）、NodeTool（local-first AI 工作流）。MCP（Model Context Protocol）2026 已成 LLM 工具標準（10k+ 公開伺服器、Anthropic 捐贈給 Linux Foundation）。
> - 戰略主軸不變：**教育/可解釋性護城河**仍是 CodefyUI 唯一不可被快速複製的優勢。本次新增 SWOT 與「延伸方向」章節。

---

## 目錄

1. [Executive Summary](#1-executive-summary)
2. [現況分析](#2-現況分析)
   - 2.1 技術棧 / 2.2 架構分析 / 2.3 功能清單 / 2.4 內建節點清單 / 2.5 程式碼品質
   - 2.6 教育功能（A1 Step Trace / A2 Weight Persistence / A3 Backward）
   - **2.7 Plugin 系統 + 安全架構 + 可執行 Python 匯出（2026-05 新增章節）★**
3. [競品分析](#3-競品分析)（含 2026-05 GitHub Stars 重排、Adobe Project Graph / NodeTool / MCP）
4. [Gap Analysis — 差距分析](#4-gap-analysis--差距分析)
5. [目標定位與願景](#5-目標定位與願景)
6. [建議發展路線圖](#6-建議發展路線圖)
7. [技術架構建議](#7-技術架構建議)
8. **[SWOT 分析](#8-swot-分析) ★ 新增**
9. [風險評估](#9-風險評估)
10. **[延伸與改進方向](#10-延伸與改進方向) ★ 新增**
11. [附錄：競品功能對照表](#11-附錄競品功能對照表)

---

## 1. Executive Summary

CodefyUI 在 2026-04 確立**「視覺化 ML 學習與實驗平台」**的定位後，2026-05 三週內把「能站上市場」的工程基礎一次補齊：**Plugin 系統**讓教學內容可由社群擴充、**安全架構**讓 desktop 模式不再裸奔、**可執行 Python 匯出**讓「教學原型 → 真實 PyTorch 程式」這條路通了、**no-Node 發佈管線**讓 end users 只需 `uv + Python` 就能裝起來，無須裝 Node/pnpm。

三大教育核心（A1 Step Trace、A2 Weight Persistence、A3 Backward Pass + Grad Health）仍是市場上沒有任何競品提供的能力，現在搭配 Plugin 章節包與安全沙盒驗證，意味著教師 / 課程作者可以**寫一個自訂教學節點 → 透過 `cdui plugin install` 分發給學生**，且學生端不會因為一個惡意 `import os; os.system(...)` 而中招。

**核心優勢：**

*基礎能力（既有 + 本期擴大）*
- 完整 DAG 圖形編輯器、拓撲排序 + 並行執行引擎、12 種類型安全連線、節點快取（hash-based）、Dirty Node Tracking、執行取消、錯誤恢復
- 現代化技術棧（React 19 + @xyflow/react 12 + FastAPI + WebSocket + PyTorch 2）
- 多分頁工作區、自訂節點熱重載、Preset 巢狀展開
- **91 個內建節點、15 類別**（CNN/RNN/Transformer/RL/Data/Training/IO/Control/Dataflow/Utility/Normalization/TensorOps/**LLM**/**Diffusion**/**Classical**）
- 28 個 CI 驗證範例工作流（9 個新分類目錄）
- 後端 128 個 test files、前端 13 個 test files
- AGPL-3.0 + 商業授權雙軌（PR #9 正式化）

*教育與可解釋性（仍跨平台無對手）*
- **A1 步驟追蹤（Verbose Step Trace）**：節點執行時逐步記錄中間張量並附 LaTeX 公式
- **A2 權重持續化（Weight Persistence）**：StatefulModuleMixin + NodeStateStore（LRU 200 modules），可在 UI 互動式訓練
- **A3 反向傳播 + 梯度健康度**：自動 retain_grad、BackwardOnce 顯式選 loss、按 norm/mean/max 分類為 vanishing / exploding / healthy
- **MathText（KaTeX）**：節點描述、Step Trace、ConfigPanel、Tooltip 全面支援 LaTeX
- **Inspector 三分頁**（Forward / Steps / Backward）+ Compare Segment 路徑高亮
- 互動式視覺化節點：AttentionHeatmap / AttentionMask / EmbeddingScatter / TokenizerViz / EduSelfAttention / EduMultiHeadAttention / EduCrossAttention / EduKNN 等 11 個 viz node

*2026-05 新增戰略基礎*
- **Plugin 系統（PR #21）**：`cdui plugin install C2 | owner/repo[@ref] | <url>`、`plugin list/uninstall/update/info/search`；命名空間 `cdui_plugins.<id>`，built-in 章節包就地啟用、第三方包下載到 `<USER_DATA>/plugins/`；lockfile 記錄 SHA pin 與使用者批准的 allowed_modules。6 個內建章節包 c1–c6 含 12 個 Edu 教學節點。
- **安全架構（commit 09ca115）**：
  - **Session token**：每個 backend process 啟動產一個 0600 權限 token file（`<USER_DATA>/codefyui/session.token`），所有 mutating 請求必須在 `X-CodefyUI-Token` 帶回；防 CSRF / DNS rebinding。
  - **Host header 白名單**：阻擋 DNS rebinding 把 attacker.com 重綁到 127.0.0.1。
  - **AST 驗證器**（`plugin_validator.py`）：對 plugin / 自訂節點 Python 原始碼做 AST 掃描，封鎖 `os/subprocess/socket/pickle/ctypes/...` 進口、`exec/eval/getattr` 呼叫、`__class__/__bases__/__subclasses__` 沙盒逃逸 dunder。
  - **Codegen 白名單**（`codegen.py`）：Python 匯出時所有插入到生成原始碼的 user-controlled 型別名（dataset / optimizer / loss / layer 名稱）都會比對白名單，未列入者改成 `# TODO` placeholder，杜絕透過 graph.json 注入 RCE。
- **可執行 Python 匯出（PR #20/#22）**：支援 graph spec v2，匯出檔案使用 graph 名稱當 header / filename；舊 v1 仍 fallback 支援。
- **no-Node 發佈管線**：GitHub Releases 上傳 `frontend-dist.tar.gz`，end users 安裝時 cdui 從 release 下載 prebuilt，**完全不需 Node.js / pnpm**。`cdui install` 流程支援 `--gpu auto|cu118|cu121|cu124|cu128|rocm6.x|cpu|mps|skip` 與 `--dev` 旗標、CI 三平台 install-check + tag-triggered asset pipeline。
- **i18n 完成度**：所有 91 個內建節點完成 zh-TW 翻譯（PR #23、#3）。
- **節點測試覆蓋**：commit 0f2473c 一次為 60 個先前未測過的節點補上專屬單元測試。

**主要差距（更新版）：**

| 差距 | 說明 |
|------|------|
| LLM 雲端推理節點 | 仍缺 OpenAI / Anthropic / Ollama 客戶端節點；Tokenizer/WordVector/AttentionHeatmap 等屬於 LLM 教學節點而非生產推理。**Anthropic 目前佔企業 LLM 支出 40%，是 first-class 整合的優先選擇** |
| Vector DB / RAG | 完全缺席（Chroma/FAISS/PGVector/DocumentLoader/TextSplitter/Retriever） |
| Agent 編排 | 完全缺席（Tool 使用、Memory、Planning） |
| **MCP 協定** | 完全缺席（既無 client 也無 server 模式）；MCP 2026 已是 LLM 工具標準 |
| 容器化部署 | 仍無 Dockerfile / docker-compose / Helm Chart（教育場景影響小，但阻礙課堂部署） |
| 資料庫儲存 | 仍為 JSON / 記憶體；Graph / Run / NodeState 重啟即遺失 |
| 跨 Run Dashboard | Step Trace 解單 run 的可解釋性，但跨 run 比較仍弱 |
| 使用者認證 + 團隊協作 | 完全缺席（session token 是「同 user 證明」非真正多帳號） |
| Python SDK | 無 `pip install codefyui`；plugin CLI 已是雛形但僅供 plugin 管理 |
| GPU 進階管理 | 仍只基本 cpu/cuda 切換，無 VRAM 監控、無多 GPU |
| Step Trace 覆蓋率 | 仍僅約 1/4 節點啟用；RL / Tensor Ops / Diffusion / Classical 多數未 instrument |

---

## 2. 現況分析

### 2.1 技術棧

| 層級 | 技術 | 版本 |
|------|------|------|
| **前端框架** | React | 19.1 |
| **類型系統** | TypeScript | 5.8 |
| **圖形引擎** | React Flow (@xyflow/react) | 12.6 |
| **狀態管理** | Zustand | 5.0 |
| **建構工具** | Vite | 6.3 |
| **數學渲染** | KaTeX + react-katex | 0.16 / 3.1 |
| **自動排版** | @dagrejs/dagre | 3.0 |
| **後端框架** | FastAPI | 0.115+ |
| **ML 框架** | PyTorch + torchvision | 2.0+ / 0.15+ |
| **經典 ML** | scikit-learn | 1.3+（2026-05 新增依賴，支援 Classical 節點） |
| **LLM tokenization** | tiktoken + tokenizers | 0.7+ / 0.15+ |
| **HuggingFace** | datasets + huggingface_hub | 2.14+ / 0.20+ |
| **即時通訊** | WebSocket（原生，含同源檢查 + token query param） | — |
| **數據驗證** | Pydantic + pydantic-settings | 2.0+ |
| **Plugin TOML** | tomllib（3.11+）+ tomli backport（3.10） | — |
| **跨平台路徑** | platformdirs | 4.0+ |
| **環境管理** | uv (Astral) — 自動安裝 Python 3.11 | — |
| **Python** | 3.10+（CI 矩陣 3.10 / 3.11 / 3.12） | — |
| **跨平台執行** | scripts/dev.py + Makefile + cdui launcher + install.sh/.ps1 | — |
| **授權** | AGPL-3.0-only（SPDX, PEP 639）+ 商業授權雙軌 | — |
| **發佈** | GitHub Releases + frontend-dist.tar.gz asset | — |

### 2.2 架構分析

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Frontend (React 19)                          │
│  ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────┐ ┌──────────┐ │
│  │ NodePal-│ │FlowCanvas│ │ConfigPnl │ │ Inspector   │ │ Results  │ │
│  │  ette   │ │ + Smart- │ │ +MathText│ │ Forward /   │ │  Panel   │ │
│  │ +Quick- │ │ DataEdge │ │ (KaTeX)  │ │ Steps /     │ │ +Loss    │ │
│  │ Search  │ │ +Trigger │ │ +Dialog- │ │ Backward    │ │ Chart    │ │
│  └─────────┘ │ +Note    │ │ Container│ └─────────────┘ └──────────┘ │
│              │ +VizNodes│ └──────────┘                               │
│              └──────────┘                                            │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Viz Nodes (11)：AttentionHeatmap / AttentionMask /            │  │
│  │  EmbeddingScatter / TokenizerViz / TextInputViz /              │  │
│  │  EduSelfAttention / EduMultiHeadAttention / EduCrossAttention /│  │
│  │  EduKNN（每個都有對應後端節點 + 互動式 HeatmapModal/Scatter）   │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │          Zustand Stores                                         │  │
│  │  tabStore (nodes/edges/run/segment) | nodeDefStore | uiStore    │  │
│  │  + WS reconnect logic + persistence robustness                  │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                  │ REST API (X-CodefyUI-Token)  │ WS (token query)   │
└──────────────────┼──────────────────────────────┼─────────────────────┘
                   ▼                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          Backend (FastAPI)                           │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  AuthMiddleware (session token + Host whitelist + bootstrap)    │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ routes_nodes | routes_graph | routes_presets | routes_examples │  │
│  │ routes_models | routes_images | routes_custom_nodes            │  │
│  │ routes_execution_state | routes_execution_outputs              │  │
│  │ routes_plugins ★ (install/list/uninstall/reload)                │  │
│  │                       ws_execution                              │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Core Engine                                                    │  │
│  │  NodeRegistry | PresetRegistry | GraphEngine | TypeSystem       │  │
│  │  BaseNode (ABC) | StatefulModuleMixin                           │  │
│  │  ── Security ── ★ NEW                                           │  │
│  │  auth.py            (session token + 0600 file + Host check)    │  │
│  │  plugin_validator   (AST scan: blocked dunders/modules/calls)   │  │
│  │  codegen.py         (whitelist-protected Python export)         │  │
│  │  ── Plugin ── ★ NEW                                              │  │
│  │  plugin_loader      (cdui_plugins.* namespace, lockfile)        │  │
│  │  ── Educational Subsystems ──                                   │  │
│  │  ExecutionContext (verbose/weights_persistent/backward_mode)    │  │
│  │  StepRecorder + step_trace  (A1)                                │  │
│  │  NodeStateStore (LRU 200)   (A2)                                │  │
│  │  backward_pass + grad_health (A3)                               │  │
│  │  RunOutputStore (per-run forward + steps + grads)               │  │
│  │  ExecutionCache (hash-based) | DirtyNodeTracker                 │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Built-in Node Packages (91 nodes, 15 categories)               │  │
│  │  CNN(9) | RNN(3) | Transformer(4) | RL(5) | Data(11)            │  │
│  │  Training(5) | IO(9) | Control(1) | Dataflow(3) | Utility(8)    │  │
│  │  Normalization(4) | TensorOps(11) | LLM(7) | Diffusion(6) ★ NEW │  │
│  │  Classical(5) ★ NEW                                              │  │
│  │  ↑ 15+ 個層級節點以 StatefulModuleMixin 持續化權重                │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Plugin Discovery (NEW)                                         │  │
│  │  <REPO>/plugins/c1..c6/ (built-in chapter packs)                │  │
│  │  <USER_DATA>/plugins/<id>/ (downloaded packs)                   │  │
│  │  <USER_DATA>/plugins/installed.json (lockfile)                  │  │
│  │  cdui plugin install C2 | owner/repo[@ref] | https://...        │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.3 功能清單

> 表格按主題分組；★ = 2026-04 新增、◆ = 自上次報告以來大幅強化、⚒ = **2026-05 新增**。

**圖形編輯與互動**

| 功能 | 狀態 | 備註 |
|------|------|------|
| DAG 圖形編輯器 | 已完成 | React Flow 12，拖放、連線、多選 |
| 類型安全連線 | 已完成 | 12 種 DataType（含 TRIGGER）、相容性矩陣 |
| 節點參數面板 | 已完成 | 7 種 ParamType（int/float/string/bool/select/model_file/image_file）|
| 快速節點搜尋 | 已完成 | 雙擊畫布開啟，支援節點 + Preset |
| 多分頁工作區 | 已完成 | 獨立執行環境、localStorage 持久化 |
| Preset 子圖系統 | 已完成 | 巢狀展開、自動偵測外露埠口、◆ 支援匯入主編輯器工作流 |
| MiniMap / 右鍵選單 | 已完成 | 類別顏色、刪除/複製/重命名 |
| ★ Trigger 入口系統 | 已完成 | Start 節點 + DataType.TRIGGER 取代舊 isEntryPoint flag |
| ★ SmartDataEdge 智慧路由 | 已完成 | hash-based jitter、弧線/skip/step 自動路由 |
| ★ Note Node + 繫結 | 已完成 | 文字 / 影像便利貼可繫結節點 |
| ★ Auto-layout (dagre) | 已完成 | Network-simplex、>2400px 自動換行為多列 grid |
| ★ Empty Canvas Examples | 已完成 | 空畫布顯示分組範例卡片 |
| ⚒ 互動式視覺化節點 | 已完成 | 11 個 viz nodes（AttentionHeatmap/Scatter/EduSelfAttention/...）含 HeatmapModal |
| ⚒ In-app Dialog（替代 window.confirm/prompt）| 已完成 | DialogContainer + Toast，PR #17 |
| ⚒ Settings popover + FontSize menu | 已完成 | "Crafted dark" 工具列 |
| ⚒ WS reconnect + persistence 強化 | 已完成 | PR #8，斷線自動重連 |

**執行引擎**

| 功能 | 狀態 | 備註 |
|------|------|------|
| 拓撲排序執行 | 已完成 | Kahn + 循環偵測 + 並行同層級（asyncio.gather） |
| 節點輸出快取 | 已完成 | hash-based + ExecutionCache |
| 部分重新執行 | 已完成 | Dirty Node Tracking |
| 執行取消 | 已完成 | ExecutionContext._cancel_event |
| 錯誤恢復 | 已完成 | fail_fast / continue / retry |
| 圖表驗證 | 已完成 | 邊類型 + 循環檢查 |
| WebSocket 即時進度 | 已完成 | per-node running/completed/error |
| ★ Educational ExecutionContext | 已完成 | verbose / weights_persistent / backward_mode / auto_backward |
| ★ RunOutputStore | 已完成 | per-run forward + steps + grads，支援大張量切片回退 |
| CLI 圖表執行器 | 已完成 | `run_graph.py` |
| ⚒ AdamW + Rprop PyTorch 2.11 相容 | 已完成 | PR #13 |

**教育與可解釋性**

| 功能 | 狀態 | 備註 |
|------|------|------|
| ★ A1 Verbose Step Trace | 已完成 | 目前約 1/4 節點 instrumented |
| ★ A2 Weight Persistence | 已完成 | 15+ 個層級節點遷移至 StatefulModuleMixin |
| ★ A3 Backward Pass + Grad Health | 已完成 | attach_retain_grad / select_backward_target / capture_grads / vanishing-exploding-healthy 分類 |
| ★ MathText (KaTeX) | 已完成 | 描述 / Step Trace / Tooltip / ConfigPanel |
| ★ Inspector 三分頁 | 已完成 | Forward / Steps / Backward |
| ★ Compare Segment | 已完成 | 選 head/tail 兩節點高亮路徑 + 跨節點比較 |
| ★ TensorInput 節點 | 已完成 | 前端建構張量輸入 |
| ⚒ Attention 互動視覺化 | 已完成 | AttentionHeatmap / AttentionMask / PositionalEncoding 節點 + Modal |
| ⚒ Edu Viz nodes（章節包配套）| 已完成 | EduSelfAttention/MultiHeadAttention/CrossAttention/KNN 等 viz 元件 |

**Plugin 系統（⚒ 2026-05 新增）**

| 功能 | 狀態 | 備註 |
|------|------|------|
| ⚒ `cdui plugin install` | 已完成 | 支援 catalog 短名（C2）、`owner/repo[@ref]`、完整 URL；GitHub tarball 下載 |
| ⚒ `cdui plugin list/uninstall/update/info/search` | 已完成 | lockfile-driven |
| ⚒ 內建章節包 c1-c6 | 已完成 | 6 packs 含 12 個 Edu 節點 |
| ⚒ Plugin 命名空間 | 已完成 | `cdui_plugins.<id>` 同享 NodeRegistry.discover；無需 `uv pip install -e` |
| ⚒ Plugin lockfile | 已完成 | `<USER_DATA>/plugins/installed.json`：source_kind、SHA pin、allowed_modules |
| ⚒ Plugin AST 驗證 | 已完成 | 安裝時對所有 Python 源做 AST 掃描，封鎖 RCE 模式 |
| ⚒ 官方 Plugin 模板 repo | 已完成 | `treeleaves30760/CodefyUI-Plugin-Official` |

**安全 / 部署 / 匯出（⚒ 2026-05 新增）**

| 功能 | 狀態 | 備註 |
|------|------|------|
| ⚒ Session token 認證 | 已完成 | 0600 token file + `X-CodefyUI-Token` header + WS query param |
| ⚒ Host header 白名單 | 已完成 | 防 DNS rebinding |
| ⚒ WS 同源檢查 | 已完成 | `cdui start` 同源放行（PR #12 修 403） |
| ⚒ Codegen 白名單 | 已完成 | dataset / optimizer / loss / layer 型別名比對白名單 |
| ⚒ 可執行 Python 匯出 | 已完成 | 真正可跑（v1 + v2 graph spec），檔名用 graph 名稱 |
| ⚒ no-Node release pipeline | 已完成 | frontend-dist.tar.gz GitHub asset + `cdui install` 自動下載 |
| ⚒ `cdui install --gpu/--dev` 旗標 | 已完成 | auto/cu118-cu128/rocm6.x/cpu/mps/skip + dev extras 選擇 |
| ⚒ Friendly install preflight + PATH auto-patch | 已完成 | PR #7 |
| ⚒ Hot-reload custom node edits | 已完成 | PR #6 |

**檔案 / 資源管理**

| 功能 | 狀態 | 備註 |
|------|------|------|
| 模型權重管理 | 已完成 | REST API 上傳/列表/下載/刪除（.pt/.pth/.safetensors/.ckpt/.bin） |
| MODEL_FILE 參數類型 | 已完成 | 節點下拉選 |
| ★ 影像檔案管理 | 已完成 | REST API + IMAGE_FILE 參數類型 |
| ★ ImageReader / Writer / BatchReader | 已完成 | 對接上傳列表 |
| Custom Node Manager | 已完成 | GUI 管理（含⚒AST 驗證） |
| 匯入/匯出 Graph JSON | 已完成 | 後端儲存 + ⚒ 安全 Path 驗證 |

**範例與內容**

| 功能 | 狀態 | 備註 |
|------|------|------|
| 範例工作流 | 已完成 | **28 個**（含⚒新增 Classical/Diffusion/LLM/RNN/Transformer/RL 分類目錄）全部 CI 驗證 |
| 範例分類 | 已完成 | Classical / Diffusion / LLM / Model_Architecture / Others / RL / RNN / Transformer / Usage_Example |

**安裝與部署**

| 功能 | 狀態 | 備註 |
|------|------|------|
| ★ 跨平台一鍵安裝 | 已完成 | install.sh / install.ps1 |
| ★ cdui launcher | 已完成 | `cdui install/dev/start/stop/test/clean/uninstall/plugin ...` |
| ⚒ `cdui start` production 模式 | 已完成 | 單 uvicorn + 預構 frontend，end users 無需 Node |
| ⚒ `cdui dev` 開發模式 | 已完成 | backend :8000 + Vite HMR :5173（contributors） |
| ⚒ CI matrix | 已完成 | Python 3.10–3.12、三平台、pnpm@9、cdui smoke、release-build draft 修復 |

**其他**

| 功能 | 狀態 | 備註 |
|------|------|------|
| ⚒ Python 腳本匯出 | **已完成（runnable）** | PR #20/#22，graph spec v2 + 白名單保護 |
| i18n 多語言 | 已完成 | en + ⚒ zh-TW 完整覆蓋 91 個內建節點 |
| 深色主題 | 已完成 | 固定深色（"Crafted dark"） |
| ResultsPanel | 已完成 | 分頁 Log/Training + 可調高度 + Loss 曲線 |
| 結構化日誌 | 已完成 | JsonFormatter + 輪轉 |
| ⚒ AGPL-3.0 + 商業雙授權 | 已完成 | SPDX PEP 639 字串、PR #9 / #11 |

### 2.4 內建節點清單 (91 個，15 類別)

> ★ = StatefulModuleMixin（持續化權重） · ✦ = 含 Verbose Step Trace · ⚙ = 2026-04 新增 · ⚛ = 2026-05 LLM 教學 · ⚒ = 2026-05 新增（其他） · ☆ = Diffusion 教學 · ▲ = sklearn-backed Classical

| 類別 | 節點 | 數量 |
|------|------|------|
| **CNN** | Conv1d★✦, Conv2d★✦, ConvTranspose2d★, BatchNorm2d★, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, Dropout, Activation | 9 |
| **RNN** | LSTM★, GRU★, ⚒ RNNCell（vanilla, 教學用） | 3 |
| **Transformer** | MultiHeadAttention★✦, TransformerEncoder★, TransformerDecoder★, ⚒ MoELayer（Top-K Routing） | 4 |
| **Normalization** | BatchNorm1d★, LayerNorm★✦, GroupNorm★, InstanceNorm2d★ | 4 |
| **Utility** | Linear★, Embedding★, Flatten, Concat, Reshape, Print, Visualize, SequentialModel | 8 |
| **Tensor Ops** | Add, MatMul, Mean, Multiply, Permute, Softmax✦, Split, Squeeze, Stack, TensorCreate, Unsqueeze | 11 |
| **Training** | Optimizer, Loss, LRScheduler, TrainingLoop, ⚙ BackwardOnce | 5 |
| **IO** | ImageReader, ⚙ ImageWriter, ⚙ ImageBatchReader, FileReader, ModelLoader, ModelSaver, CheckpointSaver, CheckpointLoader, Inference | 9 |
| **Data** | Dataset, DataLoader, Transform, ⚙ TensorInput, ⚛ TextInput, HuggingFaceDataset, KaggleDataset, **⚒ CSVReader, ⚒ ColumnSelector, ⚒ Normalize, ⚒ TrainTestSplit** | 11 |
| **Control** | ⚙ Start（DataType.TRIGGER 入口） | 1 |
| **Dataflow** | Map, Reduce, Switch | 3 |
| **RL** | DQN, PPO, EnvWrapper, **⚒ RewardModel, ⚒ KLDivergence**（RLHF teaching） | 5 |
| **LLM** | ⚛ Tokenizer, ⚛ WordVector, ⚛ EmbeddingScatter, ⚛ CosineSimilarity, **⚒ AttentionHeatmap, ⚒ AttentionMask, ⚒ PositionalEncoding** | 7 |
| **★ NEW Diffusion** | ☆ GaussianNoise, ☆ TimestepEmbedding, ☆ DDPMSampler, ☆ Lerp, ☆ Upsample, ☆ DiffusionUNet | 6 |
| **★ NEW Classical** | ▲ KNN, ▲ LinearRegression, ▲ LogisticRegression, ▲ DecisionTreeClassifier, ▲ SVMClassifier | 5 |
| **合計** | | **91** |

**Plugin 章節包節點（12 個 Edu 節點，透過 `cdui plugin install c1..c6` 啟用）：**

| 章節 | 節點 | 主題 |
|------|------|------|
| C1 基本資訊 | EduColumnStats | 表格欄位統計、逐步紀錄 |
| C2 經典 AI | EduKNN, EduLinearRegression, EduLogisticRegression | 手寫經典分類器（對應 sklearn-backed Classical 節點） |
| C3 圖像處理 | EduCrossAttention, EduResBlock | Diffusion / U-Net 教學積木 |
| C4 序列處理 | EduTokenEmbedding, EduSelfAttention, EduMultiHeadAttention, EduFFN | Transformer 拆解 |
| C5 強化學習 | EduPolicyGradient | REINFORCE 拆解 softmax→gather→log→baseline→loss |
| C6 前沿技術 | EduPatchify | ViT 的 unfold→permute→flatten |

**節點分布觀察：**
- **CNN / RNN / Transformer / Normalization / Utility** 構成傳統深度學習教學骨幹（28 個）
- **Tensor Ops + Dataflow + Control** 提供圖形編輯所需的低階運算與控制流（15 個）
- **Training + IO** 是「能訓練 + 能存讀」最小生產集合（14 個）
- **Data 從 7 → 11**：本期補上 Tabular IO（CSVReader/ColumnSelector/Normalize/TrainTestSplit），打通 sklearn 場景
- **RL 從 3 → 5**：補上 RewardModel + KLDivergence，可組裝 RLHF 教學流程
- **LLM 從 4 → 7**：本期擴大為 Transformer 內部視覺化（Attention Heatmap / Mask / PositionalEncoding）
- **Diffusion / Classical 從零起步**：本期分別建立教學節點集（Diffusion 6 / Classical 5）
- **Plugin 章節包讓 Edu 節點與內建分離**：built-in 保持精簡（91），課程內容由 plugin 擴充
- **Step Trace 啟用率仍僅約 25%**：CNN 部分 / Transformer / Normalization / Softmax；RNN / RL / Tensor Ops / Diffusion / Classical 多數未 instrument——這是教育能力深化的最直接 KPI

### 2.5 程式碼品質評估

**優點：**
- 後端架構分層清晰：BaseNode ABC → NodeRegistry → GraphEngine 三層；StatefulModuleMixin 為 mixin 不破壞 ABC；plugin_loader 用 `cdui_plugins.<id>` 命名空間注入，零修改 NodeRegistry.discover
- 前端 Zustand per-tab 隔離，新增 segment / inspectorTab / dialog / toast 等 UI 狀態時無侵入式改動
- 類型系統完整（DataType enum 含 12 種）
- 教育子系統與既有引擎正交分離（旗標式開關，零開銷 fallback）
- ⚒ **安全分層明確**：auth 處理身分、plugin_validator 處理可信度、codegen 處理輸出注入——三者互不耦合
- ⚒ **PR-only workflow** 落實：所有改動透過 PR（連 CI 修復都不例外），git 歷史可追溯

**2026-05 完成的重大改善：**

*Plugin 系統（PR #21, 803010f）*
- `app/core/plugin_loader.py` + `app/core/plugin_validator.py` + `app/api/routes_plugins.py` + `scripts/plugins.py`
- 6 個 built-in chapter packs（c1-c6）含 12 個 Edu 節點
- lockfile 機制 + GitHub tarball 下載 + SHA pin
- 官方 plugin 模板 repo

*安全架構（commit 09ca115, 09ca115）*
- `app/core/auth.py`：session token 生成 / 寫入 / middleware
- `app/core/plugin_validator.py`：AST 掃描封鎖危險 imports / dunders / 呼叫
- `app/core/codegen.py`：whitelist-protected Python 匯出
- `routes_plugins.py` 強制 token 驗證

*可執行 Python 匯出（PR #20, #22, d20f862, 402a15f）*
- Graph spec v2（含 nodes / edges 結構）支援 + 舊 v1 fallback
- Layer / dataset / optimizer / loss 名稱白名單
- 檔名與 header 使用 graph 名稱
- 18 個 codegen 測試

*新節點與測試（PR #14-18, 0f2473c）*
- LLM 視覺化：attention/mask/positional（PR #14）
- Diffusion 教學：6 節點（PR #15）
- Classical sklearn + Tabular IO + RNN cell（PR #16）
- RLHF（RewardModel + KLDivergence）+ MoE Layer（PR #18）
- 0f2473c 一次補 60 個節點專屬單元測試

*發佈管線（commit 3e09c58, e68ab98, a841ae6, e84ed04, 65eba15）*
- GitHub Releases `frontend-dist.tar.gz` asset
- pre-merge build check + tag-triggered asset pipeline
- `cdui install --gpu/--dev` 旗標（PR #5）
- 三平台 install-check + softprops duplicate-draft 修復 + gh CLI 取代

*UX 強化（PR #8, #17, 0686fea）*
- WS reconnect + persistence robustness
- In-app dialogs 取代 window.confirm/prompt
- Settings popover + FontSize menu（"Crafted dark"）

**仍待改善：**
- Graph / Run / NodeState 仍使用記憶體 + JSON，重啟即遺失執行歷史與權重
- Step Trace 僅約 1/4 節點啟用（RL/RNN/Tensor Ops/Diffusion/Classical 多數未 instrument）
- backward_pass 已存在但 TrainingLoop 內未自動使用（需手動切換 backward_mode）
- WebSocket 執行串流仍為單實例（教育場景影響小，但阻礙多用戶部署）
- 無 Docker / K8s（教育場景影響小，阻礙課堂雲端部署）
- 無真正多帳號認證（session token 是「同 user 證明」非帳號管理）
- 無 LLM 雲端推理節點（Anthropic / OpenAI / Ollama），Vector DB / RAG / Agent 完全缺席
- 無 MCP（client 與 server 模式皆無）
- 無 Python SDK（`pip install codefyui`）

### 2.6 教育功能 (Educational Features)

> 三大教育核心（A1/A2/A3）合稱「Learn-by-Run」架構：按一次「Run」就能同時看到模型結構、中間張量、權重變化、梯度健康度。本節內容自 2026-04-26 起架構穩定，2026-05 主要進展在「擴大覆蓋」（新增視覺化節點 + chapter pack 教學節點），未對核心子系統做破壞性變更。

#### 2.6.1 A1 — Verbose Step Trace（步驟追蹤）

**架構：**
```
ExecutionContext.verbose=True
   │
   ▼
Node.execute()
   ├─► StepRecorder.record(name, description, scalars={}, **tensors)
   │      e.g. recorder.record("scaled_scores",
   │              "Compute attention: $S = QK^T / \\sqrt{d_k}$",
   │              scalars={"d_k": 64.0}, scores=scores)
   ▼
result = {"output": ..., "__steps__": [Step(...), ...]}
   ▼
GraphEngine 將 __steps__ 展開成 RunOutputStore 鍵
   ▼
Frontend StepTraceView (GET /api/execution/{run_id}/{node_id}/__step_index__)
   ├─ MathText 渲染 LaTeX 描述
```

**已啟用節點：** MultiHeadAttention、Conv2d、LayerNorm、Softmax、BatchNorm 等約 1/4 節點。
**待擴充：** RNN（LSTM/GRU/RNNCell unrolling）、Tensor Ops、RL（DQN/PPO/PolicyGradient）、Diffusion（DDPM forward/reverse）、Classical（KNN voting、Logistic Regression iterations）。

#### 2.6.2 A2 — Weight Persistence（權重持續化）

**架構：**
```
StatefulModuleMixin
   ├─ structural_params: ClassVar[tuple]
   ├─ build_module(params) → nn.Module
   └─ get_or_build_module(context, params)
        ▼ NodeStateStore
            ├─ 索引：(graph_id, node_id, structure_hash) → nn.Module
            ├─ LRU 上限 200 modules
            ├─ 結構參數變更 → sibling 模組全清
            └─ Per-key lock 支援 reentrant autograd
```

**設計亮點：**
- 結構參數改變時自動重建；非結構參數變更不重建
- ExecutionContext 沒給時 fallback fresh build（CLI / 測試環境）
- 與 ExecutionCache 共存：`cacheable=False` 跳過快取避免回傳舊權重

#### 2.6.3 A3 — Backward Pass + Gradient Health

**架構：**
```
ExecutionContext.backward_mode=True
   ├─ zero_module_grads()
   ├─ Forward pass → attach_retain_grad() for non-leaf float tensors
   ├─ select_backward_target() — BackwardOnce > TrainingLoop > auto leaf
   ├─ loss.backward()
   ├─ capture_grads() async — 寫入 {port}__grad / __weight_grad__{name}
   └─ grad_health(): 紅 vanishing / 黃 exploding / 綠 healthy
```

#### 2.6.4 教育功能 vs 既有引擎

| 既有能力 | 教育功能銜接 |
|---------|-------------|
| Dirty Node Tracking | weights_persistent + dirty 節點重跑時權重仍保留 |
| ExecutionCache | StatefulModuleMixin 設 `cacheable=False` 跳過快取 |
| 並行執行 | StepRecorder per-node-local 不衝突；NodeStateStore per-key lock |
| 執行取消 | backward_pass 走 await 可被取消 |
| WebSocket 進度 | 每 step record 可回報「running step N/M」(目前未串接) |

#### 2.6.5 仍未做的（教育能力擴大重要待辦）

1. **更廣的 Step Trace 覆蓋率**：RL / RNN / Diffusion / Classical / Tensor Ops
2. **Gradient 視覺化升級**：grad histogram、跨 step 演進曲線、layer-wise 比較
3. **教學模式 Onboarding**：第一次開啟啟用 verbose + 引導 tour
4. **教師工具**：Step Trace → Markdown / PDF 教材匯出
5. **內建概念解釋庫**：每節點 `concept.md`（LaTeX 推導）
6. **Quiz / 互動挑戰**：載入特定圖、調參使 grad 變 healthy（gamification）
7. **Lesson Capsule**（借 Adobe Project Graph 概念）：bundle graph + 限縮 UI 作為「一堂課」

### 2.7 Plugin 系統 + 安全架構 + 可執行 Python 匯出（★ 2026-05 新增章節）

> 本節記錄 2026-05 三項戰略基礎建設。它們之間的共同主題是：**讓 CodefyUI 從「能跑起來」進化為「能安全地被別人擴充與分發」**。

#### 2.7.1 Plugin 系統

**Why now：** 94 個內建節點已涵蓋 CNN/RNN/Transformer/RL/Diffusion/Classical/LLM 的核心，再往下擴充會撐爆 built-in 集合的可維護性。教育場景的需求是「按章節、按課程、按學派」分離節點集——這就是 plugin 系統的核心目的。

**架構：**
```
<REPO>/plugins/                     ← built-in (first-party) packs
  ├─ registry.json                  ← catalog: id → name / description / kind / chapters
  ├─ c1/  c2/  c3/  c4/  c5/  c6/   ← 6 chapter packs
  │    └─ nodes/                    ← edu_*.py 教學節點
  │
<USER_DATA>/plugins/                ← downloaded (third-party) packs
  ├─ installed.json                 ← lockfile（schema=1）
  └─ <plugin-id>/
       ├─ cdui.plugin.toml          ← manifest（id/name/security.allowed_modules）
       └─ nodes/                    ← *_node.py

sys.modules['cdui_plugins'] = SyntheticNamespacePackage(
    'cdui_plugins.c1' → <REPO>/plugins/c1/nodes/,
    'cdui_plugins.c2' → <REPO>/plugins/c2/nodes/,
    'cdui_plugins.user-plugin-xyz' → <USER_DATA>/plugins/user-plugin-xyz/nodes/,
    ...
)
NodeRegistry.discover([(nodes_dir, package_name), ...])
```

**CLI 介面：**
```bash
cdui plugin install C2                              # built-in 章節包（catalog id）
cdui plugin install foo/bar                         # GitHub 短形式，default branch
cdui plugin install foo/bar@v1.2.3                  # GitHub 標籤版本
cdui plugin install https://github.com/foo/bar      # 完整 URL
cdui plugin list
cdui plugin uninstall <id>
cdui plugin update <id>
cdui plugin info <id>
cdui plugin search <query>
```

**設計亮點：**
- **零 `uv pip install`**：透過 sys.modules namespace 注入，不污染 venv，uninstall 純粹是檔案刪除
- **lockfile-driven**：source_kind（builtin/github/url）、SHA pin、使用者批准的 allowed_modules 都進 installed.json
- **GitHub tarball 下載**：用 codeload endpoint，無需 git clone
- **內建章節包與第三方包同框架**：兩者只有 root path 與是否複製檔案的差異

**官方模板：** `treeleaves30760/CodefyUI-Plugin-Official`

#### 2.7.2 安全架構

**威脅模型：** CodefyUI 預設是 desktop 工具（127.0.0.1 監聽），但瀏覽器是預設 client，這帶來三類典型威脅：

1. **Browser CSRF**：惡意網頁從 attacker.com 對 `http://127.0.0.1:8000/api/...` 發 cross-origin 請求
2. **DNS rebinding**：把 attacker.com 重綁到 127.0.0.1 取得 same-origin
3. **惡意 plugin / 自訂節點**：使用者下載第三方 plugin，內含 `import os; os.system(...)` 或 pickle.loads 等 RCE 模式

**Layer 1：Session token**
- backend process 啟動時用 `secrets.token_urlsafe(32)` 生 token
- 寫入 `<USER_DATA>/codefyui/session.token`，mode 0600（CLI 工具如 `cdui plugin install` 呼叫 `/api/plugins/reload` 可讀回）
- 所有 mutating 請求必須在 `X-CodefyUI-Token` header 帶回；WS 走 query param `?token=...`
- Browser 透過 `GET /api/auth/bootstrap`（read-only，允許）取得 token
- 每次 server 重啟 token 重生——刻意行為，避免遺留 token 殘留

**Layer 2：Host header 白名單**
- 即使 attacker 透過 DNS rebinding 取得 127.0.0.1 socket，`Host` header 仍是 `attacker.com`
- middleware 拒絕非白名單的 Host

**Layer 3：Plugin / Custom Node AST 驗證（`plugin_validator.py`）**
- 安裝 plugin 或上傳自訂節點時，對 Python 源做 AST 掃描
- 封鎖：
  - 危險模組進口：`os / subprocess / shutil / sys / importlib / ctypes / socket / http / urllib / pickle / dill / runpy / multiprocessing / threading / ...`
  - 危險呼叫名：`exec / eval / compile / __import__ / breakpoint / globals / locals / getattr / setattr / delattr / vars / dir`
  - 危險屬性葉節點：`system / popen / spawn* / loads / load / execfile / compile_command`
  - 禁止 dunder：`__class__ / __bases__ / __mro__ / __subclasses__ / __builtins__ / __globals__ / __code__ / __reduce__ / ...`
  - `getattr(<dunder>, ...)` 即使字串字面值也封鎖
- **明確不是沙盒**：堅持要 RCE 的攻擊者仍可繞過，目標是讓 casual / drive-by RCE 變得 non-trivial，並把宣告式（pure declarative）plugin 凸顯為「免額外風險」選項

**Layer 4：Codegen 白名單（`codegen.py`）**
- Python 匯出時，所有插入到生成原始碼的 user-controlled 型別名都比對白名單：
  - Dataset：`MNIST / FashionMNIST / CIFAR10 / ImageFolder / ...`
  - Optimizer：`Adam / AdamW / SGD / Rprop / ...`
  - Loss：`CrossEntropyLoss / MSELoss / KLDivLoss / ...`
  - Layer：`Linear / Conv1d / Conv2d / BatchNorm2d / LayerNorm / Dropout / LSTM / GRU / MultiheadAttention / TransformerEncoderLayer / ...`
- 未列入白名單的型別名改成 `# TODO: unsupported layer type 'XXX'` placeholder
- **防的就是這種攻擊：** graph.json 內 `dataset.name = "MNIST(); __import__('os').system('rm -rf /')"`，下載 .py 後執行就 RCE

#### 2.7.3 可執行 Python 匯出（PR #20 / #22）

**Why：** 「教學原型 → 真實 PyTorch 程式碼」是教育平台的長線價值——學生在畫布上學會了，要能帶走可在 Colab / Kaggle / 自家環境跑的程式碼。先前的匯出只產 skeleton，學生要手動補完，這條路就斷了。

**現在的能力：**
- 支援 graph spec v1（flat layer list）與 v2（nested nodes/edges）
- 自動處理 SequentialModel 的子圖展開
- Activation（ReLU/GELU/Sigmoid/Tanh/LeakyReLU/SiLU/Mish/...）含 inplace flag
- Optimizer / Loss / Dataset 透過白名單呼叫（防注入見 §2.7.2 Layer 4）
- 檔名與檔頭 header 使用 graph 名稱（PR #22）
- 18 個 `test_codegen.py` 涵蓋常見模型架構

**仍未做的：**
- 完整訓練 loop 含 backward / optimizer.step（目前較依賴 TrainingLoop 節點轉譯）
- 多 Optimizer / Multi-task 場景
- Diffusion / RL 場景的訓練 loop 轉譯
- 自訂 plugin 節點的轉譯（僅 built-in 節點有對應 codegen）

---

## 3. 競品分析

> **2026-05 重大變化：** n8n 一年內爆漲到 188k stars（成為視覺化 AI 工作流 stars 第一）、Dify 從 55k → 90k+、Adobe Project Graph（2025-11）正式進場、MCP 成 LLM 工具標準。CodefyUI 在「教育/可解釋性」5 項指標仍全面領先，但在「LLM/RAG/Agent/MCP」場景的差距持續擴大。

### 3.1 ComfyUI — 圖像生成工作流引擎

| 面向 | 詳情 |
|------|------|
| **GitHub Stars** | **~100k**（從 92.5k 持續成長；Comfy-Org/ComfyUI 與 comfyanonymous/ComfyUI 兩 repo 合計影響聲量） |
| **生態** | 150k+ 月雲端用戶、8,500+ 貢獻者、2,500+ 社群節點透過 ComfyUI Manager |
| **核心優勢** | VRAM 智慧管理、Stable Diffusion / SDXL / Flux / 影片 LoRA 訓練 / regional prompting / inpainting / outpainting / style transfer |
| **2026 新發展** | Comfy-Org 推出 Desktop 客戶端、Cloud 服務（ComfyICU、RunComfy） |
| **弱點** | 學習曲線陡峭、macOS 安裝複雜、僅限影像/影片、自訂節點生態品質參差 |

**對 CodefyUI 啟示：** ComfyUI 已經把「節點式圖像生成」做到無可撼動，CodefyUI 不要去碰這塊；但 **ComfyUI Manager 模式**（一鍵裝社群節點 + 工作流分享）是 CodefyUI 已有 Plugin 系統雛形可以借鏡擴大的方向。

### 3.2 Langflow — LLM 視覺化工作流（與 CodefyUI 同技術棧）

| 面向 | 詳情 |
|------|------|
| **GitHub Stars** | **100k+**（PyPI 月下載 210 萬） |
| **技術棧** | Python + FastAPI + React Flow（**與 CodefyUI 幾乎一致**） |
| **2026 新發展** | Langflow 1.9 推出：**Langflow Runtime**（standalone productionize）、雙 Helm Chart（IDE + Runtime）、Langflow Assistant API + streaming、SSE webhook、MCP 整合 |
| **企業支持** | DataStax 收購支持 |
| **核心優勢** | LangGraph 整合、自訂 Python 節點、Kubernetes-ready |
| **弱點** | 無排程 / 重試 / 觀測性；無 ML 模型訓練；無教育性 |

**對 CodefyUI 啟示：** Langflow 已穩固「LLM 視覺化工作流 ✕ Production deployment」這條路；CodefyUI 不該與它直接競爭該場景，而是錯位以「教育 + LLM 學習場景」滲入，等使用者熟悉後再橫向擴展 LLM 應用建構能力。**Runtime / Helm Chart** 模式值得在 Phase 3 借鏡。

### 3.3 Dify — 全方位 LLM 應用平台

| 面向 | 詳情 |
|------|------|
| **GitHub Stars** | **90k+**（從 55k 在一年內成長近 70%） |
| **核心功能** | 知識庫、Prompt 管理、Workflow 編排、Agent、API 發佈、Analytics |
| **部署** | 自建版功能無實質限制（搶占自建市場） |
| **2026 評價** | 「For most teams in 2026, Dify is the best starting point」——知識庫、debug、發佈一條龍 |

**對 CodefyUI 啟示：** Dify 是「LLM 應用建構」的綜合冠軍，已超越 Langflow 的功能廣度；CodefyUI 不可能短期競爭 Dify 的 LLM 應用市場，但 Dify 的 **YAML DSL workflow 分享** 與 **App publishing** 模式值得參考（特別是用於「lesson capsule 分享」場景）。

### 3.4 Flowise — 輕量 LLM 工作流

| 面向 | 詳情 |
|------|------|
| **GitHub Stars** | ~30k（穩定，未隨大盤成長） |
| **定位** | 「The simplicity choice」——文檔檢索 chatbot 最快路徑 |
| **弱點** | 對比 Langflow / Dify 在生產與 LangGraph 落後 |

### 3.5 n8n — 工作流自動化平台（**2026 的最大贏家**）

| 面向 | 詳情 |
|------|------|
| **GitHub Stars** | **188k**（從 40k 在一年內 +112k 暴增，已是 GitHub 全平台 Top 50 專案） |
| **AI Agent** | 70+ native AI nodes、LangChain-based agent builder、memory + tools + guardrails |
| **整合** | 500+ 內建整合（OpenAI/Anthropic/Google/Slack/Salesforce/HubSpot 等） |
| **2026 定位** | 「The leading platform for AI agent workflows」 |
| **部署** | Docker / K8s / n8n Cloud |

**對 CodefyUI 啟示：** n8n 已從「automation 工具」進化為「AI agent workflow 平台」，是這一年 hype 最強的視覺化工具——若不擋住它的氣勢，n8n 會像 Slack 之於企業通訊一樣**吞掉所有視覺化 workflow 場景**。CodefyUI 必須堅守「ML 訓練 + 教育」這兩條 n8n 不會去碰的窄路（n8n 不是 ML 訓練工具）。

### 3.6 Kubeflow Pipelines — 企業級 ML 管線

| 面向 | 詳情 |
|------|------|
| **GitHub Stars** | ~14k（Kubeflow 整體） |
| **特色** | K8s native、容器化每步、實驗追蹤、自動擴縮 |
| **適合** | 大規模 MLOps、生產管線 |

**對 CodefyUI 啟示：** Kubeflow 是 MLOps 重型武器，門檻高、視覺化僅輔助；CodefyUI 的「視覺化優先 + 教育友善」是長線差異化定位。

### 3.7 Airflow / Prefect / ZenML — DAG 排程與 MLOps 框架

| 工具 | GitHub Stars | 特色 |
|------|--------------|------|
| Airflow | ~40k | Python-first、Operator 生態、傳統工作流 |
| Prefect | ~17k | Modern Python、ControlFlow AI 任務、3.0（2024）後更輕量 |
| ZenML | ~5k | 可插拔 Stack、orchestrator-agnostic |

### 3.8 MLflow / W&B — 實驗追蹤

| 工具 | GitHub Stars | 特色 |
|------|--------------|------|
| MLflow | 20k+ | 自建友善、開源廣度；**3.0 (2025-06) 轉型為 unified AI engineering platform**（含 agent / LLM） |
| Weights & Biases | 50k+ | 開發者體驗、自動圖表、5M+ 使用者 |

### 3.9 ⚒ Adobe Project Graph（2025-11 預覽，重大新進場玩家）

| 面向 | 詳情 |
|------|------|
| **發佈** | Adobe MAX 2025（2025-10），2026-Q1 開始 invite-only |
| **定位** | Creative Cloud 節點式 AI 工作流（Firefly + Gemini + OpenAI 整合） |
| **獨特能力** | **Cross-App Integration**（Illustrator vector → Firefly gen → Premiere motion 鏈接） |
| **★ Capsule packaging** | 把複雜 graph 打包成簡化的 panel，讓技術設計師建好「引擎」、其他團員用簡單 UI 操作 |
| **威脅程度** | 對 CodefyUI 直接影響低（不在創意領域），但 **Capsule 概念是教育場景的重要借鏡**——「一個 lesson 就是一個有限參數可調的 graph capsule」 |

### 3.10 ⚒ NodeTool — Local-first AI 工作流（新興競品）

| 面向 | 詳情 |
|------|------|
| **定位** | 開源 local-first AI workspace |
| **支援** | 影像 / 影片 / 音訊 / 文字 / 文檔搜尋 / agent |
| **特色** | 「Every major model from every major provider, called with your own keys」——強調 BYO key + local 執行 |
| **與 CodefyUI 對比** | 兩者都強調 local execution；NodeTool 更通用、更接近 NotebookLM/ComfyUI 的 prosumer 路線，**完全不碰教育/可解釋性** |

### 3.11 ⚒ Model Context Protocol (MCP) — 2026 LLM 工具標準

| 面向 | 詳情 |
|------|------|
| **狀態** | 近乎全面採用：Anthropic / OpenAI / Google / Microsoft 旗艦模型皆原生支援 |
| **規模** | 10,000+ 公開 MCP servers（2026-03）、97M monthly SDK 下載 |
| **治理** | 2025-12 Anthropic 捐贈給 Linux Foundation 新成立的 Agentic AI Foundation（含 Anthropic / OpenAI / Google / Microsoft / AWS / Cloudflare） |
| **架構** | JSON-RPC、3 個基本元素（tools / resources / prompts）、2 個 transport（stdio / HTTP-SSE） |

**對 CodefyUI 的影響：** MCP 已是 LLM ecosystem 的「USB-C」，任何 LLM 整合工作流必走這條路。CodefyUI 的延伸機會：
1. **MCP Client（LLM 節點）**：把 OpenAI / Anthropic / Ollama 客戶端節點直接用 MCP 通訊
2. **MCP Server（最大差異化）**：**把 CodefyUI 的 graph 暴露為 MCP tools**，讓 Claude Desktop / Cursor / IDE 可以呼叫 CodefyUI 訓練好的小模型；這是別人都還沒做的方向

### 3.12 其他值得關注的工具

| 工具 | 定位 | 啟示 |
|------|------|------|
| Gradio | ML 模型 demo | 快速分享、HF 整合 |
| Streamlit | 資料應用 | Python-native 即時預覽 |
| Haystack | NLP/RAG | Pipeline 抽象 |
| Kedro | ML 管線 | 資料目錄、視覺化 |
| Metaflow | ML 工程框架 | Netflix 出品 |
| TF Playground | toy MLP 視覺化 | 啟發但太簡化 |
| Teachable Machine | no-code ML | 教育場景已有但功能有限 |
| Runchat | 視覺化 AI（新） | 與 CodefyUI/Langflow 同類但更年輕 |

### 3.13 市場規模數據（2026-05 update）

| 市場區間 | 2024-2025 規模 | 預估 2033-2035 | CAGR |
|----------|---------------|---------------|------|
| 視覺分析工具 | $150 億 | $600 億 (2033) | 20% |
| 資料管線工具 | $639 億 | $5,145 億 (2034) | 26.8% |
| MLOps | $17-30 億 | $390-890 億 (2034) | 37-40% |
| LLM 市場 | $77.7 億 | $1,498 億 (2035) | 34.4% |
| 多模態 AI | $16 億 | 快速增長 (2034) | 32.7% |
| 低代碼平台 | $287.5 億 | $2,644 億 (2032) | 32.2% |

**2026 LLM 企業支出版圖（Menlo Ventures, 2025-12 / Ramp Apr 2026）：**
- **Anthropic 40%**（一年前 ~10%）
- **OpenAI 27%**（一年前 ~50%）
- Google 21%
- 其他 12%

**關鍵市場趨勢：**
- 2025 GenAI 企業支出 $370 億（vs 2024 $115 億，3.2× 增長）
- 開源 LLM 即將突破 50% 生產環境市佔率
- 76% 技術組織增加開源 AI 工具投資
- 2025 年 70% 新應用使用低代碼/無代碼技術
- **MCP 成 LLM 工具標準（2026）**
- **節點式編輯器在 2026 被多家媒體稱為「Year of the Node-Based Editor」**——Adobe Project Graph、ComfyUI 持續擴張、n8n 暴漲、Langflow 上 100k、NodeTool 興起

---

## 4. Gap Analysis — 差距分析

### 4.1 關鍵差距矩陣（更新版）

| 功能領域 | CodefyUI 現況 | 行業標準 | 差距嚴重度 |
|----------|---------------|----------|-----------|
| **執行引擎** | 並行 + 快取 + dirty + 取消 + 錯誤恢復 | 增量/並行/分散式 | 低（僅剩分散式） |
| **快取系統** | hash-based + dirty tracking | 節點級快取 | 低 |
| **GPU 管理** | cpu/cuda 切換 + install 期 wheel 選擇 | VRAM 管理、多 GPU、自動 | 嚴重 |
| **實驗追蹤** | Step Trace + RunOutputStore（無跨 run UI） | MLflow/W&B | 中 |
| **模型/影像/檔案管理** | 完整 REST API + MODEL_FILE/IMAGE_FILE param | 統一資源管理 | 低（已解決） |
| **LLM 雲端推理** | 無客戶端節點（Anthropic/OpenAI/Ollama） | 全競品都有 | **嚴重** |
| **Vector DB / RAG** | 無 | 全 LLM 競品都有 | **嚴重** |
| **Agent 編排** | 無 | n8n/Langflow/Dify 都有 | **嚴重** |
| **MCP 協定** | 無（client + server 皆無） | 2026 已標準化 | **嚴重** |
| **使用者認證** | ⚒ session token（單機證明） | OAuth2, RBAC | 中（單機足夠，多人不夠） |
| **團隊協作** | 無 | 共享工作區、版本控制、評論 | 高 |
| **容器化部署** | 無 | Docker, K8s, Helm | 高 |
| **跨平台安裝** | install.sh / install.ps1 / cdui / no-Node release pipeline | 一鍵安裝 | **低（已解決，且優於多數競品）** |
| **錯誤處理** | fail_fast/continue/retry | 重試、fallback | 低 |
| **排程執行** | 無 | Cron、Webhook | 中 |
| **節點生態** | ⚒ 91 內建 + 12 plugin Edu + Custom Node Manager + plugin install | 社群市集、ComfyUI Manager | 中（plugin 系統雛形已備，缺遠端市集） |
| **資料庫儲存** | JSON + 記憶體 | PostgreSQL/SQLite | 中 |
| **Run / State 持久層** | 記憶體 | 持久化執行歷史 | 中 |
| **API/SDK** | REST + WebSocket + plugin CLI；無 Python SDK | 完整 API + Python SDK | 中 |
| **監控/可觀測性** | WebSocket + 結構化日誌 + Step Trace | 指標、告警、Dashboard | 中 |
| **測試框架** | **128 後端 + 13 前端 + 18 codegen + 12 plugin** | 節點單元、管線整合 | **低（已大幅領先同類）** |
| **文件** | README 雙語 + docs/ + plugin template repo | 文件站、教學、API 文件 | 中 |
| **Diffusion 模型節點** | ⚒ 6 個教學節點（DDPM/UNet/...） | ComfyUI 級生產 | 視定位（教學足夠） |
| **Classical ML 節點** | ⚒ 5 個 sklearn-backed | scikit-learn / KNIME | 低（已解決基本場景） |
| **安全架構** | ⚒ session token + Host whitelist + AST + codegen 白名單 | 視為基線；ComfyUI 也有 | 低 |
| **Python 匯出可執行** | ⚒ 已可執行（v1 + v2 spec） | n8n / Langflow 有 export | 低 |
| **★ 教育與可解釋性** | Step Trace + Backward Health + Weight Persistence + LaTeX + Compare Segment | **無競品提供** | **CodefyUI 全面領先** |
| **★ Lesson Capsule / 教學打包** | 無（Plugin 是接近的雛形但定位不同） | Adobe Project Graph 有 Capsule | 中（高機會差異化） |
| **★ MCP Server 模式** | 無 | 無競品有「graph as MCP tools」 | 高（差異化機會） |

### 4.2 最關鍵的 5 個差距（重新排序）

按「市場需求 × 競爭壓力 × 投入產出比」更新：

1. **MCP 整合（client + server）** — MCP 已是 2026 LLM 工具標準；競品都還沒做「graph as MCP tools」，這是 CodefyUI 「教育模型 → 真實 LLM workflow tool」的天然延伸，且**首發者紅利大**。
2. **LLM / Vector DB / RAG 節點** — Langflow/Dify/n8n 全有；應從 Anthropic Claude 客戶端起步（享 40% 企業支出紅利），再 OpenAI / Ollama / Chroma / FAISS。
3. **教育場景的擴大化** — 教育能力仍是領先項目，但 Step Trace 仍僅 25% 覆蓋率、無 Onboarding、無教師工具（教材匯出）、無 Lesson Capsule。**這是把「領先 → 護城河」的關鍵一步**，無競品壓力。
4. **容器化 + 部署 + Run 持久層** — 無 Dockerfile 阻礙所有共享部署場景；RunOutputStore 重啟即失阻礙教學分享；SQLite + Dockerfile 是 Phase 0 必做。
5. **社群節點 Marketplace + Python SDK** — Plugin 系統已備，但缺遠端可瀏覽 catalog 與 `pip install codefyui` 程式化 API；ComfyUI Manager 模式證明生態系才能撐長尾節點需求。

### 4.3 已從關鍵差距清單移除的項目（2026-05 解決）

- ~~Plugin / 節點擴充框架~~ — PR #21 解決
- ~~安全裸奔（CSRF / DNS rebinding / RCE plugin）~~ — commit 09ca115 解決
- ~~Python 匯出不可執行~~ — PR #20 / #22 解決
- ~~end users 需要裝 Node~~ — frontend-dist 發佈管線解決
- ~~Window.confirm / prompt 體驗差~~ — PR #17 解決
- ~~測試框架不足~~ — 0f2473c 60 個節點測試 + plugin 測試
- ~~i18n 未完成~~ — PR #23 起累積完成 94/94 個內建節點 zh-TW
- ~~AGPL 授權未正式化~~ — PR #9 完成

### 4.4 反向觀察：CodefyUI 已建立的領先優勢（2026-05 update）

| 領先項目 | CodefyUI 做了什麼 | 競品狀況 |
|---------|-----------------|---------|
| 步驟追蹤 (Step Trace) | 節點 instrumented + LaTeX 描述 | 全無 |
| 反向梯度健康度 | 自動 retain_grad + grad_health 分類 | 全無（W&B grad histogram 非節點級即時） |
| 權重持續化 + 互動訓練 | StatefulModuleMixin + NodeStateStore | 全無 |
| LaTeX 公式渲染 | KaTeX 整合到節點描述 / Step Trace / Tooltip | 全無 |
| Compare Segment | 高亮路徑 + 跨節點輸出比較 | 全無 |
| Smart 邊路由 | hash-based jitter + 弧線/skip/step | 競品用 React Flow 預設邊 |
| 多入口 Trigger 系統 | Start 節點 + 綠色觸發邊 | n8n 有觸發器但非節點 |
| Note Node + 繫結 | 可固定附在某節點的便利貼 | 部分工具有 sticky note 但無繫結 |
| 跨平台一鍵安裝 | uv + cdui + no-Node 發佈管線 | ComfyUI 安裝困難；Langflow Docker 為主 |
| ⚒ **Plugin 系統 + AST 驗證** | catalog + GitHub install + lockfile + 安裝期 RCE 阻擋 | ComfyUI Manager 有 install 無安全；n8n marketplace 是 cloud 才有 |
| ⚒ **Codegen 白名單** | 防 graph.json 注入的 Python 匯出 | n8n/Langflow 匯出無此考量 |
| ⚒ **Session token + Host 白名單** | desktop 模式正確的安全層 | ComfyUI 為 dev 模式裸奔 |
| ⚒ **互動式視覺化節點** | AttentionHeatmap / EmbeddingScatter / EduKNN 等 11 個 viz node | LLM 工具有 chat 預覽無模型內部視覺 |

---

## 5. 目標定位與願景

### 5.1 建議定位（沿用 2026-04 確立的版本）

> **CodefyUI：可視化 ML 學習與實驗平台 — 看得見每一步張量、梯度、權重變化**
>
> 從教學示範到模型實驗的一站式視覺化工具：每按一次「Run」，使用者就能同時看見模型結構、中間張量、權重變化、梯度健康度——這是市場上沒有任何工具提供的能力。
>
> **2026-05 加註：** 隨著 Plugin 系統與安全架構就位，定位向「教育平台（含 marketplace + 安全擴充）」自然延伸。長期願景的「LLM 應用建構」場景應透過 **MCP server 模式**（CodefyUI graph as tools）切入，避開與 Langflow / Dify / n8n 的正面競爭。

### 5.2 為什麼以「教育/可解釋性」為主軸是正確選擇

1. **市場真空（仍未變）**：ComfyUI / Langflow / Kubeflow / MLflow / W&B / n8n / Dify / NodeTool / Adobe Project Graph 都沒有節點級 step trace、grad health、weight persistence。
2. **競品難複製（仍未變）**：要做這個，需要：BaseNode 抽象 + StatefulModuleMixin + ExecutionContext 旗標式控制 + retain_grad 自動化 + 前端 Inspector 三分頁 + LaTeX 渲染——所有競品都得重大重構才能追上。
3. **市場需求真實（仍未變）**：5,000 萬 AI/ML 學習者、大學 AI 課程急需互動工具、企業內部 AI Literacy 培訓需求。
4. **清晰的成長路徑**：教育用戶 →（ML 工程師背景者）→ 實驗 / 研究 →（少數團隊）→ 商業；類似 Figma 路徑。
5. **AGPL-3.0 + 商業雙授權**：個人 / 教育 / 研究免費，閉源 / SaaS / OEM 走商業授權。

### 5.3 三層次的定位演進策略（小幅微調）

```
┌──────────────────────────────────────────────────────────────────┐
│  Layer 3: 全流程 ML 平台（長期，2-3 年）                          │
│  + 團隊協作 + K8s + Marketplace + Python SDK + MLflow 整合       │
│  Target: 中小型 ML 團隊 / Edu/Research Lab                       │
├──────────────────────────────────────────────────────────────────┤
│  Layer 2: 視覺化 AI 應用建構（中期，6-12 月）                     │
│  + LLM/Embedding/VectorDB/RAG/Agent + **MCP server/client**      │
│  + **Lesson Capsule 打包**                                       │
│  Target: AI 應用開發者 / Prototype 工作坊 / 進階課程作者          │
├──────────────────────────────────────────────────────────────────┤
│  Layer 1: 互動式 ML 學習平台（短期，0-6 月） ← **目前主戰場**     │
│  + 擴大 Step Trace + 教學模式 + 教師工具 + Concept 庫            │
│  + Plugin 章節包擴大（c7-c10）                                   │
│  Target: ML 學生 / 教師 / 自學者 / 課程作者                      │
└──────────────────────────────────────────────────────────────────┘
```

### 5.4 差異化策略

| 競品 | 其限制 | CodefyUI 的機會 |
|------|--------|----------------|
| ComfyUI | 僅圖像生成；無教育性 | 通用 AI/ML + 教育獨特性 |
| Langflow/Flowise | 僅 LLM/RAG；無模型訓練；無教育性 | 全流程 + 教育獨特性 |
| Dify | LLM 應用為主；無模型訓練；無教育性 | **錯位以教育切入**，後期透過 MCP 雙向互通 |
| n8n | 通用自動化；ML 訓練不是強項；無教育性 | **錯位以 ML 訓練 + 教育切入** |
| Kubeflow | K8s 門檻高；視覺化僅輔助 | 視覺化優先 + 漸進複雜度 |
| Airflow | 非視覺化優先 | 拖放式 + 即時可觀察 |
| MLflow/W&B | 無管線建構；無節點級 trace | 內建 Step Trace / Grad Health |
| TF Playground | toy MLP | 真實 PyTorch + 同樣直覺 |
| Jupyter Notebook | 程式碼為主 | 零程式碼 + 立即視覺化 |
| ONNX Tutorials / Netron | 僅結構 | 結構 + 執行 + 梯度 |
| **Adobe Project Graph** | 創意領域；商業閉源 | **教育領域的 Capsule equivalent**（lesson packaging） |
| **NodeTool** | 通用 AI prosumer；無教育性 | 教育獨特性 |

### 5.5 目標使用者（小幅更新）

| 使用者層級 | 描述 | 需求 | CodefyUI 現況契合度 |
|-----------|------|------|-------------------|
| ★ **AI/ML 學生** | 大學/研究所、自學者、轉職者 | 視覺化內部、實驗、分享學習成果 | **高**（A1-A3 + LaTeX + Compare Segment） |
| ★ **AI/ML 教師** | 大學教授、線上課程作者、Bootcamp 講師 | 演示、出題、教材、批改 | **中高**（Plugin 章節包是教材分發雛形；缺 Quiz / 教材匯出） |
| **研究者** | 學術、企業 R&D | prototype、實驗比較、論文草圖 | **中-高** |
| **ML 工程師** | 日常模型開發 | 訓練管線、實驗追蹤、模型比較 | 中（缺 MLflow 整合、GPU 進階） |
| **AI 應用開發者** | LLM 應用 | RAG、Agent、API、**MCP 整合** | **低**（無 LLM/VectorDB/Agent 節點；無 MCP） |
| **MLOps 工程師** | 管線自動化 | 排程、監控、部署、CI/CD | **低**（無 K8s/容器/排程） |
| **團隊/企業** | 協作 | 共享、權限、版本 | **低** |

### 5.6 推薦的「First 1000 User」聚焦策略（沿用）

**主要鎖定：「ML 學習 + 教學」相關使用者群**

- **獲取管道：** GitHub Awesome Lists、Reddit r/MachineLearning、HN、知乎/小紅書、大學 PyTorch 課程教案
- **內容行銷：** 30 個短影片「視覺化解釋 X 概念」
- **教師合作：** 找 2-3 位線上課程作者試用
- **學生競賽：** 「最佳 CodefyUI 教學圖」Kaggle 風格競賽
- **⚒ 新管道：Plugin 章節包社群分享**——鼓勵教師把自己的課程做成 chapter pack，貢獻到官方 catalog

**驗證指標：**
- MAU 3 個月內達 500+
- 每用戶 sessions/月 ≥ 4
- GitHub stars 增長率（過去 90 天比較）
- 社群貢獻範例工作流 ≥ 30 個
- **⚒ 社群貢獻 plugin packs ≥ 5 個**

---

## 6. 建議發展路線圖

> **基於 §5 的「教育為主軸」策略，2026-05 update**：原 Phase 0 的 Plugin / 安全 / Python 匯出 / no-Node 發佈管線全數完成，剩 SQLite + Dockerfile + Run 持久層。Phase 1 拆為 1A（教育擴大）、1B（LLM + MCP），並新增 1C（Lesson Capsule + Marketplace 雛形）。

### Phase 0：基礎強化 — **預估 1-2 週**（從 2-3 週減半，大半已完成）

| 項目 | 說明 | 優先級 |
|------|------|--------|
| ~~執行引擎快取~~ | | ~~P0~~ 已完成 |
| ~~真正的執行取消~~ | | ~~P0~~ 已完成 |
| ~~錯誤處理增強~~ | | ~~P0~~ 已完成 |
| ~~並行節點執行~~ | | ~~P1~~ 已完成 |
| ~~跨平台安裝器~~ | | ~~P1~~ 已完成 |
| ~~影像/檔案管理 API~~ | | ~~P1~~ 已完成 |
| ~~完整測試套件~~ | | ~~P1~~ 已完成（128 後端 + 13 前端） |
| ~~日誌系統~~ | | ~~P2~~ 已完成 |
| ~~Plugin 系統~~ | | ~~P0~~ 已完成（PR #21） |
| ~~安全架構（session/Host/AST/codegen 白名單）~~ | | ~~P0~~ 已完成（commit 09ca115） |
| ~~Python 匯出可執行~~ | | ~~P0~~ 已完成（PR #20/#22） |
| ~~no-Node 發佈管線~~ | | ~~P0~~ 已完成 |
| **SQLite 持久層** | **替換 JSON 存儲（Graph、Run、State、設定）** | **P0** |
| **Dockerfile + docker-compose** | **前後端 + 可選 DB** | **P1** |
| **RunOutputStore 持久層** | **學生重連可看舊 step trace** | **P1** |
| **NodeStateStore 磁碟快照** | **教師可發布「訓練到 N 步」checkpoint** | **P2** |

### Phase 1：教育擴大 + LLM/MCP 第一波 + Capsule 雛形 — 預估 6-10 週

**1A. 教育擴大（首要，戰略護城河）**

| 項目 | 說明 | 優先級 |
|------|------|--------|
| Step Trace 全面覆蓋 | RNN unrolling / RL policy update / Tensor Ops / Diffusion / Classical | P0 |
| Backward Visualization 升級 | grad histogram、跨 step 演進、layer-wise 比較 | P0 |
| 教學模式 Onboarding | 第一次開啟自動 verbose + 5 步引導 | P0 |
| 內建概念解釋庫 | 每節點 `concept.md`（LaTeX 推導） | P0 |
| 教師工具：教材匯出 | Step Trace + Inspector → Markdown / PDF | P1 |
| Quiz / 互動挑戰 | 調參使 grad 變 healthy（gamification） | P1 |
| TrainingLoop 整合 backward_pass | 訓練中即時 grad health monitoring | P1 |
| ⚒ Plugin 章節包擴大 | c7-c10（時序模型 / Vision 進階 / RAG 教學 / Eval 指標） | P1 |
| GPU 智慧管理 | 自動裝置、VRAM 監控、模型卸載 | P1 |

**1B. LLM + MCP 第一波（並行）**

| 項目 | 說明 | 優先級 |
|------|------|--------|
| **MCP Server 模式** | **CodefyUI graph 暴露為 MCP tools，可被 Claude Desktop/Cursor 呼叫** | **P0**（首發者紅利） |
| **Anthropic Claude 節點** | **企業 40% 支出，第一選擇** | **P0** |
| OpenAI / Ollama 節點 | 補完主流 LLM client | P1 |
| Embedding 節點 | OpenAI / SentenceTransformers / local | P1 |
| Vector DB 節點 | ChromaDB / FAISS（local 先） | P1 |
| RAG 管線節點 | DocumentLoader / TextSplitter / Retriever / PromptTemplate | P1 |
| MLflow 整合節點 | 實驗追蹤、模型登錄 | P2 |
| HuggingFace 模型節點 | 下載 / 推理 / 微調 | P2 |
| Prompt Engineering 節點 | Template / Chain / OutputParser | P2 |
| 資料視覺化增強 | 互動式圖表（訓練曲線/混淆矩陣/ROC/特徵重要性） | P2 |

**1C. Lesson Capsule + Marketplace 雛形（新增）**

| 項目 | 說明 | 優先級 |
|------|------|--------|
| **Lesson Capsule 格式** | **bundle graph + 限縮可調參數 + 預期輸出 + 評分 criteria** | **P0**（教師工具雛形） |
| Capsule 載入器 | "Open as Lesson" 模式：唯讀畫布、僅露允許參數、自動驗收 | P0 |
| Plugin Catalog 遠端瀏覽 | `cdui plugin search` 已有但需 catalog index 改進 | P1 |
| Workflow Marketplace 雛形 | examples 目錄 + plugin catalog 整合，提供 GitHub-based 瀏覽 UI | P1 |
| Capsule 評分 + 提交 | 學生提交 → JSON 比對標準答案 | P2 |

### Phase 2：社群生態 + 專業化功能 — 預估 8-10 週

| 項目 | 說明 | 優先級 |
|------|------|--------|
| Python SDK | `pip install codefyui` + Python API 操作 | P0 |
| Workflow Marketplace 完整版 | catalog + 評分 + 安裝統計 | P0 |
| 使用者認證（基本） | OAuth2/JWT，個人帳號（先不做 RBAC） | P1 |
| 工作區與專案 | 多專案、專案級設定 | P1 |
| 工作流版本控制 | Git-like 版本、diff、分支 | P1 |
| 排程執行 | Cron、Webhook | P1 |
| AI Agent 節點 | Agent 編排、Tool、Memory、Planning | P1 |
| 模型部署節點 | ONNX 匯出、FastAPI endpoint 生成 | P2 |
| 資料集管理 | 版本、標註、探索 | P2 |
| Python 腳本匯出（進階） | 訓練 loop / 多 optimizer / 自訂 plugin 節點 | P2 |
| Diffusion 模型節點（生產） | SD/SDXL/Flux（若定位包含） | P2（視策略） |

### Phase 3：企業級與雲端 — 預估 10-12 週

| 項目 | 說明 | 優先級 |
|------|------|--------|
| 團隊協作 | 即時編輯、評論、審核 | P0 |
| RBAC 權限系統 | 角色、專案、節點權限 | P0 |
| Kubernetes 部署 | Helm Chart、分散式執行 | P1 |
| 監控 Dashboard | 執行歷史、資源使用、告警 | P1 |
| API Gateway | 工作流發佈為 REST API | P1 |
| 教育版 SaaS | 學校 / Bootcamp 托管環境 | P1 |
| 審計日誌 | 操作記錄、合規 | P2 |
| SSO 整合 | SAML / LDAP | P2 |
| 多租戶 | 組織隔離 | P2 |
| 外掛系統（前端） | 前端 plugin API（後端已有） | P2 |

### 6.5 路線圖視覺化（Gantt 概念）

```
Phase 0  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ (1-2 週) ── DB + Docker + Run 持久層
Phase 1A ░░██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ (6-8 週) ── 教育擴大化（首要）
Phase 1B ░░░░░░██████░░░░░░░░░░░░░░░░░░░░░░░░░░ (6-8 週並行) ── LLM/MCP 第一波
Phase 1C ░░░░░░██████░░░░░░░░░░░░░░░░░░░░░░░░░░ (4-6 週並行) ── Lesson Capsule + Marketplace 雛形
Phase 2  ░░░░░░░░░░░░░░██████████░░░░░░░░░░░░░░ (8-10 週) ── 社群生態 + 專業化
Phase 3  ░░░░░░░░░░░░░░░░░░░░░░░░██████████████ (10-12 週) ── 企業級
                          ▲
                          └─ Phase 1 結束後重新評估市場反應
```

---

## 7. 技術架構建議

### 7.1 執行引擎現況（已升級）

```
┌──────────────────────────────────────────────────────────┐
│              Execution Engine (Current)                  │
│                                                          │
│  ┌───────────┐  ┌───────────┐  ┌─────────────────────┐  │
│  │  Graph    │  │ Execution │  │  Topological Levels │  │
│  │  Compiler │→│   Cache   │→│  + asyncio.gather   │  │
│  │ + Cycle   │  │  (hash-   │  │  (parallel/level)   │  │
│  │  Detect   │  │   based)  │  │                     │  │
│  └───────────┘  └───────────┘  └──────────┬──────────┘  │
│                                            │             │
│                       ┌────────────────────┼──────────┐  │
│                       ▼                    ▼          │  │
│                 ┌──────────┐       ┌──────────────┐   │  │
│                 │ BaseNode │       │  Stateful-   │   │  │
│                 │ execute()│       │  ModuleMixin │   │  │
│                 │  (async) │       │  + NodeState │   │  │
│                 │          │       │     Store    │   │  │
│                 └──────────┘       └──────────────┘   │  │
│                                                       │  │
│  ┌────────────────────────────────────────────────────┘  │
│  │  ExecutionContext                                     │
│  │  ├─ _cancel_event (cancellation)                      │
│  │  ├─ verbose       → StepRecorder → __steps__          │
│  │  ├─ weights_persistent → NodeStateStore               │
│  │  ├─ backward_mode → grad_targets, capture_grads       │
│  │  ├─ DirtyNodeTracker (incremental re-run)             │
│  │  └─ ExecutionCache (hash-based)                       │
│  ├──────────────────────────────────────────────────────  │
│  │  Progress Reporter (WebSocket + token query)          │
│  │  RunOutputStore (per-run forward/steps/grads)         │
│  │  ⚒ AuthMiddleware (session token + Host whitelist)    │
│  │  ⚒ PluginLoader  (cdui_plugins.* namespace + lockfile)│
│  │  ⚒ PluginValidator (AST scan, install-time + upload)  │
│  │  ⚒ Codegen (whitelist-protected Python export)        │
│  └──────────────────────────────────────────────────────  │
│                                                          │
│  ✓ Hash-based 變更偵測                                    │
│  ✓ 並行執行同層級                                          │
│  ✓ 可中斷                                                 │
│  ✓ 錯誤恢復                                               │
│  ✓ Verbose Step Trace                                    │
│  ✓ 權重持續化                                             │
│  ✓ 自動梯度健康度                                          │
│  ✓ Plugin discovery 同框架（無 venv 污染）                 │
│  ✓ 安全層完整（session/Host/AST/codegen）                  │
│  ☐ GPU Worker Pool / VRAM 排程（待做）                    │
│  ☐ 執行歷史 DB 持久化（仍記憶體）                          │
│  ☐ 分散式執行（Celery / Ray）（長期）                     │
│  ☐ MCP Server 模式（graph as tools）（戰略待做）          │
└──────────────────────────────────────────────────────────┘
```

### 7.2 節點系統現況（已升級）

```python
class BaseNode(ABC):
    NODE_NAME: ClassVar[str] = ""
    CATEGORY: ClassVar[str] = ""
    DESCRIPTION: ClassVar[str] = ""
    cacheable: ClassVar[bool] = True

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]: ...
    @classmethod
    def define_outputs(cls) -> list[PortDefinition]: ...
    @classmethod
    def define_params(cls) -> list[ParamDefinition]: ...

    async def execute(self, inputs, params, *, context=None, progress_callback=None):
        """✓ async + ExecutionContext 已實作"""
        ...

class StatefulModuleMixin:
    structural_params: ClassVar[tuple[str, ...]]
    cacheable: ClassVar[bool] = False

    def build_module(self, params) -> nn.Module: ...
    def get_or_build_module(self, context, params): ...
```

**仍可改進：**
- `validate_params(cls, params) -> list[str]` 介面未存在；驗證散在 execute()
- `compute_hash` 由 ExecutionCache 統一，未開放節點自訂
- 無 `metadata()` 介面回傳 LaTeX 描述 / concept 連結（教育擴大化後可考慮）
- 無 `mcp_tool_spec()` 介面（若走 MCP server 模式可考慮）

### 7.3 前端架構現況與待辦

| 現況 | 建議 | 理由 |
|------|------|------|
| ~~Inline styles → CSS Modules~~ | ~~已完成~~ | |
| ~~window.prompt/alert → Dialog~~ | ~~已完成（PR #17）~~ | |
| 無路由 | React Router | 多頁面（Editor/Dashboard/Settings/Tutorial/Marketplace） |
| localStorage | IndexedDB + 後端 DB | 大型圖、可靠性、跨裝置 |
| 固定深色主題 | 主題系統 | 教育場景投影機需淺色 |
| 無快捷鍵系統 | 完整框架 | 教師演講必備 |
| ~~LaTeX 渲染~~ | ~~已完成~~ | |
| ~~Auto-layout~~ | ~~已完成~~ | |
| ~~Smart Edge~~ | ~~已完成~~ | |
| ~~Note 系統~~ | ~~已完成~~ | |
| Inspector 跨節點比較 | 跨 run / 跨 segment dashboard | Compare Segment 已就位 |
| 教學模式 Onboarding | 第一次開啟引導 | 教育定位 |
| ⚒ Lesson Capsule mode | 唯讀畫布 + 露允許參數 + 自動評分 | Phase 1C 戰略項目 |
| ⚒ Marketplace UI | Plugin catalog 瀏覽 + 一鍵裝 + 評分 | Phase 1C-2 |

### 7.4 後端架構現況與待辦

| 現況 | 建議 | 理由 |
|------|------|------|
| JSON 檔案儲存 | SQLAlchemy + SQLite → PostgreSQL | 查詢、並發、完整性 |
| RunOutputStore 記憶體 | SQLite + 大張量 → Local Disk / Object Storage | 教學分享 |
| NodeStateStore LRU | 磁碟快照（save/restore） | 教師發 checkpoint |
| ⚒ session token | + OAuth2/JWT（多帳號階段） | Phase 2 |
| ~~print() 日誌~~ | ~~結構化日誌~~ | ~~已完成~~ |
| 無任務隊列 | ARQ / Celery + Redis | 長訓練、排程 |
| 無 API 版本管理 | API v1 前綴 | SDK 出來後向後相容 |
| 無設定管理 | 分層設定（env / file / DB） | 部署靈活性 |
| WebSocket 單實例 | Redis pub/sub 抽象 | 水平擴展（Phase 3） |
| ⚒ Plugin loader 已備 | 加 cloud catalog API + signature 驗證 | Marketplace 階段 |
| ⚒ MCP Server adapter | 新增 module，把 graph 暴露為 MCP tools | Phase 1B 戰略 |

### 7.5 教育引擎子系統

```
ExecutionContext (per-run shared state)
   ├─ verbose, weights_persistent, backward_mode, auto_backward
   │
   ├─► A1: StepRecorder (per-node-call)
   │      └─► result["__steps__"] = list[Step]
   │             └─► RunOutputStore writes "__step__N__{tensor_name}"
   │                    └─► Frontend StepTraceView (lazy fetch)
   │
   ├─► A2: NodeStateStore (singleton, LRU 200)
   │      ├─► Index: (graph_id, node_id, structure_hash) → nn.Module
   │      ├─► Per-key lock for reentrant autograd
   │      └─► Sibling eviction on structural param change
   │
   └─► A3: backward_pass module
          ├─► attach_retain_grad(result, ctx)
          ├─► select_backward_target(ctx)
          ├─► run_backward(loss)
          └─► capture_grads(ctx) → grad_health
```

**Phase 1 升級建議：**

| 升級 | 說明 |
|------|------|
| StepRecorder.tag(category) | 步驟分類（"forward", "shape_change", "numerical_check"），前端可篩選 |
| StepRecorder.diff_with_previous() | 自動計算與前一步張量差異 |
| GradHistogram | 不只 norm/mean/max，回傳完整 bucket 分布 |
| Per-step WebSocket push | 即時推 step trace 給前端 |
| Concept registry | `nodes/cnn/conv2d_node.py` 旁 `concept.md`，API 提供 |
| Snapshot API | `POST /api/runs/{run_id}/snapshot` 打包 NodeStateStore + RunOutputStore |
| Capsule API | `POST /api/capsules/from-graph` 把 graph + 限縮參數打包為 lesson |

### 7.6 整體架構演進

```
Phase 0-1 (簡單部署):
  Browser → Nginx → Frontend (React, prebuilt dist)
                  → Backend (FastAPI + SQLite + session token)
                  → Local file storage (uploads/, models/, images/, plugins/)
                  → (Optional) Redis 任務隊列

Phase 2 (專業部署):
  Browser → Nginx/Traefik
          → Frontend (React, CDN)
          → API Server (FastAPI, 多實例 + OAuth2)
          → MCP Server adapter (graph as tools)
          → Worker Pool (GPU nodes)
          → PostgreSQL
          → Redis
          → Object Storage (S3/MinIO — 模型/資料/Run snapshots/Plugin packs)

Phase 3 (企業部署):
  Browser → Load Balancer
          → API Gateway (認證/限流)
          → Frontend (CDN)
          → API Cluster (K8s)
          → GPU Worker Cluster (K8s)
          → PostgreSQL (HA)
          → Redis Cluster (含 WS pub/sub)
          → Object Storage
          → Prometheus/Grafana
          → ELK/Loki
          → MCP Server cluster (multi-tenant tools)
```

---

## 8. SWOT 分析（★ 2026-05 新增章節）

### 8.1 Strengths（優勢）

| 優勢 | 說明 | 證據 |
|------|------|------|
| **教育/可解釋性護城河** | Step Trace + Backward Health + Weight Persistence + LaTeX + Compare Segment 五項，全競品都沒有 | §4.4 |
| **完整 Python ML 全鏈** | 91 個內建節點橫跨 CNN/RNN/Transformer/RL/Diffusion/Classical/LLM 教學/Tabular IO，相對 Langflow 純 LLM、ComfyUI 純圖像生成更全面 | §2.4 |
| **Plugin 系統 + 安全** | catalog + GitHub install + AST 驗證 + lockfile + 12 個內建 Edu 節點；ComfyUI Manager 有 install 但無安全層 | §2.7 |
| **真正可執行的 Python 匯出** | 含 codegen 白名單防注入；學生可帶程式碼到 Colab/Kaggle | §2.7.3 |
| **跨平台一鍵安裝（含 no-Node）** | uv + cdui + frontend-dist asset，end users 連 Node 都不用裝 | §2.3 |
| **PR-only workflow + CI 完整** | 三平台 install-check + Python 3.10-3.12 矩陣 + 128 後端測試 + 19 個 codegen 測試 | §2.5 |
| **AGPL-3.0 + 商業雙授權** | 適合教育免費 + 企業付費，類似 Mongo / Elastic 商業模式 | §5.2 |
| **i18n 完成度** | en + zh-TW 完整覆蓋 91 節點，繁中市場無對手 | §2.3 |
| **單一技術棧（與 Langflow 同）** | React + FastAPI + React Flow，使用者可遷移過來，社群貢獻者熟悉 | §3.2 |

### 8.2 Weaknesses（弱點）

| 弱點 | 影響 | 緩解（已在路線圖中） |
|------|------|---------------------|
| **無 LLM 雲端推理節點** | 失去所有 AI 應用建構場景；Anthropic 40% 企業 LLM 支出紅利沒接到 | Phase 1B P0 |
| **無 Vector DB / RAG / Agent** | RAG 場景完全不能做；對比 Langflow/Dify 落差大 | Phase 1B P1 |
| **無 MCP（client + server）** | 2026 已標準化，未來 LLM 整合都走這條 | Phase 1B P0（server 是差異化機會） |
| **無 Docker / K8s** | 教師無法在課堂雲端部署、企業無法 onboard | Phase 0 P1 |
| **無資料庫持久化** | Run / NodeState / 設定重啟即遺失，阻礙教學分享 | Phase 0 P0 |
| **Step Trace 僅 25% 覆蓋** | 教育護城河只發揮 1/4 戰力 | Phase 1A P0 |
| **無真多帳號認證** | session token 是「同 user 證明」非帳號管理 | Phase 2 |
| **無 Python SDK** | 程式化整合阻塞；CI/外部 trigger 沒對接管道 | Phase 2 P0 |
| **無遠端 plugin marketplace UI** | Plugin 系統有但 catalog 僅 GitHub 短形式，缺評分/分類/推薦 | Phase 1C-2 |
| **Bus factor: 單一維護者** | AGPL + 商業授權雖增加買單可能，但社群信心仍需貢獻者多元化 | 開展 CONTRIBUTING / governance |
| **GitHub stars 起點低** | 對比 n8n 188k / Langflow 100k+，可見度需大量推廣 | 內容行銷 + Awesome Lists |
| **無互動式訓練曲線跨 run 比較** | W&B 等已成標準；CodefyUI 只有單 run loss chart | Phase 1A 後期 |
| **WebSocket 單實例** | 阻礙水平擴展 | Phase 3 P2 |

### 8.3 Opportunities（機會）

| 機會 | 描述 | 接法 |
|------|------|------|
| **「Year of the Node-Based Editor」(2026)** | Adobe Project Graph 進場 + n8n 暴漲 + ComfyUI 持續擴張，節點式 UX 成主流敘事 | 借此趨勢做品牌定位；Awesome Lists / Trending 內容行銷 |
| **MCP server「graph as tools」首發者紅利** | MCP 已標準化但無人把「視覺化建構的 ML pipeline」做成 MCP tool 提供 | Phase 1B P0；可被 Claude Code / Cursor / Continue 直接調用——這是 LLM 工程師也會用 CodefyUI 的橋梁 |
| **Anthropic 企業 40% 支出** | Claude API 是企業首選；CodefyUI 若有第一流 Claude 節點 + Step Trace 顯示 token 流，會吸引 Anthropic 生態使用者 | Phase 1B P0；同時與 Anthropic Cookbook 例子聯動 |
| **Adobe Project Graph 的 Capsule 概念** | 把複雜 graph 打包成簡化 panel；對教育場景就是「lesson capsule」 | Phase 1C P0 |
| **AI/ML 學習者 5,000 萬+ 規模** | Coursera / Kaggle Learn / HuggingFace 持續成長 | 教育擴大化路線（Onboarding / Concept 庫 / Quiz） |
| **大學課程急需互動工具** | TF Playground 太陽春、Colab 仍偏程式碼 | Plugin 章節包讓教授可寫課件 |
| **MLflow 3.0 (2025-06) 轉向 LLM/agent unified** | 證明 ML/LLM 分割越來越模糊；CodefyUI「橫跨教學 + LLM」定位逆勢正確 | 不放棄傳統 ML 教學 + 補 LLM 節點 |
| **開源 LLM 即將 50%+ 生產市佔** | Ollama / vLLM / LM Studio 全勢頭強；本地 LLM 教學需求成長 | Ollama 節點 + 本地 LLM trace 範例 |
| **MCP 10k+ 公開 servers + 97M 月下載** | 生態系即將爆發 | 同時做 MCP client 與 server，雙向受益 |
| **教育版 SaaS** | 學校 / Bootcamp 託管教育環境（避免 IT 管轄） | Phase 3 P1，借鏡 Replit/Gradescope |
| **plugin marketplace 模式** | 教育圈本來就有教材分發習慣（PowerPoint / Slido / Kahoot），plugin 章節包是天然對應 | Phase 1C-2 |
| **企業教育市場** | 企業 AI Literacy 培訓需求暴增（Claude/ChatGPT 普及推動） | 商業授權目標客戶 |

### 8.4 Threats（威脅）

| 威脅 | 描述 | 緩解 |
|------|------|------|
| **n8n 188k stars 巨人壓境** | 一年 +112k stars，將吞噬所有「視覺化 workflow」品類 | 守住「ML 訓練 + 教育」窄門；n8n 不會去碰 nn.Module 訓練 |
| **Langflow 1.9 + Runtime + Helm** | 生產化能力快速追上、Helm 雙 Chart 進企業 | 不正面競爭 LLM 應用市場，走教育切入 |
| **Dify 90k+ 知識庫 + Agent** | 「LLM 應用平台」最強通解 | 同上錯位 |
| **Adobe Project Graph 進場** | 企業背書 + Creative Cloud 通路 | 創意領域不重疊，借 Capsule 概念為己用 |
| **NodeTool 興起** | 同走 local-first，定位類似 | 教育獨特性是 NodeTool 不會碰的 |
| **ComfyUI 持續擴張 + 2,500 社群節點** | 圖像生成穩固外溢到通用 | 視覺化 ML 教學是 ComfyUI 不會做的 |
| **官方 IDE 內建視覺化** | Claude Code / Cursor / Continue 等未來可能內建 simple graph builder | 與其競爭 in-IDE 簡易版，提供 MCP server 讓他們調用 CodefyUI |
| **AGPL 對企業招待性偏弱** | 部分企業避開 AGPL（GPL 病毒擴散） | 商業授權路徑 + 為企業準備 OEM | 
| **單一維護者 bus factor** | Issue 累積、PR review 延遲、社群信心 | 開放 contributor 進入、寫 CONTRIBUTING、code owners |
| **Step Trace 25% 覆蓋率不夠戲劇性** | 教師示範時容易卡到「這節點還沒 trace」 | Phase 1A P0 全面 instrument |
| **MCP 採用速度比 CodefyUI 補 LLM 節點快** | 標準會比競品實作先到位，可能讓 first-mover 變灰色 | 立即啟動 MCP server，不只 client |
| **教育用戶付費意願低** | 個人學生不付錢，靠商業授權與企業培訓 | Phase 3 教育 SaaS 給機構，不向學生收費 |
| **PR-only 流程偶有摩擦** | 修小東西也要走 PR，速度受 CI 限制 | 已是有意識的選擇，換來可審計性 |

### 8.5 SWOT 戰略象限（TOWS）

| | Opportunities | Threats |
|--|---------------|---------|
| **Strengths** | **SO 加碼策略**：用教育護城河 + Plugin 系統 → 推 Lesson Capsule 模式 + MCP server graph-as-tools；趁「Year of Node Editor」趨勢做品牌敘事 | **ST 防禦策略**：守住「ML 訓練 + 教育」窄門；n8n/Langflow/Dify 攻不進這塊；用安全架構建立 enterprise trust |
| **Weaknesses** | **WO 補位策略**：補 LLM 節點時優先 Anthropic（享 40% 紅利）+ MCP server（搶首發紅利）；Phase 0 Docker 完成後申請進 Awesome Lists | **WT 緊縮策略**：不去碰 RAG / Agent 與 n8n/Dify 正面對打；Python SDK 與 Docker 是必補不可的最小防線 |

---

## 9. 風險評估

### 9.1 技術風險

| 風險 | 影響 | 可能性 | 緩解策略 |
|------|------|--------|---------|
| Step Trace 全覆蓋的維護成本 | 中 | 中 | 每節點獨立 PR、保留「未 trace」fallback |
| GPU 管理跨平台問題 | 中 | 高 | 已有 install 期 wheel 選擇；runtime 切換待加抽象層 |
| 社群 plugin 安全性 | 高 | 中 | AST 驗證已備；簽章 / 沙盒待 Phase 2-3 |
| 前端效能（大型圖） | 中 | 中 | React Flow 虛擬化、lazy rendering |
| WebSocket 擴縮 | 中 | 中 | Redis pub/sub（Phase 3） |
| MCP server 規格演進 | 中 | 中 | 跟緊 Anthropic / Linux Foundation 動態；用 SDK 而非手刻 |

### 9.2 產品風險

| 風險 | 影響 | 可能性 | 緩解策略 |
|------|------|--------|---------|
| 定位過廣 | 高 | 中 | 已聚焦教育為 P0；LLM 為 P1 補位；不去碰 Agent 編排 |
| 與 ComfyUI 直接競爭 | 低 | 低 | 領域錯位（ML/教育 vs 圖像生成） |
| 與 n8n 直接競爭 | 低 | 低 | 領域錯位（ML 訓練 vs 通用自動化） |
| 社群建設困難 | 高 | 中 | Plugin 模板 repo + 教師合作 + 內容行銷 |
| 企業功能分散注意力 | 中 | 中 | Phase 3 才開始；Layer 1-2 先做 |

### 9.3 市場/商業風險

| 風險 | 影響 | 可能性 | 緩解策略 |
|------|------|--------|---------|
| AGPL 阻擋企業採用 | 中 | 中 | 商業授權 + OEM 通道 |
| 教育市場付費難 | 高 | 高 | 教育 SaaS 給機構（學校 / Bootcamp / 公司培訓），不向個人學生收費 |
| Anthropic / OpenAI 推官方視覺化 | 高 | 低-中 | MCP server 模式讓 CodefyUI 成為他們的延伸（被調用而非競爭） |
| Bus factor（單維護者） | 高 | 中 | 開放 contributor + governance 文件 + 模組化責任 |

### 9.4 建議的初期聚焦場景

考慮資源限制，建議先聚焦 **2 個場景**：

1. **互動式 ML 學習** — 教育護城河最深、無競品壓力
2. **MCP server（graph as tools）+ Anthropic Claude 節點** — 首發者紅利大、與 LLM 生態互通而非競爭

---

## 10. 延伸與改進方向（★ 2026-05 新增章節）

> 本節為「下一步」具體可執行的延伸方向，按優先順序與分類組織。每項皆對應 §6 路線圖中的某個 Phase。

### 10.1 短期（0-3 月）：把領先做深

**A. 教育能力擴大化（最高 ROI，無競品壓力）**

| 行動 | 預期成果 | KPI |
|------|---------|-----|
| Step Trace 全面覆蓋（RNN / RL / Tensor Ops / Diffusion / Classical） | 任何教學情境都能展示 step-by-step | Step Trace 啟用率 25% → 80%+ |
| 內建 `concept.md` 庫 | 每節點有 LaTeX 推導 + 為何重要 + 常見坑 | 100% 內建節點覆蓋 |
| 教學模式 Onboarding | 第一次開啟自動啟用 verbose + 5 步引導 tour | 新使用者完成 tour 率 ≥ 50% |
| 教師工具：Step Trace → Markdown / PDF 匯出 | 老師可直接用 export 成講義 | 1 個試用教師 + 5 篇匯出範例 |
| Quiz / Gamification | 「調參數使 grad 變 healthy」、「補上缺失節點使模型 train 收斂」 | 5-10 個內建 quiz |
| Backward Visualization 升級 | grad histogram、跨 step 演進、layer-wise 比較 | 視覺化元件上線 |
| Plugin 章節包擴大 c7-c10 | 時序模型 / Vision 進階 / RAG 教學 / Eval 指標 | 4 個新章節包 |

**B. 基礎設施補完**

| 行動 | 預期成果 |
|------|---------|
| SQLite 持久層（取代 JSON）| Run / NodeState / 設定不再重啟丟失 |
| RunOutputStore 落地（SQLite + 大張量 → 磁碟）| 學生重連可看舊 step trace |
| Dockerfile + docker-compose | 教師可一鍵在課堂雲端部署 |
| NodeStateStore 磁碟快照 | 教師可發布「訓練到 N 步」checkpoint |

### 10.2 中期（3-9 月）：橋接到 LLM 生態 + 首發 MCP

**C. LLM 第一波（補位）**

| 行動 | 預期成果 |
|------|---------|
| **Anthropic Claude 客戶端節點**（優先）| 享企業 40% 支出紅利；附 Step Trace 顯示 token usage / system prompt / tool calls |
| OpenAI / Ollama 節點 | 主流 LLM 客戶端補完 |
| Embedding 節點（OpenAI / SentenceTransformers / local） | RAG 前置 |
| Vector DB 節點（ChromaDB / FAISS）| RAG 核心 |
| RAG 管線節點（DocumentLoader / TextSplitter / Retriever / PromptTemplate） | 一條 RAG 教學流程可在畫布上拖出 |
| PromptEngineering 節點（Template / Chain / OutputParser）| Prompt 教學場景 |

**D. MCP 整合（差異化機會）**

| 行動 | 預期成果 |
|------|---------|
| **MCP Server 模式**（首發者紅利）| 把 CodefyUI graph 暴露為 MCP tools，讓 Claude Desktop / Cursor / Continue 等可直接呼叫 |
| MCP Client 節點 | 在 CodefyUI 內呼叫其他 MCP 公開 server |
| 與 MCP 公開 catalog 整合 | 讓 CodefyUI plugin marketplace 與 MCP 生態互通 |

**E. Lesson Capsule（教育平台關鍵差異化）**

| 行動 | 預期成果 |
|------|---------|
| Capsule 格式：bundle graph + 可調參數 + 預期輸出 + 評分 criteria | 教師可發「lesson」如同 PowerPoint slide |
| Capsule 載入模式：唯讀畫布 + 露允許參數 + 即時驗收 | 學生開啟即進入做題模式 |
| Capsule 內建範例（從 28 個 examples 抽 5-10 個轉換）| Day-one 內容 |
| Capsule 提交 + JSON 比對標準答案 | 簡易自動評分（不需 backend 服務） |
| Capsule 分享連結（git-friendly）| 教師發 GitHub gist 或 plugin pack 即可分享 |

### 10.3 長期（9-24 月）：社群生態 + 企業

**F. 社群與生態**

| 行動 | 預期成果 |
|------|---------|
| Python SDK（`pip install codefyui`）| 程式化操作；CI / 外部 trigger 入口 |
| Workflow Marketplace UI | catalog 瀏覽 + 一鍵裝 + 評分 + 安裝統計 |
| Plugin signature / 簽章 | 進一步降低 plugin 安全風險 |
| Plugin sandbox（subprocess + IPC）| AST 驗證之上的執行時隔離 |
| Discord / 論壇 | 教師、學生、教材作者社群 |
| Educator Partnership Program | 與 5-10 位線上 ML 課程作者合作試用 |
| Open governance | CONTRIBUTING.md + code owners + RFC 流程 |

**G. 企業 / 雲端**

| 行動 | 預期成果 |
|------|---------|
| 多帳號認證（OAuth2 / JWT） | 多人共用部署 |
| RBAC + 多租戶 | 學校 / 公司獨立空間 |
| Kubernetes Helm Chart | 企業部署 |
| 教育版 SaaS | 學校 / Bootcamp 託管 |
| 監控 Dashboard（Prometheus / Grafana） | 企業運維 |
| SSO（SAML / LDAP） | 企業 IT 整合 |

### 10.4 探索（可選 / 高風險）

| 方向 | 描述 | 風險 |
|------|------|------|
| **Multi-modal 教學節點** | Whisper 語音節點 + 多模態 RAG 教學 | 領域擴大可能稀釋教育聚焦 |
| **AutoML 教學** | 視覺化 hyperparam search 與 trial 比較 | 與 Optuna / W&B 重疊；定位需精確 |
| **AI 助教（in-app LLM）** | Claude / OpenAI 在 ConfigPanel 旁回答「這節點是什麼」、「為何 grad vanishing」| 需處理離線降級、Token 成本 |
| **與 Jupyter 互通** | `.ipynb` 匯入 / 匯出；Cell ↔ Node 雙向 | 規格複雜，使用者基礎不一定重疊 |
| **CodefyUI as MCP Tools Server**（Phase 1B 已列） | 把訓練好的小模型暴露為 MCP tool 給 LLM 調用 | 首發者紅利；技術門檻中等 |
| **「Recipe」分享平台** | 類似 Kaggle Notebook 的工作流分享平台（Cloud） | 與 GitHub Gists / examples 模式比較需評估 |
| **與 CodefyUI-OJ 整合**（見 memory）| 學生在 OJ 系統作答時用 CodefyUI 出題 / 解題 | 同一團隊兩個產品互通，雙方加值 |

### 10.5 「下一個 PR」推薦清單（如果要從今天起選 3 個動）

1. **Anthropic Claude 客戶端節點 + MCP Server adapter scaffold**——一次完成 LLM 補位 + MCP 首發 server。先寫 `nodes/llm/claude_client_node.py` 與 `core/mcp_server.py`（minimum: tools list 暴露 + 一個 graph endpoint）。
2. **Step Trace 全面覆蓋 RNN + Tensor Ops**——把護城河從 25% 推到 50%+。一個 PR 一個類別，~1 週可完成 RNN+Tensor Ops 兩類。
3. **Dockerfile + docker-compose**——解鎖所有共享部署場景；同時為未來教育 SaaS 鋪路。

---

## 11. 附錄：競品功能對照表

> 重新組織為「核心 / 教育可解釋性 / LLM AI 應用 / 部署企業 / 生態」五大群組。★ = CodefyUI 領先項目。

### 11.1 核心圖形編輯與執行

| 功能 | CodefyUI | ComfyUI | n8n | Langflow | Dify | Kubeflow | Airflow | NodeTool |
|------|----------|---------|-----|----------|------|----------|---------|----------|
| 視覺化圖形編輯 | V | V | V | V | V | 部分 | 部分 | V |
| 節點拖放 | V | V | V | V | V | X | X | V |
| 類型安全連線（含 TRIGGER） | V | V | V | V | V | N/A | N/A | 部分 |
| 增量執行 / 節點快取 | V (hash + dirty) | V | X | X | X | X | X | 部分 |
| 並行執行同層級 | V | V | 部分 | 部分 | 部分 | V | V | 部分 |
| 錯誤處理 / 重試 | V (3 模式) | 部分 | V | 部分 | V | V | V | 部分 |
| 執行取消 | V | V | V | V | V | V | V | V |
| WebSocket 即時進度 | V | V | X | V | V | X | X | V |
| 多分頁工作區 | V | X | X | X | X | X | X | X |
| Preset / 子圖 | V（含匯入主編輯器） | V | V | V | V | V | X | V |
| 自訂節點 + GUI 管理 | V（Custom Node Manager） | V (CLI) | V | V | V | V | V | V |
| ⚒ Plugin 系統 + lockfile | V（cdui plugin install） | V（Manager） | V（cloud） | V（pip） | V | V | V | V |
| ⚒ Plugin 安裝期 AST 驗證 | **V** | X | N/A | X | X | X | X | X |

### 11.2 ★ 教育與可解釋性（CodefyUI 全面領先）

| 功能 | CodefyUI | ComfyUI | n8n | Langflow | Dify | Kubeflow | Airflow | W&B / MLflow | NodeTool |
|------|----------|---------|-----|----------|------|----------|---------|--------------|----------|
| ★ Verbose Step Trace | V | X | X | X | X | X | X | X | X |
| ★ Backward Pass 即時 | V | X | X | X | X | X | X | X | X |
| ★ Gradient Health 分類 | V | X | X | X | X | X | X | 部分（W&B） | X |
| ★ Weight Persistence（互動訓練） | V | X | X | X | X | X | X | X | X |
| ★ LaTeX 公式渲染 | V | X | X | X | X | X | X | X | X |
| ★ Compare Segment | V | X | X | X | X | X | X | X | X |
| 節點概念說明 / LaTeX | V + LaTeX | 部分 | 部分 | 部分 | 部分 | X | 部分 | N/A | X |
| 互動式訓練曲線 | 部分（Loss） | X | X | X | X | V | X | V | X |
| 跨 Run Dashboard | X | X | X | X | X | V | X | V | X |
| ⚒ 互動式視覺化節點（Attention/Embedding） | **V**（11 個 viz） | X | X | 部分（chat） | 部分 | X | X | 部分（trace） | X |
| ⚒ Lesson Capsule（教學打包）| X（計畫 Phase 1C） | X | X | X | X | X | X | X | X |

### 11.3 LLM / AI 應用

| 功能 | CodefyUI | ComfyUI | n8n | Langflow | Dify | NodeTool |
|------|----------|---------|-----|----------|------|----------|
| LLM 推理節點（OpenAI/Anthropic/HF） | X（教學節點有） | 社群 | V（500+ 整合） | V | V | V |
| 本地 LLM（Ollama/llama.cpp） | X | 社群 | V | V | V | V |
| Embedding 節點 | X | X | V | V | V | V |
| Vector DB 節點 | X | X | V | V | V | V |
| RAG 管線（Loader/Splitter/Retriever） | X | X | V | V | V | V |
| AI Agent 編排 | X | X | V（70+ AI nodes） | V | V | V |
| **MCP 協定（client）** | X | X | V | V | X | 部分 |
| **MCP 協定（server）** | X | X | X | X | X | X |
| Diffusion 模型節點 | 教學（6 個） | V（生產） | X | X | X | V |
| 模型微調節點 | 部分 | X | X | 部分 | 部分 | 部分 |
| Tokenization / Attention 視覺 | **V**（7 個 LLM 節點） | X | X | X | X | X |

### 11.4 部署與企業

| 功能 | CodefyUI | ComfyUI | n8n | Langflow | Dify | Kubeflow | Airflow |
|------|----------|---------|-----|----------|------|----------|---------|
| ★ 跨平台一鍵安裝（含 no-Node） | **V**（cdui + uv + frontend-dist asset） | 部分 | V | V | V | X | X |
| Docker 部署 | X | V | V | V | V | V | V |
| K8s 部署 | X | 社群 | V | V (1.9 Helm) | V | V | V |
| GPU 管理 | 基本（install 期 wheel） | 進階（VRAM 自動） | X | X | X | V | X |
| ⚒ 安全架構（CSRF/DNS rebinding/RCE 防護） | **V**（session token + Host + AST + codegen） | 基本 | V | V | V | V | V |
| 使用者認證 | ⚒ session token | X | V | V | V | V | V |
| RBAC / 團隊協作 | X | X | V | X | V | V | V |
| 排程執行（Cron） | X | X | V | X | X | V | V |
| 模型部署（API endpoint） | X | X | X | V (Runtime) | 部分 | V | X |
| 監控 Dashboard | 基本（ResultsPanel） | X | V | 部分 | V | V | V |
| ⚒ 可執行 Python 匯出（含白名單） | **V** | X | V（JSON） | V | V | V | V |

### 11.5 生態與開發

| 功能 | CodefyUI | ComfyUI | n8n | Langflow | Dify | Kubeflow | Airflow | NodeTool |
|------|----------|---------|-----|----------|------|----------|---------|----------|
| 內建節點數 | **91** | ~50 核心 | 500+ | ~100 | ~80 | ~30 | 100+ Operator | ~200 |
| 社群節點數 | ⚒ 12（plugin packs c1-c6） | 2,500+ | cloud | V | V | 部分 | V | V |
| Plugin install CLI | **V** (`cdui plugin install`) | V（Manager） | cloud-only | V (pip) | V | V | V | V |
| Python SDK / `pip install` | X | X | X | V | V | V | V | X |
| REST API | V | V | V | V | V | V | V | V |
| 資料版本控制 | X | X | X | X | X | V | X | X |
| 實驗追蹤整合（MLflow/W&B） | X | X | X | X | X | V | X | X |
| 範例工作流 | **28**（CI 驗證 + 分類目錄） | 2.5M（社群） | 數百 | 數十 | 數十 | 數十 | 數十 | 數十 |
| i18n 多語言 | **V**（en + zh-TW 完整） | 社群 | V | V | V | X | V | X |
| 後端測試覆蓋 | **128 test files** | 部分 | V | V | V | V | V | 部分 |

### 11.6 授權與規模

| 項目 | CodefyUI | ComfyUI | n8n | Langflow | Dify | Flowise | Kubeflow | Airflow | NodeTool |
|------|----------|---------|-----|----------|------|---------|----------|---------|----------|
| 開源授權 | AGPL-3.0 + 商業 | GPL-3 | Fair-Code | MIT | Apache-2 | MIT | Apache-2 | Apache-2 | AGPL-3.0 |
| **GitHub Stars (2026-05)** | — | **~100k** | **188k** | **100k+** | **90k+** | ~30k | 14k | 40k | ~10k |
| 主要差異化 | **教育 + 可解釋性 + 安全擴充** | 圖像生成 | 自動化整合 + AI agent | LLM/RAG | LLM 應用平台 | 簡易 LLM | K8s MLOps | DAG 排程 | Local-first AI |

### 11.7 競品對照觀察（2026-05 update）

1. **CodefyUI 在「教育/可解釋性」5+1 項仍全面領先** —— 加上互動式視覺化節點與 Lesson Capsule（計畫中），這是真正的差異化。
2. **n8n 是 2026 最大威脅** —— 188k stars + 70+ AI agent nodes，視覺化 workflow 通用品類已被吞下。CodefyUI 必須堅守「ML 訓練 + 教育」窄門。
3. **Langflow / Dify 走自己的路** —— 兩者都已成熟到「LLM 應用」場景；CodefyUI 不可能正面打。MCP server 是與它們互通的橋梁。
4. **Adobe Project Graph + NodeTool 進場** —— 證明節點式 UX 是 2026 主流敘事；CodefyUI 借此趨勢做品牌，並借 Capsule 概念為己用。
5. **MCP 是 2026 標準** —— 任何 LLM 工具不做 MCP 就會落伍；MCP server graph-as-tools 是 CodefyUI 唯一可以與 LLM 巨頭互通而不被吞掉的策略。
6. **節點數中位但品質優** —— 91 個 + 12 plugin Edu 比 ComfyUI 核心（~50）多、比 n8n（500+）少，但每節點有 type safety + 教育性 + 測試覆蓋。
7. **CI 驗證範例 + 完整測試** —— 28 個範例全 CI 驗證、128 個後端測試、19 個 codegen 測試，品質遠超 ComfyUI 社群分享。
8. **Python SDK 仍是缺口** —— Langflow / Dify / Kubeflow / Airflow 都有 `pip install`，CodefyUI 沒有會擋下程式化整合與 CI 對接。
9. **安全層已超越多數開源競品** —— session token + Host + AST + codegen 白名單，這個四層安全在開源視覺化工具中是少見的完整度。
10. **i18n（en + zh-TW 全節點）是繁中市場的無對手優勢** —— 競品多僅英文，繁中是教育場景的關鍵。

---

## 總結

CodefyUI 自上次報告（2026-04-26）至本次更新（2026-05-17）的三週內完成了**戰略基礎建設的最後一塊拼圖**：Plugin 系統讓教學內容可由社群擴充、安全架構讓 desktop 模式不再裸奔、可執行 Python 匯出讓「教學原型 → 真實程式碼」這條路通了、no-Node 發佈管線讓 end users 連 Node 都不需要裝。同期內節點從 69 → 91（新增 Diffusion 與 Classical 兩大類別），後端測試從 24 → 128。

過去三週的 35 次 commit 帶來四組高密度進化：

**(1) Plugin 系統與安全架構（戰略基礎）**
- `cdui plugin install` 支援 catalog 短名 / GitHub 短形式 / 完整 URL
- 6 個內建章節包（c1-c6）含 12 個 Edu 教學節點
- 四層安全（session token、Host whitelist、AST 驗證、codegen 白名單）
- 官方 plugin 模板 repo

**(2) 節點集大幅擴張**
- Diffusion 教學節點集（6 個）
- Classical sklearn-backed 節點集（5 個）
- LLM 視覺化擴大（+3：AttentionHeatmap / AttentionMask / PositionalEncoding）
- Data 補 Tabular IO（+4：CSVReader / ColumnSelector / Normalize / TrainTestSplit）
- RNN cell、Transformer MoE、RL RLHF（RewardModel / KLDivergence）
- 60 個節點補上專屬單元測試（commit 0f2473c）

**(3) 部署與發佈**
- no-Node release pipeline（frontend-dist GitHub asset）
- `cdui install --gpu/--dev` 旗標
- 三平台 install-check + Python 3.10-3.12 CI 矩陣
- AGPL-3.0 + 商業雙授權正式化

**(4) UX 與可執行 Python 匯出**
- In-app dialogs（PR #17 取代 window.confirm/prompt）
- WS reconnect + persistence robustness（PR #8）
- Python 匯出 v2（PR #20/#22）：runnable + graph 名稱命名 + 白名單防注入

### 戰略路徑建議（重申並補強）

> **核心邏輯：先以「教育獨特性」建立護城河與使用者基數，再以 MCP server 模式橋接 LLM 生態，最後橫向擴展 MLOps/企業能力。**

| Phase | 重點 | 期間 | 主要動作 |
|-------|------|------|---------|
| **Phase 0** | DB 持久層 + Docker | 1-2 週 | SQLite + Dockerfile + RunOutputStore 持久化 |
| **Phase 1A** | 教育擴大化 | 6-8 週 | Step Trace 全覆蓋 + Backward 升級 + Onboarding + 教師工具 |
| **Phase 1B** | LLM + MCP 第一波（並行） | 6-8 週 | **MCP Server** + Anthropic Claude + Vector DB + RAG |
| **Phase 1C** | Lesson Capsule + Marketplace 雛形 | 4-6 週並行 | Capsule 格式 + 載入器 + plugin catalog UI |
| **Phase 2** | 社群生態 + 專業化 | 8-10 週 | Python SDK + Marketplace 完整版 + 認證 |
| **Phase 3** | 企業 + 雲端 | 10-12 週 | RBAC + K8s + 教育版 SaaS |

### 本次更新 vs 上次的關鍵轉變

| 議題 | 上次（2026-04-26） | 本次（2026-05-17） |
|------|---------------------|---------------------|
| 戰略主軸 | 教育/可解釋性為主軸 | **不變**（教育仍是護城河） |
| Phase 0 內容 | Plugin / 安全 / Docker / 持久層 | **大幅縮減**（前三項完成，僅剩 DB + Docker + Run 持久層） |
| Phase 1 內容 | 1A 教育 + 1B LLM | **新增 1C** Lesson Capsule + Marketplace 雛形 |
| MCP 提及 | 列為市場趨勢 | **升級為 P0 戰略行動**（MCP Server 首發者紅利） |
| 競品威脅排序 | Langflow 同技術棧威脅最大 | **n8n 188k 暴漲為最大威脅**，但領域錯位仍能守住 |
| LLM 節點優先 | OpenAI / HuggingFace / Ollama | **Anthropic Claude 第一**（享 40% 企業支出紅利） |
| SWOT 分析 | 無 | **新增完整 SWOT + TOWS 戰略象限** |
| 延伸方向章節 | 散落在路線圖 | **集中為 §10 含「下一個 PR 推薦」** |

### 一句話結論

> CodefyUI 過去三週把「能讓別人安全地擴充自己」（Plugin + 安全 + 可執行匯出 + no-Node 發佈）這四件事做完之後，**已從「能用的教育工具」躍升為「可以分發、可以託管、可以被 LLM 工程師當作 MCP tool 調用」的視覺化 ML 平台基礎**。下一步的關鍵不是寫更多節點，而是**把教育護城河擴到 80%+ 覆蓋率、首發 MCP server 模式、再以 Lesson Capsule 把教師變成內容生產者**。

---

> **附：本次更新涉及的核心檔案**
>
> *Plugin / 安全 / 匯出（戰略基礎）：*
> - `backend/app/core/plugin_loader.py`、`plugin_validator.py`、`auth.py`、`codegen.py`
> - `backend/app/api/routes_plugins.py`
> - `scripts/plugins.py`
> - `plugins/registry.json` + `plugins/c1..c6/`
>
> *新增節點：*
> - LLM：`attention_heatmap_node.py`、`attention_mask_node.py`、`positional_encoding_node.py`
> - Diffusion：`gaussian_noise_node.py`、`timestep_embedding_node.py`、`ddpm_sampler_node.py`、`lerp_node.py`、`upsample_node.py`、`diffusion_unet_node.py`
> - Classical：`knn_node.py`、`linear_regression_node.py`、`logistic_regression_node.py`、`decision_tree_classifier_node.py`、`svm_classifier_node.py`
> - Data：`csv_reader_node.py`、`column_selector_node.py`、`normalize_node.py`、`train_test_split_node.py`
> - RL：`reward_model_node.py`、`kl_divergence_node.py`
> - RNN：`rnn_cell_node.py`
> - Transformer：`moe_layer_node.py`
>
> *新增前端元件：*
> - `Nodes/AttentionHeatmapVizNode.tsx`、`AttentionMaskVizNode.tsx`、`EmbeddingScatterVizNode.tsx`、`TokenizerVizNode.tsx`、`TextInputVizNode.tsx`
> - `Nodes/EduSelfAttentionVizNode.tsx`、`EduMultiHeadAttentionVizNode.tsx`、`EduCrossAttentionVizNode.tsx`、`EduKNNVizNode.tsx`
> - `shared/HeatmapModal.tsx`、`HeatmapPlot.tsx`、`ScatterPlot.tsx`、`TokenChip.tsx`
> - `shared/DialogContainer.tsx`、`Toast.tsx`
> - `Toolbar/SettingsPopover.tsx`、`FontSizeMenu.tsx`
>
> *新增 / 強化 CI 工具：*
> - frontend-dist release pipeline + tag-triggered asset
> - Python 3.10-3.12 矩陣 + pnpm@9 + cdui smoke + softprops 修復
> - `scripts/dev.py` 強化（friendly preflight + PATH auto-patch）
