# CodefyUI 專案分析報告

> 撰寫日期：2026-03-21（最後更新：2026-04-26）
> 目標：評估 CodefyUI 現況、分析競品、規劃專業 AI/ML 視覺化管線工具的發展路徑
>
> **本次更新（2026-04-26）重點：**
> - 過去一個月（2026-03-22 → 2026-04-26）共 35 次 commit，主軸為 **教育/可解釋性功能**（A1 步驟追蹤、A2 權重持續化、A3 反向傳播健康度）、跨平台安裝器、影像/註解節點與 SmartDataEdge 路由。
> - 節點數由 59 → 69（13 類別），測試檔案由 10 → 28。
> - 新增重大策略思考：CodefyUI 已從「節點式 ML 工具原型」演進為**具備教育獨特性的視覺化 ML 學習平台**，這是相對 ComfyUI / Langflow / Kubeflow 真正可建立護城河的差異化方向。

---

## 目錄

1. [Executive Summary](#1-executive-summary)
2. [現況分析](#2-現況分析)
   - 2.1 技術棧 / 2.2 架構分析 / 2.3 功能清單 / 2.4 內建節點清單 / 2.5 程式碼品質
   - **2.6 教育功能（A1 Step Trace / A2 Weight Persistence / A3 Backward）★ 新增**
3. [競品分析](#3-競品分析)
4. [Gap Analysis — 差距分析](#4-gap-analysis--差距分析)
   - 4.1 差距矩陣 / 4.2 5 大關鍵差距（更新版）/ 4.3 已解決項目 / 4.4 領先優勢
5. [目標定位與願景](#5-目標定位與願景)
   - 5.1 定位 / 5.2 為何主軸是教育 / 5.3 三層次演進 / 5.4 差異化 / 5.5 目標使用者 / 5.6 First 1000 User 策略
6. [建議發展路線圖](#6-建議發展路線圖)
   - Phase 0 基礎強化 / Phase 1A 教育擴大 + 1B LLM / Phase 2 社群生態 / Phase 3 企業
7. [技術架構建議](#7-技術架構建議)
   - 7.1 執行引擎現況 / 7.2 節點系統 / 7.3 前端 / 7.4 後端 / 7.5 教育引擎子系統 / 7.6 整體架構演進
8. [風險評估](#8-風險評估)
9. [附錄：競品功能對照表](#9-附錄競品功能對照表)
   - 核心 / 教育與可解釋性 / LLM / 部署 / 生態 / 授權

---

## 1. Executive Summary

CodefyUI 已從**早期原型** (v0.1.0) 演進為**具備獨特教育價值的視覺化 ML 學習與實驗平台**。基礎設施大幅完善（執行引擎、跨平台安裝、影像/檔案管理、SmartEdge 路由），同時在 2026-04 推出三大教育核心：步驟追蹤（Step Trace）、權重持續化（Weight Persistence）、反向傳播健康度可視化（Backward Health）。這三項功能在所有競品中都未出現，是真正可建立護城河的差異化方向。

**核心優勢：**

*基礎能力（既有）*
- 完整的 DAG 圖形編輯器、拓撲排序 + 並行執行引擎、類型安全連線系統
- 架構設計清晰（BaseNode 抽象、Registry 模式、Preset 子圖、StatefulModuleMixin）
- 現代化技術棧（React 19 + @xyflow/react 12 + FastAPI + WebSocket + PyTorch 2）
- 多分頁工作區、自訂節點熱重載、Preset 巢狀展開
- 節點快取（hash-based）、Dirty Node Tracking、執行取消、錯誤恢復（fail_fast/continue/retry）
- 結構化日誌、24 個後端測試檔案、8 個前端測試
- 模型權重 (.pt/.pth/.safetensors/.ckpt/.bin) 與影像檔案 (.png/.jpg/.webp/.gif/.bmp/.tiff) 的上傳/列表/下載/刪除 API
- 7 種 ParamType（int/float/string/bool/select/model_file/image_file）

*教育與可解釋性（2026-04 新增，跨平台無對手）*
- **A1 步驟追蹤（Verbose Step Trace）**：節點執行時逐步記錄中間張量（如 MultiHeadAttention 的 Q/K/V → scaled scores → softmax → weighted sum），每步可附 LaTeX 公式說明
- **A2 權重持續化（Weight Persistence）**：StatefulModuleMixin + NodeStateStore 讓 nn.Module 在多次「Run」之間保留權重（LRU 200 modules），可在 UI 上互動式訓練
- **A3 反向傳播 + 梯度健康度（Backward Pass + Grad Health）**：自動 retain_grad、BackwardOnce 顯式選 loss、按 norm/mean/max 分類為 vanishing / exploding / healthy
- **15 個層級節點已遷移至 StatefulModuleMixin**（Conv1d/2d/Transpose、BatchNorm 1d/2d、LayerNorm/GroupNorm/InstanceNorm2d、LSTM/GRU、MHA/Encoder/Decoder、Linear、Embedding）
- **MathText（KaTeX）**：節點描述、Step Trace、ConfigPanel、Tooltip 全面支援 `$...$` / `$$...$$` 公式
- **三分頁 Inspector**：Forward / Steps / Backward 切換，配合 segment 路徑高亮做 Compare Segment

*UX 與部署（2026-04 新增）*
- **Trigger Edge 系統**：Start 節點 + DataType.TRIGGER 取代 isEntryPoint flag，視覺上以綠色虛線顯示
- **SmartDataEdge**：基於 hash 的 jitter + 弧線/skip/step 路由，自動避免 edge 重疊
- **Note Node**：可繫結節點的文字/影像便利貼，便於工作流註解與教學
- **Auto-layout（dagre）**：含寬度自動換行（>2400px wrap to multi-row grid）
- **跨平台安裝器**：`./cdui install`、`install.sh` / `install.ps1` 一行安裝、uv 自動管理 Python、`scripts/dev.py` 跨平台 task runner
- **範例工作流 19 個**（15 個 Model_Architecture 模型參考 + 4 個 Usage_Example 可訓練範例），全部驗證可執行（CI gate）

**主要差距：**

| 差距 | 說明 |
|------|------|
| GPU/分散式計算 | 僅基本 cpu/cuda 切換，無 VRAM 管理、無多 GPU、無分散式訓練 |
| LLM/GenAI 節點 | 完全缺席（OpenAI/Anthropic/HF/Ollama、Embedding、Vector DB、RAG、Agent） |
| 實驗追蹤整合 | 無 MLflow / W&B 整合（Step Trace 算內建版可解釋性，但跨 run 比較仍弱） |
| 使用者認證 + 團隊協作 | 完全缺席（無 OAuth/RBAC/多人即時編輯/評論） |
| 容器化部署 | 無 Dockerfile / docker-compose / Helm Chart |
| 資料庫儲存 | 仍為 JSON 檔案；Run/State 資料無持久層，重啟後遺失 |
| 社群節點生態 | Custom Node Manager GUI 已有，但無 Marketplace / 遠端搜尋 / 安裝 |
| Python SDK / API Gateway | 無 `pip install codefyui` 程式化操作管道 |

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
| **即時通訊** | WebSocket (原生) | — |
| **數據驗證** | Pydantic + pydantic-settings | 2.0+ |
| **資料適配** | datasets, kagglehub, safetensors | — |
| **環境管理** | uv (Astral) — 自動安裝 Python 3.11 | — |
| **Python** | 3.10+ (實際用 3.14) | — |
| **跨平台執行** | scripts/dev.py + Makefile + cdui launcher | — |

### 2.2 架構分析

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Frontend (React 19)                          │
│  ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────┐ ┌──────────┐ │
│  │ NodePal-│ │FlowCanvas│ │ConfigPnl │ │ Inspector   │ │ Results  │ │
│  │  ette   │ │ + Smart- │ │ +MathText│ │ Forward /   │ │  Panel   │ │
│  │ +Quick- │ │ DataEdge │ │ (KaTeX)  │ │ Steps /     │ │ +Loss    │ │
│  │ Search  │ │ +Trigger │ │          │ │ Backward    │ │ Chart    │ │
│  └─────────┘ │ +NoteNode│ └──────────┘ └─────────────┘ └──────────┘ │
│              └──────────┘                                            │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │          Zustand Stores                                         │  │
│  │  tabStore (nodes/edges/run/segment) | nodeDefStore | uiStore    │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                  │ REST API              │ WebSocket                  │
└──────────────────┼──────────────────────┼─────────────────────────────┘
                   ▼                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          Backend (FastAPI)                           │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ routes_nodes | routes_graph | routes_presets | routes_examples │  │
│  │ routes_models | routes_images | routes_custom_nodes            │  │
│  │ routes_execution_state | routes_execution_outputs              │  │
│  │                       ws_execution                              │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Core Engine                                                    │  │
│  │  NodeRegistry | PresetRegistry | GraphEngine | TypeSystem       │  │
│  │  BaseNode (ABC) | StatefulModuleMixin                           │  │
│  │                                                                 │  │
│  │  ── Educational Subsystems (新增) ──                            │  │
│  │  ExecutionContext (verbose/weights_persistent/backward_mode)    │  │
│  │  StepRecorder + step_trace  (A1)                                │  │
│  │  NodeStateStore (LRU 200)   (A2)                                │  │
│  │  backward_pass + grad_health (A3)                               │  │
│  │  RunOutputStore (per-run forward + steps + grads)               │  │
│  │  ExecutionCache (hash-based) | DirtyNodeTracker                 │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Node Packages (69 nodes, 13 categories)                        │  │
│  │  CNN(9) | RNN(2) | Transformer(3) | RL(3) | Data(7)             │  │
│  │  Training(6) | IO(9) | Control(1) | Dataflow(3) | Utility(7)    │  │
│  │  Normalization(4) | TensorOps(11) | LLM(4)                      │  │
│  │  ↑ 15 個層級節點以 StatefulModuleMixin 持續化權重                │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.3 功能清單

> 表格按主題分組；★ 標示 2026-04 新增、◆ 標示自上次報告以來大幅強化。

**圖形編輯與互動**

| 功能 | 狀態 | 備註 |
|------|------|------|
| DAG 圖形編輯器 | 已完成 | React Flow 12, 拖放、連線、多選 |
| 類型安全連線 | 已完成 | 12 種 DataType（含新增 TRIGGER）, 相容性矩陣 |
| 節點參數面板 | 已完成 | 7 種 ParamType（int/float/string/bool/select/model_file/image_file） |
| 快速節點搜尋 | 已完成 | 雙擊畫布開啟，支援節點 + Preset |
| 多分頁工作區 | 已完成 | 獨立執行環境、localStorage 持久化 |
| Preset 子圖系統 | 已完成 | 巢狀展開、自動偵測外露埠口、◆ 支援匯入主編輯器工作流 |
| MiniMap | 已完成 | 類別顏色標示 |
| 右鍵選單 | 已完成 | 刪除/複製/重命名 |
| ★ Trigger 入口系統 | 已完成 | Start 節點 + DataType.TRIGGER 取代舊 isEntryPoint flag；綠色虛線顯示；支援多重入口 |
| ★ SmartDataEdge 智慧路由 | 已完成 | hash-based jitter、弧線/skip/step 自動路由，避免 edge 重疊 |
| ★ Trigger Edge 重連 | 已完成 | 觸發邊可拖曳重連到不同 handle |
| ★ Note Node | 已完成 | 文字 / 影像便利貼，可繫結節點，支援顏色與大小自訂；NoteBindingLines 顯示連線 |
| ★ Auto-layout (dagre) | 已完成 | Network-simplex 排版；超過 2400px 自動換行為多列 grid；以 Start 節點為錨點 |
| ★ Empty Canvas Examples | 已完成 | 空畫布顯示分組範例卡片（Trainable Workflows / Architecture Reference / Other） |

**執行引擎**

| 功能 | 狀態 | 備註 |
|------|------|------|
| 拓撲排序執行 | 已完成 | Kahn's algorithm, 循環偵測, 並行同層級節點（asyncio.gather） |
| 節點輸出快取 | 已完成 | hash-based 變更偵測 + ExecutionCache |
| 部分重新執行 | 已完成 | Dirty Node Tracking, 僅重跑變更節點及其下游 |
| 執行取消 | 已完成 | ExecutionContext._cancel_event, asyncio.Task cancellation |
| 錯誤恢復 | 已完成 | fail_fast / continue / retry 三種模式 |
| 圖表驗證 | 已完成 | 邊緣類型檢查 + 循環偵測 |
| WebSocket 即時進度 | 已完成 | 每節點 running/completed/error 狀態 |
| ★ Educational ExecutionContext | 已完成 | verbose / weights_persistent / backward_mode / auto_backward 四個教育旗標 |
| ★ RunOutputStore | 已完成 | 每次 run 的 forward + steps + grads 統一儲存；支援大張量切片回退 |
| CLI 圖表執行器 | 已完成 | `run_graph.py` 命令列驗證 + 執行 graph.json |

**教育與可解釋性（2026-04 新增）**

| 功能 | 狀態 | 備註 |
|------|------|------|
| ★ A1 Verbose Step Trace | 已完成 | StepRecorder 逐步記錄中間張量；支援 LaTeX 描述；目前已在 MHA / Conv2d / LayerNorm / Softmax / BatchNorm 等節點啟用 |
| ★ A2 Weight Persistence | 已完成 | StatefulModuleMixin + NodeStateStore（LRU 200）；15 個層級節點已遷移；支援結構參數變更時自動釋放舊模組 |
| ★ A3 Backward Pass + Grad Health | 已完成 | attach_retain_grad / select_backward_target / capture_grads / grad_health（vanishing / exploding / healthy）；支援 BackwardOnce 顯式選 loss 與自動 leaf 合成 loss |
| ★ MathText (KaTeX) | 已完成 | 解析 `$...$` 與 `$$...$$`；ConfigPanel 描述、Step Trace 步驟說明、Tooltip 全面支援 |
| ★ Inspector 三分頁 | 已完成 | Forward / Steps / Backward 切換；保留 ValueDiff（cache hit 比較） |
| ★ Compare Segment | 已完成 | 選 head/tail 兩節點高亮路徑、跨節點輸出比較、cache-hit 記錄、多 input segment、per-bubble clear |
| ★ TensorInput 節點 | 已完成 | 直接在前端建構張量輸入，作為教學入口 |

**檔案 / 資源管理**

| 功能 | 狀態 | 備註 |
|------|------|------|
| 模型權重管理 | 已完成 | REST API 上傳/列表/下載/刪除 (.pt/.pth/.safetensors/.ckpt/.bin) |
| MODEL_FILE 參數類型 | 已完成 | 節點參數下拉選取已上傳模型檔案，含上傳按鈕 |
| ★ 影像檔案管理 | 已完成 | REST API 上傳/列表/下載/刪除 (.png/.jpg/.jpeg/.bmp/.webp/.gif/.tiff) |
| ★ IMAGE_FILE 參數類型 | 已完成 | 一般化 file param field，圖像參數可選取已上傳檔案 |
| ★ ImageReader 整合 | 已完成 | ImageReader 節點直接從上傳列表挑檔，自動正方化 |
| ★ Image Writer / Batch Reader | 已完成 | 新增寫入與批次讀取節點 |
| ★ 完成節點檔案下載 | 已完成 | 節點輸出包含檔案路徑時，UI 顯示下載按鈕 |
| Custom Node Manager | 已完成 | GUI 管理自訂節點 (上傳/啟用/停用/刪除) |
| 自訂節點 | 已完成 | Python 檔案放入 custom_nodes/, 熱重載 |
| 匯入/匯出 Graph JSON | 已完成 | 檔案上傳 + 後端儲存 |

**範例與內容**

| 功能 | 狀態 | 備註 |
|------|------|------|
| ★ 範例工作流 | 已完成 | 19 個範例（15 Model_Architecture + 4 Usage_Example），全部 CI 驗證可執行 |
| 範例分類組織 | 已完成 | Model_Architecture / Usage_Example / Others 三類 |

**安裝與部署（2026-04 新增）**

| 功能 | 狀態 | 備註 |
|------|------|------|
| ★ 跨平台一鍵安裝 | 已完成 | `install.sh` (Unix) + `install.ps1` (Windows)，自動安裝 uv + Python 3.11 |
| ★ cdui launcher | 已完成 | 包裝 `scripts/dev.py`，支援 `cdui install/dev/test/stop` |
| ★ scripts/dev.py | 已完成 | 跨平台 task runner（install/update/dev/stop/test/clean/uninstall）；UTF-8 強制；自動 venv re-exec |
| ★ Makefile | 已完成 | Unix 慣例入口；委派 dev.py |
| ★ Release 安裝驗證 CI | 已完成 | 三平台 install-check 工作流 |

**其他**

| 功能 | 狀態 | 備註 |
|------|------|------|
| Python 腳本匯出 | 部分完成 | 僅產生骨架，非可執行程式碼 |
| i18n 多語言 | 已完成 | 英文 + 繁體中文 + 節點分散式 locale 檔 |
| 深色主題 | 已完成 | 固定深色主題 |
| ResultsPanel 增強 | 已完成 | 分頁 (Log/Training), 可調整高度, Loss 曲線圖表 |
| 結構化日誌 | 已完成 | JsonFormatter + 輪轉檔案 |

### 2.4 內建節點清單 (69 個，13 類別)

> ★ = StatefulModuleMixin（持續化權重） · ✦ = 含 Verbose Step Trace · ⚙ = 2026-04 新增 · ⚛ = 2026-05 新增 (LLM/PR #4)

| 類別 | 節點 | 數量 |
|------|------|------|
| **CNN** | Conv1d★✦, Conv2d★✦, ConvTranspose2d★, BatchNorm2d★, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, Dropout, Activation | 9 |
| **RNN** | LSTM★, GRU★ | 2 |
| **Transformer** | MultiHeadAttention★✦, TransformerEncoder★, TransformerDecoder★ | 3 |
| **Normalization** | BatchNorm1d★, LayerNorm★✦, GroupNorm★, InstanceNorm2d★ | 4 |
| **Utility** | Linear★, Embedding★, Flatten, Concat, Reshape, Print, Visualize | 7 |
| **Tensor Ops** | Add, MatMul, Mean, Multiply, Permute, Softmax✦, Split, Squeeze, Stack, TensorCreate, Unsqueeze | 11 |
| **Training** | Optimizer, Loss, LRScheduler, TrainingLoop, SequentialModel, ⚙ BackwardOnce | 6 |
| **IO** | ImageReader, ⚙ ImageWriter, ⚙ ImageBatchReader, FileReader, ModelLoader, ModelSaver, CheckpointSaver, CheckpointLoader, Inference | 9 |
| **Data** | Dataset, DataLoader, Transform, ⚙ TensorInput, ⚛ TextInput, HuggingFaceDataset, KaggleDataset | 7 |
| **Control** | ⚙ Start (DataType.TRIGGER 入口) | 1 |
| **Dataflow** | Map, Reduce, Switch | 3 |
| **RL** | DQN, PPO, EnvWrapper | 3 |
| **LLM** | ⚛ Tokenizer, ⚛ WordVector, ⚛ EmbeddingScatter, ⚛ CosineSimilarity | 4 |

**節點分布觀察：**
- **15 個 ★ 節點**（Conv1d/2d/Transpose、BatchNorm 1d/2d、LayerNorm/GroupNorm/InstanceNorm2d、LSTM/GRU、MHA/Encoder/Decoder、Linear、Embedding）已採用 StatefulModuleMixin，是教育互動性的核心。
- **Step Trace** 目前主要在 Transformer、CNN、Normalization 類別啟用；RNN、RL、Tensor Ops 多數未啟用，是後續可擴充重點。
- **Control 類別僅剩 Start**：原 If / ForLoop / Compare 節點已被 Dataflow 類別（Map/Reduce/Switch）取代，並加上 Trigger 系統定位入口。
- **Dataflow** 為新類別：取代原本散落在 Control 的高階控制流節點。
- **缺失但市場急需的節點**：LLM 推理 / Embedding / Vector DB / Document Loader / Text Splitter / Retriever / Prompt Template / Agent / Tool。

### 2.5 程式碼品質評估

**優點：**
- 後端架構清晰：BaseNode ABC → NodeRegistry → GraphEngine 層次分明，新增 StatefulModuleMixin 不破壞既有抽象
- 前端 Zustand 單一 store 設計合理，per-tab 隔離良好；新增 segment / inspectorTab 等 UI 狀態時無侵入式改動
- 類型系統完整（DataType enum 含 12 種，TRIGGER 為新增型別）
- Preset 系統設計精巧（自動外露偵測、巢狀展開、參數覆蓋、◆ 支援匯入主編輯器工作流）
- 教育子系統與既有引擎正交分離：ExecutionContext 旗標控制是否啟用，未開啟時無效能負擔
- backward_pass 將 PyTorch autograd 細節（retain_grad、leaf 偵測、grad health）封裝在後端，前端只需消費結果

**2026-04 已完成的重大改善：**

*教育引擎（commit 77abf0a, 0983c31, 0c21ebc, ffd8714, e61fbf4, 4383a27, 9245d22）*
- ExecutionContext 加入 verbose / weights_persistent / backward_mode / auto_backward 四旗標
- StepRecorder（core/step_trace.py）+ `__steps__` 結果展開到 RunOutputStore
- NodeStateStore（core/node_state_store.py）—— LRU 200 modules、per-key lock、結構參數變更時自動釋放
- backward_pass.py —— attach_retain_grad / select_backward_target / capture_grads / grad_health
- 15 個層級節點遷移至 StatefulModuleMixin（CNN 4 + Norm 4 + RNN 2 + Transformer 3 + Utility 2）
- Inspector 三分頁（Forward / Steps / Backward）+ BackwardView + Compare Segment 路徑高亮
- MathText 元件（KaTeX）+ 在 ConfigPanel / Step Trace / Tooltip 全面支援 LaTeX
- 後端測試擴充：test_step_trace.py / test_backward_pass.py / test_node_state_store.py / test_stateful_modules.py

*UX 與 Canvas（commit a2e1fa1, 8350795, 45ad0d6, b582c4c, c28950e, d0a1002, 4a0715a）*
- Note Node（文字 + 影像 + 顏色 + 繫結節點）
- SmartDataEdge（hash-based jitter + 弧線/skip/step 路由）
- TriggerEdge（綠色虛線）+ trigger handle UX + 邊重連
- Auto-layout 採 dagre（network-simplex），>2400px 自動換行成多列 grid
- EmptyCanvasOverlay 將範例分組為 Trainable Workflows / Architecture Reference / Other
- SubgraphEditor 改用 dagre + 支援匯入主編輯器工作流

*入口系統重構（commit 5473a01, f3a34cf, 2f14b01, 909c43b, a67896e）*
- 移除 isEntryPoint flag，改用 Start 節點 + DataType.TRIGGER；前後端與所有 19 個範例同步遷移
- 移除 MigrationModal 與已死的 executionStore / flowStore re-exports

*影像/檔案 API（commit f78fefa, e448191, 8b146eb, 0183b71, 021e82c）*
- routes_images.py + ALLOWED_EXTENSIONS + 安全路徑驗證
- ImageReader / ImageWriter / ImageBatchReader 節點對接 API
- ParamField 一般化為支援 model_file / image_file 兩種選擇器
- 完成節點顯示輸出檔案下載按鈕

*跨平台安裝與 CI（commit d0b74c5, 149d345, 713cb9b, fb94443, 73d0299, 7cae018, 0c078ae, 32b064d, 350a680, 59d1dfe, a88c820, 640030d, c561e81, 05ab4f7, 3f7fed8）*
- uv 自動安裝 Python 3.11、cdui launcher、scripts/dev.py 跨平台 task runner、Makefile
- Release 安裝驗證 CI（三平台 install-check）
- Windows 修復：PNPM_HOME、@types/node、pnpm.cmd shell=True、checkout v6 / upload-artifact v7

*文件與整理（commit a504097, a6bcf86, 5662865, 7a177cd, 7f079c0, 0bf4f1d, 56101f8）*
- docs/ 目錄重構與 SETUP 同步、README 雙語更新、.gitignore 擴充、移除未用資料

**仍待改善：**
- Graph / Run / NodeState 仍使用記憶體 + JSON 檔案，重啟即遺失執行歷史與權重
- Python 腳本匯出功能不完整（僅骨架程式碼，無可執行的資料流程式碼）
- Step Trace 僅約 1/4 節點啟用，RNN / RL / Tensor Ops 多數未instrument
- backward_pass 已存在但 TrainingLoop 內未自動使用（需手動切換 backward_mode）
- WebSocket 執行串流仍為單實例，未抽象為 Redis pub/sub 以利水平擴展

### 2.6 教育功能 (Educational Features) — 重大新方向

> 2026-04 期間最重要的策略性演進。三大教育核心（A1/A2/A3）合稱「Learn-by-Run」架構：使用者按一次「Run」就能同時看到模型結構、中間張量、權重變化、梯度健康度，這是市面上沒有任何競品提供的能力。

#### 2.6.1 A1 — Verbose Step Trace（步驟追蹤）

**架構：**
```
ExecutionContext.verbose=True
   │
   ▼
Node.execute()
   │
   ├─► StepRecorder.record(name, description, scalars={}, **tensors)
   │      e.g. recorder.record("scaled_scores",
   │              "Compute attention: $S = QK^T / \\sqrt{d_k}$",
   │              scalars={"d_k": 64.0},
   │              scores=scores)
   │
   ▼
result = {"output": ..., "weights": ..., "__steps__": [Step(...), ...]}
   │
   ▼
GraphEngine 將 __steps__ 展開成 RunOutputStore 鍵：
   "__step__0__Q", "__step__0__K", "__step__1__scores", ...
   │
   ▼
Frontend StepTraceView (GET /api/execution/{run_id}/{node_id}/__step_index__)
   ├─ 取得步驟索引
   ├─ 點擊步驟卡 → 取對應張量（fallback 切片大張量）
   └─ MathText 渲染 LaTeX 描述
```

**已啟用節點：** MultiHeadAttention（Q/K/V→scaled scores→softmax→weighted sum 共 4-5 步）、Conv2d、LayerNorm、Softmax、BatchNorm 等。
**待擴充：** RNN（LSTM unrolling）、Tensor Ops、RL（policy/value 計算）。

**為什麼重要（教育場景）：**
- 學生選擇 Conv2d 節點 → 看到 padding 後的 input、kernel sliding 過程、each output element 計算 → 比讀 PyTorch 文件直觀十倍
- 教師可在課堂上即時演示「為什麼 attention 要除以 √d_k」（觀察 softmax 飽和的差異）
- 研究者可比較不同 normalization 在 forward 中的數值差異

#### 2.6.2 A2 — Weight Persistence（權重持續化）

**架構：**
```
StatefulModuleMixin (core/stateful_module.py)
   │
   ├─ structural_params: ClassVar[tuple] (e.g. ("in_channels", "out_channels", "kernel_size"))
   ├─ build_module(params) → 由子類別實作，回傳 nn.Module
   └─ get_or_build_module(context, params)
        │
        ▼
   NodeStateStore (core/node_state_store.py)
       ├─ 索引：(graph_id, node_id, structure_hash) → nn.Module
       ├─ LRU 上限 200 modules
       ├─ 結構參數變更 → 同 (graph_id, node_id) 兄弟模組全清
       └─ Per-key lock 支援 reentrant autograd
```

**設計亮點：**
- 結構參數（影響張量 shape，例如 channels）改變時自動重建；非結構參數（例如 lr）變更不重建
- ExecutionContext 沒給時 fallback 為 fresh build，CLI / 測試環境不受影響
- 與 ExecutionCache 共存：`cacheable=False` 讓 stateful 節點每次重跑（避免 cache 回傳舊權重）

**教育價值：**
- 連續按「Run」可看到 loss 真的下降（而非每次重置）→ 這是視覺化訓練最關鍵的體驗
- 配合 Compare Segment 可比對「訓練前 vs 訓練 N 步後」同一節點的輸出差異
- 配合 NoteNode 可在畫布上標記「此處 BatchNorm running_mean 已收斂」等觀察

#### 2.6.3 A3 — Backward Pass + Gradient Health（反向傳播與梯度健康度）

**架構：**
```
ExecutionContext.backward_mode=True
   │
   ▼
1. zero_module_grads() — 清除 stateful module 累積梯度
   │
   ▼
2. Forward pass（每個節點執行後）
       │
       └─► attach_retain_grad()
              ├─ 對非 leaf 浮點張量呼叫 .retain_grad()
              └─ 記錄 (node_id, port) → tensor 至 ctx.grad_targets
   │
   ▼
3. select_backward_target()
       ├─ Priority 1: BackwardOnce 節點輸出（顯式選 loss）
       ├─ Priority 2: 若已有 TrainingLoop，跳過（其自管 backward）
       └─ Priority 3: 自動合成 loss = largest_leaf_tensor.sum()
   │
   ▼
4. loss.backward()
   │
   ▼
5. capture_grads() (async)
       ├─ 對 ctx.grad_targets 寫入 {port}__grad + {port}__grad__meta
       └─ 對每個 stateful module 寫入 __weight_grad__{param_name} + meta
   │
   ▼
Frontend BackwardView
   ├─ 顯示 port grad 的 norm / mean / max
   ├─ 顯示 weight grad 統計
   └─ 健康度標籤：
       - 紅 vanishing  (norm < 1e-7 或 mean < 1e-8)
       - 黃 exploding  (norm > 1e3 或 max > 1e2)
       - 綠 healthy
```

**教育價值（這是最大的差異化）：**
- 學生可立即看到「梯度消失」與「梯度爆炸」的視覺化證據——而非只在 textbook 看圖
- 比較不同初始化策略（Xavier / Kaiming）對梯度健康度的影響
- 比較有 / 沒有 BatchNorm 的網路深層梯度差異
- DeBugging 場景：訓練 loss 不降，可立即定位是哪個節點 grad vanishing

**目前限制（待改進）：**
- TrainingLoop 內部仍走自己的 backward，未整合 grad health 視覺化
- 沒有 grad heatmap / histogram（僅 scalar 統計）
- 沒有跨 epoch 的 grad 演進曲線

#### 2.6.4 教育功能與既有引擎的關係

| 既有引擎能力 | 教育功能如何銜接 |
|--------------|-----------------|
| Dirty Node Tracking | 配合 weights_persistent，只重跑 dirty 節點時權重仍保留 |
| ExecutionCache | StatefulModuleMixin 設 `cacheable=False` 跳過快取，避免回傳舊張量 |
| 並行執行 | StepRecorder 為節點 local，不衝突；NodeStateStore 用 per-key lock |
| 執行取消 | backward_pass 走 await，可在 capture_grads 階段被取消 |
| WebSocket 進度 | 可在每個 step record 時回報「running step 2/5」等細節（目前未串接） |

#### 2.6.5 教育功能尚未做的（重要待辦）

1. **更廣的 Step Trace 覆蓋率**：將 LSTM unrolling、PPO advantage、DQN target Q 都 instrument，使所有教學場景都受惠
2. **Gradient 視覺化升級**：grad histogram、跨 step 演進曲線、layer-wise 梯度比較
3. **教學模式 Onboarding**：第一次開啟啟用 verbose+ 提示，引導使用者點 Step Trace
4. **教師工具**：將 Step Trace 結果匯出為 Markdown / PDF 教學講義
5. **內建概念解釋庫**：每個節點配一段 `concept.md`，於 ConfigPanel 展開（含 LaTeX 推導）
6. **Quiz / 互動挑戰**：載入特定圖，要求使用者調整參數使 grad 變 healthy（gamification）

---

## 3. 競品分析

### 3.1 ComfyUI — 圖像生成工作流引擎

**概述：** ComfyUI 是最成功的開源節點式 AI 工作流工具，專注於 Stable Diffusion 圖像生成。GitHub 92.5k+ stars，4M+ 活躍使用者。

| 面向 | 詳情 |
|------|------|
| **技術棧** | 後端 Python (PyTorch), 前端 Vue 3 + TypeScript (獨立套件 comfyui-frontend-package), REST + WebSocket |
| **執行引擎** | 智慧快取系統 — 僅重新執行變更的節點，大幅提升效率 |
| **GPU 管理** | 智慧 VRAM 管理，自動卸載不需要的模型，支援多 GPU |
| **節點數量** | 核心 ~50+，社群擴充 2,000+ 套件 (透過 ComfyUI Manager)，2.5M 共享工作流 |
| **社群生態** | ComfyUI Manager 一鍵安裝社群節點，50k+ Discord 月活，300+ 核心貢獻者 |
| **API** | 完整 REST API，可程式化驅動工作流 |
| **部署** | Docker 支援, 雲端服務 (RunComfy, ComfyICU), 內建隊列 |

**ComfyUI 成功關鍵因素：**
1. **增量執行 + 快取** — 只重跑改動的部分，互動體驗極佳
2. **社群節點生態** — ComfyUI Manager 讓安裝第三方節點像 npm install 一樣簡單
3. **專注單一領域** — 圖像生成做到極致
4. **GPU 智慧管理** — 自動 VRAM 分配、模型切換
5. **工作流分享文化** — JSON 工作流直接匯入匯出

**ComfyUI 弱點：**
- 學習曲線陡峭，對非技術使用者不友善
- 僅限圖像/影片生成，不適合通用 ML
- 自訂節點生態碎片化（維護品質參差、版本相容性問題）
- 行動裝置體驗差
- 安裝設定複雜（特別是 macOS）

### 3.2 n8n — 工作流自動化平台

**概述：** 開源工作流自動化工具，定位類似 Zapier 但可自建。GitHub 40k+ stars，45k+ 社群論壇成員。

| 面向 | 詳情 |
|------|------|
| **技術棧** | TypeScript, Vue.js 前端, Node.js 後端, SQLite/PostgreSQL |
| **節點數量** | 400+ 內建整合節點 |
| **AI 功能** | AI Agent 節點、LangChain 整合、向量儲存、MCP 支援、多 Agent 系統編排 |
| **部署** | Docker, Kubernetes, 雲端託管 (n8n Cloud) |
| **定價** | Community (免費自建) / Cloud ($20/mo+) / Enterprise (客製) |
| **特色** | 視覺化除錯、版本控制、子工作流、錯誤處理流程、Agentic AI 四大模式 |

**與 CodefyUI 的關聯：**
- n8n 展示了節點式工具如何擴展到企業級
- AI Agent 功能值得參考（LLM 整合、工具使用、記憶機制）
- 子工作流 = CodefyUI 的 Preset 概念，但更成熟
- 錯誤處理流程是 CodefyUI 缺失的關鍵功能

### 3.3 Kubeflow Pipelines — 企業級 ML 管線

**概述：** Google 主導的 Kubernetes 原生 ML 管線平台。

| 面向 | 詳情 |
|------|------|
| **技術棧** | Python SDK, Kubernetes, Argo Workflows |
| **UI** | 有視覺化介面但非主要互動方式，以 Python SDK 為主 |
| **特色** | 容器化每個步驟、自動擴縮、實驗追蹤、Artifacts 管理 |
| **適合** | 大規模 MLOps、生產環境管線 |
| **GitHub Stars** | ~14k (Kubeflow 整體) |

**啟示：**
- Pipeline = 容器化步驟的 DAG，每個步驟是獨立容器
- 實驗追蹤是必要功能
- 生產環境部署需要容器化
- 與 K8s 的深度整合帶來強大的擴縮能力

### 3.4 Apache Airflow — DAG 排程引擎

**概述：** Python 原生的工作流排程平台。Airflow 是 DAG 式工作流的標竿。

| 面向 | 詳情 |
|------|------|
| **技術棧** | Python, Flask, Celery/Kubernetes Executor |
| **定義方式** | Python 程式碼定義 DAG（非視覺化優先） |
| **特色** | 排程、重試、SLA、告警、豐富的 Operator 生態 |
| **GitHub Stars** | ~40k |

**啟示：**
- DAG 排程（Cron-based）是 CodefyUI 缺失的
- 重試機制、SLA 監控是生產環境必須
- Operator/Hook/Sensor 模式值得參考
- Airflow 的視覺化介面是輔助而非主要入口 — CodefyUI 的「視覺化優先」是差異化

### 3.5 MLflow — ML 生命週期管理

**概述：** ML 實驗追蹤、模型註冊、部署的標準工具。

| 面向 | 詳情 |
|------|------|
| **核心功能** | Tracking (實驗記錄), Models (模型版本), Registry (模型註冊), Serving |
| **GitHub Stars** | ~19k |
| **整合** | PyTorch, TensorFlow, scikit-learn, HuggingFace |

**啟示：**
- CodefyUI 應整合 MLflow 而非重新發明
- 實驗追蹤 (metrics, params, artifacts) 是核心需求
- 模型版本管理是生產化的前提

### 3.6 Weights & Biases (W&B) — 實驗追蹤平台

**概述：** ML 實驗追蹤與協作平台，商業化成功案例。

| 面向 | 詳情 |
|------|------|
| **核心功能** | 實驗追蹤、超參搜尋、模型評估、資料版本、報告 |
| **協作** | 團隊 Dashboard、報告分享、模型審核 |
| **特色** | Sweeps (超參最佳化), Artifacts (資料版本), Tables (互動式資料分析) |

**啟示：**
- 互動式訓練曲線、即時指標是必備功能
- 團隊協作功能對企業客戶至關重要
- W&B 的 Dashboard 和報告功能可作為 UI 參考

### 3.7 Flowise / Langflow — LLM 視覺化工作流

**概述：** 專為 LLM 應用設計的節點式工作流建構工具。

| 工具 | 技術棧 | Stars | 特色 |
|------|--------|-------|------|
| **Flowise** | TypeScript, React Flow, Node.js | ~30k | LangChain 視覺化, 拖放建構 Chatbot, 100+ 整合, RBAC/SSO |
| **Langflow** | Python, React Flow, FastAPI | **100k+** | DataStax 支持, 多框架支援, MCP 支援, 最快速的 RAG 原型開發 |
| **Dify** | Python, React | ~55k | 全方位 LLM 應用建構, RAG + Agent, YAML DSL 工作流分享 |

**關鍵共通點：**
- 都使用 React Flow（與 CodefyUI 相同）
- 都聚焦 LLM/RAG 應用場景
- Langflow 的技術棧（Python + FastAPI + React Flow）與 CodefyUI 幾乎相同

**啟示：**
- LLM/GenAI 節點是當前市場最大需求
- RAG 管線建構是高需求場景
- CodefyUI 可以覆蓋 Langflow 的場景並擴展到傳統 ML

### 3.8 其他值得關注的工具

| 工具 | 定位 | 值得借鏡之處 |
|------|------|-------------|
| **Gradio** | ML 模型 Demo 介面 | 快速原型驗證、分享、HuggingFace 整合 |
| **Streamlit** | 資料應用快速開發 | Python-native 體驗、即時預覽 |
| **Dify** | LLM 應用開發平台 | Agent 編排、知識庫、工作流、YAML DSL |
| **NodeTool** | 本地優先節點式 AI 工作流 | 支援圖像/影片/文字/資料/自動化 |
| **Adobe Project Graph** | Creative Cloud 節點式 AI 系統 | 原生 Creative Cloud 整合 (2025 發布) |
| **Haystack** | NLP/RAG 管線 | Pipeline 抽象、Component 系統 |
| **ZenML** | MLOps 管線框架 | 可插拔的 Stack（orchestrator/artifact store/model deployer） |
| **Prefect** | 現代工作流引擎 | Python-native, ControlFlow AI 任務抽象 |
| **Kedro** | ML 管線框架 | 資料目錄、管線視覺化、可重現性 |
| **Metaflow** | ML 工程框架 | Netflix 出品, 專注 ML 工程師體驗 |

### 3.9 市場規模數據

| 市場區間 | 2024-2025 規模 | 預估 2033-2035 | CAGR |
|----------|---------------|---------------|------|
| 視覺分析工具 | $150 億 | $600 億 (2033) | 20% |
| 資料管線工具 | $639 億 | $5,145 億 (2034) | 26.8% |
| MLOps | $17-30 億 | $390-890 億 (2034) | 37-40% |
| LLM 市場 | $77.7 億 | $1,498 億 (2035) | 34.4% |
| 多模態 AI | $16 億 | 快速增長 (2034) | 32.7% |
| 低代碼平台 | $287.5 億 | $2,644 億 (2032) | 32.2% |

**關鍵市場趨勢：**
- 企業 AI 支出：2025 年 GenAI 支出 $370 億，較 2024 年 $115 億增長 3.2 倍
- Anthropic 取代 OpenAI 成為企業 LLM 支出第一（40% vs 27%）
- 開源 LLM 即將突破 50% 生產環境市佔率
- 76% 技術組織正在增加開源 AI 工具投資
- 2025 年 70% 新應用將使用低代碼/無代碼技術
- Model Context Protocol (MCP) 正成為 LLM 整合標準協議

---

## 4. Gap Analysis — 差距分析

### 4.1 關鍵差距矩陣

> 「教育/可解釋性」為新增維度，是 CodefyUI 已建立的反向領先優勢。

| 功能領域 | CodefyUI 現況 | 行業標準 | 差距嚴重度 |
|----------|---------------|----------|-----------|
| **執行引擎** | 並行 + 快取 + dirty tracking + 取消 + 錯誤恢復 | 增量/並行/分散式 | 低（僅剩分散式） |
| **快取系統** | hash-based 節點快取 + dirty node tracking | 節點級快取、增量執行 | 低（已解決） |
| **GPU 管理** | 僅基本 cpu/cuda 選擇 | VRAM 管理、多 GPU、自動裝置分配 | 嚴重 |
| **實驗追蹤** | Step Trace + RunOutputStore（內建版可解釋性，無跨 run dashboard） | MLflow/W&B 整合 | 高（部分內建解決） |
| **模型管理** | 模型檔案上傳/列表/下載/刪除 API + MODEL_FILE param | 版本控制、註冊、部署 | 中（已部分解決） |
| **影像/資源管理** | 影像檔案上傳/列表/下載/刪除 API + IMAGE_FILE param | 統一資源管理 | 低（已解決） |
| **LLM/GenAI 節點** | 無 | LLM 推理、RAG、Embedding、Agent | 嚴重 |
| **使用者認證** | 無 | OAuth2, RBAC | 高 |
| **團隊協作** | 無 | 共享工作區、版本控制、評論 | 高 |
| **容器化部署** | 無 | Docker, K8s, Helm | 高 |
| **跨平台安裝** | install.sh / install.ps1 / cdui / scripts/dev.py / Makefile | 一鍵安裝 + venv 管理 | 低（已解決） |
| **錯誤處理** | fail_fast/continue/retry 三種模式 | 重試、fallback、錯誤分支 | 低（已解決） |
| **排程執行** | 無 | Cron、事件觸發 | 中 |
| **節點生態** | 69 個內建（13 類別）+ Custom Node Manager GUI | 社群市場、包管理器 | 中（管理工具已有，無遠端市集） |
| **資料庫儲存** | JSON 檔案 + 記憶體 | PostgreSQL/SQLite | 中 |
| **Run / State 持久層** | RunOutputStore 為記憶體實現，重啟即遺失 | 持久化執行歷史 | 中 |
| **API/SDK** | 基本 REST + WebSocket | 完整 API + Python/JS SDK | 中 |
| **監控/可觀測性** | WebSocket 進度 + 結構化日誌 + Step Trace | 指標、日誌、告警、Dashboard | 中（日誌與 trace 已解決） |
| **測試框架** | 24 個後端測試檔（pytest + asyncio） + 8 個前端測試 | 節點單元測試、管線整合測試 | 低（已大幅強化） |
| **文件** | README 雙語 + docs/ 設定指南 | 完整文件站、教學、API 文件 | 中（仍缺教學/API 文件） |
| **Diffusion 模型節點** | 無 | SD/SDXL/Flux/ComfyUI 等級 | 視定位 |
| **★ 教育與可解釋性** | **Step Trace + Backward Health + Weight Persistence + LaTeX + Compare Segment** | **無競品提供** | **CodefyUI 領先** |
| **★ 圖形 UX** | SmartDataEdge + Trigger 系統 + Note Node + Auto-layout（dagre 多列） | React Flow 基本拖放 | CodefyUI 領先 |

### 4.2 最關鍵的 5 個差距（更新版）

按「市場需求 × 競爭壓力 × 投入產出比」重新排序：

1. **LLM/GenAI 節點** — 仍是市場最大需求；Langflow 已 100k stars 與 CodefyUI 同技術棧。可在現有 BaseNode 框架快速擴充 OpenAI / Anthropic / HF / Ollama / Embedding / VectorDB 節點。
2. **教育場景的擴大化** — 教育能力已是領先項目，但僅 1/4 節點啟用 Step Trace、無 Onboarding、無教師工具（教材匯出）。**這是把「領先 → 護城河」的關鍵一步**，且無競品壓力。
3. **容器化 + 部署 + Run 持久層** — 無 Dockerfile 即無法進入任何生產或共享部署場景；RunOutputStore 重啟即失也阻礙教學分享（學生重連無法看到先前 run）。
4. **實驗追蹤整合 / 跨 Run Dashboard** — Step Trace 解決了「單 run 內」可解釋性，但跨 run 比較（不同超參、不同初始化）仍弱；MLflow/W&B 整合或自建 Run History 都是必要。
5. **社群節點生態 + Python SDK** — Custom Node Manager GUI 已就位，但缺遠端市集與 Python 程式化 API；ComfyUI Manager 模式證明生態系才能撐起長尾節點需求。

### 4.3 已從關鍵差距清單移除的項目（過去 30 天解決）

- ~~執行引擎升級（快取 / 並行 / 取消 / 錯誤恢復 / dirty tracking）~~ — 全數完成
- ~~跨平台安裝門檻~~ — install.sh / cdui / dev.py 解決
- ~~影像資源管理~~ — routes_images + IMAGE_FILE param 解決
- ~~測試框架不足~~ — 24 個後端 + 8 個前端測試檔，含教育引擎完整覆蓋
- ~~結構化日誌~~ — JsonFormatter + 輪轉檔案
- ~~CSS Modules 遷移~~ — 已完成

### 4.4 反向觀察：CodefyUI 已建立的領先優勢

這些是競品都沒做、但 CodefyUI 做得很好的項目，應視為「不可被替代的賣點」：

| 領先項目 | CodefyUI 做了什麼 | 競品狀況 |
|---------|-----------------|---------|
| 步驟追蹤 (Step Trace) | 節點 instrumented 後逐步記錄 + LaTeX 描述 | 全無 |
| 反向梯度健康度 | 自動 retain_grad + grad_health 分類 | 全無（W&B 有 grad histogram，但非節點級即時） |
| 權重持續化 + 互動訓練 | StatefulModuleMixin + NodeStateStore，連按 Run 即可訓練 | 全無 |
| 視覺化 LaTeX 公式 | KaTeX 整合到節點描述 / Step Trace / Tooltip | 全無 |
| Compare Segment | 選 head/tail 高亮路徑 + 跨節點輸出比較 | 全無 |
| Smart 邊路由 | hash-based jitter + 弧線/skip/step | ComfyUI/Langflow 用 React Flow 預設邊 |
| 多入口 Trigger 系統 | 顯式 Start 節點 + 綠色觸發邊 | n8n 有觸發器但非節點 |
| Note Node + 繫結 | 可固定附在某節點的便利貼 | 部分工具有 sticky note 但無繫結 |
| 跨平台一鍵安裝 | uv + cdui + dev.py 三平台統一 | ComfyUI 安裝困難（特別 macOS）；Langflow Docker 為主 |

---

## 5. 目標定位與願景

### 5.1 建議定位（更新版）

> **CodefyUI：可視化 ML 學習與實驗平台 — 看得見每一步張量、梯度、權重變化**
>
> 從教學示範到模型實驗的一站式視覺化工具：每按一次「Run」，使用者就能同時看見模型結構、中間張量、權重變化、梯度健康度——這是市場上沒有任何工具提供的能力。
>
> 長期願景仍是「AI/ML 全流程管線」（訓練 + 推理 + 部署 + LLM 應用），但**短中期應以「教育與可解釋性」為主軸建立護城河**，再以此為基礎擴展到 LLM/RAG 與 MLOps 場景。

### 5.2 為什麼以「教育/可解釋性」為主軸是正確選擇

1. **市場真空**：ComfyUI / Langflow / Kubeflow / MLflow / W&B 都沒有節點級 step trace、grad health、weight persistence。
2. **競品難複製**：要做這個，需要：（a）統一的 BaseNode 抽象 + StatefulModuleMixin、（b）ExecutionContext 旗標式控制、（c）retain_grad 自動化、（d）前端 Inspector 三分頁 + LaTeX 渲染。CodefyUI 已全數擁有，新進入者要追上需要重大重構。
3. **市場需求真實**：
   - 全球 AI/ML 學習者 2025 年估 5,000 萬+（Coursera、Kaggle Learn、HuggingFace 學習者統計）
   - 大學 AI 課程急需互動工具（PyTorch / TF playground 太陽春，Colab notebook 仍偏程式碼）
   - 企業內部訓練：新人 onboarding、跨部門 AI Literacy 課程需要直觀工具
4. **有清晰的成長路徑**：教育用戶 →（其中 ML 工程師背景者）→ 實驗 / 研究用戶 →（少數團隊）→ 商業用戶；類似 Figma 從個人設計師起家、最後賣到企業設計團隊。
5. **AGPL-3.0 + 商業授權雙軌 + 自建社群**：讓個人開發者、小型團隊、教育與研究使用者能以開源方式採用，同時為閉源、SaaS、OEM 與企業部署保留商業授權路徑。

### 5.3 三層次的定位演進策略

```
┌──────────────────────────────────────────────────────────────────┐
│  Layer 3: 全流程 ML 平台（長期，2-3 年）                          │
│  + 團隊協作 + K8s 部署 + 節點市集 + Python SDK + MLflow 整合      │
│  Target: 中小型 ML 團隊 / Edu/Research Lab                       │
├──────────────────────────────────────────────────────────────────┤
│  Layer 2: 視覺化 AI 應用建構（中期，6-12 月）                     │
│  + LLM/Embedding/VectorDB/RAG/Agent 節點 + 部署 endpoint         │
│  Target: AI 應用開發者 / Prototype 工作坊                        │
├──────────────────────────────────────────────────────────────────┤
│  Layer 1: 互動式 ML 學習平台（短期，0-6 月） ← **目前主戰場**     │
│  + 擴大 Step Trace 覆蓋 + 教學模式 + 教師工具 + 內建概念解釋庫    │
│  Target: ML 學生 / 教師 / 自學者 / 課程作者                      │
└──────────────────────────────────────────────────────────────────┘
```

### 5.4 差異化策略（更新版）

相較於競品，CodefyUI 的差異化空間：

| 競品 | 其限制 | CodefyUI 的機會 |
|------|--------|----------------|
| ComfyUI | 僅限圖像生成；無教育性 | **通用 AI/ML + 教育獨特性**，涵蓋 NLP/CV/RL/GenAI |
| Langflow/Flowise | 僅限 LLM/RAG；無模型訓練 | **全流程**，含訓練 + 推理 + 部署，再以教育獨特性吸引長期使用者 |
| Kubeflow | 複雜、K8s 門檻高；視覺化僅輔助 | **視覺化優先 + 漸進複雜度**，從互動學習開始建立習慣 |
| Airflow | 非視覺化優先 | **拖放式設計 + 即時可觀察** |
| MLflow/W&B | 無管線建構；無節點級 trace | **整合管線建構 + 內建 Step Trace / Grad Health** |
| TensorFlow Playground | 過於簡化、僅 toy MLP | **真實 PyTorch 模型 + 同樣直覺的視覺化** |
| Jupyter Notebook | 程式碼為主、視覺化需手寫 | **零程式碼建構 + 立即視覺化** |
| ONNX Tutorials / Netron | 僅顯示模型結構、無執行 | **同時顯示結構 + 執行 + 梯度** |

### 5.5 目標使用者（重排序）

| 使用者層級 | 描述 | 需求 | CodefyUI 現況契合度 |
|-----------|------|------|-------------------|
| ★ **AI/ML 學生** | 大學/研究所學生、自學者、轉職者 | 視覺化理解模型內部、隨手實驗、可分享學習成果 | **高**（Step Trace + Backward Health + LaTeX 已就位） |
| ★ **AI/ML 教師** | 大學教授、線上課程作者、Bootcamp 講師 | 課堂演示、出題、教材製作、學生作業批改 | **中**（缺 Quiz / 教材匯出 / 學生提交） |
| **研究者** | 學術研究、企業 R&D | 快速 prototype、實驗變因比較、論文草圖 | **中-高**（已可用，但跨 run dashboard 弱） |
| **ML 工程師** | 日常模型開發 | 訓練管線、實驗追蹤、模型比較 | **中**（缺 MLflow 整合、缺 GPU 進階管理） |
| **AI 應用開發者** | LLM 應用建構 | RAG 管線、Agent 編排、API 整合 | **低**（無 LLM/Vector DB/Agent 節點） |
| **MLOps 工程師** | 管線自動化 | 排程、監控、部署、CI/CD | **低**（無 K8s/容器/排程） |
| **團隊/企業** | 協作開發 | 共享工作區、權限、版本控制 | **低**（無認證/協作/RBAC） |

### 5.6 推薦的「First 1000 User」聚焦策略

短期不要試圖同時打所有使用者層級。建議選擇：

**主要鎖定：「ML 學習 + 教學」相關使用者群**
- **獲取管道：** GitHub Awesome Lists（ML 視覺化、PyTorch tools）、Reddit r/MachineLearning、HackerNews、知乎/小紅書 ML 學習板塊、大學 PyTorch 課程教案
- **內容行銷：** 30 個短影片「視覺化解釋 X 概念」（如「梯度消失看起來像什麼？」「BatchNorm 真的有用嗎？來看數值差異」），每支 1-2 分鐘
- **教師合作：** 找 2-3 位線上 ML 課程作者試用，提供「免費 + 列名感謝」交換意見
- **學生競賽：** 辦「最佳 CodefyUI 工作流範例」競賽（Kaggle 風格），獎品為 GPU credit / 書籍

**驗證指標：**
- 月活躍使用者（MAU）3 個月內達 500+
- 每位使用者平均 sessions/月 ≥ 4
- GitHub stars 增長率（過去 90 天比較）
- 社群貢獻範例工作流 ≥ 30 個

---

## 6. 建議發展路線圖

> **基於 §5 的「教育為主軸」策略，重排 Phase 1**：原本以 LLM 為 Phase 1 P0，現改為「教育能力擴大化」為 Phase 1 P0，並將 LLM 節點降為 Phase 1 P1。理由是先把領先優勢做深、形成護城河，再橫向擴展生態。

### Phase 0：基礎強化 (Foundation) — **預估 2-3 週**（原 4-6 週，已大部分完成）

**目標：** 修正現有短板，為後續功能打基礎

| 項目 | 說明 | 優先級 |
|------|------|--------|
| ~~執行引擎快取~~ | ~~節點級輸出快取，hash-based 變更偵測，增量執行~~ | ~~P0~~ 已完成 |
| ~~真正的執行取消~~ | ~~asyncio.Task cancellation, 可中斷的節點執行~~ | ~~P0~~ 已完成 |
| ~~錯誤處理增強~~ | ~~節點級重試、失敗繼續、錯誤分支~~ | ~~P0~~ 已完成 |
| ~~並行節點執行~~ | ~~拓撲排序後，同層級節點並行執行~~ | ~~P1~~ 已完成 |
| ~~跨平台安裝器~~ | ~~install.sh / cdui / dev.py / Makefile~~ | ~~P1~~ 已完成 |
| ~~影像/檔案管理 API~~ | ~~routes_images / routes_models + IMAGE_FILE / MODEL_FILE param~~ | ~~P1~~ 已完成 |
| **SQLite 持久層** | **替換 JSON 檔案存儲（Graph、Run、State、使用者設定）** | **P0** |
| ~~完整測試套件~~ | ~~後端單元測試 + 整合測試, 前端元件測試~~ | ~~P1~~ 已完成（24 後端 + 8 前端） |
| **Docker 化** | **Dockerfile + docker-compose（前端 + 後端 + DB）** | **P1** |
| ~~日誌系統~~ | ~~結構化日誌 (JsonFormatter), 取代 print()~~ | ~~P2~~ 已完成 |
| **RunOutputStore 持久層** | **教學分享所需：學生關閉瀏覽器後重連仍能看到 step trace** | **P1** |

### Phase 1：教育能力擴大化 + LLM 第一波 — 預估 6-8 週

**目標：** 把「教育獨特性」從領先變成護城河；同時補上 LLM 缺口避免被 Langflow 完全取代。

**1A. 教育擴大（首要）**

| 項目 | 說明 | 優先級 |
|------|------|--------|
| Step Trace 全面覆蓋 | RNN（LSTM unrolling）/ RL（policy update）/ Tensor Ops 全部 instrument | P0 |
| Backward Visualization 升級 | grad histogram、跨 step 演進曲線、layer-wise 比較 | P0 |
| 教學模式 Onboarding | 第一次開啟自動啟用 verbose、5 步驟引導 tour | P0 |
| 內建概念解釋庫 | 每節點配 concept.md（含 LaTeX 推導、為何重要、常見坑） | P0 |
| 教師工具：教材匯出 | Step Trace + Inspector 結果匯出 Markdown / PDF | P1 |
| Quiz / 互動挑戰 | 載入特定圖，調整參數使 grad 變 healthy 等任務 | P1 |
| TrainingLoop 整合 backward_pass | 訓練中即時 grad health monitoring | P1 |
| GPU 智慧管理 | 自動裝置分配、VRAM 監控、模型卸載（與教育場景脫離 Colab 必要） | P1 |

**1B. LLM 第一波（並行）**

| 項目 | 說明 | 優先級 |
|------|------|--------|
| LLM 推理節點 | OpenAI/Anthropic/HuggingFace/Ollama API 整合 | P1 |
| Embedding 節點 | 文字向量化 (OpenAI/Sentence-Transformers/local) | P1 |
| 向量資料庫節點 | ChromaDB/FAISS 兩種（先做 local，再做 cloud） | P1 |
| RAG 管線節點 | Document Loader/Text Splitter/Retriever/Prompt Template | P1 |
| MLflow 整合 | 實驗追蹤、模型登錄 (透過專用節點) | P2 |
| HuggingFace 整合 | 模型下載/推理/微調節點 | P2 |
| 更多傳統 ML 節點 | scikit-learn 節點 (分類/迴歸/聚類/前處理) | P2 |
| Prompt Engineering 節點 | Template/Chain/Output Parser | P2 |
| 資料視覺化增強 | 互動式圖表 (訓練曲線/混淆矩陣/ROC/特徵重要性) | P2 |

### Phase 2：社群生態 + 專業化功能 — 預估 8-10 週

**目標：** 讓使用者把 CodefyUI 當成可長期投入的工具

| 項目 | 說明 | 優先級 |
|------|------|--------|
| Python SDK | `pip install codefyui` + Python API 操作 | P0 |
| Node Package Manager | 社群節點搜尋/安裝/更新/評分（類似 ComfyUI Manager） | P0 |
| Workflow Marketplace | 範例 / 教學工作流發布 + 一鍵載入 | P0 |
| 使用者認證（基本） | OAuth2/JWT，個人使用者管理（先不做 RBAC） | P1 |
| 工作區與專案 | 多專案管理, 專案級設定 | P1 |
| 工作流版本控制 | Git-like 版本歷史, diff 檢視, 分支/合併 | P1 |
| 排程執行 | Cron 排程、事件觸發、Webhook | P1 |
| AI Agent 節點 | Agent 編排, Tool 使用, Memory, Planning | P1 |
| 模型部署節點 | ONNX 匯出, TorchServe, FastAPI endpoint 生成 | P2 |
| 資料集管理 | 資料版本, 標註, 探索 | P2 |
| Diffusion 模型節點 | SD/SDXL/Flux 整合 (如果定位包含圖像生成) | P2 |
| Python 腳本匯出 | 完整可執行程式碼，非僅骨架 | P2 |

### Phase 3：企業級與雲端 — 預估 10-12 週

**目標：** 適合團隊和企業使用

| 項目 | 說明 | 優先級 |
|------|------|--------|
| 團隊協作 | 即時協作編輯, 評論, 審核 | P0 |
| RBAC 權限系統 | 角色權限, 專案權限, 節點權限 | P0 |
| Kubernetes 部署 | Helm Chart, 分散式執行 | P1 |
| 監控 Dashboard | 執行歷史, 資源使用, 告警 | P1 |
| API Gateway | 將工作流發布為 REST API | P1 |
| 教育版 SaaS | 為學校 / Bootcamp 提供托管環境 | P1 |
| 審計日誌 | 操作記錄, 合規追蹤 | P2 |
| SSO 整合 | SAML, LDAP | P2 |
| 多租戶 | 組織隔離 | P2 |
| 外掛系統 | 前端/後端外掛 API | P2 |

### 6.5 路線圖視覺化（Gantt 概念）

```
Phase 0  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ (2-3 週) ── DB + Docker + Run 持久層
Phase 1A ░░░░████████░░░░░░░░░░░░░░░░░░░░░░░░░░ (6-8 週) ── 教育擴大化（首要）
Phase 1B ░░░░░░░░░░██████░░░░░░░░░░░░░░░░░░░░░░ (與 1A 並行) ── LLM 第一波
Phase 2  ░░░░░░░░░░░░░░░░░░██████████░░░░░░░░░░ (8-10 週) ── 社群生態 + 專業化
Phase 3  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░██████████ (10-12 週) ── 企業級
                          ▲
                          └─ 預期 Phase 1 結束後重新評估市場反應
```

---

## 7. 技術架構建議

### 7.1 執行引擎現況（已完成的升級）

執行引擎已在 2026-04 大幅升級，現有架構：

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
│  │  └─ ExecutionCache (hash-based output cache)          │
│  ├──────────────────────────────────────────────────────  │
│  │  Progress Reporter (WebSocket)                        │
│  │  RunOutputStore (per-run forward/steps/grads)         │
│  └──────────────────────────────────────────────────────  │
│                                                          │
│  ✓ Hash-based 變更偵測（節點 params + 輸入 hash）         │
│  ✓ 並行執行同層級節點                                      │
│  ✓ 可中斷執行（asyncio.Event + Task.cancel）              │
│  ✓ 錯誤恢復（fail_fast / continue / retry）              │
│  ✓ Verbose Step Trace（教育性增量資訊）                   │
│  ✓ 持久化 nn.Module 權重（互動式訓練）                    │
│  ✓ 自動梯度健康度（vanishing / exploding / healthy）      │
│  ☐ GPU Worker Pool / VRAM 感知排程（待做）                │
│  ☐ 執行歷史 DB 持久化（RunOutputStore 仍為記憶體）        │
│  ☐ 分散式執行（Celery / Ray）（長期）                     │
└──────────────────────────────────────────────────────────┘
```

### 7.2 節點系統現況（已升級）

實際 BaseNode 已具備：

```python
class BaseNode(ABC):
    NODE_NAME: ClassVar[str] = ""
    CATEGORY: ClassVar[str] = ""
    DESCRIPTION: ClassVar[str] = ""
    cacheable: ClassVar[bool] = True  # ✓ 已實作（StatefulModuleMixin 設 False）

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]: ...
    @classmethod
    def define_outputs(cls) -> list[PortDefinition]: ...
    @classmethod
    def define_params(cls) -> list[ParamDefinition]: ...

    async def execute(self, inputs, params, *, context=None, progress_callback=None):
        """✓ async + ExecutionContext 已實作"""
        ...

# StatefulModuleMixin（教育用權重持續化）
class StatefulModuleMixin:
    structural_params: ClassVar[tuple[str, ...]]  # 影響 module shape 的 params
    cacheable: ClassVar[bool] = False              # 由 mixin 強制設定

    def build_module(self, params) -> nn.Module: ...        # 子類別實作
    def get_or_build_module(self, context, params): ...     # 透過 NodeStateStore 取/建
```

**仍可改進：**
- `validate_params(cls, params) -> list[str]` 介面尚未存在；目前驗證散落在 execute() 中
- `compute_hash` 由 ExecutionCache 統一處理，未開放給節點自訂（影響特殊類型如自訂 dataclass 輸入）

### 7.3 前端架構現況與待辦

| 現況 | 建議 | 理由 |
|------|------|------|
| ~~Inline styles~~ | ~~CSS Modules~~ | ~~已完成~~ |
| ~~window.prompt/alert~~ | ~~自訂 Modal 元件~~ | ~~已大部分改善（CustomNodeManager / PresetModal）~~ |
| 無路由 | React Router | 多頁面（Editor/Dashboard/Settings/Tutorial） |
| localStorage | IndexedDB + 後端 DB | 大型圖表、可靠性、跨裝置同步 |
| 固定深色主題 | 主題系統（明/暗） | 使用者偏好（教育場景投影機需淺色） |
| 無快捷鍵系統 | 完整快捷鍵框架 | 專業工具必備（特別是教師/演講場景） |
| ~~LaTeX 渲染~~ | ~~MathText (KaTeX)~~ | ~~已完成（描述/Step Trace/Tooltip）~~ |
| ~~Auto-layout~~ | ~~dagre + 多列換行~~ | ~~已完成~~ |
| ~~Smart Edge~~ | ~~SmartDataEdge + TriggerEdge~~ | ~~已完成~~ |
| ~~Note 系統~~ | ~~NoteNode + NoteBindingLines~~ | ~~已完成~~ |
| Inspector 跨節點比較 | 跨 run / 跨 segment dashboard | Compare Segment 已就位，跨 run 待做 |
| 教學模式 Onboarding | 第一次開啟引導、提示式 tour | 教育定位需要 |

### 7.4 後端架構現況與待辦

| 現況 | 建議 | 理由 |
|------|------|------|
| JSON 檔案儲存 | SQLAlchemy + SQLite (預設) → PostgreSQL (規模化) | 查詢、並發、完整性 |
| RunOutputStore 記憶體 | 改 SQLite + 大張量改 Object Storage / Local Disk | 教學分享需要持久化 |
| NodeStateStore LRU | 加上磁碟快照（save/restore），讓教學用權重可分享 | 教師可發布「訓練到 N 步」的 checkpoint |
| 無認證 | FastAPI Security + JWT/OAuth2 | 多使用者、安全性 |
| ~~print() 日誌~~ | ~~結構化日誌（JsonFormatter）~~ | ~~已完成~~ |
| 無任務隊列 | ARQ / Celery + Redis | 長時間訓練、排程 |
| 無 API 版本管理 | API v1 前綴 + 版本策略 | 向後相容（特別是 SDK 出來後） |
| 無設定管理 | 分層設定（env / file / DB） | 部署靈活性 |
| WebSocket 單實例 | Redis pub/sub 抽象 | 水平擴展（Phase 3） |

### 7.5 教育引擎子系統（新增章節，記錄當前架構）

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
          ├─► attach_retain_grad(result, ctx)  ← After each node
          ├─► select_backward_target(ctx)      ← After forward
          ├─► run_backward(loss)
          └─► capture_grads(ctx)               ← Async, writes {port}__grad
                 └─► grad_health(grad) → vanishing/exploding/healthy
                        └─► Frontend BackwardView
```

**Phase 1 升級建議（教育能力擴大化）：**

| 升級 | 說明 |
|------|------|
| StepRecorder.tag(category) | 步驟分類（"forward", "shape_change", "numerical_check"），前端可篩選 |
| StepRecorder.diff_with_previous() | 自動計算與前一步的張量差異，避免使用者手動比較 |
| GradHistogram | 不只回傳 norm/mean/max，回傳完整 bucket 分布 |
| Per-step WebSocket push | 即時把 step trace 推給前端（目前是後端先全跑完才能拉） |
| Concept registry | `nodes/cnn/conv2d_node.py` 旁加 `concept.md`，後端 API 提供 |
| Snapshot API | `POST /api/runs/{run_id}/snapshot` 將 NodeStateStore + RunOutputStore 打包匯出 |

### 7.6 建議的整體架構演進

```
Phase 0-1 (簡單部署):
  Browser → Nginx → Frontend (React)
                  → Backend (FastAPI + SQLite)
                  → Local file storage (uploads/, models/, images/)
                  → (Optional) Redis 任務隊列

Phase 2 (專業部署):
  Browser → Nginx/Traefik
          → Frontend (React, CDN)
          → API Server (FastAPI, 多實例)
          → Worker Pool (GPU nodes — Phase 1B 起需要)
          → PostgreSQL
          → Redis
          → Object Storage (S3/MinIO — 模型/資料集/Run snapshots)

Phase 3 (企業部署):
  Browser → Load Balancer
          → API Gateway (認證/限流)
          → Frontend (CDN)
          → API Cluster (K8s)
          → GPU Worker Cluster (K8s)
          → PostgreSQL (HA)
          → Redis Cluster (含 WebSocket pub/sub)
          → Object Storage
          → Prometheus/Grafana (監控)
          → ELK/Loki (日誌)
```

---

## 8. 風險評估

### 8.1 技術風險

| 風險 | 影響 | 可能性 | 緩解策略 |
|------|------|--------|---------|
| 執行引擎重寫複雜度 | 高 | 高 | 漸進式改造，先加快取再加並行 |
| GPU 管理跨平台問題 | 中 | 高 | 抽象層設計, 針對 CUDA/MPS/CPU 各自實作 |
| 社群節點安全性 | 高 | 中 | 沙盒執行、程式碼審核、簽章機制 |
| 前端效能(大型圖) | 中 | 中 | React Flow 虛擬化、lazy rendering |
| WebSocket 擴縮性 | 中 | 中 | WebSocket 連線管理器、Redis pub/sub |

### 8.2 產品風險

| 風險 | 影響 | 可能性 | 緩解策略 |
|------|------|--------|---------|
| 定位過廣 | 高 | 高 | 先聚焦 2-3 個場景做深，逐步擴展 |
| 與 ComfyUI 直接競爭 | 中 | 中 | 差異化定位（通用 AI/ML vs 圖像生成） |
| 社群建設困難 | 高 | 中 | 提供優秀的節點開發體驗、完善文件 |
| 企業功能分散注意力 | 中 | 中 | Phase 3 才開始企業功能 |

### 8.3 建議的初期聚焦場景

考慮到資源限制，建議先聚焦以下 **2 個場景**：

1. **LLM/RAG 應用建構** — 市場增長最快，與 Langflow/Flowise 競爭
   - 目標：讓使用者用拖放方式建構 RAG 管線和 AI Agent
   - 優勢：CodefyUI 已有訓練管線，可覆蓋 Langflow 做不到的微調場景

2. **ML 模型訓練管線** — 差異化優勢，競品較少
   - 目標：視覺化設計模型、配置訓練、追蹤實驗
   - 優勢：已有的 CNN/RNN/Transformer/RL 節點基礎

---

## 9. 附錄：競品功能對照表

> 重新組織為「核心」「教育/可解釋性」「LLM/AI 應用」「企業/部署」「生態」五大群組，便於識別 CodefyUI 的領先項目（標 ★）。

### 9.1 核心圖形編輯與執行

| 功能 | CodefyUI | ComfyUI | n8n | Langflow | Dify | Kubeflow | Airflow |
|------|----------|---------|-----|----------|------|----------|---------|
| 視覺化圖形編輯 | V | V | V | V | V | 部分 | 部分 |
| 節點拖放 | V | V | V | V | V | X | X |
| 類型安全連線（含 TRIGGER 型別） | V | V | V | V | V | N/A | N/A |
| 增量執行 / 節點快取 | V (hash + dirty tracking) | V | X | X | X | X | X |
| 並行執行同層級節點 | V | V | 部分 | 部分 | 部分 | V | V |
| 錯誤處理 / 重試（三模式） | V | 部分 | V | 部分 | V | V | V |
| 執行取消 | V (asyncio Event) | V | V | V | V | V | V |
| WebSocket 即時進度 | V | V | X | V | V | X | X |
| 多分頁工作區 | V | X | X | X | X | X | X |
| Preset / 子圖 | V (含匯入主編輯器) | V | V | V | V | V | X |
| 自訂節點 + GUI 管理 | V (Custom Node Manager) | V (CLI) | V | V | V | V | V |

### 9.2 ★ 教育與可解釋性（CodefyUI 全面領先）

| 功能 | CodefyUI | ComfyUI | n8n | Langflow | Dify | Kubeflow | Airflow | W&B / MLflow |
|------|----------|---------|-----|----------|------|----------|---------|--------------|
| ★ Verbose Step Trace（節點內部步驟） | V | X | X | X | X | X | X | X |
| ★ Backward Pass 即時顯示 | V | X | X | X | X | X | X | X |
| ★ Gradient Health 分類 | V | X | X | X | X | X | X | 部分 (W&B grad histogram) |
| ★ Weight Persistence（互動訓練） | V | X | X | X | X | X | X | X |
| ★ LaTeX 公式渲染（KaTeX） | V | X | X | X | X | X | X | X |
| ★ Compare Segment（跨節點比較） | V | X | X | X | X | X | X | X |
| 節點概念說明 / 文件 | 描述 + LaTeX | 部分 | 部分 | 部分 | 部分 | X | 部分 | N/A |
| 互動式訓練曲線 | 部分 (ResultsPanel Loss) | X | X | X | X | V | X | V |
| 跨 Run Dashboard | X | X | X | X | X | V | X | V |

### 9.3 LLM / AI 應用

| 功能 | CodefyUI | ComfyUI | n8n | Langflow | Dify |
|------|----------|---------|-----|----------|------|
| LLM 推理節點 (OpenAI/Anthropic/HF) | X | 社群 | V | V | V |
| 本地 LLM (Ollama/llama.cpp) | X | 社群 | V | V | V |
| Embedding 節點 | X | X | V | V | V |
| Vector Database 節點 | X | X | V | V | V |
| RAG 管線（Loader/Splitter/Retriever） | X | X | V | V | V |
| AI Agent 編排 | X | X | V | V | V |
| MCP 協議支援 | X | X | V | V | X |
| Diffusion 模型節點 | X | V | X | X | X |
| 模型微調節點 | 部分（需手動接） | X | X | 部分 | 部分 |

### 9.4 部署與企業

| 功能 | CodefyUI | ComfyUI | n8n | Langflow | Dify | Kubeflow | Airflow |
|------|----------|---------|-----|----------|------|----------|---------|
| ★ 跨平台一鍵安裝 | V (cdui + uv) | 部分 | V | V | V | X | X |
| Docker 部署 | X | V | V | V | V | V | V |
| K8s 部署 | X | 社群 | V | V | V | V | V |
| GPU 管理 | 基本 | 進階（VRAM 自動） | X | X | X | V | X |
| 使用者認證 | X | X | V | V | V | V | V |
| RBAC / 團隊協作 | X | X | V | X | V | V | V |
| 排程執行（Cron） | X | X | V | X | X | V | V |
| 模型部署（API endpoint） | X | X | X | 部分 | 部分 | V | X |
| 監控 Dashboard | 基本（ResultsPanel） | X | V | 部分 | V | V | V |

### 9.5 生態與開發

| 功能 | CodefyUI | ComfyUI | n8n | Langflow | Dify | Kubeflow | Airflow |
|------|----------|---------|-----|----------|------|----------|---------|
| 內建節點數 | 69 | ~50 核心 | 400+ | ~100 | ~80 | ~30 | 100+ Operator |
| 社群節點市場 | X | V (Manager 一鍵裝) | V | V | V | 部分 | V |
| Python SDK / pip install | X | X | X | V | V | V | V |
| REST API | V | V | V | V | V | V | V |
| 資料版本控制 | X | X | X | X | X | V | X |
| 實驗追蹤整合（MLflow/W&B） | X | X | X | X | X | V | X |
| 範例工作流 | 20（CI 驗證） | 2.5M（社群） | 數百 | 數十 | 數十 | 數十 | 數十 |
| i18n 多語言 | V (en/zh-TW) | 社群 | V | V | V | X | V |

### 9.6 授權與規模

| 項目 | CodefyUI | ComfyUI | n8n | Langflow | Dify | Kubeflow | Airflow |
|------|----------|---------|-----|----------|------|----------|---------|
| 開源授權 | AGPL-3.0 + 商業授權 | GPL-3 | Fair-Code | MIT | Apache-2 | Apache-2 | Apache-2 |
| **GitHub Stars** | — | 92.5k | 40k | 100k | 55k | 14k | 40k |
| 主要差異化 | **教育 + 可解釋性** | 圖像生成 | 自動化整合 | LLM/RAG | LLM 應用 | K8s MLOps | DAG 排程 |

### 9.7 競品對照觀察

1. **CodefyUI 在「教育/可解釋性」5 項功能全面領先** — 這 5 項其他工具幾乎全為 X，是真正的差異化。
2. **LLM/AI 應用是最大缺口** — n8n / Langflow / Dify 全面普及，CodefyUI 完全沒有，Phase 1B 必須補上。
3. **企業/部署是中期挑戰** — Docker / 認證 / RBAC / 協作都缺，但教育場景可暫時不需要（學生個人使用）。
4. **節點數量是中位** — 69 個比 ComfyUI 核心多，但遠少於 n8n（400+）；不過 CodefyUI 節點都有 type safety + 教育性，品質 > 數量。
5. **CI 驗證範例是品質優勢** — 19 個範例全 CI 驗證可執行，相對 ComfyUI 社群分享品質參差。
6. **Python SDK 是專業使用者的剛需** — Langflow / Dify / Kubeflow / Airflow 都有，CodefyUI 沒有會擋下程式化整合場景。

---

## 總結

CodefyUI 自上次報告（2026-03-22）至本次更新（2026-04-26）期間經歷了一次**戰略性質變**：從「節點式 ML 工具原型」進化為**具備獨特教育價值的視覺化 ML 學習與實驗平台**。

過去 35 天內，35 次 commit 帶來了三組高密度的能力擴張：

**(1) 教育與可解釋性子系統（戰略護城河）**
- A1 Verbose Step Trace、A2 Weight Persistence、A3 Backward + Grad Health
- 15 個層級節點遷移至 StatefulModuleMixin
- KaTeX LaTeX 渲染、Inspector 三分頁、Compare Segment 路徑高亮
- 這四項功能在所有調研競品中皆不存在，CodefyUI 是市場唯一

**(2) UX 與圖形編輯**
- SmartDataEdge 智慧路由、TriggerEdge 入口系統、Note Node 繫結
- dagre 自動排版（含寬度自動換行）、EmptyCanvasOverlay 範例分組

**(3) 部署與基礎設施**
- 跨平台一鍵安裝（cdui + uv + dev.py + Makefile + install.sh/.ps1）
- 影像/模型檔案 API、IMAGE_FILE / MODEL_FILE param
- 24 個後端測試 + 8 個前端測試
- 19 個 CI 驗證的範例工作流（Model_Architecture / Usage_Example）

### 戰略路徑建議（重排版）

> **核心邏輯：先以「教育獨特性」建立護城河與使用者基數，再橫向擴展 LLM/MLOps/企業能力。**

| Phase | 重點 | 期間 | 主要動作 |
|-------|------|------|---------|
| **Phase 0** | 持久層 + Docker 化 | 2-3 週 | SQLite + Dockerfile + RunOutputStore 持久化 |
| **Phase 1A** | 教育擴大化 | 6-8 週 | Step Trace 全覆蓋 + Backward 升級 + Onboarding + 教師工具 |
| **Phase 1B** | LLM 第一波（並行） | 6-8 週 | LLM/Embedding/Vector DB/RAG 節點 |
| **Phase 2** | 社群生態 + 專業化 | 8-10 週 | Python SDK + Node Marketplace + 認證 |
| **Phase 3** | 企業 + 雲端 | 10-12 週 | RBAC + K8s + 教育版 SaaS |

### 為什麼這次調整：原報告 vs 本次

| 議題 | 原報告（3/22） | 本次（4/26） |
|------|---------------|--------------|
| 戰略主軸 | LLM/RAG + ML 訓練管線並重 | **教育/可解釋性為主軸**，LLM 為平行擴展 |
| 護城河 | 「同時覆蓋 ML 訓練 + LLM」 | 「節點級 Step Trace + Backward Health」（市場唯一） |
| 第一個 1000 用戶 | 未明確 | **ML 學習者 / 教師**（明確獲取管道） |
| 競品威脅 | Langflow 同技術棧威脅大 | Langflow 在 LLM 領先，但無法複製教育能力；可錯位競爭 |
| Phase 1 P0 | LLM/Embedding/RAG | **教育能力擴大化**，LLM 降為 P1 |

### 一句話結論

CodefyUI 已經偶然抵達一個少有人做、且難以複製的位置——**「能讓你看見每一步張量、權重、梯度的視覺化 ML 平台」**。下一步是**把這個位置做深、做得無可取代**，而不是急於追趕 Langflow 在 LLM 場景的領先。教育場景成功後，LLM/MLOps/企業場景的機會自然會跟上。

---

> **附：本次更新涉及的核心檔案**
>
> - 新增模組：`backend/app/core/step_trace.py`、`node_state_store.py`、`stateful_module.py`、`backward_pass.py`、`run_output_store.py`
> - 新增 API：`backend/app/api/routes_images.py`、`routes_examples.py`、`routes_execution_state.py`、`routes_execution_outputs.py`
> - 新增節點：`backend/app/nodes/training/backward_once_node.py`、`io/image_writer_node.py`、`io/image_batch_reader_node.py`、`data/tensor_input_node.py`、`control/start_node.py`
> - 新增前端元件：`frontend/src/components/shared/MathText.tsx`、`Canvas/SmartDataEdge.tsx`、`Canvas/TriggerEdge.tsx`、`Nodes/NoteNode.tsx`、`InspectorPanel/StepTraceView.tsx`、`InspectorPanel/BackwardView.tsx`
> - 新增工具：`scripts/dev.py`、`cdui` / `cdui.cmd`、`install.sh`、`install.ps1`、`Makefile`
