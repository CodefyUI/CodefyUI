# CodefyUI

[![en](https://img.shields.io/badge/Language-English-blue)](../README.md)

視覺化、節點式的深度學習管線建構工具。透過拖曳節點到畫布上，連接成 DAG，直接在瀏覽器中設計 CNN、RNN、Transformer 和 RL 架構並執行管線。

![CodefyUI 截圖](../Assets/UI-zh-TW.png)

## 功能特色

- **視覺化圖形編輯器** — 拖放節點、型別安全的連線、即時驗證
- **63 個內建節點**，涵蓋 12 大類別（CNN、RNN、Transformer、RL、資料、資料流、訓練、IO、控制、工具、正規化、張量運算）
- **教學檢視器 (Teaching Inspector)** — 記錄每個節點的完整輸出，在右側面板對照輸入與輸出張量的差異；用 **段落比較** 功能以淺橘色泡泡包住一段子圖，只比較 head 的輸入與 tail 的輸出。搭配新的 `TensorInput` 節點（可在網頁直接編輯格子）餵入資料，看資料逐節點變化
- **預設模組系統** — 內建模型模板快速開始；可將子圖匯出為可重用的預設模組
- **多分頁工作區** — 多個獨立畫布，各自擁有獨立的執行環境
- **WebSocket 即時執行** — 即時顯示每個節點的進度，Print 節點的輸出會顯示在執行紀錄面板
- **部分重新執行** — 髒節點追蹤：僅重新執行已變更的節點及其下游依賴
- **快速搜尋節點** — 在畫布上雙擊開啟即時搜尋面板，快速新增節點與預設模組
- **自訂節點管理器** — 上傳、啟用/停用、刪除自訂節點的 GUI 介面
- **模型檔案管理** — 透過 REST API 上傳、列出、刪除模型權重檔（.pt、.pth、.safetensors、.ckpt、.bin）
- **CLI 圖形執行器** — 使用 `run_graph.py` 直接從命令列執行 graph.json
- **結果面板** — 分頁面板（執行紀錄 / 訓練），可調整大小，包含即時 loss 圖表
- **多語言支援** — 英文與繁體中文，使用響應式 `rem` 字型大小
- **自動儲存** — 所有分頁自動存入 `localStorage`；支援匯入/匯出 graph JSON 檔案
- **深色主題** — 完整的深色 UI，類別以顏色區分

## 快速開始

**一行指令安裝**：

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/treeleaves30760/CodefyUI/main/install.sh | bash
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/treeleaves30760/CodefyUI/main/install.ps1 | iex"
```

自動安裝缺少的依賴（git、Node.js、pnpm、uv；Python 由 uv 提供，不需要額外安裝），並把 `cdui` launcher 加到你的 PATH。安裝完成後，**請重新開啟 terminal**，然後在任何目錄執行：

```bash
cdui dev
```

開啟 [http://localhost:5173](http://localhost:5173)。前端會將 API/WS 請求代理到後端的 `:8000` 埠。

| 指令 | 說明 |
|------|------|
| `cdui install` | 安裝 backend + frontend 依賴 |
| `cdui update` | 拉取最新 `main` 並重新安裝依賴 |
| `cdui dev` | 啟動 backend :8000 + frontend :5173 |
| `cdui stop` | 停止所有服務 |
| `cdui test` | 執行 backend 測試 |
| `cdui clean` | 移除虛擬環境與 node_modules |
| `cdui uninstall` | clean + 移除 PATH 上的 launcher |

> `cdui` 是 install 腳本放到 `~/.local/bin/cdui` 的輕量 launcher（Windows 為 `cdui.cmd`）。若你還沒重開 terminal，可改用絕對路徑：`~/CodefyUI/cdui dev`。`python scripts/dev.py <cmd>` 也一樣能用——`dev.py` 會自動切換到 venv 的 Python。

> 以上快速開始假設使用 **NVIDIA 顯卡 + CUDA 12.4**。若使用 CPU、Apple Silicon、AMD 或需要更詳細的疑難排解，請參考[完整安裝指南](./SETUP_zh-TW.md)。

### CLI 執行

無需啟動伺服器，直接從命令列執行 graph：

```bash
cd backend
python run_graph.py ../examples/Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json
python run_graph.py ../examples/Model_Architecture/ResNet-SkipConnection-CNN/graph.json --validate-only
```

## 架構

```
frontend/   React 19 · TypeScript · React Flow 12 · Zustand 5 · Vite 6
backend/    Python 3.10+ · FastAPI · PyTorch
```

| 原則 | 說明 |
|------|------|
| **後端權威** | `GET /api/nodes` 回傳所有節點定義。後端新增節點後 UI 自動出現。 |
| **單一 BaseNode 元件** | 一個 React 元件渲染所有節點類型，由後端定義參數化。 |
| **WebSocket 執行** | `ws://host/ws/execution` 串流每個節點的狀態。REST 處理圖表 CRUD。 |
| **拓撲排序執行** | 使用 Kahn 演算法進行 DAG 排序 + 循環偵測。支援獨立節點的平行執行。 |

## 內建節點

| 類別 | 節點 | 數量 |
|------|------|------|
| **CNN** | Conv2d、Conv1d、ConvTranspose2d、MaxPool2d、AvgPool2d、AdaptiveAvgPool2d、BatchNorm2d、Dropout、Activation | 9 |
| **RNN** | LSTM、GRU | 2 |
| **Transformer** | MultiHeadAttention、TransformerEncoder、TransformerDecoder | 3 |
| **RL** | DQN、PPO、EnvWrapper | 3 |
| **資料** | Dataset、DataLoader、Transform、HuggingFaceDataset、KaggleDataset、TensorInput | 6 |
| **資料流** | Map、Reduce、Switch | 3 |
| **訓練** | Optimizer、Loss、TrainingLoop、LRScheduler、SequentialModel | 5 |
| **IO** | ImageReader、ImageWriter、ImageBatchReader、FileReader、CheckpointSaver、CheckpointLoader、ModelLoader、ModelSaver、Inference | 9 |
| **控制** | Start | 1 |
| **工具** | Print、Reshape、Concat、Flatten、Linear、Visualize、Embedding | 7 |
| **正規化** | BatchNorm1d、LayerNorm、GroupNorm、InstanceNorm2d | 4 |
| **張量運算** | Add、MatMul、Mean、Multiply、Permute、Softmax、Split、Squeeze、Stack、TensorCreate、Unsqueeze | 11 |

## 範例

預建的範例工作流程位於 `examples/`：

| 類別 | 範例 |
|------|------|
| **模型架構** | ResNet、ConvNeXt、EfficientNet、UNet、ViT、SwinTransformer、BERT、GPT、LLaMA、DiT、LSTM TimeSeries、BiGRU SpeechRecognition、Seq2Seq Attention、DQN Atari、PPO Robotics |
| **使用範例** | CNN-MNIST 訓練、CNN-MNIST 推論 |

## 教學檢視器（Teaching Inspector）

CodefyUI 可作為互動式教材 — 讓學生看到資料流過每一個節點的真實張量變化。

1. 把 **TensorInput** 節點（資料類別）拖進畫布。把 `value_mode` 設成 `explicit`，直接在內嵌格子編輯器裡填入想餵給管線的數值。
2. 用任意張量運算節點串起來（例如 `Reshape → Softmax → Print`）。
3. **從節點面板拖一個 `Start` 節點到畫布，把它右邊的 trigger 輸出（菱形 handle）連到你想要開始執行的第一個節點（通常就是 `TensorInput`）。** 少了 Start → 起始節點的 trigger 連線，圖會被當作草稿，按「執行」會出現「尚未定義開始節點」的錯誤提示。只有從 Start 可到達的節點才會被執行。
4. 確認工具列的 **錄製 ON**，然後按 **執行**。每個完成的節點的完整輸出會被保存在伺服器記憶體中，以該次執行的 run id 當作索引。
5. 點任一節點 — 右側 **檢視器** 面板會抓取該節點的輸入與輸出，上下堆疊顯示 shape、dtype、min/max/mean 與實際值。數值變動的格子會以熱力色標示。
6. Shift 選兩個節點 → 按 **段落比較**，畫布上會以淺橘色泡泡把這段包起來並加上 **HEAD** / **TAIL** 標籤，檢視器切換成只顯示頭部輸入與尾部輸出。
7. 跑重度訓練前想省資源可以先按 **錄製 OFF** — 已經記錄過的 run id 仍然可以查閱，直到伺服器重啟。

被保存的資料是伺服器當下記憶體（LRU，最近 20 次 run）。段落標記會跟著 graph JSON 一起儲存。

## 自訂節點

將 `.py` 檔案放入 `backend/app/custom_nodes/`，繼承 `BaseNode`：

```python
from app.core.node_base import BaseNode, DataType, PortDefinition

class MyNode(BaseNode):
    NODE_NAME = "MyNode"
    CATEGORY = "Custom"
    DESCRIPTION = "自訂節點"

    @classmethod
    def define_inputs(cls):
        return [PortDefinition(name="input", data_type=DataType.TENSOR)]

    @classmethod
    def define_outputs(cls):
        return [PortDefinition(name="output", data_type=DataType.TENSOR)]

    def execute(self, inputs, params):
        return {"output": inputs["input"]}
```

透過 `POST /api/nodes/reload` 或工具列的 **重新載入節點** 按鈕進行熱重載。也可以使用 **自訂節點管理器** GUI 上傳、啟用/停用和刪除自訂節點。

## 快捷鍵

| 操作 | 按鍵 |
|------|------|
| 刪除節點 | `Delete` |
| 多選 | `Shift` + 點擊 |
| 快速新增節點 | 雙擊畫布 |
| 重新命名節點 | 右鍵 → 重新命名 |
| 複製節點 | 右鍵 → 複製 |
| 復原 | `Ctrl/Cmd` + `Z` |
| 重做 | `Ctrl/Cmd` + `Shift` + `Z` / `Ctrl/Cmd` + `Y` |
| 複製節點 | `Ctrl/Cmd` + `C` |
| 貼上節點 | `Ctrl/Cmd` + `V` |
| 自動排版 | `Shift` + `L` |
| 顯示快捷鍵 | `?` |

## API 端點

| 端點 | 方法 | 說明 |
|------|------|------|
| `/api/nodes` | GET | 列出所有節點定義 |
| `/api/nodes/{node_name}` | GET | 取得單一節點定義 |
| `/api/nodes/reload` | POST | 熱重載所有節點 |
| `/api/presets` | GET | 列出預設模組定義 |
| `/api/presets/{name}` | GET | 取得單一預設模組定義 |
| `/api/presets/create` | POST | 從選取的節點建立新預設模組 |
| `/api/graph/validate` | POST | 驗證圖形 |
| `/api/graph/save` | POST | 儲存圖形 |
| `/api/graph/load/{name}` | GET | 載入已儲存的圖形 |
| `/api/graph/list` | GET | 列出已儲存的圖形 |
| `/api/graph/export` | POST | 匯出圖形為 Python 腳本 |
| `/api/examples/list` | GET | 列出範例圖形 |
| `/api/examples/load` | GET | 載入範例圖形 |
| `/api/custom-nodes` | GET | 列出自訂節點 |
| `/api/custom-nodes/upload` | POST | 上傳自訂節點 |
| `/api/custom-nodes/toggle` | POST | 啟用/停用自訂節點 |
| `/api/custom-nodes/{filename}` | DELETE | 刪除自訂節點 |
| `/api/execution/outputs/{run_id}` | GET | 列出該次 run 捕獲的所有埠口 |
| `/api/execution/outputs/{run_id}/{node_id}/{port}` | GET | 取得該節點某埠口的完整張量（可用 `?slice=0,:,:` 切片） |
| `/api/execution/outputs/{run_id}` | DELETE | 清除該次 run 的捕獲資料 |
| `/api/models` | GET | 列出已上傳的模型檔案 |
| `/api/models/upload` | POST | 上傳模型權重檔 |
| `/api/models/{filename}` | DELETE | 刪除模型檔案 |
| `/ws/execution` | WebSocket | 即時圖形執行 |

## 測試

```bash
cd backend
source .venv/bin/activate
pytest tests/ -v
```

## 授權

MIT
