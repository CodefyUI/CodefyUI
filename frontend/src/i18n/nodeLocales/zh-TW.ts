import type { NodeTranslations } from './types';

const zhTW: NodeTranslations = {
  // ── Control ──
  Start: {
    description: '標記執行的進入點。將此節點連接到要執行的腳本的第一個節點，類似於「當綠旗被點擊」積木。',
  },

  // ── Classical (sklearn) ──
  Accuracy: {
    description: '計算預測標籤與真實標籤的分類正確率。會輸出 [0, 1] 的小數、答對數與樣本總數三個欄位 — 讓圖表能直接落在「50%」這種一目了然的數字上。C2 章節的標準收尾節點：線性分類器在同心圓資料上失敗時，這裡就會跳出 ≈ 0.5 的證據。',
  },
  DecisionTreeClassifier: {
    description: 'CART 決策樹（封裝 sklearn）。遞迴切分特徵以最大化純度；教學上最有價值的是 tree_text 輸出 — 直接把學到的 if/else 規則以易讀格式印出來。',
    params: {
      max_depth: '樹的最大深度。0 = 不設限（一路長到全純）。',
      criterion: '切分品質的衡量函式：gini 不純度、entropy 或 log loss。',
      random_state: '同分時打破平手用的種子；可重現的關鍵。',
    },
  },
  KNN: {
    description: 'k 近鄰分類器（封裝 sklearn）。EduKNN 的正式版替身：內部用 KD-tree 索引、可選距離加權與多種度量。要看數學去用 EduKNN，要實際跑資料就用這個。',
    params: {
      n_neighbors: '鄰居數 k。',
      weights: '投票權重：uniform 每個鄰居等權；distance 越近權重越大。',
      metric: '距離度量。minkowski 預設 p=2 等同 euclidean。',
    },
  },
  LinearRegression: {
    description: '普通最小平方法線性迴歸（封裝 sklearn）。封閉解、不需迭代。輸出係數、截距與對 query 集合的預測值。所有迴歸課程的第一行公式。',
    params: {
      fit_intercept: '若為 False，迴歸直線會通過原點（不擬合截距）。',
    },
  },
  LogisticRegression: {
    description: '多類別邏輯斯迴歸（封裝 sklearn）。擬合 softmax 分類器，支援 L2/L1 正則化；I/O 介面與 EduLogisticRegression 相同，可直接抽換。',
    params: {
      C: '正則化強度的倒數（值越小，正則化越強）。',
      max_iter: '求解器最大迭代次數。',
      penalty: '正則化類型。l1 需要 liblinear/saga 求解器；sklearn 會自動挑。',
    },
  },
  MLPClassifier: {
    description: '前饋神經網路分類器（封裝 sklearn MLPClassifier）。一層或多層隱藏層、ReLU/tanh 激活、Adam 優化器。I/O 介面與線性分類器完全相同，所以「同心圓線性分類失敗 → 換 MLP 救回」這個敘事只需換一個節點型別。要看內部步驟用 Edu 系列，要直接拿結果用這個。',
    params: {
      hidden_sizes: '逗號分隔的隱藏層大小。「16,16」代表兩層、每層 16 個神經元。',
      activation: '隱藏層激活函數。「identity」會讓整個網路退化成線性 — 用來示範為什麼需要非線性激活。',
      max_iter: '最大訓練迭代次數（在整個資料集上跑幾輪）。',
      learning_rate_init: 'Adam 的初始學習率。',
      seed: '可重現用的隨機種子。',
    },
  },
  SVMClassifier: {
    description: '支援向量分類器（封裝 sklearn SVC）。求出最大邊界超平面，並用 kernel trick 畫出非線性決策邊界。會輸出 support vectors 給下游視覺化 — 那才是真正決定邊界的點。',
    params: {
      C: '懲罰強度；C 越小邊界越寬，可容忍更多違規。',
      kernel: '核函式：linear 線性、rbf 高斯、poly 多項式、sigmoid。',
      gamma: 'rbf/poly/sigmoid 的核係數。「scale」用 1/(F·var(X))、「auto」用 1/F，也可以填數字字串。',
    },
  },

  // ── CNN ──
  Conv2d: {
    description: '對輸入張量套用 2D 卷積（封裝 nn.Conv2d）。$y[i,j]=\\sum_{k,l} x[i+k,j+l]\\cdot w[k,l] + b$',
    params: {
      in_channels: '輸入通道數',
      out_channels: '輸出通道數',
      kernel_size: '卷積核大小',
      stride: '卷積步幅',
      padding: '兩側的零填充',
    },
  },
  MaxPool2d: {
    description: '對輸入張量套用 2D 最大池化（封裝 nn.MaxPool2d）',
    params: {
      kernel_size: '池化視窗大小',
      stride: '池化視窗步幅',
    },
  },
  BatchNorm2d: {
    description: '對輸入張量套用 2D 批次正規化（封裝 nn.BatchNorm2d）。每通道：$y = \\frac{x - \\mu_C}{\\sqrt{\\sigma_C^2 + \\epsilon}} \\gamma + \\beta$',
    params: {
      num_features: '要正規化的特徵（通道）數量',
    },
  },
  Dropout: {
    description: '對輸入張量套用 Dropout 正則化（封裝 nn.Dropout）',
    params: {
      p: '元素被歸零的機率',
    },
  },
  Activation: {
    description: '對輸入張量套用激活函數',
    params: {
      function: '要套用的激活函數',
    },
  },
  Conv1d: {
    description: '對輸入張量套用 1D 卷積（封裝 nn.Conv1d）',
    params: {
      in_channels: '輸入通道數',
      out_channels: '輸出通道數',
      kernel_size: '卷積核大小',
      stride: '卷積步幅',
      padding: '兩側的零填充',
    },
  },
  ConvTranspose2d: {
    description: '對輸入張量套用 2D 轉置卷積/反卷積（封裝 nn.ConvTranspose2d）',
    params: {
      in_channels: '輸入通道數',
      out_channels: '輸出通道數',
      kernel_size: '卷積核大小',
      stride: '卷積步幅',
      padding: '兩側的零填充',
      output_padding: '輸出形狀的額外大小',
    },
  },
  AvgPool2d: {
    description: '對輸入張量套用 2D 平均池化（封裝 nn.AvgPool2d）',
    params: {
      kernel_size: '池化視窗大小',
      stride: '池化視窗步幅',
      padding: '兩側的零填充',
    },
  },
  AdaptiveAvgPool2d: {
    description: '對輸入張量套用 2D 自適應平均池化，產生固定輸出尺寸（封裝 nn.AdaptiveAvgPool2d）',
    params: {
      output_height: '目標輸出高度',
      output_width: '目標輸出寬度',
    },
  },

  // ── Normalization ──
  LayerNorm: {
    description: '套用層正規化（封裝 nn.LayerNorm）。$y = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\gamma + \\beta$',
    params: {
      normalized_shape: '要正規化的維度形狀（逗號分隔整數）',
      eps: '數值穩定性的 Epsilon',
    },
  },
  GroupNorm: {
    description: '套用群組正規化（封裝 nn.GroupNorm）。用於現代 CNN 架構。',
    params: {
      num_groups: '將通道分成的群組數',
      num_channels: '通道數（必須能被 num_groups 整除）',
    },
  },
  InstanceNorm2d: {
    description: '套用 2D 實例正規化（封裝 nn.InstanceNorm2d）。用於風格轉換和影像生成。',
    params: {
      num_features: '特徵（通道）數',
      affine: '是否使用可學習的仿射參數',
    },
  },
  BatchNorm1d: {
    description: '套用 1D 批次正規化（封裝 nn.BatchNorm1d）。用於 Linear 層之後。',
    params: {
      num_features: '要正規化的特徵數',
    },
  },

  // ── RNN ──
  RNNCell: {
    description: '單步原生 RNN cell：$h_t = \\phi(W_{ih} x_t + W_{hh} h_{t-1} + b)$。封裝 nn.RNNCell。把上一個 cell 的 hidden 輸出接到下一個 cell 的 hidden 輸入，就能手動展開遞迴。',
    params: {
      input_size: '每個時間步的輸入向量維度。',
      hidden_size: '隱藏狀態的維度。',
      nonlinearity: '套用在遞迴輸出上的激活函式。',
      seed: 'W_ih / W_hh / 偏置初始化的隨機種子。',
    },
  },
  LSTM: {
    description: '對輸入序列套用 LSTM 遞迴層（封裝 nn.LSTM）',
    params: {
      input_size: '輸入的預期特徵數',
      hidden_size: '隱藏狀態的特徵數',
      num_layers: '遞迴層數量',
      batch_first: '若為 True，輸入/輸出形狀為 (batch, seq, feature)',
      bidirectional: '若為 True，則為雙向 LSTM',
    },
  },
  GRU: {
    description: '對輸入序列套用 GRU 遞迴層（封裝 nn.GRU）',
    params: {
      input_size: '輸入的預期特徵數',
      hidden_size: '隱藏狀態的特徵數',
      num_layers: '遞迴層數量',
      batch_first: '若為 True，輸入/輸出形狀為 (batch, seq, feature)',
      bidirectional: '若為 True，則為雙向 GRU',
    },
  },

  // ── Transformer ──
  MultiHeadAttention: {
    description: '套用多頭注意力機制（封裝 nn.MultiheadAttention）。核心：$\\text{Attention}(Q,K,V)=\\text{softmax}(\\frac{QK^T}{\\sqrt{d_k}})V$',
    params: {
      embed_dim: '模型的總維度',
      num_heads: '平行注意力頭的數量',
    },
  },
  TransformerEncoder: {
    description: '對輸入張量套用 Transformer 編碼器堆疊',
    params: {
      d_model: '模型維度',
      nhead: '注意力頭的數量',
      num_layers: '編碼器層數',
      dim_feedforward: '前饋網路維度',
    },
  },
  TransformerDecoder: {
    description: '對輸入張量套用 Transformer 解碼器堆疊（含編碼器記憶）',
    params: {
      d_model: '模型維度',
      nhead: '注意力頭的數量',
      num_layers: '解碼器層數',
      dim_feedforward: '前饋網路維度',
    },
  },
  MoELayer: {
    description: '混合專家（Mixture-of-Experts）前饋層。每個 token 由 gate 以 softmax 分數挑出 top-k 個專家，輸出為這 k 個專家輸出的加權和。Switch Transformer、Mixtral、DeepSeek-MoE 都採用這個結構。',
    params: {
      num_experts: '專家 FFN 的數量。',
      top_k: '每個 token 路由到的專家數（會 clamp 在 num_experts 內）。',
      hidden_dim: 'Token 的隱藏維度 H。',
      expert_hidden_dim: '每個專家 FFN 內部的寬度。',
      seed: '初始化的隨機種子，確保可重現。',
    },
  },

  // ── RL ──
  DQN: {
    description: '建立用於強化學習的深度 Q 網路（簡單 MLP）',
    params: {
      state_dim: '狀態空間維度',
      action_dim: '動作空間維度',
      hidden_dim: '隱藏層維度',
    },
  },
  PPO: {
    description: '建立用於強化學習的 PPO Actor-Critic 網路',
    params: {
      state_dim: '狀態空間維度',
      action_dim: '動作空間維度',
      hidden_dim: '隱藏層維度',
    },
  },
  EnvWrapper: {
    description: '建立並封裝 Gymnasium 環境，回傳環境與初始觀測值',
    params: {
      env_name: 'Gymnasium 環境 ID',
    },
  },
  KLDivergence: {
    description: 'KL(p || q) 散度 — RLHF 中用來把策略約束在參考策略附近的正則項。可接受機率或 logits 作為輸入；預設輸出純量，也可以改成 per-sample。',
    params: {
      input_kind: 'p、q 是已經算好的機率，還是尚未經過 softmax 的 logits。',
      reduction: '如何把每個樣本的 KL 聚合起來。batchmean = sum / batch_size，是 RLHF 的預設用法。',
    },
  },
  RewardModel: {
    description: 'RLHF 的獎勵頭。一個小 MLP 把序列打成單一純量分數 — 你會先用人類偏好資料訓練它，再讓 PPO 去最大化它。可接受 [B, H]（每筆一個向量）或 [B, T, H]（取最後一個 token）。',
    params: {
      input_dim: '隱藏狀態的維度 H。',
      hidden_dim: 'MLP 中間層的寬度。',
      seed: '初始化的隨機種子，確保可重現。',
    },
  },

  // ── Data ──
  TensorInput: {
    description: '教學用進入點 — 內嵌張量編輯器，可使用明確值、隨機、零、一或 arange 模式。隨機模式可用 seed 重現。',
    params: {
      shape: '張量形狀，以逗號分隔的整數（例如 \'1,4,4\'）',
      dtype: '資料型別',
      value_mode: '張量填充方式',
      values: '巢狀值列表（當 value_mode=explicit 時使用）',
      seed: '可重現隨機數的種子（當 value_mode=random 時使用）',
    },
  },
  Dataset: {
    description: '載入標準資料集（MNIST、CIFAR10 或 FashionMNIST）',
    params: {
      name: '要載入的資料集',
      split: '資料分割',
      data_dir: '下載/儲存資料集的目錄',
    },
  },
  SyntheticDataset: {
    description: '用 sklearn 即時生成 2D 玩具資料集（同心圓、雙月、blob 群聚或一般 make_classification）。輸出格式與 CSVReader 一致，所以 TrainTestSplit 與下游分類器可以直接接 — 範例圖不再需要綁定 CSV 檔。C2-2 ~ C2-5 的標準資料來源。',
    params: {
      kind: 'circles 同心圓（線性不可分）；moons 雙交錯半月；blobs 等向高斯群聚（線性可分）；classification 通用 sklearn make_classification。',
      n_samples: '要生成的樣本總數。',
      noise: '加在點上的高斯噪聲（僅 circles/moons/classification 用得到）。',
      factor: 'circles 內外圈半徑比，介於 0 與 1 之間。其他 kind 會忽略。',
      centers: 'blob 群聚的中心數（只在 kind=blobs 時使用）。',
      seed: '可重現用的隨機種子。',
    },
  },
  HuggingFaceDataset: {
    description: '從 HuggingFace Hub 載入影像分類資料集（透過 datasets 套件）',
    params: {
      dataset_name: 'HuggingFace Hub 上的 repo id（例：cifar10、ylecun/mnist、uoft-cs/cifar100）',
      subset: '多 config 資料集的 config 名稱（空字串=不指定）',
      split: '資料分割：train/test/validation，亦支援切片語法（如 train[:1000]）',
      image_column: '影像欄位名（不同資料集可能是 image、img、pixel_values）',
      label_column: '標籤欄位名',
      cache_dir: '覆寫 HuggingFace 快取位置（空=用 ~/.cache/huggingface）',
    },
  },
  KaggleDataset: {
    description: '從 Kaggle 下載資料集，並以 ImageFolder 結構載入',
    params: {
      dataset_slug: 'Kaggle dataset 的 owner/slug（例：puneet6060/intel-image-classification）',
      subdir: '下載後資料夾內，包含 class 子資料夾的相對路徑',
      cache_dir: '覆寫 kagglehub 快取位置（空=用預設）',
    },
  },
  DataLoader: {
    description: '將資料集包裝為 DataLoader 以進行批次迭代',
    params: {
      batch_size: '每批次的樣本數',
      shuffle: '每個 epoch 是否隨機打亂資料',
      num_workers: '資料載入的子程序數量',
    },
  },
  Transform: {
    description: '對資料集套用常見影像變換（調整大小、正規化、轉為張量）',
    params: {
      resize: '調整大小維度（0 表示不調整）',
      normalize: '套用正規化（mean=0.5, std=0.5）',
      to_tensor: '將 PIL 影像轉為張量',
    },
  },
  CSVReader: {
    description: '把 CSV 載入為「特徵張量 + 標籤列表」。數值欄位（若有設 include_columns 會再篩選）轉成 [N, F] 的 float32 張量；target_column 指定的欄位則變成下游分類器吃的字串標籤列表。',
    params: {
      path: 'CSV 檔案路徑（絕對路徑或相對於後端工作目錄）。',
      target_column: '標籤欄位名稱（選填）。留空表示沒有標籤、純資料載入。',
      include_columns: '要保留的特徵欄位（逗號分隔，選填）。留空表示「除了 target 之外所有數值欄位」。',
      skip_header: 'True 代表第一列是欄位名稱；False 則自動把欄位命名為 0、1、2…',
    },
  },
  ColumnSelector: {
    description: '從 2D 張量中挑出部分欄位。設 indices 用位置選；設 names（需同時接 columns 輸入）則用名稱選。兩者同時設定時 names 優先。',
    params: {
      indices: '以逗號分隔的欄位索引，例：「0,2,3」。當 names 為空時使用。',
      names: '以逗號分隔的欄位名稱。一旦設定就會蓋過 indices，並且需要連上 columns 輸入。',
    },
  },
  Normalize: {
    description: '沿指定軸縮放張量。zscore = $(x-\\mu)/\\sigma$、minmax = $(x-\\min)/(\\max-\\min)$、unit_norm = $x/\\|x\\|_2$。表格資料逐欄正規化用 axis=0；逐樣本正規化用 axis=1。',
    params: {
      mode: '正規化方法。',
      axis: '計算統計量的軸。0 = 逐欄，1 = 逐列。',
    },
  },
  TrainTestSplit: {
    description: '把 (features, labels) 切成訓練集與測試集，封裝 sklearn.train_test_split。開啟 stratify 會讓兩邊保持相同的類別比例 — 對不平衡資料是必備的。',
    params: {
      test_size: '保留作為測試集的樣本比例，必須介於 (0, 1)。',
      seed: '隨機洗牌的種子，方便可重現。',
      stratify: '是否在兩邊保留每個類別的比例（分層抽樣）。',
    },
  },

  // ── Training ──
  Optimizer: {
    description: '建立優化器用於模型參數',
    params: {
      type: '優化器演算法',
      lr: '學習率',
      weight_decay: '權重衰減（L2 懲罰）',
    },
  },
  Loss: {
    description: '建立損失函數',
    params: {
      type: '損失函數類型',
    },
  },
  TrainingLoop: {
    description: '執行訓練迴圈，支援驗證、早停、學習率排程和梯度裁剪',
    params: {
      epochs: '訓練 epoch 數量',
      device: '訓練裝置',
      early_stopping_patience: '驗證損失未改善 N 個 epoch 後停止（0 = 停用）',
      grad_clip_norm: '最大梯度範數裁剪（0 = 停用）',
    },
  },
  BackwardOnce: {
    description: '標記張量為 autograd 反向傳播的目標，供 Backward 檢視器使用。僅在工具列啟用 Backward 模式時執行。反向傳播目標：$\\mathcal{L} = \\sum(\\text{input})$（合成純量）。',
  },

  LRScheduler: {
    description: '建立學習率排程器',
    params: {
      type: '排程器類型',
      step_size: 'StepLR 的步長',
      gamma: '衰減因子',
      T_max: 'CosineAnnealingLR 的最大迭代數',
      max_lr: 'OneCycleLR 的最大學習率',
      total_steps: 'OneCycleLR 的總訓練步數',
    },
  },

  // ── IO ──
  ImageReader: {
    description: '從磁碟讀取影像檔案，輸出為張量 (C, H, W)，值域 [0, 1]',
    params: {
      path: '選擇已上傳的影像，或上傳新檔案',
      mode: '載入影像的色彩模式（L = 灰階）',
      resize: '縮放為 (resize, resize) 正方形（0 = 不縮放）',
    },
  },
  ImageWriter: {
    description: '將張量儲存為影像檔案（PNG、JPEG 等）',
    params: {
      path: '輸出檔案路徑',
      format: '影像格式',
    },
  },
  ImageBatchReader: {
    description: '從目錄讀取所有影像，堆疊為批次張量 (N, C, H, W)',
    params: {
      directory: '包含影像檔案的目錄',
      pattern: '檔案比對模式（如 *.png、*.jpg）',
      resize: '將所有影像調整為此正方形大小（批次處理必需）',
      max_images: '最大載入影像數（0 = 全部）',
      mode: '色彩模式',
    },
  },
  FileReader: {
    description: '讀取文字或 CSV 檔案，輸出內容為字串或張量（數值 CSV）',
    params: {
      path: '檔案路徑',
      mode: '讀取方式',
      encoding: '文字編碼',
      csv_header: 'CSV 是否有標頭列（載入為張量時跳過）',
    },
  },

  ModelSaver: {
    description: '將模型權重（state_dict）儲存為 .pt/.pth/.safetensors 檔案',
    params: {
      path: '輸出檔案路徑（.pt、.pth 或 .safetensors）',
      save_mode: '儲存模式：state_dict（推薦）或完整模型',
      format: '檔案格式：pytorch（.pt/.pth）或 safetensors（.safetensors）',
    },
  },
  ModelLoader: {
    description: '從 .pt/.pth/.safetensors 檔案載入模型權重，或載入完整的已儲存模型',
    params: {
      path: '權重檔案路徑（.pt、.pth 或 .safetensors）',
      load_mode: '載入模式：state_dict（需要模型輸入）或完整模型',
      device: '載入權重的裝置',
      strict: '是否嚴格要求 state_dict 中的鍵值匹配（僅 state_dict 模式）',
    },
  },
  CheckpointSaver: {
    description: '儲存完整訓練檢查點（模型 + 優化器 + epoch + 損失值），用於稍後恢復訓練',
    params: {
      path: '輸出檢查點檔案路徑',
      epoch: '要儲存在檢查點中的當前 epoch 數',
    },
  },
  CheckpointLoader: {
    description: '載入訓練檢查點以恢復訓練（恢復模型 + 優化器 + epoch）',
    params: {
      path: '檢查點檔案路徑',
      device: '載入的目標裝置',
    },
  },
  Inference: {
    description: '對已訓練的模型執行推論（前向傳播）。自動設為 eval 模式並停用梯度。',
    params: {
      device: '執行推論的裝置',
    },
  },

  // ── Data Flow ──
  Switch: {
    description: '根據選擇器索引選取多個輸入之一。純資料流條件選擇：所有輸入都會被求值，選擇器決定轉發哪一個。',
  },
  Map: {
    description: '對列表中的每個元素套用子圖（預設模組）。函數式批次處理。',
    params: {
      subgraph: '要套用到每個元素的子圖/預設模組名稱',
    },
  },
  Reduce: {
    description: '將列表聚合為單一結果。支援 sum、mean、min、max、concat、stack、first、last。',
    params: {
      operation: '聚合運算',
      dim: 'concat/stack 運算的維度',
    },
  },

  // ── Tensor Operations ──
  Permute: {
    description: '排列（重新排序）張量的維度',
    params: {
      dims: '新的維度順序（逗號分隔整數）',
    },
  },
  Squeeze: {
    description: '移除大小為 1 的維度',
    params: {
      dim: '要壓縮的維度（-1 表示全部）',
    },
  },
  Unsqueeze: {
    description: '在指定位置新增大小為 1 的維度',
    params: {
      dim: '要插入的維度位置',
    },
  },
  Add: {
    description: '兩個張量的逐元素相加（支援廣播）',
    params: {
      alpha: 'tensor_b 的乘數：a + alpha * b',
    },
  },
  Multiply: {
    description: '兩個張量的逐元素相乘（支援廣播）',
  },
  MatMul: {
    description: '兩個張量的矩陣乘法（torch.matmul）',
  },
  Mean: {
    description: '沿指定維度計算張量的平均值',
    params: {
      dim: '要縮減的維度（逗號分隔整數）',
      keepdim: '是否保留被縮減的維度',
    },
  },
  Softmax: {
    description: '沿指定維度套用 Softmax：$\\text{softmax}(x_i) = \\frac{e^{x_i}}{\\sum_j e^{x_j}}$。為數值穩定，先減去 $\\max(x)$ 再取指數。',
    params: {
      dim: '要套用 Softmax 的維度',
    },
  },
  Split: {
    description: '沿指定維度將張量切分為多個區塊',
    params: {
      chunks: '要切分的區塊數',
      dim: '要切分的維度',
    },
  },
  Stack: {
    description: '沿新維度堆疊兩個張量',
    params: {
      dim: '要堆疊的維度',
    },
  },
  TensorCreate: {
    description: '建立填充零、一、隨機值或常數的張量',
    params: {
      shape: '張量形狀（逗號分隔整數）',
      fill: '填充方法',
      value: '填充值（僅 full 模式）',
      requires_grad: '張量是否需要梯度',
    },
  },

  // ── Utility ──
  Print: {
    description: '將輸入值印出到主控台並傳遞',
    params: {
      label: '標籤前綴',
    },
  },
  Reshape: {
    description: '將張量重塑為指定形狀',
    params: {
      shape: '目標形狀，以逗號分隔的整數（例如 \'-1,784\'）',
    },
  },
  Concat: {
    description: '沿指定維度串接兩個張量',
    params: {
      dim: '串接的維度',
    },
  },
  Visualize: {
    description: '將資料（張量、損失值等）生成 matplotlib 圖表，輸出為 base64 編碼的 PNG',
    params: {
      title: '圖表標題',
      plot_type: '要生成的圖表類型',
    },
  },

  Flatten: {
    description: '展平張量的維度：nn.Flatten(start_dim, end_dim)',
    params: {
      start_dim: '開始展平的維度',
    },
  },
  Linear: {
    description: '全連接（密集）層：$y = xW^T + b$。封裝 nn.Linear(in_features, out_features)。',
    params: {
      in_features: '輸入特徵大小',
      out_features: '輸出特徵大小',
    },
  },
  SequentialModel: {
    description: '從 JSON 層列表建構 nn.Sequential 模型',
    params: {
      layers: '層定義的 JSON 陣列',
    },
  },
  Embedding: {
    description: '可學習的嵌入查表（封裝 nn.Embedding）。將整數索引對應到可訓練權重矩陣 $W$ 的列：$E[i] = W[i, :]$。如需預訓練詞向量（GloVe 等），請改用 LLM 分類下的 `WordVector` 節點。',
    params: {
      num_embeddings: '詞彙表大小',
      embedding_dim: '每個嵌入向量的維度',
      padding_idx: '填充 token 的索引（-1 表示無）',
    },
  },

  TextInput: {
    description:
      '純文字輸入點。在節點本體的多行 textarea 中打字，輸出 STRING 可以接到 Tokenizer、WordVector 或任何吃 STRING 的輸入埠。對之後 RAG 的 DocumentInput 是同一條路。',
    params: {
      value: '多行文字。可以拖曳 textarea 右下角調整大小。',
    },
  },

  // ── LLM ──
  Tokenizer: {
    description:
      '把文字切成 LLM 看得懂的整數 token。不同家族用不同演算法 — BPE（GPT）、WordPiece（BERT）、SentencePiece（Llama、T5）— 同一段文字會被切成不同的樣子。',
    params: {
      family: 'Tokenizer 家族。tiktoken 完全離線可跑 cl100k/o200k/p50k/gpt2；其餘會在第一次使用時從 HuggingFace 下載 tokenizer.json。',
      text: '要切分的文字。當沒有 `text` 輸入連線時使用此欄位。',
      show_special_tokens: '是否輸出 tokenizer 的特殊 token（BOS/EOS/CLS/SEP/...）。',
    },
  },
  WordVector: {
    description:
      '為每個輸入單字查找預訓練向量。預訓練嵌入會把語意相近的字放在一起，所以 $king - man + woman \\approx queen$。預設 `demo-16d` 後端隨安裝附帶；`glove-*` 後端會在第一次使用時下載真實 GloVe 向量。',
    params: {
      backend:
        '向量來源。demo-16d 是手工打造的玩具詞彙、完全離線可跑；glove-* 會在第一次使用時下載真實 GloVe 向量；minilm-sentence-384d 需要安裝 [llm-sentence] 額外相依套件。',
      words: '以空白或逗號分隔的單字列表。當沒有 `tokens` 輸入連線時使用此欄位。',
      normalize: '對每個向量做 L2 正規化。下游若要用點積算 cosine similarity，請打開此選項。',
      keep_oov: '對詞彙表外的字輸出零向量，而不是直接略過。',
    },
  },
  EmbeddingScatter: {
    description:
      '把高維嵌入投影到 2D 來「看見」嵌入空間的幾何結構。語意相近的字會聚成一群。PCA 是線性、決定性、快；t-SNE 是非線性、會更好保留局部鄰域結構，但每次跑出來的版面都略有不同。',
    params: {
      method: 'PCA：線性、決定性、快。t-SNE：非線性、保留局部鄰域結構。',
      perplexity: '只在 t-SNE 使用 — 局部親和模型的鄰域大小。',
      seed: '隨機種子（給 t-SNE）。同樣的種子會得到一樣的版面。',
    },
  },
  CosineSimilarity: {
    description:
      '計算每個 query 與每個 key 之間的 cosine similarity。對單位向量輸入這就是點積；非單位向量會自動正規化。輸出整個相似度矩陣以及每個 query 的 top-k key — 這就是 RAG 中向量檢索的核心。',
    params: {
      top_k: '每個 query 要回傳的最相似 key 數量。',
      exclude_self_words:
        '要從 top-k 排除的標籤（以逗號分隔）。在類比示範中很有用：設成 "king,man,woman" 可以讓 top-1 直接顯示 queen。',
    },
  },
  PositionalEncoding: {
    description: '把位置資訊加到 token 嵌入上。sinusoidal 用 Vaswani et al. (2017) 的公式 PE(pos, 2i)=sin(pos/10000^(2i/d))；learnable 則回傳種子可控的亂數樣式（同樣種子會得到一樣結果）。',
    params: {
      mode: 'sinusoidal = Vaswani 公式；learnable = 種子可控的隨機初始化。',
      max_len: '支援的最大序列長度。輸入超過此值會直接報錯。',
      seed: 'learnable 模式的隨機種子。sinusoidal 會忽略此值。',
    },
  },
  AttentionMask: {
    description: '產生布林注意力遮罩（True 表示被擋）。causal 會擋掉未來位置（GPT 風格 decoder）；padding 會擋掉值等於 pad_token 的欄位，避免注意力洩漏到填補欄位。',
    params: {
      mode: 'causal：擋掉嚴格未來的位置（decoder 風格）。padding：擋掉值與 pad_token 相同的欄位。',
      pad_token: '視為填補的字串符號，只在 padding 模式下使用。',
    },
  },
  AttentionHeatmap: {
    description: '純視覺化節點：原樣轉送 attention 權重，同時把它暴露給熱圖視圖。可以串在任何 attention 節點（教學版或正式版）的 weights 輸出後面，不會改變下游圖的結構。',
    params: {
      head_index: '若權重是 per-head 形式（[H,seq,seq] 或 [B,H,seq,seq]），可指定顯示哪一個 head。-1 代表保留全部 head 並排顯示。',
      colormap: '熱圖視覺化用的色階（僅前端使用，後端會忽略）。',
    },
  },

  // ── Diffusion ──
  GaussianNoise: {
    description: '產生獨立同分布的高斯噪聲 $\\epsilon \\sim \\mathcal{N}(\\mu, \\sigma^2)$。把張量接到 shape_ref 就會自動跟隨上游形狀（diffusion 中 x_0 與 noise 配對的標準作法）；否則就讀 shape 參數。可指定種子確保可重現。',
    params: {
      shape: '逗號分隔的維度，例：「1,3,32,32」。當 shape_ref 沒接時才會用。',
      mean: '高斯分布的平均值，預設 0（標準常態）。',
      std: '標準差，預設 1（標準常態）。',
      seed: '隨機種子；同樣的種子會得到相同的噪聲。',
    },
  },
  Lerp: {
    description: '線性內插：$\\alpha\\,a + (1-\\alpha)\\,b$。$\\alpha=1$ 時等於 $a$；$\\alpha=0$ 時等於 $b$。可作為 diffusion 前向公式的教學替身。',
    params: {
      alpha: '內插權重（0..1）。只在沒接 alpha 輸入時才會用此參數。',
    },
  },
  TimestepEmbedding: {
    description: '把 diffusion 的時間步 $t$ 編碼成可餵給 U-Net 各 block 的向量。沿用 Vaswani 風格的正弦頻率組合，後面接 Linear→SiLU→Linear，是 DDPM 的標準配方。',
    params: {
      embed_dim: '時間向量的維度，必須是偶數（sin/cos 各半）。',
      max_period: '頻率組中最大的週期 — 控制能分辨多少個不同的時間步。',
      seed: '投影層初始化的隨機種子。',
    },
  },
  Upsample: {
    description: '純粹的空間 upsample（使用 F.interpolate，沒有可學習權重）。預設把空間維度放大為 2 倍。U-Net 解碼路徑若不想讓 upsample 也跟著學習，就用這個（相較之下 ConvTranspose2d 會學）。',
    params: {
      mode: '插值方式。nearest=直接複製像素、bilinear=雙線性平滑、area=平均（適合做 downsample）。',
      scale_factor: '空間維度的縮放倍數。2.0 放大兩倍、0.5 縮成一半。',
    },
  },
  DiffusionUNet: {
    description: '把整個玩具版 diffusion U-Net 封裝成單一節點。輸出一個 nn.Module，輸入 $(x, t)$ 後會回傳形狀與 x 相同的「預測噪聲」。可串到 DDPMSampler 跑反向 diffusion。若想看到架構被一塊塊明確接起來，可以改用 `Mini-UNet-Expanded` preset。',
    params: {
      in_channels: '噪聲輸入的通道數（RGB 為 3，常見 SD latent 為 4）。',
      base_channels: '經過 stem 後的通道數。每一層會以對應的 channel_mult 倍率相乘。',
      channel_mult: '每一層的通道倍率，以逗號分隔；長度決定深度（下行區塊數 + bottleneck）。',
      time_emb_dim: '時間步嵌入的維度，必須是偶數。',
      num_groups: '每個 ResBlock 內部 GroupNorm 的分組數，必須能整除所有層的通道數。',
      seed: '所有權重初始化的隨機種子。',
    },
  },
  DDPMSampler: {
    description: '執行反向 DDPM 去噪。會依照排程逐步呼叫 `model(x_t, t)` 預測噪聲，再套用 DDPM 更新公式。整個反向迴圈封在節點內部，圖才能維持無環 — 開啟 verbose 後可在 step trace 看到中間軌跡。',
    params: {
      num_steps: '反向 diffusion 的步數。步數越多軌跡越平滑，但也越慢。',
      schedule: '噪聲排程。linear 是原版 DDPM；cosine（Nichol & Dhariwal 2021）在接近資料的區域噪聲增加得更慢。',
      beta_start: '線性排程的起始 variance。cosine 模式會忽略此值。',
      beta_end: '線性排程的結束 variance。cosine 模式會忽略此值。',
      seed: '取樣時每一步加入的高斯噪聲 z 所使用的種子。',
    },
  },

  // ── Custom ──
  AddScalar: {
    description: '將純量值加到張量上（自訂節點範例）',
    params: {
      value: '要加的純量值',
    },
  },
};

export default zhTW;
