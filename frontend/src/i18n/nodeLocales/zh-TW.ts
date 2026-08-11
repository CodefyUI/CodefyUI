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
    description: '載入標準影像資料集。把變換鏈接到 train_transform / eval_transform 就能控制前處理與資料增強；沒有接的話會套用 ToTensor 與 Normalize(0.5)。',
    params: {
      name: '要載入的資料集',
      split: '資料分割',
      data_dir: '下載/儲存資料集的目錄',
    },
  },
  ImageFolderDataset: {
    description: '從「一個類別一個資料夾」的結構載入自己的影像。標籤由資料夾名稱依字母順序決定。',
    params: {
      path: '放置各個分割的資料夾。相對路徑會相對於同時放著 models/ 與 images/ 的資料目錄。',
      split: '要載入的子資料夾。如果類別資料夾直接放在 path 底下、沒有分割這一層，選「(none)」；這時沒有分割可以區分兩個 transform 埠，所以接了哪一個就用哪一個，兩個都接時以 train_transform 為準。',
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
      pin_memory: '把批次放在鎖頁記憶體中，加快傳到 GPU 的速度（在 CPU 上沒有作用）',
      drop_last: '丟掉最後不滿一批的資料，讓每一批的大小都一樣',
      persistent_workers: '在 epoch 之間保持 worker 行程存活。需要 num_workers > 0。',
      prefetch_factor: '每個 worker 預先載入幾批資料。num_workers 為 0 時會被忽略。',
    },
  },
  Transform: {
    description: '對資料集套用變換流程。三個內建步驟以外的需求，請把變換鏈接到 transform；一旦接上，下面的參數就會被忽略。',
    params: {
      resize: '調整大小維度（0 表示不調整）。接上變換鏈時會被忽略。',
      normalize: '套用正規化（mean=0.5, std=0.5）。接上變換鏈時會被忽略；資料集統計值的預設組合在 NormalizeTransform。',
      to_tensor: '將 PIL 影像轉為張量。接上變換鏈時會被忽略。',
    },
  },

  // ── Data / 變換鏈（core#136）──
  ResizeTransform: {
    description: '把每個樣本縮放成指定邊長的正方形。放在 ToTensorTransform 之前。',
    params: {
      size: '縮放後正方形的邊長（像素）',
      interpolation: '重取樣濾波器。nearest 保留硬邊緣（遮罩、標籤圖）；bicubic 在照片上比較銳利。',
    },
  },
  ToTensorTransform: {
    description: '把 PIL 影像轉成範圍 [0, 1] 的 CxHxW 浮點張量。多數變換鏈的分界點：幾何與色彩步驟放在它之前，NormalizeTransform 放在它之後。',
  },
  NormalizeTransform: {
    description: '對每個通道做 (x - mean) / std 標準化。需要張量，所以放在 ToTensorTransform 之後。',
    params: {
      preset: '用來標準化的通道統計值。Half 會把 [0, 1] 映射到 [-1, 1]，也是 CodefyUI 在有預設組合之前一直採用的做法；想重現論文結果時，請選你實際訓練的資料集。',
      mean: '每個通道的平均值，以逗號分隔。只給一個值就套用到所有通道。',
      std: '每個通道的標準差，以逗號分隔。只給一個值就套用到所有通道。',
    },
  },
  RandomCrop: {
    description: '先補邊，再隨機取一個 size x size 的視窗。size 32 搭配 padding 4 就是標準的 CIFAR-10 資料增強：物體每個 epoch 都會偏移幾個像素，模型因此不再依賴它原本的位置。',
    params: {
      size: '裁切後正方形的邊長（像素）',
      padding: '裁切前四邊各補上的像素數。設 0 會真的裁出比原圖小的視窗；補的量等於想要的位移量時，輸出大小會和輸入一樣。',
      padding_mode: '補上的邊框內容。constant 是黑色，reflect 則鏡射影像邊緣（不會留下人工邊框讓模型去學）。',
    },
  },
  RandomHorizontalFlip: {
    description: '以機率 p 左右鏡射影像。在照片上幾乎是免費的準確率；但對於左右有意義的資料（數字、文字）就是錯的。',
    params: {
      p: '每個樣本被翻轉的機率',
    },
  },
  RandomRotation: {
    description: '把每個樣本旋轉一個從 [-degrees, +degrees] 均勻抽出的角度。小角度對手寫與衛星影像有幫助；角度過大則會破壞任何方向性有意義的類別。',
    params: {
      degrees: '旋轉範圍的半寬（度）。設 15 表示每個樣本最多往任一邊轉 15 度。',
      expand: '放大輸出畫布，避免角落被裁掉。這會改變影像尺寸，所以下游任何假設固定形狀的節點後面都要再接一個縮放。',
      fill: '旋轉後空出來的角落要填什麼值。0 是黑色。',
    },
  },
  ColorJitter: {
    description: '隨機調整亮度、對比、飽和度與色相。讓模型明白，暖光燈下的貓還是貓。預設值就是多數 ImageNet 訓練腳本採用的組合。',
    params: {
      brightness: '亮度會乘上一個從 [1-b, 1+b] 抽出的係數。設 0 表示停用。',
      contrast: '範圍規則與亮度相同。設 0 表示停用。',
      saturation: '範圍規則與亮度相同。設 0 表示停用。',
      hue: '色相會平移一個從 [-h, +h] 抽出的量，色環寬度為 1，所以上限是 0.5。請設小一點：超過 0.2 左右，某個類別賴以辨識的顏色就不再是那個類別的顏色了。',
    },
  },
  RandAugment: {
    description: '從固定的操作集合（傾斜、平移、旋轉、色調分離、曝光過度、色彩、對比、亮度、銳利度、直方圖等化、自動對比、identity）隨機挑 num_ops 個套用，強度都一樣。需要 PIL 影像或 uint8 張量，所以放在 ToTensorTransform 之前。',
    params: {
      num_ops: '每個樣本要套用幾個操作。論文的預設值是 2。',
      magnitude: '所有操作的強度，範圍是 0 到 num_magnitude_bins - 1。模型與資料集越大就調越高；小模型配小資料集，通常還沒需要 15 就已經欠擬合了。',
      num_magnitude_bins: '強度刻度的解析度。torchvision 的預設值是 31；改動它會連帶改變 magnitude 的意義。',
    },
  },
  ComposeTransform: {
    description: '把數條變換鏈依照埠的順序合併成一條：step_1 先跑。節點接節點本身就會組合，所以只有在兩條鏈分開建立、又要合成同一條流程時才需要它。',
    params: {
      steps: '要合併幾條鏈',
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
      momentum: '動量係數（0 = 單純的梯度下降）',
      betas: 'Adam 家族用來計算梯度與梯度平方移動平均的係數，格式為「beta1, beta2」',
      eps: '加在分母上的項，用來維持數值穩定。Adagrad 會沿用自己的 1e-10 預設值，不受此設定影響。',
      amsgrad: '使用 Adam 的 AMSGrad 變體',
      nesterov: 'Nesterov 加速梯度。需要 momentum > 0 且 dampening = 0。',
      dampening: '施加在動量項上的阻尼',
    },
  },
  Loss: {
    description: '建立損失函數',
    params: {
      type: '損失函數類型',
      label_smoothing: '把 one-hot 目標變得平滑一些：0 表示硬目標，0.1 是常見的正則化強度',
      reduction: '如何合併每個樣本的損失：mean（平均）、sum（加總）或 none（維持逐樣本）',
      weight: '各類別的權重，以逗號分隔，例如兩類不平衡時可用「1, 5」。留空表示每個類別權重相同。',
      ignore_index: '不計入損失、也不產生梯度的目標值，例如 padding 標籤',
      pos_weight: '正類別的權重，可以是單一數字或每個輸出各一個。留空表示不加權。',
    },
  },
  TrainingLoop: {
    description: '執行訓練迴圈，支援驗證、早停、學習率排程和梯度裁剪',
    params: {
      epochs: '訓練 epoch 數量',
      device: '訓練裝置',
      early_stopping_patience: '監控的指標未改善 N 個 epoch 後停止（0 = 停用）',
      monitor: '早停監控的指標。val_loss：越低越好（預設）。val_accuracy：越高越好，僅在使用分類損失函數（CrossEntropyLoss/NLLLoss）且有接上 val_dataloader 時才會記錄；兩者缺一就會退回 val_loss 並記錄警告，而不是去監控一個從未被算出來的數值。',
      checkpoint_every: '每 N 個完整 epoch 存一次檢查點，讓伺服器當機時最多只損失 N 個 epoch，而不是整次執行。與 CheckpointSaver 互相獨立，恢復方式也相同：把 CheckpointLoader.epoch 接到 start_epoch。每個檢查點大小大約是模型加上優化器狀態（常常是模型本身的好幾倍），而且是在訓練執行緒上同步寫入；大模型搭配偏低的 N、跑很長的訓練，執行完之前可能會用掉好幾 GB 磁碟空間，因為執行中的任務目前沒有機制限制這件事（0 = 停用）',
      grad_clip_norm: '最大梯度範數裁剪（0 = 停用）',
      batch_metrics: '同時把每一批的損失記錄成 train_loss_batch 這條序列（預設關閉：每批一列資料量相當可觀）',
      precision: '混合精度。bf16 在 Ampere 以後的顯卡上可以把 activation 記憶體用量大約減半，其他都不用改；fp16 則是給更舊的顯卡用的，會額外搭配 loss scaler。val_accuracy 會用跟這個相同的精度計算（不會強制轉成 fp32），才能跟 val_loss 維持可比較性——但精度較低的 logit 在接近平手時可能讓 argmax 換邊，讓量出來的準確率有些微變動。裝置做不到的話會自動退回 fp32 並記錄下來。',
      accumulate_steps: '累積這麼多批之後才做一次優化器更新，同時把每一批的損失除以同一個數字。batch_size 8 搭配 4，梯度會等同於 batch_size 32，但記憶體裡同時只放 8 筆（1 = 關閉）。',
      max_steps: '總共跑滿這麼多次優化器更新後就停止，不論 epochs 設定為何。算的是優化器更新次數而不是批次數，所以不管 accumulate_steps 設多少意思都一樣（0 = 不限制）',
      log_interval: '開啟批次指標時，每 N 批記錄一次。長時間執行時調高可以讓圖表稀疏一點。',
      deterministic: '要求 PyTorch 使用決定性的運算核心。沒有決定性實作的運算會發出警告，而不會讓執行失敗。',
      tensorboard: '同時把指標寫成 TensorBoard 事件檔，放在這次執行專屬的資料夾裡。用 `tensorboard --logdir <路徑>` 開啟；該路徑會列在這次執行的產出檔案中。',
    },
  },
  EvaluateModel: {
    description: '算訓練好的分類模型在一個 dataset 上的準確率。吃 model + dataset，內部建 DataLoader 跑完整個資料集、對每筆取 argmax 跟標籤比，輸出 accuracy / correct / total。補上通用訓練流缺的「評估」那一塊（對應 I2-4 看 MNIST 測試準確率）。',
    params: {
      batch_size: '評估時每批跑幾筆（不影響結果，只影響速度/記憶體）。',
      device: '評估裝置',
      precision: '前向傳播用的混合精度。bf16 在 Ampere 以後的顯卡上可以把 activation 記憶體用量大約減半，其他都不用改；fp16 則是給更舊的顯卡用的。不論選哪一種，參數都維持 fp32；但降精度的前向傳播仍可能讓量出來的準確率有些微變動（精度較低的 logit 在接近平手時可能讓 argmax 換邊），所以要回報的準確率應該用 fp32 這個數字。裝置做不到的話會自動退回 fp32 並記錄下來。',
      step: 'eval_accuracy 這個指標記錄時使用的 step 值。同一張圖裡有多個 EvaluateModel 節點時（例如微調前後的比較），需要各自設定不同的 step，否則會在圖表上互相覆蓋。',
    },
  },
  BackwardOnce: {
    description: '標記張量為 autograd 反向傳播的目標，供 Backward 檢視器使用。僅在工具列啟用 Backward 模式時執行。反向傳播目標：$\\mathcal{L} = \\sum(\\text{input})$（合成純量）。',
  },

  LRScheduler: {
    description: '建立學習率排程器',
    params: {
      type: '排程器類型',
      step_size:
        'StepLR：每隔幾個 epoch 降一次學習率。MultiStepLR：在此值的 1、2、3、4 倍處各降一次。TrainingLoop 是每個 epoch 走一步排程器，所以這裡算的是 epoch，不是 batch。',
      gamma:
        'StepLR、MultiStepLR、ExponentialLR 的衰減因子；ReduceLROnPlateau 也拿它當 factor。',
      T_max:
        'CosineAnnealingLR：一個 cosine 週期的長度，單位是 epoch。通常應該設成和 TrainingLoop.epochs 一樣——排程器每個 epoch 走一步，設小了會在學習率還很高的時候就結束，設大了則是只走到曲線中途，兩者通常會少掉幾個百分點的準確率，而且畫面上完全看不出來是排程造成的。這裡刻意不強制：截斷的排程本來就是合理選擇。CosineAnnealingWarmRestarts 會把這個值當成 T_0（第一次重啟前的週期長度），那種情況下設成和 epochs 一樣反而永遠不會重啟。',
      max_lr: 'OneCycleLR 的最大學習率',
      total_steps:
        'OneCycleLR：整個 one-cycle 排程的長度。TrainingLoop 是每個 epoch 走一步，所以這裡要填 TrainingLoop.epochs，不是 batch 數（OneCycleLR 官方文件講的 step 是 batch）。預設值 1000 遠大於一般的 epoch 數，照著不改就只會走到週期的開頭：學習率稍微升上去，然後從來不會退火下來。',
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
      load_mode:
        '載入模式：state_dict（需要模型輸入）或 full_model 完整模型。full_model 會重建存檔中的模組本身，並且在 torch 的受限解序列化器下讀取，因此只接受由標準 torch.nn 層組成的模型，其餘一律拒絕',
      device: '載入權重的裝置',
      strict: '是否嚴格要求 state_dict 中的鍵值匹配（僅 state_dict 模式）',
    },
  },
  CheckpointSaver: {
    description: '儲存完整訓練檢查點（模型 + 優化器 + 學習率排程 + epoch + 損失值），用於稍後恢復訓練',
    params: {
      path: '輸出檢查點檔案路徑',
      epoch: '要儲存在檢查點中的當前 epoch 數',
    },
  },
  CheckpointLoader: {
    description: '載入訓練檢查點以恢復訓練（恢復模型 + 優化器 + 學習率排程 + epoch）',
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
  PythonScript: {
    description:
      '直接在畫布上寫 Python。定義 run(inputs, params)，回傳以輸出連接埠為鍵的字典（直接回傳單一值時視為 out1）。腳本只能使用 collections、itertools、json、math、numpy、re、statistics、torch 這幾個函式庫。這限制的是它能碰到哪些函式庫，而不是那些函式庫能做什麼：這是防護欄，不是沙箱，程式碼以你的權限在 CodefyUI 行程內執行。只執行你信任的程式碼。',
    params: {
      code: '定義 run(inputs, params) 的 Python 原始碼。每次編輯都會依 Tier-0 政策檢查。',
      input_ports: '輸入連接埠 in1..inN 的數量（1..8）',
      output_ports: '輸出連接埠 out1..outN 的數量（1..8）',
      input_types: '每個輸入連接埠的資料型別，以逗號分隔。列得比連接埠少時，最後一項會沿用到其餘連接埠。',
      output_types: '每個輸出連接埠的資料型別，以逗號分隔。ANY 可接到任何地方；填入實際型別則可讓流程驗證器替你檢查接線。',
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
  CausalLMModel: {
    description:
      '建立一個可訓練的 GPT-style decoder-only transformer 語言模型。輸出真正的 MODEL（input_ids (B,T) int64 → logits (B,T,vocab)），可直接接 Optimizer / TrainingLoop / CheckpointSaver，另輸出精確的可訓練參數量讓圖能證明模型大小。因果遮罩在模型內部；搭配 LMCrossEntropyLoss 與打包好的 LM 資料使用。預設形狀約 2.02 億參數（d_model 1024、12 層、gpt2 詞彙表、共享嵌入）。',
    params: {
      vocab_size: '詞彙表大小（50257 = tiktoken gpt2；請與 LM tokenizer 的 encoding 保持一致）。',
      d_model: '嵌入／殘差流寬度。',
      n_layers: 'Transformer 區塊層數。',
      n_heads: '注意力頭數（d_model 必須能被 n_heads 整除）。',
      d_ff: '前饋隱藏層寬度（通常是 d_model 的 4 倍）。',
      max_seq_len: '模型接受的最大序列長度。',
      dropout: '注意力／MLP／嵌入的 dropout（單一 epoch 的 LM 預訓練通常設 0）。',
      positional: '位置編碼：learned 可學習嵌入、sinusoidal 固定正弦、rope 旋轉位置編碼（RoPE）。',
      norm: '正規化層（兩種都用 pre-norm 區塊）。',
      activation: '前饋層的激活函數。',
      tie_embeddings: '讓輸出層與 token 嵌入矩陣共享權重（省下 vocab×d_model 個參數）。',
      gradient_checkpointing: '反向傳播時重算激活值，用計算換記憶體。',
      init_std: '權重初始化的標準差（殘差投影會再除以 sqrt(2×n_layers)）。',
      seed: '初始化種子 — 相同種子建出完全相同的權重。',
    },
  },
  LMCrossEntropyLoss: {
    description:
      '語言模型用的 next-token cross-entropy：內部自動攤平，接受 logits (B,T,vocab) 對 targets (B,T) — 正是 CausalLMModel 與打包 LM 資料產出的形狀。支援 ignore_index 與 label smoothing。LM 訓練請用這顆，而不是通用 Loss 節點的 CrossEntropyLoss。',
    params: {
      ignore_index: '目標值等於此數的位置不計損失（padding 慣例）。',
      label_smoothing: '標籤平滑係數（0 = 關閉）。',
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
  LMTokenizer: {
    description:
      '產生一個可重複使用的 tokenizer 物件（tiktoken BPE）給 LM 訓練管線：LMTokenizedDataset 用它編碼並打包語料，文字生成用它編碼 prompt／解碼輸出。輸出物件提供 encode / encode_batch / decode、eos_id 與 vocab_size。gpt2 = 詞彙表 50257、eos <|endoftext|>（50256），與 CausalLMModel 的預設 vocab_size 一致。',
    params: {
      encoding: 'tiktoken 編碼。gpt2 = 50257 個 token（GPT-2 詞彙表）；cl100k／o200k 是 GPT-4 世代的較大詞彙表。第一次使用會下載一次，之後離線可用。',
    },
  },
  TextCorpusDataset: {
    description:
      '把文字語料載成「每列一段原始文字」的資料集 — 來源可以是 HuggingFace Hub 的文字資料集（如 roneneldan/TinyStories、wikitext）或本機文字檔（一行一筆）。接 LMTokenizedDataset 編碼打包後就是 LM 訓練資料。',
    params: {
      source: '語料來源。',
      dataset_name: 'HuggingFace Hub repo id（例如 roneneldan/TinyStories、wikitext）。',
      subset: '多設定資料集的 config 名稱（留空 = 無；wikitext 需要如 wikitext-103-raw-v1）。',
      split: '切分：train／validation／test，或 HF 切片語法（train[:10000]）。',
      text_column: '存放文字的欄位名稱。',
      local_file: '本機文字檔，一行一筆（空行略過）。',
      max_rows: '只保留前 N 列（0 = 全部）— 試跑預算的便宜開關。',
      cache_dir: '覆寫 HuggingFace 快取目錄（留空 = HF 預設）。',
    },
  },
  LMTokenizedDataset: {
    description:
      '把文字語料編碼並打包成固定長度的 next-token 訓練塊 — 標準 LM 預訓練資料形狀。列與列之間接 EOS、整條 token 流切成 seq_len+1 的塊，輸出 (input_ids, labels) 配對（labels 右移一格）。接 TextCorpusDataset + LMTokenizer 進來，輸出接 DataLoader → TrainingLoop。打包結果以內容指紋為鍵快取在磁碟，重跑不再重新編碼。',
    params: {
      seq_len: '每塊的 token 長度（不可超過模型的 max_seq_len）。',
      append_eos: '文件之間插入 tokenizer 的 EOS，讓模型學到文件邊界。',
      max_tokens: '打包流最多 N 個 token（0 = 全部）— 試跑預算開關。',
      cache: '把打包好的塊快取到資料目錄的 lm_token_cache/。',
      cache_dir: '覆寫快取目錄（相對路徑解析在資料目錄內）。',
    },
  },
  PerplexityEvaluate: {
    description:
      '在打包好的 (input_ids, labels) 驗證資料上評估 causal LM：輸出 token 加權的平均 cross-entropy（val_loss）與 perplexity = exp(val_loss)。標籤為 -100 的位置不計入。接 TrainingLoop.model 與驗證用的 LMTokenizedDataset；perplexity 是逐 token 指標，只在相同資料集與 tokenizer 下可比較。',
    params: {
      batch_size: '評估批次大小（只影響速度／記憶體，結果相同）。',
      max_batches: '最多評估 N 個批次（0 = 整個資料集）。',
      device: '評估用的裝置（auto 跟隨全域裝置）。',
      precision: '前向的 autocast 精度。bf16 在 Ampere+ 省一半激活記憶體，是 LM 評估預設；裝置不支援會退回 fp32 並註明。要發表的數字請用 fp32。',
    },
  },
  TextGenerate: {
    description:
      '用訓練好的 causal LM 自迴歸生成文字：以 LMTokenizer 編碼 prompt、逐 token 取樣（temperature / top-k / top-p，種子化可重現）、遇 EOS 或 max_new_tokens 停止並解碼。畫布上訓練成果最直觀的質性驗證。',
    params: {
      prompt: '沒有接 prompt 輸入時使用的提示文字。',
      max_new_tokens: '最多生成的 token 數。',
      temperature: '取樣溫度（0 = greedy argmax）。',
      top_k: '取樣前只保留機率最高的 k 個 token（0 = 關閉）。',
      top_p: 'Nucleus 取樣：保留累積機率達 p 的最小 token 集合（1 = 關閉）。',
      seed: '取樣種子 — 相同種子與權重會重現相同文字。',
      device: '前向運算的裝置。',
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
