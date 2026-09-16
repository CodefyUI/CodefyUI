import type { NodeTranslations } from './types';

const zhTW: NodeTranslations = {
  // ── Control ──
  Start: {
    description: '標記執行的進入點，trigger 接到的節點就是起點',
    details:
      '相當於 Scratch 的「當綠旗被點擊」積木：節點本身在執行時不做任何事，也不傳遞資料。trigger 連線是告訴引擎要執行圖上的哪一塊，' +
      '因此通常接到 Dataset 之類的資料起點。',
  },

  // ── Classical (sklearn) ──
  Accuracy: {
    description: '比對預測標籤與真實標籤，輸出正確率、答對數與總數',
    details: '正確率是答對數除以總數，範圍在 [0, 1]。標籤會轉成字串比對，所以 0 和 \'0\' 算相同；兩份清單長度必須一致，兩邊都空的時候回傳 0。',
  },
  DecisionTreeClassifier: {
    description: '遞迴切分特徵，輸出學到的規則',
    details:
      'sklearn 的 CART 決策樹，分割準則可選 gini、entropy 或 log loss。除了預測結果，還會輸出 ' +
      'tree_text（if/else 規則的可讀文字）與 feature_importances（每個特徵一個分數，總和為 1）。' +
      'max_depth 設成 0 時會長到葉節點純淨為止。',
    params: {
      max_depth: '樹的最大深度。0 = 不設限（一路長到全純）。',
      criterion: '切分品質的衡量函式：gini 不純度、entropy 或 log loss。',
      random_state: '同分時打破平手用的種子；可重現的關鍵。',
    },
  },
  KNN: {
    description: '以最近的 k 個訓練點投票決定每筆查詢的類別',
    details:
      'sklearn 的 KNeighborsClassifier 以 KD-tree 或 ball-tree 索引訓練集，能處理較大的資料；k ' +
      '會被限制在訓練筆數之內。weights 設成 distance 時越近的鄰居權重越高。另外會輸出各類別的投票比例。foundations ' +
      '外掛的 Edu-KNN 介面相同，以暴力法做同樣的計算並輸出距離，兩者可直接互換。',
    params: {
      n_neighbors: '鄰居數 k。',
      weights: '投票權重：uniform 每個鄰居等權；distance 越近權重越大。',
      metric: '距離度量。minkowski 預設 p=2 等同 euclidean。',
    },
  },
  LinearRegression: {
    description: '以最小平方法擬合，預測查詢目標並輸出係數與截距',
    details:
      'sklearn 的封閉解最小平方法 $\\beta = (X^\\top X)^{-1} X^\\top y$，X 秩不足時改用 SVD 求解，' +
      '不需要學習率也不需要迭代。目標值可以是單欄或多欄；關掉 fit_intercept 時迴歸會通過原點。',
    params: {
      fit_intercept: '若為 False，迴歸直線會通過原點（不擬合截距）。',
    },
  },
  LogisticRegression: {
    description: 'softmax 分類器，輸出類別與各類機率',
    details:
      'sklearn 的 LogisticRegression。C 是正則化強度的倒數，越小正則化越強；penalty 可選 l2、l1 或 ' +
      'none，求解器會自動配合。y_train 至少要有兩個類別；coef 每個類別一列，剛好兩類時只有一列。介面與 foundations ' +
      '外掛的 Edu-LogisticRegression 相同，兩者可直接互換。',
    params: {
      C: '正則化強度的倒數（值越小，正則化越強）。',
      max_iter: '求解器最大迭代次數。',
      penalty: '正則化類型。l1 需要 liblinear/saga 求解器；sklearn 會自動挑。',
    },
  },
  MLPClassifier: {
    description: '訓練前饋神經網路，標註每筆查詢的類別',
    details:
      'sklearn 的 MLPClassifier，以 Adam 訓練。hidden_sizes 是以逗號分隔的各層寬度，\'16,16\' ' +
      '代表兩層各 16 個神經元；activation 設成 identity 時整個網路仍是線性的。另外會輸出 softmax ' +
      '機率與最後的訓練損失。介面與線性分類器相同，換成這個節點不必重接線路。',
    params: {
      hidden_sizes: '逗號分隔的隱藏層大小。「16,16」代表兩層、每層 16 個神經元。',
      activation: '隱藏層激活函數。「identity」會讓整個網路退化成線性 — 用來示範為什麼需要非線性激活。',
      max_iter: '最大訓練迭代次數（在整個資料集上跑幾輪）。',
      learning_rate_init: 'Adam 的初始學習率。',
      seed: '可重現用的隨機種子。',
    },
  },
  SVMClassifier: {
    description: '以最大邊界的分隔面切開類別，並標註每筆查詢',
    details:
      'sklearn 的 SVC。kernel（linear、rbf、poly、sigmoid）決定邊界能有多彎，rbf、poly、' +
      'sigmoid 靠 kernel trick 做到。C 在邊界寬度與訓練違例之間取捨，gamma 可填 scale、auto 或數字。' +
      '另外會輸出 support vectors，也就是落在邊界附近的訓練點；沒有開啟機率估計，因此沒有 probabilities 輸出。',
    params: {
      C: '懲罰強度；C 越小邊界越寬，可容忍更多違規。',
      kernel: '核函式：linear 線性、rbf 高斯、poly 多項式、sigmoid。',
      gamma: 'rbf/poly/sigmoid 的核係數。「scale」用 1/(F·var(X))、「auto」用 1/F，也可以填數字字串。',
    },
  },
  RandomForestClassifier: {
    description: '由多棵決策樹投票決定每筆查詢的類別',
    details:
      'sklearn 的 RandomForestClassifier：每棵樹在隨機重抽的樣本上訓練，投票結果以各類別機率輸出，' +
      '邊界比單棵決策樹平滑。n_estimators 決定樹的數量，越多越穩也越慢；max_depth 設成 0 代表深度不限。輸入與 ' +
      'predictions 輸出和其他 classical 分類器一致。',
  },

  // ── CNN ──
  Conv2d: {
    description: '在影像上滑動可學習的卷積核，每個產生一張特徵圖',
    details:
      '底層是 nn.Conv2d：$y[i,j]=\\sum_{k,l} x[i+k,j+l]\\cdot w[k,l] + b$。' +
      'in_channels 必須等於輸入的通道數；權重一開始是隨機的，訓練時才被學出來。',
    params: {
      in_channels: '輸入通道數',
      out_channels: '輸出通道數',
      kernel_size: '卷積核大小',
      stride: '卷積步幅',
      padding: '兩側的零填充',
    },
  },
  Conv2dExplicit: {
    description: '用你指定的卷積核做卷積，不是網路學出來的',
    details:
      '沒有可學參數，也沒有隨機初始化。內建預設為 EdgeDetection3x3、Sharpen3x3、VerticalEdge3x3；把 ' +
      'preset 設成 Custom 就能自填 NxN 矩陣，元素數量必須剛好是 kernel_size × kernel_size。' +
      '同一個卷積核以分組卷積（也就是 depthwise 卷積）逐通道套用，通道數不變。',
    params: {
      preset: '內建的 3×3 kernel，或選 Custom 自己寫一個矩陣。',
      kernel_size: 'NxN kernel 的邊長 N（只有 Custom 會用到）。',
      weights: '自訂的 kernel 矩陣（NxN）；格子大小跟著 kernel_size 走。',
      stride: '卷積步幅',
      padding: '空間維度兩側的零填充',
    },
  },
  MaxPool2d: {
    description: '取每個視窗中的最大值，縮小高與寬',
    details:
      '底層是 nn.MaxPool2d，只開放 kernel_size 與 stride，沒有 padding。用預設值（kernel_size ' +
      '2、stride 2）時，高與寬各減半。',
    params: {
      kernel_size: '池化視窗大小',
      stride: '池化視窗步幅',
    },
  },
  BatchNorm2d: {
    description: '對每個通道在批次與高寬上正規化',
    details:
      '底層是 nn.BatchNorm2d：$y = \\frac{x - \\mu_C}{\\sqrt{\\sigma_C^2 + ' +
      '\\epsilon}} \\gamma + \\beta$，其中 $\\mu_C$、$\\sigma_C^2$ 是每個通道在 (N, H, W) ' +
      '上的統計量。num_features 必須等於輸入的通道數。',
    params: {
      num_features: '要正規化的特徵（通道）數量',
    },
  },
  Dropout: {
    description: '隨機把元素歸零，其餘按比例放大',
    details:
      '底層是 nn.Dropout，留下來的元素會乘上 $1/(1-p)$，讓平均值維持不變。單獨當節點用時每次都以訓練模式重建，' +
      '評估階段也照樣丟棄；放進 SequentialModel 則跟隨模型的訓練／評估狀態。',
    params: {
      p: '元素被歸零的機率',
    },
  },
  Activation: {
    description: '套用選定的非線性函數，形狀不變',
    details:
      '共十二種函數：relu、leaky_relu、elu、gelu、silu、mish、selu、prelu、sigmoid、tanh、' +
      'hardswish、softmax。softmax 在最後一個維度上正規化；prelu 每次執行都會新建一個未訓練的斜率 0.25。',
    params: {
      function: '要套用的激活函數',
    },
  },
  Conv1d: {
    description: '沿著長度軸滑動可學習的卷積核',
    details:
      '底層是 nn.Conv1d：$y[i]=\\sum_k x[i+k]\\cdot w[k]+b$。in_channels 必須等於輸入的通道數，' +
      'out_channels 決定要學幾個濾波器。輸入為 (N, C, L)。',
    params: {
      in_channels: '輸入通道數',
      out_channels: '輸出通道數',
      kernel_size: '卷積核大小',
      stride: '卷積步幅',
      padding: '兩側的零填充',
    },
  },
  ConvTranspose2d: {
    description: '把特徵圖上採樣，放大高與寬',
    details:
      '底層是 nn.ConvTranspose2d，也稱反卷積。輸出高度為 $(H-1)\\times \\text{stride} - ' +
      '2\\,\\text{padding} + \\text{kernel\\_size} + \\text{output\\_padding}$，' +
      '寬度同理；多個輸入尺寸對應到同一輸出時，由 output_padding 決定取哪一個。',
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
    description: '取每個視窗的平均值，縮小高與寬',
    details:
      '底層是 nn.AvgPool2d。視窗裡每個數值都會納入計算，結果比 MaxPool2d 平滑。用預設值（kernel_size 2、' +
      'stride 2）時，高與寬各減半。',
    params: {
      kernel_size: '池化視窗大小',
      stride: '池化視窗步幅',
      padding: '兩側的零填充',
    },
  },
  AdaptiveAvgPool2d: {
    description: '把每個通道平均縮到指定的高與寬',
    details:
      '底層是 nn.AdaptiveAvgPool2d，池化窗口由輸入尺寸反推，因此任何輸入都會得到 (N, C, output_height, ' +
      'output_width)。預設的 1x1 會把每個通道壓成一個數字。',
    params: {
      output_height: '目標輸出高度',
      output_width: '目標輸出寬度',
    },
  },

  // ── Normalization ──
  LayerNorm: {
    description: '對每個樣本在最後幾個維度上正規化',
    details:
      '底層是 nn.LayerNorm：$y = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} ' +
      '\\gamma + \\beta$。normalized_shape 以逗號分隔整數，例如 512 或 64,32，必須與輸入最後幾個維度相符。',
    params: {
      normalized_shape: '要正規化的維度形狀（逗號分隔整數）',
      eps: '數值穩定性的 Epsilon',
    },
  },
  GroupNorm: {
    description: '把通道分成幾組，在每一組之內做正規化',
    details:
      '底層是 nn.GroupNorm。num_channels 必須等於輸入的通道數，且能被 num_groups 整除；統計量與批次大小無關，' +
      '這點比 BatchNorm2d 穩定。',
    params: {
      num_groups: '將通道分成的群組數',
      num_channels: '通道數（必須能被 num_groups 整除）',
    },
  },
  InstanceNorm2d: {
    description: '每個樣本的每個通道各自在高寬上正規化',
    details:
      '底層是 nn.InstanceNorm2d，等同於每個通道自成一組的 GroupNorm，風格轉換與影像生成用的就是這種正規化。' +
      'affine 預設關閉，除非打開，否則沒有可學的 $\\gamma$ 與 $\\beta$。',
    params: {
      num_features: '特徵（通道）數',
      affine: '是否使用可學習的仿射參數',
    },
  },
  BatchNorm1d: {
    description: '對每個特徵在批次上正規化',
    details:
      '底層是 nn.BatchNorm1d。輸入為 (N, C) 或 (N, C, L)，num_features 必須等於 C；(N, C, ' +
      'H, W) 的影像張量請改用 BatchNorm2d。正規化後會再做一次可學的縮放與平移。',
    params: {
      num_features: '要正規化的特徵數',
    },
  },

  // ── RNN ──
  RNNCell: {
    description: '結合這一步的輸入與前一步的隱藏狀態',
    details:
      '底層是 nn.RNNCell：$h_t = \\phi(W_{ih} x_t + W_{hh} h_{t-1} + b)$，$\\phi$ 由 ' +
      'nonlinearity 決定，權重依 seed 確定性初始化。hidden 可省略，預設全零；把前一個 cell 的 hidden ' +
      '接到下一個的 hidden，即可手動展開遞迴。',
    params: {
      input_size: '每個時間步的輸入向量維度。',
      hidden_size: '隱藏狀態的維度。',
      nonlinearity: '套用在遞迴輸出上的激活函式。',
      seed: 'W_ih / W_hh / 偏置初始化的隨機種子。',
    },
  },
  LSTM: {
    description: '以四個閘掃過序列，輸出每步狀態與最後狀態',
    details:
      '底層是 nn.LSTM，四個閘分別是輸入、遺忘、記憶單元與輸出。output 埠是每個時間步的隱藏狀態，hidden 埠是最後的 h_n；' +
      '細胞狀態 c_n 不對外輸出。',
    params: {
      input_size: '輸入的預期特徵數',
      hidden_size: '隱藏狀態的特徵數',
      num_layers: '遞迴層數量',
      batch_first: '若為 True，輸入/輸出形狀為 (batch, seq, feature)',
      bidirectional: '若為 True，則為雙向 LSTM',
    },
  },
  GRU: {
    description: '以兩個閘掃過序列，輸出每步狀態與最後狀態',
    details:
      '底層是 nn.GRU，兩個閘分別是重置閘與更新閘。output 埠是每個時間步的隱藏狀態，hidden 埠是最後一步，形狀為 ' +
      '(num_layers × 方向數, batch, hidden_size)。batch_first 開啟時，輸入為 (batch, ' +
      'seq_len, input_size)。',
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
    description: '以 query/key/value 計算注意力與權重',
    details:
      '本層直接使用 PyTorch 的 nn.MultiheadAttention。' +
      '$\\text{Attention}(Q,K,V)=\\text{softmax}(\\frac{QK^T}{\\sqrt{d_k}})V$，' +
      'embed_dim 會平均分給 num_heads 個頭；輸入預設為 (seq, batch, embed)，開啟 batch_first ' +
      '才是批次在前，輸出的權重是各頭的平均。',
    params: {
      embed_dim: '模型的總維度',
      num_heads: '平行注意力頭的數量',
    },
  },
  TransformerEncoder: {
    description: '以多層自注意力編碼序列',
    details:
      '共 num_layers 層，每層是自注意力加上寬度 dim_feedforward 的前饋網路，d_model 會分給 nhead 個頭。' +
      '張量固定為 (seq, batch, d_model)，沒有 batch_first 選項。',
    params: {
      d_model: '模型維度',
      nhead: '注意力頭的數量',
      num_layers: '編碼器層數',
      dim_feedforward: '前饋網路維度',
    },
  },
  TransformerDecoder: {
    description: '以編碼器記憶解碼目標序列',
    details:
      '共 num_layers 層，每層是自注意力、對 memory 的交叉注意力與寬度 dim_feedforward 的前饋網路，' +
      'd_model 會分給 nhead 個頭。張量固定為 (seq, batch, d_model)，沒有 batch_first 選項；' +
      '也不套用因果遮罩，每個目標位置都看得到整個序列。',
    params: {
      d_model: '模型維度',
      nhead: '注意力頭的數量',
      num_layers: '解碼器層數',
      dim_feedforward: '前饋網路維度',
    },
  },
  MoELayer: {
    description: '每個 token 交給 top-k 專家，加權相加',
    details:
      'gate 以線性層對每個專家評分，softmax 只在選中的 k 個專家上正規化，每個 token 的權重和為 1；' +
      'routing_weights 與 expert_indices 會回報路由結果。專家權重每次執行都依 seed 重新初始化，前向也在 ' +
      'no_grad 下計算，所以這層不會被訓練。Switch Transformer、Mixtral、DeepSeek-MoE ' +
      '都採用這個路由結構。',
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
    description: '以 MLP 把狀態轉成每個動作的 Q 值',
    details:
      '深度 Q 網路：三層 Linear、中間夾 ReLU，大小由 state_dim、hidden_dim、action_dim 決定。' +
      'state 輸入是選填的，只是拿剛建好的網路跑一次前向；沒接時 Q 值全是 0。',
    params: {
      state_dim: '狀態空間維度',
      action_dim: '動作空間維度',
      hidden_dim: '隱藏層維度',
    },
  },
  PPO: {
    description: 'Actor-Critic：共用主幹、動作頭與價值頭',
    details:
      '兩層 Linear 加 Tanh 的共用主幹，接上 softmax 的 actor 與純量 critic，forward 回傳 ' +
      '(action_probs, value)。state 輸入是選填的，只是拿剛建好的網路跑一次前向；沒接時 action 全是 0。',
    params: {
      state_dim: '狀態空間維度',
      action_dim: '動作空間維度',
      hidden_dim: '隱藏層維度',
    },
  },
  EnvWrapper: {
    description: '依 ID 建立環境並 reset',
    details:
      '封裝 Gymnasium，env_name 可填 gymnasium.make 接受的任何 ID，例如 CartPole-v1，' +
      '而且必須先安裝該套件。不想多裝套件時，改用 GridWorldEnv。reset 同時輸出第一個觀測。',
    params: {
      env_name: 'Gymnasium 環境 ID',
    },
  },
  KLDivergence: {
    description: 'KL(p || q)，輸入可為機率或 logits',
    details:
      'reduction 沿用 PyTorch 慣例，預設 batchmean（總和除以批次大小），none 則每個樣本給一個值。KL 不對稱：' +
      'p 是策略，q 是被拉近的那個凍結參考策略。',
    params: {
      input_kind: 'p、q 是已經算好的機率，還是尚未經過 softmax 的 logits。',
      reduction: '如何把每個樣本的 KL 聚合起來。batchmean = sum / batch_size，是 RLHF 的預設用法。',
    },
  },
  RewardModel: {
    description: '用 MLP 頭把隱藏狀態打成每筆一個純量分數',
    details:
      'RLHF 的獎勵頭：先用人類偏好訓練它，再讓 PPO 去最大化它給的分數。兩層 Linear 加 ReLU 收成單一輸出，依 seed ' +
      '初始化，所以初始權重可重現。可接受 [B, H] 或 [B, T, H]（取最後一個 token）；輸入是選填的，沒接時 rewards ' +
      '是空張量。',
    params: {
      input_dim: '隱藏狀態的維度 H。',
      hidden_dim: 'MLP 中間層的寬度。',
      seed: '初始化的隨機種子，確保可重現。',
    },
  },
  GridWorldEnv: {
    description: '方格世界，終點在對角，另有陷阱',
    details:
      '代理人從左上角出發，終點在對角：抵達終點以 goal_reward 結束回合，踩到陷阱以 trap_reward 結束，其餘每步給 ' +
      'step_reward（預設 0）。不需要安裝 gymnasium。觀測是格子的 one-hot，所以 state_dim 等於 size ' +
      '× size，動作固定 4 個；單層 Linear 在這個 one-hot 上就是一張表格式策略。',
    params: {
      size: '格子邊長。4 就是課本畫的 4×4。',
      traps: "陷阱格，寫成 '列,行'，多個用 ';' 分隔（例如 '1,1; 2,3'）。留空表示沒有陷阱。",
      step_reward: '沒有結束回合的那一步給多少獎勵。0 讓獎勵保持稀疏 — 整個回合只在最後結算。',
      goal_reward: '走到終點的獎勵（回合結束）。',
      trap_reward: '踩到陷阱的獎勵（回合結束）。',
      max_steps: '步數上限。走滿就結束回合，算沒走到終點。',
    },
  },
  PolicyRollout: {
    description: '讓策略跑 N 個回合，輸出狀態、動作與獎勵',
    details:
      '動作是從 softmax(logits / temperature) 抽樣而來，所以同一顆策略跑兩次不會一樣；seed 固定整批。' +
      '另外會輸出 logits、抽樣當下記錄的 log_probs（PPO 的 log_probs_old）、每回合的回報、長度與 ' +
      'episode_ids，以及文字報告和第一個回合的逐步表格。env 只需要提供 reset() 與 step(action)。',
    params: {
      episodes: '要走幾個回合。設成 1 看單一軌跡；設成 K 就是 GRPO 的一組樣本。',
      temperature: '抽樣溫度。低 = 偏向利用（每次都挑機率最高的動作）、高 = 偏向探索。',
      seed: '抽樣的隨機種子，確保可重現。',
    },
  },
  Discount: {
    description: '折扣回報 G_t = r_t + γ·G_{t+1}',
    details:
      '由後往前摺疊，所以每一步只算一次。獎勵只落在第 T 步時，G_0 = γ^(T-1)。接上 episode_ids 就會在回合邊界重新起算，' +
      '沒接時整個張量當成一個回合。',
    params: {
      gamma: '折扣因子。1 = 完全不折扣；越小則越晚拿到的獎勵越不值錢，智能體也就越偏好快點達成目標。',
    },
  },
  PPOClipObjective: {
    description: '逐樣本 min(rA, clip(r,1±ε)A)',
    details:
      '可以直接給 ratio，也可以給 log_probs_new 與 log_probs_old，比值就是 exp(new - old)。' +
      '另外把未截斷項與截斷項分開輸出，還有純量損失 -mean(objective)、被截斷的樣本遮罩與 clip_fraction。clip ' +
      '壓平的是目標而不是機率比：超出區間後再往前推得不到好處，但並沒有被禁止。',
    params: {
      epsilon: '截斷半徑。常用 0.1–0.2：太大則一個幸運樣本就可能暴衝，太小則學得很慢。',
    },
  },
  GroupRelativeAdvantage: {
    description: '組平均為基準：A_i = r_i - mean(r)',
    details:
      'GRPO 的基準：以組平均取代 PPO 學出來的 critic，基準是算出來的而不是估出來的。同組內的優勢加總為 0，這是最快的接線檢查；' +
      '一組樣本分數全部相同時優勢全為 0，這一組學不到東西。expand_index（PolicyRollout 的 episode_ids）' +
      '把優勢攤回它涵蓋的每一步，normalize 則再除以該組的標準差。',
    params: {
      normalize: '是否再除以組內標準差。預設關閉，這樣算出來的數字還能跟手算對得起來。',
    },
  },
  PreferenceDataset: {
    description: '合成偏好對，分成訓練與保留兩份',
    details:
      '模擬 RLHF 的偏好資料。每筆的真實品質是前 signal_dims 個維度的加權和，品質高的那一筆就是贏家。最後一個維度是捷徑：' +
      '在訓練集裡跟品質高度相關，在保留集裡則是純雜訊，所以獎勵模型只要抓到它，訓練準確率滿分、保留集準確率卻會掉下來，這就是可重現的獎勵作弊。' +
      'shortcut_strength 設 0 就沒有捷徑。',
    params: {
      n_pairs: '訓練用的偏好對數量。',
      holdout_pairs: '保留驗證用的偏好對數量，只拿來量、不拿來訓練。',
      feature_dim: '一個回答的特徵維度。',
      signal_dims: '有幾個維度承載真實品質。越多越分散，也就越難贏過那條捷徑。',
      shortcut_strength: '捷徑維度在訓練集裡有多響亮。設 0 等於整條拿掉，也就是對照組（保留集準確率會維持在高點）。',
      seed: '資料生成的隨機種子，確保可重現。',
    },
  },
  BradleyTerryLoss: {
    description: '從兩個獎勵分數算出偏好損失、P(w>l) 與準確率',
    details:
      '損失為 -log sigmoid(r_w - r_l)，只有兩個分數的差有意義：兩邊同時加上一個常數，機率完全不變。這就是 RLHF ' +
      '從來不需要人類給絕對分數的原因，也是兩個獎勵模型的分數不能互相比較的原因。',
    params: {},
  },
  BradleyTerryTrain: {
    description: '在偏好對上訓練獎勵模型',
    details:
      '以 Bradley-Terry 目標訓練一個兩層 MLP，每個 epoch 記錄損失與兩個準確率。兩個準確率之間的落差就是獎勵作弊：' +
      '模型可以在訓練過的那批資料上拿到滿分 1.000，學到的卻是捷徑而不是偏好。保留集輸入是選填的，沒接時保留集的數字是 NaN。',
    params: {
      epochs: '在偏好對上跑幾輪。夠把訓練集完全學起來即可。',
      hidden_dim: '獎勵頭中間層的寬度。',
      lr: '學習率。',
      seed: '訓練的隨機種子，確保可重現。',
    },
  },

  // ── Data ──
  TensorInput: {
    description: '手動輸入張量，或以隨機、零、一、arange 填滿',
    details: '隨機模式的數值由 seed 決定，同一個 seed 每次都得到相同的張量；explicit、zeros、ones、arange 不受 seed 影響。',
    params: {
      shape: '張量形狀，以逗號分隔的整數（例如 \'1,4,4\'）',
      dtype: '資料型別',
      value_mode: '張量填充方式',
      values: '巢狀值列表（當 value_mode=explicit 時使用）',
      seed: '可重現隨機數的種子（當 value_mode=random 時使用）',
    },
  },
  Dataset: {
    description: '載入影像資料集，如 MNIST、CIFAR10',
    details:
      '把變換鏈接到 train_transform 或 eval_transform 就能控制前處理與資料增強；兩個都沒接時套用 ' +
      'ToTensor 與 Normalize(0.5)。第一次執行會把檔案下載到 data_dir。',
    params: {
      name: '要載入的資料集',
      split: '資料分割',
      data_dir: '下載/儲存資料集的目錄',
    },
  },
  ImageFolderDataset: {
    description: '從「一個類別一個資料夾」的結構載入自己的影像',
    details:
      '標籤依資料夾名稱的字母順序決定，classes 輸出也照同一順序列出。split 決定要讀 path 底下哪個子目錄；類別資料夾直接放在 ' +
      'path 之下時選「(none)」。',
    params: {
      path: '放置各個分割的資料夾。相對路徑會相對於同時放著 models/ 與 images/ 的資料目錄。',
      split: '要載入的子資料夾。如果類別資料夾直接放在 path 底下、沒有分割這一層，選「(none)」；這時沒有分割可以區分兩個 transform 埠，所以接了哪一個就用哪一個，兩個都接時以 train_transform 為準。',
    },
  },
  SyntheticDataset: {
    description: '生成 2D 玩具資料集，如同心圓、雙月',
    details:
      '底層是 sklearn 的 make_circles / make_moons / make_blobs / ' +
      'make_classification，seed 固定產生的點。三個輸出與 CSVReader 相同，TrainTestSplit ' +
      '與分類器節點可以直接接上。',
    params: {
      kind: 'circles 同心圓（線性不可分）；moons 雙交錯半月；blobs 等向高斯群聚（線性可分）；classification 通用 sklearn make_classification。',
      n_samples: '要生成的樣本總數。',
      noise: '加在點上的高斯噪聲（僅 circles/moons/classification 用得到）。',
      factor: 'circles 內外圈半徑比，介於 0 與 1 之間。其他 kind 會忽略。',
      centers: 'blob 群聚的中心數（只在 kind=blobs 時使用）。',
      seed: '可重現用的隨機種子。',
    },
  },
  SyntheticSequence: {
    description: '生成整數序列資料集，標籤就在序列一端',
    details:
      '其餘位置都是不帶資訊的干擾 Token。recall_first 把答案放在第 1 個位置，依賴距離等於整條序列，是檢驗 RNN ' +
      '梯度消失的標準任務；recall_last 放在最後一格，距離只有 1。vocab_size 輸出等於 n_classes + ' +
      'n_distractors，即下游 Embedding 的 num_embeddings 最小值。',
    params: {
      kind: 'recall_first 答案在第 1 個位置（依賴距離 = seq_len）；recall_last 答案在最後一個位置（依賴距離 = 1）。',
      seq_len: '每條序列的長度 T。調大就是把依賴距離拉遠。',
      n_samples: '要生成幾筆序列。太少模型會直接背下來，建議上萬筆。',
      n_classes: '答案有幾種（也就是類別數）。亂猜的準確率是 1/n_classes、亂猜的 loss 是 ln(n_classes)。',
      n_distractors: '干擾 Token 有幾種。它們不帶任何資訊，只負責把答案和輸出隔開。',
      seed: '可重現用的隨機種子。訓練集與測試集請用不同的 seed。',
    },
  },
  HuggingFaceDataset: {
    description: '從 HuggingFace 載入影像分類資料集',
    details:
      '資料由 HuggingFace 的 datasets 套件載入，第一次執行需要網路連線。資料集欄位不是預設的 image / label ' +
      '時，請設定 image_column 與 label_column；split 也接受 train[:1000] 這種切片寫法。',
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
    description: '下載 Kaggle 資料集為 ImageFolder',
    details:
      '需要網路連線與 Kaggle 憑證：環境變數 KAGGLE_USERNAME 與 KAGGLE_KEY，或放在 ~/.kaggle 的 ' +
      'kaggle.json。類別資料夾不在下載內容的根目錄時，用 subdir 指到真正的起點。',
    params: {
      dataset_slug: 'Kaggle dataset 的 owner/slug（例：puneet6060/intel-image-classification）',
      subdir: '下載後資料夾內，包含 class 子資料夾的相對路徑',
      cache_dir: '覆寫 kagglehub 快取位置（空=用預設）',
    },
  },
  DataLoader: {
    description: '把資料集分成批次，供訓練逐批取用',
    details: 'shuffle 每個 epoch 重新排序，用的是由本次執行種子衍生的產生器，所以順序只取決於種子。',
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
    description: '為資料集裝上前處理流程',
    details:
      '把變換鏈接到 transform 之後，下面三個參數就會被忽略。這個節點是給本身沒有 transform 輸入埠的資料集用的，例如 ' +
      'HuggingFaceDataset、KaggleDataset 與自訂資料集；Dataset 與 ImageFolderDataset ' +
      '自己就能接變換鏈。SyntheticShapes 與 SyntheticSegmentation 會忽略裝上去的流程。',
    params: {
      resize: '調整大小維度（0 表示不調整）。接上變換鏈時會被忽略。',
      normalize: '套用正規化（mean=0.5, std=0.5）。接上變換鏈時會被忽略；資料集統計值的預設組合在 NormalizeTransform。',
      to_tensor: '將 PIL 影像轉為張量。接上變換鏈時會被忽略。',
    },
  },

  // ── Data / 變換鏈（core#136）──
  ResizeTransform: {
    description: '把每個樣本縮放成指定邊長的正方形',
    details: '放在 ToTensorTransform 之前。兩邊都明確指定，所以非正方形的影像會被壓扁，不會維持原本的比例。',
    params: {
      size: '縮放後正方形的邊長（像素）',
      interpolation: '重取樣濾波器。nearest 保留硬邊緣（遮罩、標籤圖）；bicubic 在照片上比較銳利。',
    },
  },
  ToTensorTransform: {
    description: 'PIL 影像轉成 0~1 的 CxHxW 浮點張量',
    details: '多數變換鏈的分界：幾何與色彩步驟放在它前面，NormalizeTransform 放在它後面。',
  },
  NormalizeTransform: {
    description: '各通道標準化為 (x - mean) / std',
    details: '需要張量，所以放在 ToTensorTransform 之後。預設組合收錄 ImageNet、CIFAR-10、CIFAR-100 公布的通道統計值。',
    params: {
      preset: '用來標準化的通道統計值。Half 會把 [0, 1] 映射到 [-1, 1]，也是 CodefyUI 在有預設組合之前一直採用的做法；想重現論文結果時，請選你實際訓練的資料集。',
      mean: '每個通道的平均值，以逗號分隔。只給一個值就套用到所有通道。',
      std: '每個通道的標準差，以逗號分隔。只給一個值就套用到所有通道。',
    },
  },
  RandomCrop: {
    description: '先補邊，再隨機取一個 size x size 的視窗',
    details: 'size 32 配 padding 4 是標準的 CIFAR-10 設定：物體每輪都會偏移幾個像素。',
    params: {
      size: '裁切後正方形的邊長（像素）',
      padding: '裁切前四邊各補上的像素數。設 0 會真的裁出比原圖小的視窗；補的量等於想要的位移量時，輸出大小會和輸入一樣。',
      padding_mode: '補上的邊框內容。constant 是黑色，reflect 則鏡射影像邊緣（不會留下人工邊框讓模型去學）。',
    },
  },
  RandomHorizontalFlip: {
    description: '以機率 p 左右鏡射影像',
    details: '用在照片上沒問題；但左右有意義的資料（數字、文字）不能這樣做。',
    params: {
      p: '每個樣本被翻轉的機率',
    },
  },
  RandomRotation: {
    description: '隨機旋轉，角度在 ±degrees 以內',
    details: '角度在範圍內均勻抽樣。小角度對手寫與衛星影像有幫助；角度太大會破壞方向本身就是特徵的類別。',
    params: {
      degrees: '旋轉範圍的半寬（度）。設 15 表示每個樣本最多往任一邊轉 15 度。',
      expand: '放大輸出畫布，避免角落被裁掉。這會改變影像尺寸，所以下游任何假設固定形狀的節點後面都要再接一個縮放。',
      fill: '旋轉後空出來的角落要填什麼值。0 是黑色。',
    },
  },
  ColorJitter: {
    description: '隨機調整亮度、對比、飽和度與色相',
    details: '預設值（亮度、對比、飽和度 0.4，色相 0.1）是常見 ImageNet 配方採用的數值。',
    params: {
      brightness: '亮度會乘上一個從 [1-b, 1+b] 抽出的係數。設 0 表示停用。',
      contrast: '範圍規則與亮度相同。設 0 表示停用。',
      saturation: '範圍規則與亮度相同。設 0 表示停用。',
      hue: '色相會平移一個從 [-h, +h] 抽出的量，色環寬度為 1，所以上限是 0.5。請設小一點：超過 0.2 左右，某個類別賴以辨識的顏色就不再是那個類別的顏色了。',
    },
  },
  RandAugment: {
    description: '隨機挑 num_ops 個操作套用，強度相同',
    details:
      '操作集合是傾斜、平移、旋轉、色調分離、曝光過度、色彩、對比、亮度、銳利度、直方圖等化、自動對比與 identity。需要 PIL 影像或 ' +
      'uint8 張量，所以放在 ToTensorTransform 之前。',
    params: {
      num_ops: '每個樣本要套用幾個操作。論文的預設值是 2。',
      magnitude: '所有操作的強度，範圍是 0 到 num_magnitude_bins - 1。模型與資料集越大就調越高；小模型配小資料集，通常還沒需要 15 就已經欠擬合了。',
      num_magnitude_bins: '強度刻度的解析度。torchvision 的預設值是 31；改動它會連帶改變 magnitude 的意義。',
    },
  },
  ComposeTransform: {
    description: '依埠的順序把多條變換鏈合併成一條',
    details:
      '節點接節點本身就會組合，所以只有在兩條鏈分開建立、又要合進同一條流程時才需要它。沒接線的埠會跳過，三個埠只接 step_1 和 ' +
      'step_3 就是依序組合這兩條。',
    params: {
      steps: '要合併幾條鏈',
    },
  },
  CSVReader: {
    description: '把 CSV 載入為特徵、標籤與欄位名稱',
    details:
      '數值欄位組成 [N, F] 的 float32 張量；include_columns 會再縮小這個範圍，沒設定時非數值欄位一律捨棄。' +
      'target_column 指定的欄位變成字串標籤列表。檔案沒有標題列時關掉 skip_header，欄位會依序命名為 0、1、2。',
    params: {
      path: 'CSV 檔案路徑（絕對路徑或相對於後端工作目錄）。',
      target_column: '標籤欄位名稱（選填）。留空表示沒有標籤、純資料載入。',
      include_columns: '要保留的特徵欄位（逗號分隔，選填）。留空表示「除了 target 之外所有數值欄位」。',
      skip_header: 'True 代表第一列是欄位名稱；False 則自動把欄位命名為 0、1、2…',
    },
  },
  ColumnSelector: {
    description: '從 2D 張量中挑出指定的欄位，依位置或名稱',
    details: '用名稱挑選時必須連上 columns 輸入；indices 與 names 同時設定時以 names 為準。',
    params: {
      indices: '以逗號分隔的欄位索引，例：「0,2,3」。當 names 為空時使用。',
      names: '以逗號分隔的欄位名稱。一旦設定就會蓋過 indices，並且需要連上 columns 輸入。',
    },
  },
  Normalize: {
    description: '沿指定軸正規化張量，並輸出所用的統計量',
    details:
      'zscore = $(x-\\mu)/\\sigma$、minmax = $(x-\\min)/(\\max-\\min)$、unit_norm = ' +
      '$x/\\|x\\|_2$。axis=0 逐欄計算，axis=1 逐列計算。整欄數值相同時除以 1 而不是 0，結果為 0 而非 NaN。',
    params: {
      mode: '正規化方法。',
      axis: '計算統計量的軸。0 = 逐欄，1 = 逐列。',
    },
  },
  TrainTestSplit: {
    description: '把特徵與標籤切成訓練集與測試集',
    details:
      '切分本身呼叫的是 sklearn.model_selection.train_test_split。開啟 stratify ' +
      '會讓兩邊維持相同的類別比例，標籤不平衡時尤其重要。',
    params: {
      test_size: '保留作為測試集的樣本比例，必須介於 (0, 1)。',
      seed: '隨機洗牌的種子，方便可重現。',
      stratify: '是否在兩邊保留每個類別的比例（分層抽樣）。',
    },
  },
  DatasetBatch: {
    description: '從資料集中取出一個批次，輸出影像與對應標籤',
    details:
      '影像輸出形狀為 (N, C, H, W)。標籤在分類資料集是 (N,) 的類別編號，在分割資料集是 (N, H, W) 的逐像素遮罩。' +
      'start_index 超過資料集長度時會繞回開頭。',
  },
  RowSelector: {
    description: '從 2D 張量中挑出指定的列，依位置或名稱',
    details:
      '用名稱挑選時必須連上 labels 輸入；indices 與 names 同時設定時以 names 為準。要依條件（例如分數 > 80）' +
      '篩選，請用 EDU 套件裡的 FilterRows。',
  },
  SyntheticSegmentation: {
    description: '生成影像與逐像素遮罩的資料集',
    details:
      '每張是單通道小圖，上面隨機畫一到兩個形狀；遮罩逐像素標記 0 為背景、1 為圓形、2 為方形，規模小到能在 CPU 上訓練 UNet。' +
      '全部由 seed 在記憶體中生成，不需要下載；訓練集與測試集請用不同的 seed。',
  },
  SyntheticShapes: {
    description: '生成小圖資料集，每張有一個柔和光斑',
    details:
      '影像為單通道，數值正規化到 [-1, 1]，也就是擴散模型訓練慣用的範圍。每個光斑是暗背景上的高斯函數，位置與大小由 seed 決定。' +
      '每筆附一個固定為 0 的假標籤，可以直接通過 DataLoader。',
  },

  // ── Training ──
  Optimizer: {
    description: '把梯度變成權重更新：9 種演算法',
    details:
      '共 9 種 torch 演算法。超參數只會在接受它的演算法下出現（momentum 屬於 SGD 與 RMSprop，betas 屬於 ' +
      'Adam 家族，amsgrad 只有 Adam 與 AdamW）；weight_decay 設成非 0 卻選了沒有這個參數的演算法（如 ' +
      'Rprop）會直接失敗，而不是悄悄忽略。',
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
    description: '交叉熵、MSE、BCE 等 11 種損失函數',
    details:
      '共 11 種 torch 損失函數。只有部分類型接受的選項（label_smoothing、類別權重、ignore_index、' +
      'pos_weight）只會在適用的類型下出現。',
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
    description: '訓練模型指定的輪數，輸出訓練後的模型與 loss',
    details:
      '驗證、早停、學習率排程、梯度裁剪、混合精度與梯度累積都是選用。接續訓練是把 CheckpointLoader.epoch 接到 ' +
      'start_epoch：輪數編號是整段訓練的絕對值，losses 只涵蓋這次執行跑過的輪。按停止會寫出中斷檢查點並回傳已算出的曲線，' +
      '不會報錯。',
    params: {
      epochs: '訓練 epoch 數量',
      device: '此節點的裝置。保持 auto 就跟隨這張圖的裝置（工具列的圖裝置，沒有指定時則用設定）。一張圖只在一個裝置上執行；若部分工作需要不同裝置，請拆成兩張圖。',
      early_stopping_patience: '監控的指標未改善 N 個 epoch 後停止（0 = 停用）',
      monitor: '早停監控的指標。val_loss：越低越好（預設）。val_accuracy：越高越好，僅在使用分類損失函數（CrossEntropyLoss/NLLLoss）且有接上 val_dataloader 時才會記錄；兩者缺一就會退回 val_loss 並記錄警告，而不是去監控一個從未被算出來的數值。',
      checkpoint_every: '每 N 個完整 epoch 存一次檢查點，讓伺服器當機時最多只損失 N 個 epoch，而不是整次執行。與 CheckpointSaver 互相獨立，恢復方式也相同：把 CheckpointLoader.epoch 接到 start_epoch。每個檢查點大小大約是模型加上優化器狀態（常常是模型本身的好幾倍），而且是在訓練執行緒上同步寫入；大模型搭配偏低的 N、跑很長的訓練，執行完之前可能會用掉好幾 GB 磁碟空間，因為執行中的任務目前沒有機制限制這件事（0 = 停用）',
      grad_clip_norm: '最大梯度範數裁剪（0 = 停用）',
      batch_metrics: '同時把每一批的損失記錄成 train_loss_batch 這條序列（預設關閉：每批一列資料量相當可觀）',
      precision: '混合精度。bf16 在 Ampere 以後的顯卡上可以把 activation 記憶體用量大約減半，其他都不用改；fp16 則是給更舊的顯卡用的，會額外搭配 loss scaler。val_accuracy 會用跟這個相同的精度計算（不會強制轉成 fp32），才能跟 val_loss 維持可比較性——但精度較低的 logit 在接近平手時可能讓 argmax 換邊，讓量出來的準確率有些微變動。裝置做不到的話會自動退回 fp32 並記錄下來。',
      accumulate_steps: '累積這麼多批之後才做一次優化器更新，同時把每一批的損失除以同一個數字。batch_size 8 搭配 4，梯度會等同於 batch_size 32，但記憶體裡同時只放 8 筆（1 = 關閉）。',
      max_steps: '總共跑滿這麼多次優化器更新後就停止，不論 epochs 設定為何。算的是優化器更新次數而不是批次數，所以不管 accumulate_steps 設多少意思都一樣（0 = 不限制）',
      log_interval: '開啟批次指標時，每 N 批記錄一次。長時間執行時調高可以讓圖表稀疏一點。',
      scheduler_step: '學習率排程器何時前進（#297）。epoch 是歷史行為；optimizer_step 會在每次優化器更新後走一步 — 以步數計價的排程（warmup_cosine、total_steps = max_steps 的 OneCycleLR）在步數預算的執行裡需要這個模式。ReduceLROnPlateau 由指標驅動，兩種模式下都維持每 epoch 一步。這個設定同時決定 LRScheduler 節點上每一個長度（T_max、total_steps、step_size）的單位是什麼，以及排程長度提醒是拿哪一把尺去量：epoch 模式量的是 epoch 數，optimizer_step 模式量的是這次執行的優化器步數預算（#308）。',
      log_grad_norm: '每 log_interval 個優化器步記錄一次裁剪前的全域梯度範數（grad_norm 序列；設定 grad_clip_norm 時另記 grad_norm_clipped）— loss 突波與穩定性鑑識的原料（#298）。',
      log_update_ratio: '每 log_interval 個優化器步記錄 ||lr×grad|| / ||weights||（全域近似）為 update_ratio 序列 — 學習率健康度的經典訊號，約 1e-3 是常見的健康量級（#298）。',
      val_every_steps: '每 N 個優化器步在接入的 val_dataloader 上做一次訓練中途驗證，記錄 val_loss_step 序列（0 = 關閉）。epochs=1 + max_steps 的跑法裡，這是取得驗證「曲線」而非單一終點的唯一方式（#298）。',
      checkpoint_every_steps: '每 N 個優化器步存一個週期檢查點（0 = 關閉）。步數里程碑快照是研究能力湧現的原料；單 epoch 的跑法裡每 epoch 檢查點永遠不會觸發（#298）。檔案大小警告同 checkpoint_every；快照的 epoch 欄位承載的是步數，不適合拿來續訓。',
      deterministic: '要求 PyTorch 使用決定性的運算核心。沒有決定性實作的運算會發出警告，而不會讓執行失敗。',
      tensorboard: '同時把指標寫成 TensorBoard 事件檔，放在這次執行專屬的資料夾裡。用 `tensorboard --logdir <路徑>` 開啟；該路徑會列在這次執行的產出檔案中。',
    },
  },
  EvaluateModel: {
    description: '分類模型在資料集上的準確率，附答對數與總數',
    details:
      '以批次跑完整個資料集，對每筆取 argmax 與標籤比對。準確率同時會以 eval_accuracy 指標記在指定的 step 上，' +
      '同一張圖有兩顆 EvaluateModel 就要設不同的 step，否則會蓋掉彼此的點。',
    params: {
      batch_size: '評估時每批跑幾筆（不影響結果，只影響速度/記憶體）。',
      device: '此節點的裝置。保持 auto 就跟隨這張圖的裝置（工具列的圖裝置，沒有指定時則用設定）。一張圖只在一個裝置上執行；若部分工作需要不同裝置，請拆成兩張圖。',
      precision: '前向傳播用的混合精度。bf16 在 Ampere 以後的顯卡上可以把 activation 記憶體用量大約減半，其他都不用改；fp16 則是給更舊的顯卡用的。不論選哪一種，參數都維持 fp32；但降精度的前向傳播仍可能讓量出來的準確率有些微變動（精度較低的 logit 在接近平手時可能讓 argmax 換邊），所以要回報的準確率應該用 fp32 這個數字。裝置做不到的話會自動退回 fp32 並記錄下來。',
      step: 'eval_accuracy 這個指標記錄時使用的 step 值。同一張圖裡有多個 EvaluateModel 節點時（例如微調前後的比較），需要各自設定不同的 step，否則會在圖表上互相覆蓋。',
    },
  },
  BackwardOnce: {
    description: '標記反向傳播的起點，張量原樣輸出',
    details:
      'Backward 模式是工具列上的開關。開啟時，引擎會對這裡輸出的張量取合成純量 $\\mathcal{L} = ' +
      '\\sum(\\text{input})$ 再做反向傳播；關閉時節點照樣執行，只是原樣傳遞、沒有副作用。',
  },

  LRScheduler: {
    description: '訓練中改變學習率：10 種排程',
    details:
      '共 10 種排程，含 StepLR、CosineAnnealingLR、ReduceLROnPlateau、OneCycleLR 與 ' +
      'warmup 系列。step_size、T_max、total_steps 的單位由 TrainingLoop 的 ' +
      'scheduler_step 決定：預設是輪數，另一種是優化器步數。長度大於整段訓練時排程跑不完。',
    params: {
      type: '排程器類型',
      step_size:
        'StepLR：每隔多久降一次學習率。MultiStepLR：在此值的 1、2、3、4 倍處各降一次。單位跟著 TrainingLoop.scheduler_step 走（#308）：預設的 epoch 模式算的是 epoch（不是 batch），optimizer_step 模式算的是優化器步數。兩種模式下都必須小於整次執行在該單位下的長度，否則第一次下降永遠不會發生，整次執行都在同一個學習率上跑。',
      gamma:
        'StepLR、MultiStepLR、ExponentialLR 的衰減因子；ReduceLROnPlateau 也拿它當 factor。',
      T_max:
        'CosineAnnealingLR：一個 cosine 週期的長度。單位跟著 TrainingLoop.scheduler_step 走（#308）：預設的 epoch 模式單位是 epoch，要設成和 TrainingLoop.epochs 一樣；optimizer_step 模式單位是優化器步數，要改成設成整次執行的步數預算（有設 max_steps 就是它，否則是 epochs × 每個 epoch 的批次數 ÷ accumulate_steps）。兩種模式下設小了會讓週期提早結束、cosine 曲線再往上翻，執行的尾段學習率反而是升的；設大了則是只走到曲線中途、結束時學習率還很高。兩者通常會少掉幾個百分點的準確率，而且畫面上完全看不出來是排程造成的。這裡刻意不強制：截斷的排程本來就是合理選擇。CosineAnnealingWarmRestarts 會把這個值當成 T_0（第一次重啟前的週期長度），那種情況下設成和整次執行的長度一樣反而永遠不會重啟。',
      max_lr: 'OneCycleLR 的最大學習率',
      total_steps:
        'OneCycleLR：整個 one-cycle 排程的長度；warmup 家族也用這個值當總長（暖身斜坡 + 衰減）。單位跟著 TrainingLoop.scheduler_step 走（#308）：預設的 epoch 模式是 epoch 數，要填 TrainingLoop.epochs，不是 batch 數（OneCycleLR 官方文件講的 step 是 batch）；optimizer_step 模式下它就是那個步數，要填整次執行的步數預算：有設 max_steps 就是它，否則是 epochs × 每個 epoch 的批次數 ÷ accumulate_steps。預設值 1000 遠大於一般的 epoch 數，在 epoch 模式照著不改就只會走到週期的開頭：學習率稍微升上去，然後從來不會退火下來。',
      warmup_steps:
        'warmup_cosine / warmup_linear / constant_with_warmup：先以線性斜坡從 ~0 升到優化器學習率的步數，之後才進入衰減段。以「排程器步」計價 — 搭配 TrainingLoop 的 scheduler_step=optimizer_step 並把 total_steps 設成該跑的 max_steps，「暖身 100 步再在 1500 步內退火」才是字面意思（#297）。預設的 scheduler_step=epoch 模式下一個排程器步就是一個 epoch，所以預設的 100 在只跑 5 個 epoch 的執行裡斜坡根本爬不完，整次執行都不會用到你設定的學習率（#308）。',
    },
  },

  // ── IO ──
  ImageReader: {
    description: '讀取影像為 (C,H,W) 張量，值域 [0,1]',
    details: '只寫檔名時會先找上傳影像目錄，找不到才依原樣相對於執行圖的行程工作目錄解析。resize 設為 0 就保留原尺寸。',
    params: {
      path: '選擇已上傳的影像，或上傳新檔案',
      mode: '載入影像的色彩模式（L = 灰階）',
      resize: '縮放為 (resize, resize) 正方形（0 = 不縮放）',
    },
  },
  ImageWriter: {
    description: '把影像張量存成 PNG/JPEG/BMP/TIFF',
    details: '副檔名跟著選定的格式走。相對路徑會寫到資料目錄下的 output 資料夾；輸入是 (N, C, H, W) 批次張量時只存第一張。',
    params: {
      path: '輸出檔案路徑',
      format: '影像格式',
    },
  },
  VideoWrite: {
    description: '把幀張量編碼成 mp4 或 gif，寫入媒體目錄',
    details:
      '輸入可以是 (T, C, H, W) 或 (T, H, W, C)。PATH 上有 ffmpeg 執行檔就輸出 mp4，沒有就退回 ' +
      'Pillow 輸出 gif。同名檔案會被覆蓋；另外輸出可在編輯器內嵌播放的參照與中間幀的 PNG。',
    params: {
      filename: '媒體目錄下的檔名（可含子資料夾）；副檔名依格式決定，同名會覆寫',
      format: 'auto：PATH 上有 ffmpeg 則 mp4，否則 gif。gif 永遠可用；mp4 需要安裝 ffmpeg',
      fps: '播放幀率（每秒幀數）',
      resize: '輸出高度（像素），等比縮放、最近鄰（0 = 原始大小）；96px 的研究畫面放大 2-3 倍較易觀看',
    },
  },
  VideoLoad: {
    description: '解碼影片為幀張量 (T,3,H,W)、fps 與幀數',
    details:
      '幀是 float、值域 [0,1]。mp4 與 webm 走 ffmpeg 執行檔解碼，gif 走 Pillow。相對路徑以媒體目錄為基準，' +
      '也就是 VideoWrite 的輸出位置。沒設 max_frames 時，長影片會整支載進記憶體。',
    params: {
      path: '影片檔案：絕對路徑，或相對於媒體目錄',
      max_frames: '最多解碼幀數（0 = 全部）；未設上限的長片會整段載入記憶體',
      stride: '每 N 幀取 1 幀（fps 輸出會等比例下降）',
    },
  },
  ImageBatchReader: {
    description: '把目錄下每張影像堆成 (N, C, H, W)',
    details: '符合 glob 樣式的檔案依檔名排序讀入，並全部縮放成同一個正方形尺寸才能堆疊；開不起來的影像會略過。max_images 可限制讀取張數。',
    params: {
      directory: '包含影像檔案的目錄',
      pattern: '檔案比對模式（如 *.png、*.jpg）',
      resize: '將所有影像調整為此正方形大小（批次處理必需）',
      max_images: '最大載入影像數（0 = 全部）',
      mode: '色彩模式',
    },
  },
  FileReader: {
    description: '讀文字或 CSV 檔為字串，數值 CSV 另出張量',
    details:
      'csv 模式把數值列轉成 2D 浮點張量，只要有一格不是數字，張量輸出就是空的；text 模式恆為空。相對路徑落在 graphs ' +
      '目錄（專案模式下是 assets/data），專案資料目錄以外的檔案一律拒讀。',
    params: {
      path: '檔案路徑',
      mode: '讀取方式',
      encoding: '文字編碼',
      csv_header: 'CSV 是否有標頭列（載入為張量時跳過）',
    },
  },

  ModelSaver: {
    description: '把模型權重或整個模組存成檔案',
    details:
      'state_dict 模式只存權重，也是預設；full_model 模式把整個模組 pickle 起來，遇到定義在函式內的層類別會直接拒存。' +
      '路徑用 .pth 也可以，safetensors 只支援 state_dict，相對路徑會寫到 models 目錄。',
    params: {
      path: '輸出檔案路徑（.pt、.pth 或 .safetensors）',
      save_mode: '儲存模式：state_dict（推薦）或完整模型',
      format: '檔案格式：pytorch（.pt/.pth）或 safetensors（.safetensors）',
    },
  },
  ModelLoader: {
    description: '從檔案載入權重到模型，或直接載入整個已儲存的模型',
    details:
      'state_dict 模式要接上模型，可讀 .pt、.pth 或 .safetensors。full_model 模式在 torch ' +
      '受限的 unpickler 下重建整個模組，只接受原生 torch.nn 層與 CodefyUI 自己的層，其餘一律拒絕，' +
      '包含自訂節點與外掛的類別。',
    params: {
      path: '權重檔案路徑（.pt、.pth 或 .safetensors）',
      load_mode:
        '載入模式：state_dict（需要模型輸入）或 full_model 完整模型。full_model 會重建存檔中的模組本身，並且在 torch 的受限解序列化器下讀取，因此只接受由標準 torch.nn 層與 CodefyUI 自己的層組成的模型，其餘一律拒絕 —— 包含來自自訂節點或外掛的類別，以及除了 transformer 層會存下的那兩個 torch 啟動函式以外的任何函式',
      device: '此節點的裝置。保持 auto 就跟隨這張圖的裝置（工具列的圖裝置，沒有指定時則用設定）。一張圖只在一個裝置上執行；若部分工作需要不同裝置，請拆成兩張圖。',
      strict: '是否嚴格要求 state_dict 中的鍵值匹配（僅 state_dict 模式）',
    },
  },
  CheckpointSaver: {
    description: '存下模型、優化器、epoch 與損失歷史',
    details: '學習率排程與 fp16 損失縮放狀態在對應輸入有接線時也會一併存入。相對路徑會寫到 models 目錄，CheckpointLoader 讀的是同一種格式。',
    params: {
      path: '輸出檢查點檔案路徑',
      epoch: '要儲存在檢查點中的當前 epoch 數',
    },
  },
  CheckpointLoader: {
    description: '還原模型與優化器，輸出存檔時的 epoch',
    details:
      'epoch 輸出接到 TrainingLoop.start_epoch，損失歷史與 fp16 損失縮放狀態也會一併輸出。只有 ' +
      'lr_scheduler 接上 LRScheduler 時才會還原排程位置；沒接就丟棄並提示，改由 TrainingLoop 重放 ' +
      'start_epoch 步數推算。',
    params: {
      path: '檢查點檔案路徑',
      device: '此節點的裝置。保持 auto 就跟隨這張圖的裝置（工具列的圖裝置，沒有指定時則用設定）。一張圖只在一個裝置上執行；若部分工作需要不同裝置，請拆成兩張圖。',
    },
  },
  Inference: {
    description: '以 eval 模式對模型做前向傳播，不計算梯度',
    details: 'eval() 與搬到指定裝置這兩件事都直接作用在接進來的模型上，節點結束後仍然有效。',
    params: {
      device: '此節點的裝置。保持 auto 就跟隨這張圖的裝置（工具列的圖裝置，沒有指定時則用設定）。一張圖只在一個裝置上執行；若部分工作需要不同裝置，請拆成兩張圖。',
    },
  },
  GraphInput: {
    description: '宣告這張圖的具名輸入，輸出外部傳進來的值',
    details:
      'API 呼叫端以 POST /api/graph/run 提供值；在畫布上執行時改用 default 參數。要接一個 Start 節點進來，' +
      '這個節點才會執行。type=image 時 API 傳 base64，畫布則讀伺服器本機的檔案路徑。',
  },
  GraphOutput: {
    description: '宣告這張圖的具名輸出，回傳接進來的值',
    details: 'POST /api/graph/run 會以這個節點的 name 為鍵回傳該值，CLI 執行器讀的是同一份契約。',
  },

  // ── Data Flow ──
  Switch: {
    description: '依 selector 索引轉發最多四個輸入中的一個',
    details: '所有輸入都會先被求值，selector 才挑出其中一個，是資料流的選擇而不是會跳過計算的分支。索引從 0 開始，超出範圍或沒有連線時會退回 input_0。',
  },
  Map: {
    description: '對清單中的每個元素各跑一次子圖，收集成結果清單',
    details:
      'subgraph 參數填的是已儲存的預設子圖名稱；子圖的第一個對外輸入接收每個元素，第一個對外輸出就是該元素的結果。一個元素算一個工作單位，' +
      '中途停止會在元素之間中斷並回傳已完成的部分。',
    params: {
      subgraph: '要套用到每個元素的子圖/預設模組名稱',
    },
  },
  Reduce: {
    description: '把清單聚合成單一值：總和、平均、最小、最大',
    details:
      'operation 可選 sum、mean、min、max、concat、stack、first、last，dim 只對 concat 與 ' +
      'stack 有作用。除了 first 與 last 之外，元素必須是數值或張量，空清單會直接報錯。',
    params: {
      operation: '聚合運算',
      dim: 'concat/stack 運算的維度',
    },
  },

  // ── Tensor Operations ──
  Permute: {
    description: '重新排列張量的維度順序',
    params: {
      dims: '新的維度順序（逗號分隔整數）',
    },
  },
  Squeeze: {
    description: '移除張量中大小為 1 的維度',
    params: {
      dim: '要壓縮的維度（-1 表示全部）',
    },
  },
  Unsqueeze: {
    description: '在指定位置插入一個大小為 1 的維度',
    params: {
      dim: '要插入的維度位置',
    },
  },
  Add: {
    description: '兩個張量逐元素相加，支援廣播',
    params: {
      alpha: 'tensor_b 的乘數：a + alpha * b',
    },
  },
  Multiply: {
    description: '兩個張量逐元素相乘，支援廣播',
  },
  MaskedFill: {
    description: '把遮罩標記的位置換成常數，預設 $-\\infty$',
    details:
      '放在 Softmax 之前：因為 $e^{-\\infty} = 0$，被擋住的位置會拿到恰好 0 的機率，其餘位置重新歸一化後加總仍為 1。' +
      '改在 softmax 之後才歸零，該列就不再加總為 1。輸入張量必須是浮點數。',
    params: {
      value: '填入被遮住位置的值。-inf 是注意力的標準做法（softmax 之後變成 0）；zero 與 custom 用於非注意力的遮罩，不會得到正確的注意力權重。',
      custom_value: '當 value 選 custom 時實際填入的數值。',
    },
  },
  MatMul: {
    description: '兩個張量的批次矩陣乘法',
    details: '行為同 torch.matmul：最後兩維做矩陣相乘，前面的維度會以批次方式廣播。兩個一維輸入則得到內積。',
  },
  Mean: {
    description: '沿一或多個維度取平均',
    params: {
      dim: '要縮減的維度（逗號分隔整數）',
      keepdim: '是否保留被縮減的維度',
    },
  },
  Softmax: {
    description: '沿指定維度把分數轉成加總為 1 的機率',
    details: '$\\text{softmax}(x_i) = \\frac{e^{x_i}}{\\sum_j e^{x_j}}$，計算時會先減去最大值，避免數值過大而溢位。',
    params: {
      dim: '要套用 Softmax 的維度',
    },
  },
  Split: {
    description: '沿指定維度把張量切成數個區塊',
    details: '切法同 torch.chunk，維度長度無法整除時最後一塊會比較小，也可能切出的區塊比輸出埠還少。',
    params: {
      chunks: '要切分的區塊數',
      dim: '要切分的維度',
    },
  },
  Stack: {
    description: '沿新的維度把兩個張量疊起來',
    details: '兩個張量的形狀必須相同。',
    params: {
      dim: '要堆疊的維度',
    },
  },
  TensorCreate: {
    description: '依指定形狀建立張量：零、一、隨機值或常數',
    details: 'arange 只會用到 shape 的第一個數字，輸出一維浮點張量。',
    params: {
      shape: '張量形狀（逗號分隔整數）',
      fill: '填充方法',
      value: '填充值（僅 full 模式）',
      requires_grad: '張量是否需要梯度',
    },
  },
  Argmax: {
    description: '沿指定維度取最大值的索引',
  },
  ScalarMultiply: {
    description: '把張量的每個元素乘上一個常數',
  },

  // ── Utility ──
  Print: {
    description: '把值印到主控台，並原樣傳出',
    params: {
      label: '標籤前綴',
    },
  },
  PythonScript: {
    description: '在畫布上寫 Python，回傳以連接埠為鍵的字典',
    details:
      '腳本要定義 run(inputs, params)；回傳的不是字典時視為 out1。只能碰 collections、itertools、' +
      'json、math、numpy、re、statistics、torch，限制的是能匯入哪些函式庫，不是它們能做什麼：這是防護欄，不是沙箱，' +
      '程式碼以你的權限在 CodefyUI 行程內執行，只跑你信任的腳本。此節點不快取，每次都重跑。',
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
    description: '把資料畫成折線、直方圖、熱圖或影像',
    details:
      '圖由 matplotlib 繪製。line 與 histogram 會先把多維資料攤平，heatmap 會把一維輸入摺成方形格子，' +
      'image 則把 (N, C, H, W) 批次排成網格。',
    params: {
      title: '圖表標題',
      plot_type: '要生成的圖表類型',
    },
  },

  Flatten: {
    description: '從 start_dim 起把張量的維度展平',
    details: '底層是 nn.Flatten，從 start_dim 到最後一維全部合併成一維。',
    params: {
      start_dim: '開始展平的維度',
    },
  },
  Linear: {
    description: '全連接層：$y = xW^T + b$',
    details: '底層是 nn.Linear，也叫密集（dense）層。in_features 必須等於輸入張量的最後一維。',
    params: {
      in_features: '輸入特徵大小',
      out_features: '輸出特徵大小',
    },
  },
  SequentialModel: {
    description: '在架構編輯器裡組出網路',
    details:
      '在節點上點兩下可編輯層的連線圖。它是 DAG，除了直線堆疊，也支援分支與合併（Add、Concat、Multiply、Subtract、' +
      'Mean、Stack）；建出的模組可接到 Optimizer 與 TrainingLoop。',
    params: {
      layers: '層定義的 JSON 陣列',
    },
  },
  Embedding: {
    description: '把整數索引對應到可訓練權重矩陣 $W$ 的列',
    details:
      '底層是 nn.Embedding，即 $E[i] = W[i, :]$，這張表會跟著整張圖一起訓練。padding_idx 小於 0 ' +
      '代表不設 padding 列。要用 GloVe 這類預訓練詞向量請改用 LLM 分類下的 WordVector 節點。',
    params: {
      num_embeddings: '詞彙表大小',
      embedding_dim: '每個嵌入向量的維度',
      padding_idx: '填充 token 的索引（-1 表示無）',
    },
  },

  TextInput: {
    description: '輸入一段文字，輸出給任何 STRING 輸入埠',
    params: {
      value: '多行文字。可以拖曳 textarea 右下角調整大小。',
    },
  },
  DecisionBoundary: {
    description: '把訓練好的分類器 2D 決策區域上色，並疊上訓練點',
    details:
      'model 要接 Classifier 節點的輸出：節點會在涵蓋 x_train 的密格上呼叫 predict()，而且只支援 2D 特徵。' +
      'grid_steps 越大邊界越平滑也越慢；show_support_vectors 只在 model 是 SVM 時有效。',
  },
  ScatterPlot2D: {
    description: '把 (N, 2) 的點畫成散點圖，依標籤上色',
    details: '沒有接 labels 時，所有點都畫成同一個顏色。',
  },

  // ── LLM ──
  Tokenizer: {
    description: '把文字切成 token，並輸出 ids 與字元位移',
    details:
      '不同家族用不同演算法：BPE（GPT）、WordPiece（BERT）、SentencePiece（Llama、T5），' +
      '同一段文字會被切成不同的樣子。cl100k_base、o200k_base、p50k_base 與 gpt2 來自 tiktoken，' +
      'BPE 表第一次載入後留在本機；其餘三個從 HuggingFace 下載 tokenizer.json。',
    params: {
      family: 'Tokenizer 家族。tiktoken 完全離線可跑 cl100k/o200k/p50k/gpt2；其餘會在第一次使用時從 HuggingFace 下載 tokenizer.json。',
      text: '要切分的文字。當沒有 `text` 輸入連線時使用此欄位。',
      show_special_tokens: '是否輸出 tokenizer 的特殊 token（BOS/EOS/CLS/SEP/...）。',
    },
  },
  WordVector: {
    description: '為每個輸入單字查一條向量，並回報查到的字',
    details:
      '預訓練向量會把語意相近的字放在一起，所以 $king - man + woman \\approx queen$。demo-16d ' +
      '是離線隨附的 59 字手工詞彙表，類比精確成立；glove-50d 是 word-vectors 套件包裡的 40 萬字 GloVe 表，' +
      '只能近似；sentence-transformer 後端（sentence-embeddings 套件包）把每個字丟進現代編碼器，' +
      '單字結果更雜亂。',
    params: {
      backend: '向量來源。灰掉的選項需要先到套件中心安裝對應的套件包；執行圖的時候永遠不會自動下載。',
      words: '以空白或逗號分隔的單字列表。當沒有 `tokens` 輸入連線時使用此欄位。',
      normalize: '對每個向量做 L2 正規化。下游若要用點積算 cosine similarity，請打開此選項。',
      keep_oov: '對詞彙表外的字輸出零向量，而不是直接略過。只對表格型後端（demo-16d、glove-50d）有意義；句子模型會為每個字都算出向量。',
    },
  },
  EmbeddingScatter: {
    description: '把 [N, D] 嵌入張量投影成 [N, 2] 座標',
    details: 'PCA 取變異數最大的方向，線性、結果固定、快。t-SNE 保留原空間中的鄰近關係，群聚通常更緊，但每個種子的版面都不同。座標會縮放到 [-1, 1]。',
    params: {
      method: 'PCA：線性、決定性、快。t-SNE：非線性、保留局部鄰域結構。',
      perplexity: '只在 t-SNE 使用 — 局部親和模型的鄰域大小。',
      seed: '隨機種子（給 t-SNE）。同樣的種子會得到一樣的版面。',
    },
  },
  CosineSimilarity: {
    description: '每個 query 對 key 的相似度與 top-k',
    details:
      '輸入若已是單位向量，結果就是內積；否則會先正規化。top-k 是每個 query 各自取。exclude_self_words ' +
      '會把指定的標籤排除在 top-k 之外，做類比時才不會撈回輸入的那幾個字。',
    params: {
      top_k: '每個 query 要回傳的最相似 key 數量。',
      exclude_self_words:
        '要從 top-k 排除的標籤（以逗號分隔）。在類比示範中很有用：設成 "king,man,woman" 可以讓 top-1 直接顯示 queen。',
    },
  },
  PositionalEncoding: {
    description: '把位置資訊加到嵌入上，並輸出位置編碼',
    details:
      'sinusoidal 就是 Vaswani et al. (2017) 的公式 $PE(pos, 2i) = \\sin(pos / ' +
      '10000^{2i/d})$，無狀態、結果固定；learnable 回傳的是種子決定的亂數樣式，不是訓練出來的。序列長度超過 ' +
      'max_len 會報錯。',
    params: {
      mode: 'sinusoidal = Vaswani 公式；learnable = 種子可控的隨機初始化。',
      max_len: '支援的最大序列長度。輸入超過此值會直接報錯。',
      seed: 'learnable 模式的隨機種子。sinusoidal 會忽略此值。',
    },
  },
  AttentionMask: {
    description: '布林遮罩 [seq, seq]，True 表示被擋',
    details:
      'causal 擋掉嚴格在未來的位置，也就是 GPT 式 decoder 的樣式；padding 擋掉 token 等於 pad_token ' +
      '的整欄。序列長度取自 tokens 清單，或 tensor 輸入的第 0 維。下游的注意力節點以 ' +
      'scores.masked_fill(mask, -inf) 使用它。',
    params: {
      mode: 'causal：擋掉嚴格未來的位置（decoder 風格）。padding：擋掉值與 pad_token 相同的欄位。',
      pad_token: '視為填補的字串符號，只在 padding 模式下使用。',
    },
  },
  AttentionHeatmap: {
    description: '原樣轉送注意力權重，或只取其中一個 head',
    details:
      '節點卡片會把權重畫成熱圖。接受 [seq, seq]、[H, seq, seq] 或 [B, H, seq, seq]；' +
      'head_index 為 -1 時保留所有 head 並排顯示，設為非負數則取出其中一個。labels 會原樣轉送，供座標軸標註。',
    params: {
      head_index: '若權重是 per-head 形式（[H,seq,seq] 或 [B,H,seq,seq]），可指定顯示哪一個 head。-1 代表保留全部 head 並排顯示。',
      colormap: '熱圖視覺化用的色階（僅前端使用，後端會忽略）。',
    },
  },
  CausalLMModel: {
    description: 'GPT 式純解碼器：下個 token logits',
    details:
      '跟其他模型一樣接到 Optimizer 與 TrainingLoop，損失函數用 LMCrossEntropyLoss。預設值會建出約 ' +
      '204M 參數的模型；把 d_model 與 n_layers 調小，筆電才訓練得動。改動任何結構參數都會丟棄已保存的權重。',
    params: {
      vocab_size: '模型認得幾種不同的 token。必須與餵進來的 tokenizer 一致 — 50257 是 GPT-2 的詞彙量。',
      d_model: 'residual stream 的寬度：每個 token 穿過整個網路時所攜帶的向量大小。必須能被 n_heads 整除。',
      n_layers: '堆疊幾層 transformer block。深度決定了模型能做幾步推理，成本隨層數線性增加。',
      n_heads: '每一層的寬度要切給幾個 attention head。head 越多、同時追蹤的關係越多，但每個 head 的子空間就越窄（寬度為 d_model / n_heads）。',
      d_ff: '每個 block 內部 MLP 的隱藏層寬度，慣例是 d_model 的 4 倍。模型有三分之二的參數住在這裡。',
      max_seq_len: '模型有位置資訊可用的最長序列長度（單位：token）。超過長度的批次會直接報錯而不是截斷；生成時會以這個大小滑動視窗。',
      tie_embeddings: '讓輸入的 embedding 與輸出的 head 共用同一個矩陣。小模型的標準作法：可以省下 vocab_size x d_model 個參數，而且通常還會更好。',
      positional: '模型如何知道一個 token 在什麼位置。learned = 每個位置一個訓練出來的向量（GPT-2）；sinusoidal = 固定的 sin/cos 表（Vaswani et al.）；rope = 依位置旋轉 query 與 key（Llama），對更長的文字外推得最好。',
      norm: '每個 sub-layer 前面的正規化層。rmsnorm 少了減平均與 bias — 稍微便宜一點，也是現代開源模型的選擇。',
      activation: 'MLP 的非線性函數。gelu 是 transformer 的預設；silu（又叫 swish）是 Llama 的選擇；relu 最便宜。',
      dropout: '訓練時被歸零的 activation 比例（0 = 關閉）。大語料預訓練通常關著；在小資料集上微調時再調高。',
      gradient_checkpointing: '反向傳播時重新計算每個 block，而不是把 activation 存下來：記憶體省很多，時間多花約 30%。當一個 batch 塞不進顯卡時再打開。',
      init_std: '權重初始化所用常態分布的標準差。0.02 是 GPT-2 的值；寫回 residual stream 的那幾個投影層還會再除以 sqrt(2 x n_layers)。',
      seed: '權重初始化的隨機種子。同樣的種子會得到同樣的起始模型，兩次跑的差別就只有你改掉的部分。',
      n_kv_heads: 'Grouped-query attention 的 key/value 頭數。0 = 與 n_heads 相同（標準多頭注意力）；較少時一組 query 頭共享一個 K/V 頭（GQA），1 就是 multi-query attention。必須能整除 n_heads。',
      qk_norm: '在注意力點積前對每個頭的 query 與 key 做 RMS 正規化 — 高學習率訓練的標準穩定手段。',
      bias: '注意力與 MLP 投影層的 bias 項。關閉即 Llama 式無 bias 線性層；參數量會有可量測的變化。',
    },
  },
  LMCrossEntropyLoss: {
    description: '每個 token 位置的平均交叉熵',
    details:
      '它會把 (batch, seq_len, vocab_size) 的 logits 與 (batch, seq_len) 的 token ' +
      'ids 攤平對齊，所以 CausalLMModel 能直接接上 TrainingLoop 的 loss_fn。等於 ' +
      'ignore_index（預設 -100）的目標不計損失、也不產生梯度。',
    params: {
      ignore_index: '這個 target id 不會產生任何損失與梯度。可以用在 padding，或指令資料中屬於提示（prompt）的那一半。-100 是各家工具共通的慣例。',
      label_smoothing: '把一小部分機率質量分給其他 token，讓「答對但過度自信」也要付一點代價（0 = 關閉，0.1 是常見值）。',
    },
  },
  LMTokenizer: {
    description: '可重用 tokenizer：文字與 token 互轉',
    details:
      '接到 LMTokenizedDataset 可把語料打包成訓練區塊，接到生成節點則讓它們使用與訓練時相同的 token ids。gpt2 ' +
      '的 50257 個 token 訓練成本最低；cl100k_base 與 o200k_base 每個 token 塞得下更多文字，' +
      '但輸出層要更寬。每種編碼只會下載一次 BPE 對照表，之後就能離線使用。',
    params: {
      encoding:
        '要使用哪一套 BPE 詞彙表。gpt2（50257 個 token）訓練成本最低；cl100k_base（GPT-3.5/4）與 o200k_base（GPT-4o）能用同樣的 token 數塞進更多文字，但輸出層也要寬得多。',
    },
  },
  TextCorpusDataset: {
    description: '從 HuggingFace 或 .txt 載入文字列',
    details:
      '這些列沒有標籤，DataLoader 無法直接批次化：請把 dataset 接到 LMTokenizedDataset，由它切成預測下一個 ' +
      'token 的訓練區塊。預設的 repo 是 roneneldan/TinyStories，約 200 萬篇簡單的童話故事。',
    params: {
      source: '文字的來源：HuggingFace Hub 上已發布的資料集，或這台機器上的 .txt 檔。',
      dataset_name:
        'HuggingFace Hub 的 repo id。TinyStories 大約有 200 萬篇簡單的兒童故事 — 小到訓練得動，也淺到可以直接用肉眼判斷輸出好不好。',
      subset: '多組態（multi-config）資料集要用的組態名稱，例如 wikitext-103-raw-v1（留空 = 該資料集的預設組態）。',
      split: '要載入哪一個 split：train/test/validation，或 HF 的切片語法（例如 train[:5000]）。',
      text_column: '存放文件文字的欄位名稱。慣例是 text；若填錯，錯誤訊息會列出實際有哪些欄位。',
      cache_dir: '覆寫 HuggingFace 下載快取的位置（留空 = HF 預設，通常是 ~/.cache/huggingface）。',
      local_path:
        '要讀取的文字檔。可以從下拉選單挑選已上傳的檔案，或直接輸入路徑（絕對路徑，或相對於後端工作目錄；在專案模式下相對路徑會相對於專案目錄解析）。',
      split_lines:
        '把每一行當成一份獨立文件，而不是把整個檔案讀成一份。每行一句／一筆的檔案請打開；散文請關著，因為換段落並不代表換文件。',
      max_rows:
        '最多保留幾份文件（0 = 全部）。Hub 來源會把它轉成 split 切片，所以其餘資料根本不會下載 — 這是在大語料上試跑整張圖最快的方式。',
    },
  },
  LMTokenizedDataset: {
    description: '把文字列 tokenize 後切成固定長度的訓練區塊',
    details:
      '文件間用 end-of-text token 串接，再把 token 流切成 (input_ids, labels) 配對，labels ' +
      '是 input_ids 往左位移一格，也就是預測下一個 token。輸出接到 DataLoader。打包好的 token 存進磁碟快取，' +
      '只有第一次要花時間 tokenize；這份快取不會自動清除，要清請用 `cdui cache prune`。',
    params: {
      seq_len:
        '每個訓練區塊有幾個 token — 也就是模型學習時看到的上下文長度。不能超過模型的 max_seq_len。越長，attention 的記憶體成本以平方成長。',
      append_eos:
        '在每份文件後面加上 end-of-text token。建議保持開啟：少了它，模型會學到一個故事會直接接到下一個故事，生成時也永遠不會停。',
      max_tokens: '取到這麼多 token 就停（0 = 整份語料）。這是讓一個 epoch 能在一堂課內跑完最快的手段。',
      cache: '把打包好的 token 存到磁碟，語料與設定沒變時就直接重用。若你正在原地編輯語料檔，請關掉。',
      cache_dir:
        '存放 token 快取檔的子目錄（留空 = 資料目錄下的共用快取）。語料、tokenizer、seq_len、append_eos、max_tokens 每換一種組合就多一個檔案，每個 token 佔 8 bytes — 一億 token 的語料每個檔案約 800 MB，而且不會自動刪除。用 `cdui cache list` 看有哪些，用 `cdui cache prune` 清掉。',
    },
  },
  DataMixDataset: {
    description: '把 2-6 個語料混成資料集：依權重抽取或依序串接',
    details:
      'interleave 依權重不重複抽取，同一個種子得到同樣順序；某個語料抽完後就不再被抽，其餘權重重新正規化。concat 則是 ' +
      'corpus_1 全部跑完再接 corpus_2。混合只記錄（來源, 列號）索引，逐列惰性讀取。輸入接 TextCorpusDataset ' +
      '的輸出，結果餵給 LMTokenizedDataset。',
    params: {
      sources: '這顆節點有幾個語料輸入埠。',
      weights: '逗號分隔的抽取權重，每個來源一個（會正規化；只在 interleave 模式使用）。抽完的來源不再被抽，其餘來源重新正規化 — 混合的尾段就是還有剩的語料。',
      mode: 'interleave：種子化的比例抽取、不重複。concat：corpus_1 全部、再 corpus_2… — 有順序的課程。',
      seed: '交錯順序的種子 — 相同種子與輸入會重現同一個混合順序。',
    },
  },
  PerplexityEvaluate: {
    description: '用沒看過的文字評估模型：損失與 perplexity',
    details:
      'perplexity 就是 $\\exp(\\text{val\\_loss})$，大致可讀成模型每一步在幾個機率相當的 token 之間猶豫，' +
      '所以在 50257 個 token 的詞彙表上亂猜就是 50257。這個平均是「每個 token」的，而且綁定這份資料集與這套 ' +
      'tokenizer，只有用同樣方式量出來的數字才能互相比較。標籤為 -100 的位置會跳過。',
    params: {
      batch_size: '一次計分幾個區塊。它不會改變結果 — 平均是以 token 數加權，而不是以批次數加權 — 只影響速度與記憶體。',
      max_batches: '跑到這麼多批次就停（0 = 整份資料集）。適合上課時快速估一下；實際量了多少可以看 `tokens` 輸出。',
      device: '此節點的裝置。保持 auto 就跟隨這張圖的裝置（工具列的圖裝置，沒有指定時則用設定）。一張圖只在一個裝置上執行；若部分工作需要不同裝置，請拆成兩張圖。',
      precision: '前向傳播使用的混合精度。在 Ampere 之後的顯卡上，bf16 大約可以省下一半的 activation 記憶體，長上下文往往得靠它才量得動；損失本身仍然以 fp32 累加。裝置若無法支援所選精度，會退回 fp32 並在 log 中說明。',
    },
  },
  TextGenerate: {
    description: '用訓練好的模型接續提示文字，逐 token 輸出',
    details:
      'temperature、top_k、top_p 決定寫出來的東西有多敢冒險：temperature 設 0 時每次都取機率最高的 ' +
      'token，調高則是拿連貫性換多樣性。遇到 end-of-text token 或達到 max_new_tokens 就停。' +
      'tokenizer 必須跟訓練時用的是同一個。',
    params: {
      prompt: '要接續的文字。當沒有 `prompt` 輸入連線時使用此欄位。提示文字的風格越接近訓練資料，小模型的表現越好。',
      max_new_tokens: '最多生成幾個 token。每一個都要對「目前已經寫出來的全部內容」重新跑一次前向傳播，所以這是決定本節點要跑多久的旋鈕。',
      temperature: '取樣前先把分數除以這個值：小於 1 會讓分布更尖銳，大於 1 會更平坦。0 = greedy，永遠取單一最可能的 token，結果可重現但容易繞圈打轉。',
      top_k: '只從分數最高的 k 個 token 中取樣（0 = 關閉）。這是避免某個五萬分之一的 token 把整句話帶偏的手段。',
      top_p: 'Nucleus 取樣：從機率最高的 token 開始累加，直到總和達到 p，就只從這些 token 取樣（1 = 關閉）。與 top_k 不同的是這個切點會自動調整 — 模型有把握時就窄，沒把握時就寬。',
      seed: '取樣所用的隨機種子。同樣的種子加上同樣的模型，在任何裝置上都會得到同樣的文字，所以比較兩個 temperature 時，差異就只來自 temperature。',
      device: '此節點的裝置。保持 auto 就跟隨這張圖的裝置（工具列的圖裝置，沒有指定時則用設定）。一張圖只在一個裝置上執行；若部分工作需要不同裝置，請拆成兩張圖。',
    },
  },
  TextEmbedding: {
    description: '用預訓練的句子編碼器把每段文字變成一條稠密向量',
    details:
      '意思相同的文字會得到方向相同的向量，cosine 接近 1 就是判準。需要套件中心的 sentence-embeddings 套件包；' +
      '四個模型都不大（2200 萬到 1.18 億參數），純 CPU 也跑得動。texts（切塊後的清單）與 text（單一字串）只能接其中一個，' +
      '兩個都接會報錯。',
    params: {
      model: 'all-MiniLM-L6-v2：最小、英文。paraphrase-multilingual-MiniLM-L12-v2：支援 50 多種語言（含繁體中文），不需要前綴。bge-small-zh-v1.5：中文專用。multilingual-e5-small：檢索效果最好，但需要 "query: " / "passage: " 前綴（見 prefix）。',
      text: '沒有任何輸入連線時使用的備用文字。',
      split_lines: '把文字輸入的每一個非空白行當成一段獨立的文字。若要讓一整份多行文件變成一條向量，請關閉。',
      prefix: '編碼前加在每段文字前面的字串。multilingual-e5 訓練時問題用 "query: "、文件用 "passage: "；其他模型會忽略它。',
      normalize: 'L2 正規化，讓下游的點積等於 cosine similarity。',
      batch_size: '一次前向傳播處理幾段文字。只影響速度與記憶體。',
      max_seq_length: '每段文字的 token 上限（0 = 模型自己的預設：paraphrase-multilingual 128、all-MiniLM 256、bge/e5 512）。超過的部分會被截掉，切塊時請把長度控制在範圍內。',
      label_chars: 'labels 輸出中每段文字保留的字元數。',
      device: '此節點的裝置。保持 auto 就跟隨這張圖的裝置（工具列的圖裝置，沒有指定時則用設定）。一張圖只在一個裝置上執行；若部分工作需要不同裝置，請拆成兩張圖。',
    },
  },
  DocumentLoader: {
    description: '從資料夾或上傳檔讀入 .txt 與 .md，附上來源',
    details:
      '每份文件輸出成 {text, source}，來源會一路帶到 TextChunker 與最後的引用標註。內附的 ' +
      'data/samples/rag 資料夾放了五篇關於 CodefyUI 與機器學習基礎的中英雙語短文，RAG 範例不必設定就能跑。RAG ' +
      '鏈從這裡開始：DocumentLoader → TextChunker → TextEmbedding → VectorStore。',
    params: {
      source: '文件來源：一個資料夾，或一個你上傳的檔案。',
      directory: '放 .txt/.md 檔的資料夾。相對路徑先以後端工作目錄解析，再以 CodefyUI 後端資料夾解析（所以內附範例在任何目錄下都找得到）；專案模式下相對路徑必須留在專案目錄內。',
      recursive: '也讀取子資料夾。',
      file: '你用旁邊按鈕上傳的 .txt 檔。',
      max_docs: '最多保留這麼多份文件，依檔名排序（0 = 全部）。',
    },
  },
  TextChunker: {
    description: '把文件切成有重疊的小塊',
    details:
      'characters 以固定字數切窗，與語言無關，中文沒有空格也適用；sentences 與 paragraphs 保留自然邊界，' +
      '再把它們塞滿到 chunk_size。chunk_overlap 只用於 characters，讓被切斷的句子在下一塊裡仍然完整。' +
      '每一塊都會記下自己的來源；檢索以塊為單位而不是整份檔案。',
    params: {
      strategy: 'characters：固定長度的字元視窗。sentences：以句號、問號、驚嘆號切句再打包。paragraphs：以空行切段再打包。',
      chunk_size: '每一塊的字元數（所有策略共用的上限）。',
      chunk_overlap: '相鄰兩塊共享的字元數，讓被切在中間的句子至少會完整出現在其中一塊；必須小於 chunk_size。',
      min_chunk_chars: '最後一塊若短於這個長度，就併進前一塊。合併後的最後一塊最多可能超過 chunk_size 達 min_chunk_chars - 1 個字元。',
    },
  },
  VectorStore: {
    description: '把各塊的嵌入向量與文字打包成可搜尋的索引',
    details:
      '索引就是一個 [N, D] 矩陣加上 N 段文字，預設度量是 cosine — 每列都存成單位長度，所以搜尋只是一次矩陣乘法。index ' +
      '接到 Retriever。它只存在記憶體裡，重新執行時會從快取的嵌入在幾毫秒內重建。',
    params: {
      metric: 'cosine 忽略向量長度，是句子嵌入訓練時所用的度量；dot 是原始內積，給長度本身有意義的嵌入用。',
      normalize: '把每列存成單位長度（此時 cosine 等於 dot）。metric 為 dot 時忽略。',
    },
  },
  Retriever: {
    description: '從索引中取出與問題向量最相似的 top_k 塊文字',
    details:
      '它用一次矩陣乘法為索引裡的每一塊評分，留下 top_k，再丟掉低於 min_score 的，最後把文字交給 PromptBuilder。' +
      '留意分數：最高分只有 0.3 左右，通常代表語料裡沒有答案。問題向量必須來自與索引相同的嵌入模型。',
    params: {
      top_k: '要撈回幾塊。',
      min_score: '低於這個分數的結果會被丟掉（cosine 的範圍是 -1 到 1）。0 表示全部保留；用 e5/MiniLM 時 0.3 到 0.5 是合理的門檻。',
    },
  },
  PromptBuilder: {
    description: '把撈回的文字塊與問題填進提示詞模板',
    details:
      '預設模板會要求模型只依據那段內容作答。模板必須包含 {context} 與 {question}；想寫多行模板，把 TextInput ' +
      '接到 template 輸入。number_contexts 會在每塊文字前標上 [1]、[2]，接上 sources ' +
      '後會在括號裡附上出處。',
    params: {
      template: '含 {context} 與 {question} 兩個佔位符的模板。連了 template 輸入時以輸入為準。',
      separator: '各塊文字之間的分隔：blank_line 空一行、newline 換行、rule 一條分隔線。',
      number_contexts: '在每一塊前面加上 [1]、[2]...，接了 sources 時也附上來源檔名。',
      max_context_chars: '把合併後的內容區塊截到這個字元數（0 = 不限制）。小型本機模型超過幾千字就會明顯變慢。',
    },
  },
  HFTextGenerate: {
    description: '用本機的指令微調模型回答提示詞',
    details:
      '模型是 rag 套件包裡的 Qwen2.5-0.5B-Instruct（Apache-2.0，約 1 GB），對話模板會自動套用。' +
      '節點會逐個 token 回報進度；筆電 CPU 大約每秒幾個 token，GPU 快很多。若要用畫布上訓練出來的模型接續文字，請改用 ' +
      'TextGenerate；這裡載入的是預訓練權重，聽從指令作答。',
    params: {
      model: '要載入的模型。灰掉的選項需要先到套件中心安裝 rag 套件包。',
      prompt: '當沒有 `prompt` 輸入連線時使用的提示詞。',
      system_prompt: '放在使用者訊息前面的系統指令。留空就用模型內建的預設系統提示。',
      max_new_tokens: '最多生成幾個 token；決定本節點要跑多久。',
      temperature: '取樣前先把分數除以這個值：0 = greedy（永遠取最可能的 token，可重現）；越高越有變化。',
      top_p: 'Nucleus 取樣：只從累積機率達到 p 的最可能 token 中取樣（1 = 關閉）。',
      top_k: '只從分數最高的 k 個 token 中取樣（0 = 關閉）。',
      seed: '取樣的隨機種子。同樣的種子與模型會得到同樣的答案。',
      device: '此節點的裝置。保持 auto 就跟隨這張圖的裝置（工具列的圖裝置，沒有指定時則用設定）。一張圖只在一個裝置上執行；若部分工作需要不同裝置，請拆成兩張圖。',
      dtype: '權重精度。auto 在 CUDA 上用 bfloat16/float16、在 CPU 與 MPS 上用 float32。',
    },
  },
  LLMChat: {
    description: '把文字、影像或陣列送給聊天 LLM，輸出它的回覆',
    details:
      '可用的供應商有 ChatGPT API、Codex、Claude API，以及本機 Ollama（走它的 OpenAI 相容 /v1 端點）' +
      '。API 金鑰只存在於本次工作階段：存檔時會被清掉，建議改用 OPENAI_API_KEY 或 ANTHROPIC_API_KEY ' +
      '環境變數。',
  },

  // ── Diffusion ──
  GaussianNoise: {
    description: '依指定形狀取樣獨立同分布的高斯噪聲',
    details:
      '$\\epsilon \\sim \\mathcal{N}(\\mu, \\sigma^2)$，由 mean、std 決定，seed ' +
      '相同就會得到相同的噪聲。把張量接到 shape_ref 時，噪聲會跟隨該張量的形狀與裝置，shape 參數則不再使用。',
    params: {
      shape: '逗號分隔的維度，例：「1,3,32,32」。當 shape_ref 沒接時才會用。',
      mean: '高斯分布的平均值，預設 0（標準常態）。',
      std: '標準差，預設 1（標準常態）。',
      seed: '隨機種子；同樣的種子會得到相同的噪聲。',
    },
  },
  Lerp: {
    description: '以 alpha 混合兩個張量',
    details: 'alpha 預設讀參數，接上 alpha 輸入時改用輸入值，可以是純量或任何能廣播的張量；輸出形狀由廣播決定。',
    params: {
      alpha: '內插權重（0..1）。只在沒接 alpha 輸入時才會用此參數。',
    },
  },
  TimestepEmbedding: {
    description: '把時間步 $t$ 編碼成條件向量',
    details:
      '先做 Vaswani 式的正弦頻率編碼（範圍由 max_period 控制），再接 Linear→SiLU→Linear，這是 DDPM ' +
      '的標準配方。輸出形狀為 [B, embed_dim]；sin 與 cos 各佔一半，所以 embed_dim 必須是偶數。',
    params: {
      embed_dim: '時間向量的維度，必須是偶數（sin/cos 各半）。',
      max_period: '頻率組中最大的週期 — 控制能分辨多少個不同的時間步。',
      seed: '投影層初始化的隨機種子。',
    },
  },
  Upsample: {
    description: '依倍率縮放空間維度，沒有可學習權重',
    details:
      '縮放由 F.interpolate 完成：mode 可選 nearest、bilinear 或 area，scale_factor 預設 ' +
      '2.0，小於 1 時會改為縮小。若需要可學習的上採樣核，請改用 ConvTranspose2d。',
    params: {
      mode: '插值方式。nearest=直接複製像素、bilinear=雙線性平滑、area=平均（適合做 downsample）。',
      scale_factor: '空間維度的縮放倍數。2.0 放大兩倍、0.5 縮成一半。',
    },
  },
  DiffusionUNet: {
    description: '玩具版噪聲預測 U-Net，輸出與輸入同形狀的噪聲',
    details:
      '先經過 stem 卷積，接著依 channel_mult 每層一個時間條件 ResBlock，下採樣到瓶頸後再帶著 skip ' +
      '連接上採樣回來。輸入的高與寬必須能被 2^(層數-1) 整除，num_groups 也必須整除每層的通道數。把 model 接給 ' +
      'DDPMSampler 即可跑反向擴散。',
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
    description: '執行反向 DDPM 迴圈，將噪聲張量去噪成影像',
    details:
      '每一步呼叫 `model(x_t, t)` 預測噪聲，再套用 DDPM 更新公式；schedule 可選原始的 linear 或 ' +
      'cosine。整個迴圈在節點內部執行，圖因此維持無環，每步加入的高斯噪聲由 seed 決定。',
    params: {
      num_steps: '反向 diffusion 的步數。步數越多軌跡越平滑，但也越慢。',
      schedule: '噪聲排程。linear 是原版 DDPM；cosine（Nichol & Dhariwal 2021）在接近資料的區域噪聲增加得更慢。',
      beta_start: '線性排程的起始 variance。cosine 模式會忽略此值。',
      beta_end: '線性排程的結束 variance。cosine 模式會忽略此值。',
      seed: '取樣時每一步加入的高斯噪聲 z 所使用的種子。',
    },
  },
  DiffusionTrainingLoop: {
    description: '訓練 U-Net 預測加入的雜訊（DDPM）',
    details:
      '每一步取一張乾淨影像、隨機挑一個時間步加上雜訊，再用預測雜訊與實際雜訊的 MSE 更新權重，輸出訓練後的模型與每輪 loss。' +
      '這裡設定的雜訊排程（schedule、num_timesteps、beta_start、beta_end）必須和之後取樣的 ' +
      'DDPMSampler 一致，否則生成會壞掉。',
  },

  // ── VLA ──
  PushWorldEnv: {
    description: '2D 推物環境，指令指定 puck 與目標',
    details:
      '輸出回合工廠給 PushWorldDemos 與 VLARollout 依種子建立回合，畫面尺寸要與 VLAModel 的 ' +
      'image_size 一致。n_distractors 為 1 以上時會多出干擾 puck 與第二個目標，指令成為判斷目標的唯一依據。以純 ' +
      'torch 重現 PushT 的精神，不需安裝模擬器。',
    params: {
      image_size: '渲染畫面邊長（像素，正方形）；96 對齊 PushT 慣例',
      n_distractors: '目標 puck 之外的干擾 puck 數；0 時語言只是裝飾，≥1 時指令是唯一的目標線索',
      max_steps: '單回合步數上限；腳本專家平均約 23 步，VLARollout 評估時可另設預算',
    },
  },
  PushWorldDemos: {
    description: '執行腳本專家的回合，輸出行為複製樣本',
    details:
      '樣本是 ((影像, 指令位元組, 動作區塊), 動作區塊)：動作區塊也放在輸入裡，flow matching 才能在 forward ' +
      '內對它加噪。demo_noise 是 DART 式擾動：執行帶噪動作、但記錄專家動作，藉此產生回復狀態；holdout_episodes ' +
      '以不重疊的種子另收一份，demo_video 可接 VideoWrite。',
    params: {
      episodes: '訓練回合數（每回合約 25 個樣本；600 回合約 1.5 萬樣本、約 0.4 GB）',
      chunk: '每個樣本的動作數（動作區塊長度 H，須與 VLAModel 的 chunk 一致）；超過回合結尾時重複最後一個動作',
      demo_noise: 'DART 擾動強度：每步以 1/2 機率對執行動作加 N(0, noise)，標註仍為專家動作；實測關掉會讓閉環成功率崩潰（4% vs 24%）',
      holdout_episodes: '驗證回合數（獨立種子流，與訓練集永不重疊；0 = 空）',
      video_episodes: '錄進 demo_video 的回合數（0 = 不錄）',
      seed: '基礎種子；同種子同參數可完全重現資料集',
    },
  },

  VLAModel: {
    description: '迷你視覺-語言-動作策略：由影像與指令預測一段動作',
    details:
      'head_type 在 flow_matching（pi0/SmolVLA 路線：對動作區塊加噪、學習速度場、推論時 Euler 積分）與 ' +
      'regression（直接預測、用 MSE）之間切換，其餘設定相同。配對的 loss_fn 由輸出埠提供，與 model 一起接到 ' +
      'TrainingLoop。chunk、image_size 需與 demos 和 env 節點一致，預設約 3.2M 參數。',
    params: {
      head_type: 'flow_matching：pi0/SmolVLA 式速度場 + Euler 取樣。regression：直接預測區塊、MSE。loss_fn 輸出自動跟隨此選擇',
      d_model: '所有 token 流的寬度（須能被 n_heads 整除）',
      n_layers: '主幹深度（作用在 [視覺; 文字] token 上）',
      n_heads: '注意力頭數（主幹與 expert 共用）',
      expert_layers: '動作 expert 深度（區塊 query 自注意 + 對主幹交叉注意）',
      chunk: '每次預測的動作數（區塊視野 H）——須與 PushWorldDemos 的 chunk 一致',
      image_size: '輸入畫面邊長——須與 PushWorldEnv 的 image_size 一致',
      vision_stem: 'conv：三層 stride-2 3x3 卷積。patchify：經典 ViT stem。1200 回合／45 epochs 預算下的控制 A/B 量到 patchify 領先（成功率 0.85 vs 0.45，同資料同 seed）——與 NeurIPS 2021 的 conv-stem 結論及本節點早先的混淆筆記相反；2400／110 完整預算目前只跑過 conv（0.97）。這顆旋鈕存在的目的就是把這件事定案',
      patch_size: '僅 patchify stem：方形 patch 邊長',
      action_dim: '動作向量寬度（PushWorld 為 2：dx, dy）',
      max_text_len: '指令長度（位元組）——須與資料集編碼一致（PushWorldDemos 用 48）',
      flow_steps: '僅 flow_matching：推論時的 Euler 積分步數（SmolVLA 用 10）。屬執行期旋鈕——修改不會丟棄已保存權重',
      flow_time_dist: '僅 flow_matching：訓練取樣流時間 t 的分布。beta 偏重雜訊較大的一端（pi0 式）。執行期旋鈕——不影響已保存權重',
      dropout: '主幹與 expert 全程的 dropout',
      seed: '權重初始化種子',
    },
  },
  VLARollout: {
    description: '閉環評估：成功率、平均步數與影片',
    details:
      '策略每次預測一個動作區塊，只執行其中 execute_k 步就重新規劃，即所謂的後退視野（receding horizon）；把 ' +
      'execute_k 調到接近區塊長度，量到的就是開環的誤差累積。instruction_mode=swapped 改用干擾 puck ' +
      '的顏色下指令，不看語言的策略成績不會改變。frames 會記錄回合畫面，成功鑲綠邊、逾時鑲紅邊。',
    params: {
      episodes: '評估回合數（種子流與訓練資料不相交）',
      execute_k: '每個預測區塊執行幾步後重新規劃（後退視野）。同一策略實測：2 -> 46%、4 -> 34%、整塊 8 -> 20%——往區塊長度調大即可研究 open-loop 誤差累積',
      max_steps: '單回合步數預算（覆蓋 env 設定）。學到的策略比腳本專家慢，預算太緊會把控制誤差記成逾時',
      instruction_mode: 'normal：回合真實指令。swapped：改講干擾 puck 的顏色——語言接地消融。只看畫面的策略兩者同分；讀語言的策略在 swapped 下崩潰',
      record_episodes: '錄進 frames 的前 N 個回合（0 = 不錄）',
      seed: '評估種子（回合種子取自與 PushWorldDemos 預設不相交的偏移流）',
      device: '此節點的裝置。保持 auto 就跟隨這張圖的裝置（工具列的圖裝置，沒有指定時則用設定）。一張圖只在一個裝置上執行；若部分工作需要不同裝置，請拆成兩張圖。',
    },
  },
  VLAActionEval: {
    description: '在保留樣本上比較預測動作與專家動作的均方誤差',
    details:
      '請接 PushWorldDemos 的 holdout 輸出；max_samples 限制樣本數，seed 固定 flow head ' +
      '的取樣噪聲。這是開環量測，誤差低但 VLARollout 成功率低時，通常代表誤差累積。',
    params: {
      max_samples: '最多評估的樣本數',
      batch_size: '推論批次大小',
      seed: '固定 flow head 的取樣噪聲讓數字可重現（regression 不受影響）',
      device: '此節點的裝置。保持 auto 就跟隨這張圖的裝置（工具列的圖裝置，沒有指定時則用設定）。一張圖只在一個裝置上執行；若部分工作需要不同裝置，請拆成兩張圖。',
    },
  },

  // ── Plugins ──
  // The first-party packs that ship from this repo. Keyed by the QUALIFIED node
  // name the registry serves (`edu:Classifier`), which is what `tn` looks up;
  // a pack that is not installed simply never reaches these lines. A
  // third-party plugin has no way to add translations of its own yet, so its
  // nodes read in English whatever the locale.
  'deep:Edu-CrossAttention': {
    description: '多頭注意力：query 關注 context',
    details:
      '$Q$ 取自 `query`，$K$、$V$ 取自 `context`。`query` 與 `context` 的序列長度不必相同，' +
      '所以輸出形狀跟隨 `query`，權重則是長方形的 [H, Q_seq, K_seq]。可選的 [Q_seq, K_seq] ' +
      '布林遮罩會擋住指定位置，True 代表擋住。',
  },
  'deep:Edu-MultiHeadAttention': {
    description: '多頭自注意力，輸出 [H, seq, seq] 權重',
    details:
      'embed_dim 會平分給 num_heads 個頭，每個頭各自做縮放點積注意力，再把串接後的結果經 $W_o$ 混合。`causal` ' +
      '會擋住每個位置右側的內容，可選的 [seq, seq] 遮罩與它以 OR 合併。',
  },
  'deep:Edu-Patchify': {
    description: '影像切成 P×P 區塊，輸出 [B,N,C·P·P]',
    details:
      'H 與 W 都必須能被 patch_size 整除。第二個輸出是區塊格點數 [grid_h, grid_w]；把 `flatten` ' +
      '關掉則每個區塊保留 [C, P, P] 形狀，而不是壓成一條向量。verbose 模式會逐步記錄 unfold → permute → ' +
      'flatten 這條把影像變成 token 序列的過程。',
  },
  'deep:Edu-ResBlock': {
    description: '兩段 GN→SiLU→Conv，加回 skip 路徑',
    details:
      '擴散模型 U-Net 的基本組成單元。把 `TimestepEmbedding` 接到 `time_emb`，' +
      '兩個卷積之間就會加入時間步投影；不接就是一般的 ResNet 區塊。GroupNorm 的 `groups` 必須同時整除 ' +
      'in_channels 與 out_channels；兩者不同時，skip 路徑會補一個 1×1 卷積。',
  },
  'deep:Edu-SelfAttention': {
    description: '單頭自注意力，輸出 [seq, seq] 注意力權重',
    details:
      'Q、K、V 由三個 Linear 投影而來，分數為 $QK^T/\\sqrt{d}$，對分數取 softmax 得到權重，輸出則是權重乘上 ' +
      'V。分數在 softmax 前會先除以 `temperature`：小於 1 會讓分布更尖銳，大於 1 更平坦。`causal` ' +
      '會擋住每個位置右側的內容，可選的 [seq, seq] 遮罩與它以 OR 合併。',
  },
  'edu:ActivationLayer': {
    description: '往正在組裝的網路尾端加一個激活函數',
    details:
      '`function` 可選 relu、tanh、sigmoid、identity。identity 完全不加非線性，整疊 FFNLayer ' +
      '會塌回單一個線性映射。`model` 輸入必須接上，這顆要放在 FFNLayer 後面。',
  },
  'edu:AdvancedClassifier': {
    description: '以 SVM、決策樹或隨機森林分類 x_query',
    details:
      '`kind` 可選 SVM（kernel 為 rbf、linear 或 poly）、Decision Tree（`max_depth`，0 ' +
      '代表不限）或 Random Forest（`n_estimators`），三者都來自 scikit-learn。`model` 輸出可接到 ' +
      'DecisionBoundary，SVM 的 model 還帶著支持向量座標。',
  },
  'edu:Classifier': {
    description: '以 KNN、線性或邏輯斯分類 x_query',
    details:
      '`kind` 可選 knn（`n_neighbors`）、logistic 或 linear，三者底層都是 scikit-learn。' +
      'linear 是對 one-hot 標籤做最小平方再取 argmax，所以它的 `probabilities` 是回歸分數而不是機率。' +
      '`model` 輸出可接到 DecisionBoundary。',
  },
  'edu:FFNLayer': {
    description: '往正在組裝的網路尾端加一個全連接層',
    details:
      '每接一顆就往尾端加一個 `nn.Linear(in_features, out_features)`。輸入維度會從上一層推得，' +
      '所以只有鏈上第一顆會用到 `in_features`。每層參數量 = out_features × in_features + ' +
      'out_features。跟 ActivationLayer 交錯串成 MLP，鏈的尾端接 TrainAndEvaluate。',
  },
  'edu:FilterRows': {
    description: '只保留 2D 表格中指定欄位通過比較的列',
    details:
      '用 `column_name` 選欄時需要連上 `columns` 輸入，或改用 `column_index` 指定位置。除了篩選後的表，' +
      '還會輸出布林的列遮罩與通過的列數。verbose 模式會逐一記錄取欄、與門檻比較、計數、索引四個步驟。',
  },
  'edu:SentenceEmbedding': {
    description: '把文字的語意編碼成一條 (d,) 向量',
    details:
      '意思接近的句子，向量方向也接近。底層是 model2vec 靜態嵌入：純 CPU 的查表加平均，模型與結果都會在行程內快取。`model` ' +
      '可選多語言模型（支援中文與跨語言比較）或更小的純英文模型；第一次使用會下載權重。把兩條向量接給 CosineSimilarity ' +
      '就能量兩句話意思有多接近。',
  },
  'edu:SlidingWindow2D': {
    description: '把 kernel 在影像上滑動，每個位置做加權加總',
    details:
      '吃 (C, H, W) 影像，單通道 (H, W) 也接，每個通道套用同一個 kernel。`preset` 提供四個 3×3 ' +
      'kernel（模糊、邊緣偵測、銳化、垂直邊緣），或切到 Custom 自填 N×N 數字；padding=0 時邊長各縮 k-1。運算是 ' +
      'cross-correlation，kernel 不會翻轉。',
  },
  'edu:TrainAndEvaluate': {
    description: '訓練堆好的網路，再預測 x_query 標籤',
    details:
      '尾端會自動補上一個壓到類別數的線性輸出層，再以 cross-entropy 與 Adam 訓練 `epochs` 輪、學習率為 `lr`。' +
      '除了 predictions，還會輸出可接 DecisionBoundary 的 `model` 與每輪的 `losses`。',
  },
  'foundations:Edu-ColumnStats': {
    description: '每欄的平均、標準差、最小值、最大值與列數',
    details:
      '標準差預設是母體標準差（除以 N）；勾選 unbiased 改成除以 N−1 的樣本標準差。開啟「顯示內部步驟」後，還會記錄每個中間步驟：' +
      '欄總和、除以列數、平方差、開根號，檢視器就能看到每個統計量是怎麼算出來的。',
  },
  'foundations:Edu-FFN': {
    description: '兩層線性層夾一個激活函數，另外輸出隱藏層激活值',
    details:
      '權重由 seed 初始化，同一個 seed 得到同一組權重。預設維度很小：embed_dim 8、hidden_dim 16；實務上 ' +
      'hidden_dim 通常是 embed_dim 的 4 倍。',
  },
  'foundations:Edu-KNN': {
    description: '由最近的 k 個訓練點多數決，輸出預測標籤',
    details:
      '距離對每個訓練點逐一計算，成本隨 N_train × N_query 成長；資料量大時改用有樹狀索引的 KNN 節點。度量可選 ' +
      'euclidean、manhattan 或 cosine（1 − cos），k 會被限制在訓練集大小內。另外會輸出每筆查詢的前 k ' +
      '個距離與鄰居索引。',
  },
  'foundations:Edu-LinearRegression': {
    description: '以封閉解或梯度下降擬合 $y=Xw+b$',
    details:
      'closed_form 解正規方程 $w = (X^T X + \\lambda I)^{-1} X^T y$，矩陣奇異時退回最小平方求解；' +
      'gradient_descent 用 lr 跑 epochs 次迭代。regularization 在兩種解法下都只對權重加 ' +
      'L2（ridge）懲罰。',
  },
  'foundations:Edu-LogisticRegression': {
    description: '以梯度下降訓練 softmax，輸出標籤與機率',
    details:
      '損失函數是交叉熵，regularization 對權重（不含 bias）加上 L2 衰減。二元問題走同一條兩欄 softmax，等價於 ' +
      'sigmoid 形式。預測標籤是字串，取自排序後的訓練標籤集合；classes 輸出對應 weights 的欄位順序。',
  },
  'foundations:Edu-TokenEmbedding': {
    description: '以固定種子的向量表，把 token 列表轉成向量',
    details:
      '輸入可以是 tokens（字串）或 token_ids（整數）。hash 用穩定雜湊把任何 token 映到某一列；ordinal ' +
      '依首次出現順序配號，並從 vocab 輸出對照。向量表隨機、由 seed 固定且不會被訓練；需要可學習的查表請用 Utility 的 ' +
      'Embedding 節點。',
  },
  'rl:Edu-PolicyGradient': {
    description: '一次 REINFORCE 步驟的損失與每個中間量',
    details:
      'probs = softmax(logits / temperature)，在實際採取的動作上取值再取 log，loss = ' +
      '-mean(log_probs · advantages)；baseline 設 mean 會減掉批次平均獎勵，設 none ' +
      '則直接用原始獎勵當優勢。不會呼叫 backward()，那一步由 BackwardOnce 負責。',
  },
  'stats:Stats-ChartView': {
    description: '把圖表資料或表格畫成長條、折線、散佈或熱圖',
    details:
      'auto 沿用傳入圖表原本的類型；表格有 row_labels 時畫長條圖，沒有就畫折線圖。熱圖與其他類型之間不做轉換，而是在圖上加註，' +
      '不讓整次執行失敗。columns_filter 決定要畫哪些欄位：散佈圖取前兩欄，長條圖取第一欄。',
  },
  'stats:Stats-ConfusionMatrix': {
    description: '真實類別對預測類別的計數，另輸出正確率與熱圖',
    details:
      '列是真實類別、欄是預測類別，與 sklearn 的 confusion_matrix 一致。normalize 可除以真實列（對角線即 ' +
      'recall）、預測欄（precision）或總數，0/0 一律當 0。兩個輸入都接受標籤列表、1D 類別索引張量，或取每列 argmax ' +
      '的 2D 分數矩陣。',
  },
  'stats:Stats-Correlation': {
    description: 'Pearson 或 Spearman 相關矩陣',
    details:
      'drop_nan 開啟時（同 pandas 預設），每一對只取兩欄都有值的列，一個缺值不會縮短其他配對；關閉則讓 NaN 傳播。' +
      'Spearman 是對平均排名做 Pearson，同分共用排名。熱圖固定在 -1..1，不隨資料伸縮；常數欄的結果是 NaN。',
  },
  'stats:Stats-Describe': {
    description: '每欄的筆數、平均、標準差、最小值、百分位數與最大值',
    details:
      '列名與順序沿用 pandas 的 describe()，std 是樣本標準差（ddof=1）。NaN 視為缺值，±Inf 則是有效值，含 ' +
      'inf 的欄位平均也是 inf。axis 可改成逐列描述或把整張表當成一條序列；percentiles 只回報指定的那幾個，' +
      '不會自動補上中位數。',
  },
  'stats:Stats-GroupByAggregate': {
    description: '每組一列，欄位取平均、總和、計數、極值或標準差',
    details:
      '分組可接每列一個標籤的 keys（例如 CSVReader 的 labels），或用 group_by 指定表格自己的欄位；兩者都有時以 ' +
      'group_by 為準，被分組的欄位不再參與彙總。agg_overrides 以 column=agg 指定個別例外。NaN 照 ' +
      'pandas 跳過，count 算有值的筆數，每組列數看 counts 輸出。',
  },
  'stats:Stats-Histogram': {
    description: '把張量分箱計數，另輸出長條圖',
    details:
      '每列是 bin_start、bin_end 與 count；開啟 density 時第三欄改成機率密度，樣本數不同的分布才能互相比較。' +
      'range_mode 設為 manual 可固定分箱範圍，讓連續幾次執行對得起來。非有限值（NaN、±Inf）在分箱前剔除，數量從 ' +
      'dropped 輸出。',
  },
  'stats:Stats-Percentile': {
    description: '任意組百分位數，可取整個張量、逐欄或逐列',
    details:
      'q 是以逗號分隔的 0-100 百分位數，會排序並去重。內插是線性，與 np.percentile、DataFrame.quantile ' +
      '相同，但 NaN 逐條序列跳過，行為接近 np.nanpercentile。axis 選 all 得到 1D 結果，選 columns 或 ' +
      'rows 則是 [q, series] 表格。',
  },
  'stats:Stats-TableView': {
    description: '把表格排版成對齊的文字，含欄位名稱與列標籤',
    details:
      'max_rows 限制顯示的列數，結尾會有一行交代少了幾列；設為 0 則全部顯示。precision 決定非整數的小數位數；' +
      '同一份文字也會從 text 輸出埠傳出。',
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
