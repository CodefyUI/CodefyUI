---
sidebar_position: 3
title: 外掛
description: 安裝教育節點的外掛包，並學習如何撰寫與發布你自己的外掛。
---

# 外掛包

教育（「Edu」）節點以可安裝的**外掛包**形式提供，**依方向**組織，因此每一個都對應到一個動手實作的教科書模組，並在你逐步學習時累進安裝。

```bash
cdui plugin install foundations deep rl   # full textbook companion
cdui plugin list
cdui plugin info deep                      # manifest, lessons covered, node names
cdui plugin search attention               # query the catalog
cdui plugin install foo/bar                # third-party pack from GitHub
cdui plugin uninstall deep
```

## 有哪些可用的外掛包

| 外掛包 | 動手實作模組 | Edu 節點 |
|------|------------------|-----------|
| `foundations` | I1 Data Representation · I2 Classical ML | Edu-ColumnStats、Edu-KNN、Edu-LinearRegression、Edu-LogisticRegression、Edu-TokenEmbedding、Edu-FFN |
| `deep` | I3 Vision · I4 Sequences | Edu-CrossAttention、Edu-ResBlock、Edu-SelfAttention、Edu-MultiHeadAttention、Edu-Patchify |
| `rl` | I5 Reinforcement Learning | Edu-PolicyGradient |

每個 Edu 節點都把單一課程概念分解成一連串具名步驟，由 [Teaching Inspector](/usage/teaching-inspector) 一次渲染一列——`Edu-ColumnStats` 將母體標準差公式呈現為 `sum → divide → deviations² → variance → sqrt`；`Edu-PolicyGradient` 暴露 `softmax → gather → log → baseline → loss`；`Edu-Patchify` 讓 `unfold → permute → flatten` 變得可見。在 Settings popover 中開啟 **Verbose mode** 即可擷取它們。

## 外掛包如何儲存

- **內建方向外掛包**位於 repo 內的 `plugins/<id>/`，並就地啟用（不複製）。
- **第三方外掛包**會以固定 SHA 的 tarball 下載到 `<USER_DATA>/plugins/<id>/`，並在安裝前經過 **AST 驗證**。
- 位於 `<USER_DATA>/plugins/installed.json` 的 lockfile 會記錄每一次安裝，因此 `cdui start` 會在下次啟動時重新探索它們。

外掛節點會加上命名空間，以避免衝突並讓圖能自我說明——內建節點使用像 `Conv2d` 這樣的裸名稱，而外掛節點則會像 `foundations:Edu-KNN` 這樣加上限定。

## 撰寫你自己的外掛

Fork **[官方外掛模板](https://github.com/treeleaves30760/CodefyUI-Plugin-Official)**——一個可運作、採 MIT 授權的外掛，包含兩個範例節點、一張範例圖、一套測試，以及一份完整註解的資訊清單 (manifest)。它的 README 逐欄解說每個欄位與 AST 安全閘門。

```bash
# Install the template itself to see the pattern live
cdui plugin install treeleaves30760/CodefyUI-Plugin-Official

# After forking
cdui plugin install your-username/your-fork
```

一個外掛包可隨附下列任意內容：一個 `nodes/` 目錄（自動探索）、一個 `presets/` 目錄、一個 `examples/` 目錄，以及一個 `assets/` 目錄（掛載於 `/plugins/<id>/assets/<file>`）。一份 `cdui.plugin.toml` 資訊清單宣告 id、版本、`requires_codefyui`、內容目錄與課程 metadata。

:::warning 破壞性變更（v0.3）
章節外掛包 `c1`–`c6` 已重新封裝為三個方向外掛包 `foundations` / `deep` / `rl`，而且每個 Edu 節點的型別 id 都加上了一個破折號（`EduKNN` → `Edu-KNN`）。引用舊有 `cN:EduFoo` 型別的已儲存圖必須更新為 `<pack>:Edu-Foo`，並以 `cdui plugin install foundations deep rl` 重新安裝這些外掛包。
:::

## REST API

| 端點 | 方法 | 說明 |
|----------|--------|-------------|
| `/api/plugins` | GET | 列出已安裝的外掛包。 |
| `/api/plugins/{id}` | GET | 取得某外掛的資訊清單 (manifest) 與 README。 |
| `/api/plugins/reload` | POST | 熱重載所有節點與預設模組來源。 |
