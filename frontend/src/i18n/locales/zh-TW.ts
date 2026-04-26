import type { TranslationKey } from './en';

const zhTW: Record<TranslationKey, string> = {
  // Toolbar
  'toolbar.run': '執行',
  'toolbar.running': '執行中...',
  'toolbar.stop': '停止',
  'toolbar.run.title': '執行管線',
  'toolbar.stop.title': '停止執行',
  'toolbar.reloadNodes': '重新載入節點',
  'toolbar.reloadNodes.title': '從後端重新載入節點定義',
  'toolbar.reload.fail': '重新載入失敗：{error}',

  // Menu: File
  'toolbar.menu.file': '檔案',
  'toolbar.save': '儲存',
  'toolbar.save.title': '儲存圖表',
  'toolbar.save.prompt': '請輸入圖表名稱：',
  'toolbar.save.success': '圖表「{name}」儲存成功。',
  'toolbar.save.fail': '儲存失敗：{error}',
  'toolbar.load': '載入',
  'toolbar.load.title': '載入已儲存的圖表',
  'toolbar.load.fail': '載入失敗：{error}',
  'toolbar.load.loading': '載入中...',
  'toolbar.load.empty': '沒有已儲存的圖表',
  'toolbar.import': '匯入 JSON...',
  'toolbar.import.fail': '匯入失敗：{error}',
  'toolbar.clear': '清除畫布',
  'toolbar.clear.title': '清除畫布',
  'toolbar.clear.confirm': '確定要清除畫布嗎？所有未儲存的內容將會遺失。',

  // Menu: Export
  'toolbar.menu.export': '匯出',
  'toolbar.export.empty': '畫布為空 — 請先新增一些節點再匯出。',
  'toolbar.exportJson': '匯出為 JSON',
  'toolbar.exportJson.title': '將圖表下載為 JSON 檔案（包含子圖）',
  'toolbar.exportJson.empty': '畫布為空 — 請先新增一些節點再匯出。',
  'toolbar.export': '匯出為子圖',
  'toolbar.export.title': '將目前圖表匯出為可重用的子圖/預設模組',
  'toolbar.export.prompt': '請輸入子圖名稱：',
  'toolbar.export.success': '子圖「{name}」匯出成功！已出現在節點面板中。',
  'toolbar.export.fail': '匯出失敗：{error}',
  'toolbar.exportPython': '匯出為 Python',
  'toolbar.exportPython.title': '將圖表下載為獨立的 Python 腳本',
  'toolbar.exportPython.empty': '畫布為空 — 請先新增一些節點再匯出。',
  'toolbar.exportPython.fail': 'Python 匯出失敗：{error}',

  // Status
  'status.idle': '閒置',
  'status.running': '執行中',
  'status.completed': '已完成',
  'status.error': '錯誤',
  'status.skipped': '已跳過',
  'status.cached': '已快取',

  // Node Palette
  'palette.title': '節點',
  'palette.search': '搜尋節點...',
  'palette.loading': '載入節點中...',
  'palette.loadFail': '載入節點失敗：{error}',
  'palette.retry': '重試',
  'palette.noMatch': '找不到符合的節點',
  'palette.empty': '沒有可用的節點',
  'palette.hint': '拖曳節點到畫布上',
  'palette.composite': '復合',
  'palette.basic': '基本',

  // Config Panel
  'config.title': '節點設定',
  'config.selectNode': '請選擇一個節點進行設定',
  'config.parameters': '參數',
  'config.noParams': '沒有可設定的參數',
  'config.ports': '連接埠',
  'config.inputs': '輸入',
  'config.outputs': '輸出',
  'config.optional': '可選',
  'config.execution': '執行狀態',
  'config.range': '範圍：{min} — {max}',

  // Node
  'node.opt': '可選',
  'node.running': '執行中...',
  'node.completed': '已完成',
  'node.cached': '已快取',
  'node.error': '錯誤：{error}',

  // Results Panel
  'results.title': '執行紀錄',
  'results.training': '訓練',
  'results.trainingConfig': '訓練參數',
  'results.trainingEmpty': '尚無訓練資料。',
  'results.clear': '清除',
  'results.empty': '尚無紀錄。請執行管線以查看輸出。',

  // Preset
  'preset.badge': '預設',
  'preset.configure': '設定預設模組',
  'preset.nodeCount': '內含 {count} 個節點',
  'preset.nodesInside': '個內部節點',
  'preset.apply': '套用',
  'preset.cancel': '取消',
  'preset.generalGroup': '一般',

  // Empty Canvas
  'empty.title': '建立你的第一個深度學習模型',
  'empty.subtitle': '選擇一個範例快速開始',
  'empty.hint': '或從左側面板拖曳節點',
  'empty.loading': '載入範例中...',
  'empty.loadError': '載入範例失敗',
  'empty.section.trainable': '可訓練的完整流程',
  'empty.section.architecture': '架構視覺化範例',

  // Context Menu
  'contextMenu.rename': '重新命名',
  'contextMenu.duplicate': '複製',
  'contextMenu.delete': '刪除',
  'contextMenu.rename.prompt': '請輸入節點的新名稱：',
  'contextMenu.addTextNote': '新增文字註記',
  'contextMenu.addImageNote': '新增圖片註記',

  // Notes
  'note.placeholder': '點擊以編輯...',
  'note.imagePlaceholder': '點擊上傳圖片',
  'note.bind': '綁定到最近的節點',
  'note.unbind': '解除綁定',
  'note.changeColor': '更改顏色',
  'note.layoutWarning': '未綁定的註記不會被自動排版重新定位。',

  // Tabs
  'tabs.add': '新增分頁',
  'tabs.closeRunning': '此分頁仍在執行中，確定要關閉嗎？',

  // Subgraph Editor (SequentialModel)
  'subgraph.title': '模型架構編輯器',
  'subgraph.palette': '層級',
  'subgraph.apply': '套用',
  'subgraph.cancel': '取消',
  'subgraph.import': '匯入',
  'subgraph.export': '匯出',
  'subgraph.import.title': '匯入已儲存的模型架構',
  'subgraph.export.title': '將目前架構匯出為 JSON',
  'subgraph.empty': '從左側面板拖曳層級來建構你的模型',
  'subgraph.layerCount': '{count} 個層級',
  'subgraph.params': '參數',
  'subgraph.noParams': '無參數',
  'subgraph.deleteLayer': '刪除',
  'subgraph.hint': '雙擊以編輯架構',
  'subgraph.import.fail': '匯入失敗：{error}',
  'subgraph.import.selectModel': '選擇要匯入的 SequentialModel',
  'subgraph.import.noContent': '此檔案中找不到可匯入的層級或 SequentialModel 節點。',
  'subgraph.searchLayers': '搜尋層級...',
  'subgraph.snapOn': '吸附 ON',
  'subgraph.snapOff': '吸附 OFF',
  'subgraph.snapTitle': '切換網格吸附',
  'subgraph.autoLayout': '自動排列',
  'subgraph.autoLayoutTitle': '依連接順序從上到下排列節點',
  'subgraph.category.io': '輸入/輸出',
  'subgraph.category.merge': '合併',
  'subgraph.validation.cycle': '圖形包含循環',
  'subgraph.validation.noInput': '圖形必須有一個 Input 節點',
  'subgraph.validation.noOutput': '圖形必須有一個 Output 節點',
  'subgraph.port.add': '+ 新增 port',
  'subgraph.port.remove': '移除',
  'subgraph.port.namePlaceholder': 'port 名稱',
  'subgraph.port.duplicate': 'port 名稱重複',
  'subgraph.port.list': 'Ports',

  // Tooltips
  'toolbar.tooltips.on': '提示 ON',
  'toolbar.tooltips.off': '提示 OFF',
  'toolbar.tooltips.title': '切換節點描述提示',

  // Custom Node Manager
  'customNodes.title': '自訂節點管理',
  'customNodes.loading': '載入中...',
  'customNodes.empty': '沒有自訂節點。上傳 .py 檔案開始使用。',
  'customNodes.enabled': '啟用',
  'customNodes.disabled': '停用',
  'customNodes.delete': '刪除',
  'customNodes.delete.confirm': '確定要刪除「{name}」嗎？此操作無法復原。',
  'customNodes.upload': '上傳 .py',
  'toolbar.customNodes': '自訂節點',
  'toolbar.customNodes.title': '管理自訂節點',

  // Grid Snap
  'toolbar.gridSnap.on': '吸附 ON',
  'toolbar.gridSnap.off': '吸附 OFF',
  'toolbar.gridSnap.title': '切換節點網格吸附',

  // Auto Layout
  'toolbar.autoLayout': '自動排版',
  'toolbar.autoLayout.experiments': '排版實驗',
  'toolbar.autoLayout.all': '排版全部',
  'toolbar.autoLayout.selected': '排版選取 ({count})',

  // Execution errors
  'execution.error.noEntryPoints': '尚未定義起始節點。請從節點面板拖曳一個 Start 節點，並連接到要開始執行的節點。',

  // Node palette — control category / start node
  'palette.category.control': '控制',
  'palette.start.description': '標記執行的起點。連接到你想執行的第一個節點。',

  // Keyboard Shortcuts
  'shortcuts.title': '鍵盤快捷鍵',
  'shortcuts.undo': '復原',
  'shortcuts.redo': '重做',
  'shortcuts.redoAlt': '重做（替代）',
  'shortcuts.copy': '複製選取的節點',
  'shortcuts.paste': '貼上節點',
  'shortcuts.delete': '刪除選取項目',
  'shortcuts.quickSearch': '快速搜尋節點',
  'shortcuts.help': '顯示此說明',
  'shortcuts.doubleClickKey': '雙擊',

  // Training Summary
  'results.epoch': '輪次',
  'results.currentLoss': '損失',
  'results.bestLoss': '最佳',

  // Beginner Mode
  'toolbar.beginnerMode.on': '入門模式',
  'toolbar.beginnerMode.off': '所有節點',
  'toolbar.beginnerMode.title': '切換入門模式（僅顯示基本節點類別）',

  // Results Panel — expandable errors
  'results.clickToExpand': '點擊展開錯誤詳情',
  'results.clickToHighlight': '點擊以高亮節點',

  // Language
  'lang.label': '中',

  // Teaching Inspector — Record toggle
  'toolbar.record.on': '錄製 ON',
  'toolbar.record.off': '錄製 OFF',
  'toolbar.record.title': '記錄節點輸出（關閉後已捕獲的資料仍會保留）',

  // Teaching Inspector — Compare Segment
  'toolbar.compareSegment': '段落比較',
  'toolbar.clearSegment': '取消段落',
  'toolbar.clearActiveSegment': '取消此段落',
  'toolbar.compareSegment.title': '選取兩個節點後按下，建立新段落；多個段落可同時存在，按泡泡右上角的 × 只會清除該段落',
  'toolbar.compareSegment.needTwo': '請先選取恰好兩個節點',
  'segment.noPath': '段落：頭節點與尾節點之間沒有資料連線',

  // Inspector panel
  'inspector.title': '檢視器',
  'inspector.collapse': '收合檢視器',
  'inspector.expand': '展開檢視器',
  'inspector.empty.notRun': '執行圖以捕獲資料',
  'inspector.empty.notRunHint': '確認「錄製」已開啟，然後按下「執行」',
  'inspector.empty.noSelection': '選取節點或段落以檢視',
  'inspector.empty.noSelectionHint': '點擊任一節點，或 Shift 選兩個後按「段落比較」',
  'inspector.segment.inputs': '段落輸入（{count}）',
  'inspector.segment.outputs': '段落輸出（{count}）',
  'segment.removeThis': '移除此段落',

  // A1 — 詳細／步驟追蹤模式
  'toolbar.verbose.on': '詳細',
  'toolbar.verbose.off': '簡潔',
  'toolbar.verbose.title': '在檢視器中顯示演算法逐步細節（Q、K、V、scores 等）',
  'inspector.tabs.forward': '前向',
  'inspector.tabs.steps': '步驟',
  'inspector.tabs.backward': '反向',
  'inspector.steps.empty': '此節點不記錄步驟',
  'inspector.steps.requireVerbose': '啟用詳細模式並重新執行以查看步驟',

  // A2 — 節點權重持久化
  'toolbar.weights.on': '保留',
  'toolbar.weights.off': '重置',
  'toolbar.weights.title': '在多次執行間保留層的權重（讓 Conv2d / Linear / Attention 真的能學習而不是每次重置）',
  'toolbar.weights.resetAll': '重置所有權重',
  'toolbar.weights.resetAllConfirm': '確定要清除此圖所有已保存的權重嗎？',
  'toolbar.weights.resetAllOk': '已清除保存的權重',
  'contextMenu.resetWeights': '重置權重',
  'node.weightsPersistedBadge': '權重已保存',

  // A3 — 反向傳播 / 梯度檢視器
  'toolbar.backward.on': '∂ 梯度 ON',
  'toolbar.backward.off': '∂ 梯度 OFF',
  'toolbar.backward.title': '在下次執行時擷取梯度（前向 + .backward()）',
  'toolbar.autoBackward.on': '自動損失',
  'toolbar.autoBackward.off': '手動',
  'toolbar.autoBackward.title': '未提供 Loss / BackwardOnce 節點時自動合成損失',
  'toolbar.backward.trainingLoopHint': 'TrainingLoop 已自帶反向；如需獨立檢視，請插入 BackwardOnce 節點',
  'inspector.backward.empty': '未擷取到梯度',
  'inspector.backward.disabled': '啟用反向模式並重新執行以查看梯度',
  'inspector.backward.weightSection': '權重梯度',
  'inspector.backward.portSection': '輸出梯度',
  'inspector.backward.health.vanishing': '梯度消失',
  'inspector.backward.health.exploding': '梯度爆炸',
  'inspector.backward.health.healthy': '健康',
};

export default zhTW;
