# CodefyUI Windows 安裝腳本
# 用法：irm https://raw.githubusercontent.com/latteine1217/CodefyUI/main/install.ps1 | iex

$REPO    = "https://github.com/latteine1217/CodefyUI.git"
$INSTALL = if ($env:CODEFYUI_DIR) { $env:CODEFYUI_DIR } else { "$HOME\CodefyUI" }

function Step { Write-Host "`n==> $args" -ForegroundColor Cyan }
function Ok   { Write-Host "  v $args" -ForegroundColor Green }
function Warn { Write-Host "  ! $args" -ForegroundColor Yellow }

function Pause-And-Exit {
    param([string]$msg, [int]$code = 1)
    Write-Host "`nx 錯誤：$msg" -ForegroundColor Red
    Write-Host "`n按 Enter 關閉視窗..." -NoNewline
    $null = Read-Host
    exit $code
}

try {
    Write-Host ""
    Write-Host "╔══════════════════════════════════════╗" -ForegroundColor White
    Write-Host "║        CodefyUI  安裝程式            ║" -ForegroundColor White
    Write-Host "╚══════════════════════════════════════╝" -ForegroundColor White
    Write-Host "  安裝目錄：$INSTALL"

    # ── winget 可用性 ──────────────────────────────────────────────────
    $hasWinget = [bool](Get-Command winget -ErrorAction SilentlyContinue)

    # ── git ────────────────────────────────────────────────────────────
    Step "git"
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Warn "未安裝，透過 winget 安裝..."
        if (-not $hasWinget) { Pause-And-Exit "找不到 winget，請先從 Microsoft Store 安裝『應用程式安裝程式』" }
        winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements
        $env:PATH += ";C:\Program Files\Git\cmd"
    }
    Ok "$(git --version)"

    # ── Python 3 ───────────────────────────────────────────────────────
    Step "Python 3"
    $PYTHON = $null
    foreach ($cmd in @("python", "python3", "py")) {
        if (Get-Command $cmd -ErrorAction SilentlyContinue) {
            $ok = & $cmd -c "import sys; print(sys.version_info >= (3,8))" 2>$null
            if ($ok -eq "True") { $PYTHON = $cmd; break }
        }
    }
    if (-not $PYTHON) {
        Warn "未安裝，透過 winget 安裝..."
        if (-not $hasWinget) { Pause-And-Exit "找不到 winget，請先從 Microsoft Store 安裝『應用程式安裝程式』" }
        winget install --id Python.Python.3.11 -e --source winget --accept-package-agreements --accept-source-agreements
        $env:PATH += ";$HOME\AppData\Local\Programs\Python\Python311"
        $env:PATH += ";$HOME\AppData\Local\Programs\Python\Python311\Scripts"
        $PYTHON = "python"
    }
    Ok "$(& $PYTHON --version)"

    # ── Node.js ────────────────────────────────────────────────────────
    Step "Node.js"
    if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
        Warn "未安裝，透過 winget 安裝..."
        if (-not $hasWinget) { Pause-And-Exit "找不到 winget，請先從 Microsoft Store 安裝『應用程式安裝程式』" }
        winget install --id OpenJS.NodeJS.LTS -e --source winget --accept-package-agreements --accept-source-agreements
        $env:PATH += ";C:\Program Files\nodejs"
    }
    Ok "Node.js $(node --version)"

    # ── pnpm ───────────────────────────────────────────────────────────
    Step "pnpm"
    if (-not (Get-Command pnpm -ErrorAction SilentlyContinue)) {
        Warn "未安裝，透過 npm 安裝..."
        npm install -g pnpm
    }
    Ok "pnpm $(pnpm --version)"

    # ── Clone / Update ─────────────────────────────────────────────────
    Step "下載 CodefyUI"
    if (Test-Path "$INSTALL\.git") {
        Warn "目錄已存在，執行更新..."
        git -C $INSTALL pull --ff-only
        Ok "已更新至最新版本"
    } else {
        git clone --depth 1 $REPO $INSTALL
        Ok "Clone 完成"
    }

    # ── 安裝依賴 ───────────────────────────────────────────────────────
    Step "安裝專案依賴"
    Set-Location $INSTALL
    & $PYTHON dev.py install

    # ── 完成 ───────────────────────────────────────────────────────────
    Write-Host ""
    Write-Host "╔══════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "║          安裝完成！                  ║" -ForegroundColor Green
    Write-Host "╚══════════════════════════════════════╝" -ForegroundColor Green
    Write-Host ""
    Write-Host "  啟動開發伺服器："
    Write-Host "    cd $INSTALL" -ForegroundColor White
    Write-Host "    python dev.py dev" -ForegroundColor White
    Write-Host ""

} catch {
    Write-Host ""
    Write-Host "x 安裝失敗：$($_.Exception.Message)" -ForegroundColor Red
    Write-Host "  行數：$($_.InvocationInfo.ScriptLineNumber)" -ForegroundColor Red
    Write-Host ""
    Write-Host "按 Enter 關閉視窗..." -NoNewline
    $null = Read-Host
    exit 1
}

Write-Host "按 Enter 關閉視窗..." -NoNewline
$null = Read-Host
