# CodefyUI Windows 安裝腳本
# 用法：irm https://raw.githubusercontent.com/latteine1217/CodefyUI/main/install.ps1 | iex

$REPO    = "https://github.com/latteine1217/CodefyUI.git"
$INSTALL = if ($env:CODEFYUI_DIR) { $env:CODEFYUI_DIR } else { "$HOME\CodefyUI" }

function Step { Write-Host "`n==> $args" -ForegroundColor Cyan }
function Ok   { Write-Host "  v $args" -ForegroundColor Green }
function Warn { Write-Host "  ! $args" -ForegroundColor Yellow }

# ── 自動提升管理員權限 ────────────────────────────────────────────────
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "  ! 需要管理員權限，正在重新啟動..." -ForegroundColor Yellow
    $cmd = "-NoExit -ExecutionPolicy Bypass -Command `"irm https://raw.githubusercontent.com/latteine1217/CodefyUI/main/install.ps1 | iex`""
    Start-Process PowerShell -Verb RunAs -ArgumentList $cmd
    exit
}

function Finish {
    Write-Host "`n按 Enter 關閉視窗..." -NoNewline
    $null = Read-Host
}

# ── 套件管理器：winget 優先，fallback 到 Chocolatey ──────────────────
function Ensure-ChocoInPath {
    $chocoBin = "$env:ProgramData\chocolatey\bin"
    if ((Test-Path $chocoBin) -and ($env:PATH -notlike "*$chocoBin*")) {
        $env:PATH += ";$chocoBin"
    }
}

function Get-PackageManager {
    if (Get-Command winget -ErrorAction SilentlyContinue) { return "winget" }
    Ensure-ChocoInPath
    if (Get-Command choco  -ErrorAction SilentlyContinue) { return "choco"  }
    return $null
}

function Install-Choco {
    Warn "winget 不可用，改安裝 Chocolatey..."
    Ensure-ChocoInPath
    if (Get-Command choco -ErrorAction SilentlyContinue) {
        Ok "Chocolatey 已安裝（$(choco --version)）"
        return
    }
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor 3072
    iex ((New-Object Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    Ensure-ChocoInPath
}

function Pkg-Install {
    param([string]$wingetId, [string]$chocoId)
    $pm = Get-PackageManager
    if (-not $pm) { Install-Choco; $pm = "choco" }
    switch ($pm) {
        "winget" { winget install --id $wingetId -e --source winget --accept-package-agreements --accept-source-agreements }
        "choco"  { choco install $chocoId -y }
    }
}

try {
    Write-Host ""
    Write-Host "╔══════════════════════════════════════╗" -ForegroundColor White
    Write-Host "║        CodefyUI  安裝程式            ║" -ForegroundColor White
    Write-Host "╚══════════════════════════════════════╝" -ForegroundColor White
    Write-Host "  安裝目錄：$INSTALL"

    # ── git ────────────────────────────────────────────────────────────
    Step "git"
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Warn "未安裝，正在安裝..."
        Pkg-Install "Git.Git" "git"
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
        Warn "未安裝，正在安裝..."
        Pkg-Install "Python.Python.3.11" "python311"
        $env:PATH += ";$HOME\AppData\Local\Programs\Python\Python311"
        $env:PATH += ";$HOME\AppData\Local\Programs\Python\Python311\Scripts"
        $PYTHON = "python"
    }
    Ok "$(& $PYTHON --version)"

    # ── Node.js ────────────────────────────────────────────────────────
    Step "Node.js"
    if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
        Warn "未安裝，正在安裝..."
        Pkg-Install "OpenJS.NodeJS.LTS" "nodejs-lts"
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

} catch {
    Write-Host ""
    Write-Host "x 安裝失敗：$($_.Exception.Message)" -ForegroundColor Red
    Write-Host "  行數：$($_.InvocationInfo.ScriptLineNumber)" -ForegroundColor Red
}

Finish
