# CodefyUI Windows 安裝腳本
# 用法：irm https://raw.githubusercontent.com/latteine1217/CodefyUI/main/install.ps1 | iex

$REPO    = "https://github.com/latteine1217/CodefyUI.git"
$INSTALL = if ($env:CODEFYUI_DIR) { $env:CODEFYUI_DIR } else { "$HOME\CodefyUI" }

function Step { Write-Host "`n==> $args" -ForegroundColor Cyan }
function Ok   { Write-Host "  v $args" -ForegroundColor Green }
function Warn { Write-Host "  ! $args" -ForegroundColor Yellow }
function Finish {
    Write-Host "`n按 Enter 關閉視窗..." -NoNewline
    $null = Read-Host
}

# ── 套件管理器偵測（winget > Scoop > Chocolatey）────────────────────────

function Add-ToPath ([string]$dir) {
    if ((Test-Path $dir) -and $env:PATH -notlike "*$dir*") {
        $env:PATH = "$dir;$env:PATH"
    }
}

function Get-PackageManager {
    if (Get-Command winget -ErrorAction SilentlyContinue) { return "winget" }

    # Scoop（user-level，不需要 admin）
    Add-ToPath "$HOME\scoop\shims"
    if (Get-Command scoop -ErrorAction SilentlyContinue) { return "scoop" }

    # Chocolatey — 直接檢查 exe 而非依賴 PATH
    $chocoExe = "$env:ProgramData\chocolatey\bin\choco.exe"
    if (Test-Path $chocoExe) {
        Add-ToPath "$env:ProgramData\chocolatey\bin"
        return "choco"
    }

    return $null
}

function Install-Scoop {
    Warn "安裝 Scoop（不需要管理員）..."
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
    iwr -useb get.scoop.sh | iex
    Add-ToPath "$HOME\scoop\shims"
}

function Pkg-Install ([string]$wingetId, [string]$scoopId, [string]$chocoId) {
    $pm = Get-PackageManager
    if (-not $pm) {
        Install-Scoop
        $pm = "scoop"
    }
    switch ($pm) {
        "winget" { winget install --id $wingetId -e --source winget --accept-package-agreements --accept-source-agreements }
        "scoop"  { scoop install $scoopId }
        "choco"  { & "$env:ProgramData\chocolatey\bin\choco.exe" install $chocoId -y }
    }
}

# ─────────────────────────────────────────────────────────────────────

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
        Pkg-Install "Git.Git" "git" "git"
        Add-ToPath "C:\Program Files\Git\cmd"
        Add-ToPath "$HOME\scoop\apps\git\current\cmd"
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
        Pkg-Install "Python.Python.3.11" "python" "python311"
        Add-ToPath "$HOME\AppData\Local\Programs\Python\Python311"
        Add-ToPath "$HOME\AppData\Local\Programs\Python\Python311\Scripts"
        Add-ToPath "$HOME\scoop\apps\python\current"
        $PYTHON = "python"
    }
    Ok "$(& $PYTHON --version)"

    # ── Node.js ────────────────────────────────────────────────────────
    Step "Node.js"
    if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
        Warn "未安裝，正在安裝..."
        Pkg-Install "OpenJS.NodeJS.LTS" "nodejs-lts" "nodejs-lts"
        Add-ToPath "C:\Program Files\nodejs"
        Add-ToPath "$HOME\scoop\apps\nodejs-lts\current"
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
