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

function Add-ToPath ([string]$dir) {
    if ((Test-Path $dir) -and $env:PATH -notlike "*$dir*") {
        $env:PATH = "$dir;$env:PATH"
    }
}

# ── Scoop ─────────────────────────────────────────────────────────────
function Ensure-Scoop {
    Add-ToPath "$HOME\scoop\shims"
    if (Get-Command scoop -ErrorAction SilentlyContinue) { return $true }
    Warn "安裝 Scoop..."
    try {
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
        iwr -useb get.scoop.sh | iex
        Add-ToPath "$HOME\scoop\shims"
        return [bool](Get-Command scoop -ErrorAction SilentlyContinue)
    } catch {
        return $false
    }
}

# ── 安裝單一工具 ──────────────────────────────────────────────────────
function Install-Tool {
    param(
        [string]$name,
        [string]$wingetId,
        [string]$scoopId,
        [string]$manualUrl
    )

    # winget
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install --id $wingetId -e --source winget `
            --accept-package-agreements --accept-source-agreements
        return
    }

    # Scoop
    if (Ensure-Scoop) {
        scoop install $scoopId
        return
    }

    # 手動安裝提示
    Write-Host ""
    Write-Host "  無法自動安裝 $name，請手動下載：" -ForegroundColor Red
    Write-Host "  $manualUrl" -ForegroundColor Cyan
    Finish
    exit 1
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
        Install-Tool "Git" "Git.Git" "git" "https://git-scm.com/download/win"
        Add-ToPath "C:\Program Files\Git\cmd"
        Add-ToPath "$HOME\scoop\apps\git\current\cmd"
        Add-ToPath "$HOME\scoop\shims"
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
        Install-Tool "Python 3.11" "Python.Python.3.11" "python" "https://www.python.org/downloads/"
        Add-ToPath "$HOME\AppData\Local\Programs\Python\Python311"
        Add-ToPath "$HOME\AppData\Local\Programs\Python\Python311\Scripts"
        Add-ToPath "$HOME\scoop\shims"
        $PYTHON = "python"
    }
    Ok "$(& $PYTHON --version)"

    # ── Node.js ────────────────────────────────────────────────────────
    Step "Node.js"
    if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
        Warn "未安裝，正在安裝..."
        Install-Tool "Node.js LTS" "OpenJS.NodeJS.LTS" "nodejs-lts" "https://nodejs.org/en/download"
        Add-ToPath "C:\Program Files\nodejs"
        Add-ToPath "$HOME\scoop\shims"
    }
    Ok "Node.js $(node --version)"

    # ── pnpm ───────────────────────────────────────────────────────────
    Step "pnpm"
    if (-not (Get-Command pnpm -ErrorAction SilentlyContinue)) {
        Warn "未安裝，透過 npm 安裝..."
        npm install -g pnpm
        Add-ToPath "$HOME\AppData\Roaming\npm"
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
