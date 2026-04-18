# CodefyUI Windows 安裝腳本
# 用法：irm https://raw.githubusercontent.com/latteine1217/CodefyUI/main/install.ps1 | iex

$INSTALL  = if ($env:CODEFYUI_DIR) { $env:CODEFYUI_DIR } else { "$HOME\CodefyUI" }
$TOOLS    = "$HOME\.codefyui"
$REPO_ZIP = "https://github.com/latteine1217/CodefyUI/archive/refs/heads/main.zip"
$PY_URL   = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
$NODE_VER = "v22.16.0"
$NODE_URL = "https://nodejs.org/dist/$NODE_VER/node-$NODE_VER-win-x64.zip"

$ProgressPreference = "SilentlyContinue"

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
function Download ([string]$url, [string]$dest) {
    Warn "下載中..."
    (New-Object Net.WebClient).DownloadFile($url, $dest)
}

New-Item -ItemType Directory -Force -Path $TOOLS | Out-Null

try {
    Write-Host ""
    Write-Host "╔══════════════════════════════════════╗" -ForegroundColor White
    Write-Host "║        CodefyUI  安裝程式            ║" -ForegroundColor White
    Write-Host "╚══════════════════════════════════════╝" -ForegroundColor White
    Write-Host "  安裝目錄：$INSTALL"

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
        $pyExe = "$TOOLS\python.exe"
        Download $PY_URL $pyExe
        Warn "安裝 Python 3.11..."
        Start-Process $pyExe -ArgumentList "/quiet InstallAllUsers=0 PrependPath=1 Include_test=0 Include_doc=0" -Wait
        Remove-Item $pyExe -Force
        Add-ToPath "$HOME\AppData\Local\Programs\Python\Python311"
        Add-ToPath "$HOME\AppData\Local\Programs\Python\Python311\Scripts"
        $PYTHON = "python"
    }
    Ok "$(& $PYTHON --version)"

    # ── Node.js (portable) ─────────────────────────────────────────────
    Step "Node.js"
    $nodeDir = "$TOOLS\node"
    if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
        $zip = "$TOOLS\node.zip"
        Download $NODE_URL $zip
        Warn "解壓 Node.js $NODE_VER..."
        if (Test-Path $nodeDir) { Remove-Item $nodeDir -Recurse -Force }
        Expand-Archive $zip $TOOLS -Force
        Rename-Item "$TOOLS\node-$NODE_VER-win-x64" $nodeDir
        Remove-Item $zip -Force
        Add-ToPath $nodeDir
    } else {
        Add-ToPath $nodeDir
    }
    Ok "Node.js $(node --version)"

    # ── pnpm ───────────────────────────────────────────────────────────
    Step "pnpm"
    if (-not (Get-Command pnpm -ErrorAction SilentlyContinue)) {
        Warn "安裝 pnpm..."
        npm install -g pnpm --silent
        Add-ToPath "$HOME\AppData\Roaming\npm"
    }
    Ok "pnpm $(pnpm --version)"

    # ── 下載 CodefyUI ──────────────────────────────────────────────────
    Step "下載 CodefyUI"
    $zip = "$TOOLS\codefyui.zip"
    Download $REPO_ZIP $zip
    Warn "解壓..."
    if (Test-Path $INSTALL) { Remove-Item $INSTALL -Recurse -Force }
    Expand-Archive $zip $TOOLS -Force
    Move-Item "$TOOLS\CodefyUI-main" $INSTALL
    Remove-Item $zip -Force
    Ok "下載完成"

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
