# CodefyUI Windows 安裝腳本
# 用法：irm https://raw.githubusercontent.com/latteine1217/CodefyUI/main/install.ps1 | iex

$REPO    = "https://github.com/latteine1217/CodefyUI.git"
$INSTALL = if ($env:CODEFYUI_DIR) { $env:CODEFYUI_DIR } else { "$HOME\CodefyUI" }
$TOOLS   = "$HOME\.codefyui"

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
    Warn "下載 $([System.IO.Path]::GetFileName($dest))..."
    $ProgressPreference = "SilentlyContinue"
    Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
    $ProgressPreference = "Continue"
}

New-Item -ItemType Directory -Force -Path $TOOLS | Out-Null

try {
    Write-Host ""
    Write-Host "╔══════════════════════════════════════╗" -ForegroundColor White
    Write-Host "║        CodefyUI  安裝程式            ║" -ForegroundColor White
    Write-Host "╚══════════════════════════════════════╝" -ForegroundColor White
    Write-Host "  安裝目錄：$INSTALL"

    # ── Git ────────────────────────────────────────────────────────────
    Step "Git"
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        $headers = @{ "User-Agent" = "CodefyUI-Installer" }
        $release = Invoke-RestMethod -Uri "https://api.github.com/repos/git-for-windows/git/releases/latest" -Headers $headers
        $asset   = $release.assets | Where-Object { $_.name -match "64-bit\.exe$" -and $_.name -notmatch "Portable" } | Select-Object -First 1
        $gitExe  = "$TOOLS\git.exe"
        Download $asset.browser_download_url $gitExe
        Warn "安裝 Git..."
        Start-Process $gitExe -ArgumentList "/VERYSILENT /NORESTART /NOCANCEL /SP-" -Wait
        Add-ToPath "C:\Program Files\Git\cmd"
        Remove-Item $gitExe -Force
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
        $pyUrl = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
        $pyExe = "$TOOLS\python.exe"
        Download $pyUrl $pyExe
        Warn "安裝 Python 3.11..."
        Start-Process $pyExe -ArgumentList "/quiet InstallAllUsers=0 PrependPath=1 Include_test=0 Include_doc=0" -Wait
        Add-ToPath "$HOME\AppData\Local\Programs\Python\Python311"
        Add-ToPath "$HOME\AppData\Local\Programs\Python\Python311\Scripts"
        Remove-Item $pyExe -Force
        $PYTHON = "python"
    }
    Ok "$(& $PYTHON --version)"

    # ── Node.js (portable zip) ─────────────────────────────────────────
    Step "Node.js"
    if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
        $v      = (Invoke-RestMethod https://nodejs.org/dist/index.json | Where-Object { $_.lts } | Select-Object -First 1).version
        $zipUrl = "https://nodejs.org/dist/$v/node-$v-win-x64.zip"
        $zip    = "$TOOLS\node.zip"
        $nodeDir = "$TOOLS\node"

        Download $zipUrl $zip
        Warn "解壓 Node.js $v..."
        if (Test-Path $nodeDir) { Remove-Item $nodeDir -Recurse -Force }
        $ProgressPreference = "SilentlyContinue"
        Expand-Archive $zip $TOOLS -Force
        $ProgressPreference = "Continue"
        Rename-Item "$TOOLS\node-$v-win-x64" $nodeDir
        Remove-Item $zip -Force

        Add-ToPath $nodeDir
    }
    Ok "Node.js $(node --version)"

    # ── pnpm ───────────────────────────────────────────────────────────
    Step "pnpm"
    if (-not (Get-Command pnpm -ErrorAction SilentlyContinue)) {
        Warn "透過 npm 安裝 pnpm..."
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
