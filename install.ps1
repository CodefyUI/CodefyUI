# CodefyUI 一鍵安裝腳本 (Windows / PowerShell)
# 用法：
#   powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/treeleaves30760/CodefyUI/main/install.ps1 | iex"
#
# 環境變數：
#   $env:CODEFYUI_DIR           自訂安裝路徑（預設 $HOME\CodefyUI）
#   $env:CODEFYUI_RELEASE_TAG   指定要下載的 release tag（預設 latest）
#   $env:CODEFYUI_FORCE_BUILD   設為 1 強制本地 build（會額外裝 Node + pnpm）

$ErrorActionPreference = 'Stop'

$Repo = 'https://github.com/treeleaves30760/CodefyUI.git'
$ReleaseRepo = 'treeleaves30760/CodefyUI'
$ReleaseAsset = 'frontend-dist.tar.gz'
$InstallDir = if ($env:CODEFYUI_DIR) { $env:CODEFYUI_DIR } else { Join-Path $HOME 'CodefyUI' }
$ReleaseTag = if ($env:CODEFYUI_RELEASE_TAG) { $env:CODEFYUI_RELEASE_TAG } else { 'latest' }
$ForceBuild = ($env:CODEFYUI_FORCE_BUILD -eq '1')

# ── Helpers ───────────────────────────────────────────────────────────────────
function Step($msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Blue }
function Ok($msg)   { Write-Host "  OK  $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "  !   $msg" -ForegroundColor Yellow }
function Die($msg)  { Write-Host ""; Write-Host "  X Error: $msg" -ForegroundColor Red; exit 1 }

function Test-Cmd($name) {
    return $null -ne (Get-Command $name -ErrorAction SilentlyContinue)
}

function Refresh-Path {
    $machine = [System.Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user    = [System.Environment]::GetEnvironmentVariable('Path', 'User')
    # User PATH entries added by installers often contain literal
    # `%VAR%` references (e.g. pnpm adds `%PNPM_HOME%;…`). These are
    # resolved automatically on shell start, but not when we read via
    # GetEnvironmentVariable — expand manually so Test-Cmd / Get-Command
    # can actually find the binaries.
    if ($user)    { $user    = [System.Environment]::ExpandEnvironmentVariables($user) }
    if ($machine) { $machine = [System.Environment]::ExpandEnvironmentVariables($machine) }
    $env:Path = ($machine, $user, $env:Path | Where-Object { $_ }) -join ';'
}

function Install-Winget($id, $friendlyName) {
    if (-not (Test-Cmd winget)) {
        Die "winget not found. Install '$friendlyName' manually or install 'App Installer' from Microsoft Store, then re-run."
    }
    winget install --id $id --silent --accept-source-agreements --accept-package-agreements --exact
    if ($LASTEXITCODE -ne 0) { Die "winget install $id failed (exit $LASTEXITCODE)" }
    Refresh-Path
}

function Install-NodeToolchain {
    Step "pnpm（僅本地 build 路徑使用）"
    if (-not (Test-Cmd pnpm)) {
        Warn "Not installed, running standalone installer..."
        Invoke-WebRequest -UseBasicParsing -Uri 'https://get.pnpm.io/install.ps1' | Invoke-Expression
        $PnpmHome = [System.Environment]::GetEnvironmentVariable('PNPM_HOME', 'User')
        if (-not $PnpmHome) { $PnpmHome = Join-Path $env:LOCALAPPDATA 'pnpm' }
        $env:PNPM_HOME = $PnpmHome
        $env:Path = "$PnpmHome;$env:Path"
        if (-not (Test-Cmd pnpm)) { Die "pnpm not found on PATH after install. Open a new shell and re-run." }
    }
    Ok "pnpm $(pnpm --version)"

    Step "Node.js（透過 pnpm env）"
    $nodeMin = 24
    $nodeOk = $false
    if (Test-Cmd node) {
        $currentMajor = ((node --version) -replace '^v','' -split '\.')[0]
        if ($currentMajor -match '^\d+$' -and [int]$currentMajor -ge $nodeMin) {
            $nodeOk = $true
        }
    }
    if (-not $nodeOk) {
        Warn "Not installed or version < $nodeMin, installing Node $nodeMin via 'pnpm env use --global $nodeMin'..."
        pnpm env use --global $nodeMin
        if ($LASTEXITCODE -ne 0) { Die "pnpm env use --global $nodeMin failed" }
        Refresh-Path
        if (-not (Test-Cmd node)) { Die "node not found on PATH after install. Open a new shell and re-run." }
    }
    Ok "Node.js $(node --version)"
}

function Fetch-ReleaseDist {
    param([string]$DistDir)

    $url = if ($ReleaseTag -eq 'latest') {
        "https://github.com/$ReleaseRepo/releases/latest/download/$ReleaseAsset"
    } else {
        "https://github.com/$ReleaseRepo/releases/download/$ReleaseTag/$ReleaseAsset"
    }

    Write-Host "  下載：$url"

    $tmpdir = Join-Path $env:TEMP "cdui-dist-$([guid]::NewGuid().ToString('N'))"
    New-Item -ItemType Directory -Path $tmpdir -Force | Out-Null
    $tarball = Join-Path $tmpdir $ReleaseAsset

    try {
        # -UseBasicParsing avoids loading IE engine (Server Core / nano).
        # GitHub redirects 302 → S3 — Invoke-WebRequest follows by default.
        Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $tarball -TimeoutSec 60
    } catch {
        Warn "下載失敗（網路問題或 release 還沒附這個 asset）：$($_.Exception.Message)"
        Remove-Item -Recurse -Force $tmpdir -ErrorAction SilentlyContinue
        return $false
    }

    # Windows 10 1803+ ships tar.exe in System32 — extracts .tar.gz natively.
    if (-not (Test-Cmd tar)) {
        Warn "找不到 tar.exe（Windows 10 1803+ 應內建）"
        Remove-Item -Recurse -Force $tmpdir -ErrorAction SilentlyContinue
        return $false
    }

    if (Test-Path $DistDir) { Remove-Item -Recurse -Force $DistDir }
    New-Item -ItemType Directory -Path $DistDir -Force | Out-Null

    & tar -xzf $tarball -C $DistDir
    if ($LASTEXITCODE -ne 0) {
        Warn "Tarball 解壓失敗（exit $LASTEXITCODE）"
        Remove-Item -Recurse -Force $DistDir, $tmpdir -ErrorAction SilentlyContinue
        return $false
    }
    Remove-Item -Recurse -Force $tmpdir -ErrorAction SilentlyContinue

    if (-not (Test-Path (Join-Path $DistDir 'index.html'))) {
        Warn "解壓後找不到 index.html，asset 內容可能有誤"
        Remove-Item -Recurse -Force $DistDir -ErrorAction SilentlyContinue
        return $false
    }

    Ok "Prebuilt dist 解壓至 $DistDir"
    return $true
}

# ══════════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "+======================================+"
Write-Host "|        CodefyUI Installer (Windows)  |"
Write-Host "+======================================+"
Write-Host "  Install dir:  $InstallDir"
Write-Host "  Release tag:  $ReleaseTag"
if ($ForceBuild) { Write-Host "  強制本地 build (CODEFYUI_FORCE_BUILD=1)" -ForegroundColor Yellow }

# ── git ───────────────────────────────────────────────────────────────────────
Step "git"
if (-not (Test-Cmd git)) {
    Warn "Not installed, installing via winget..."
    Install-Winget 'Git.Git' 'Git'
}
Ok (git --version)

# ── uv ────────────────────────────────────────────────────────────────────────
Step "uv"
if (-not (Test-Cmd uv)) {
    Warn "Not installed, running standalone installer..."
    Invoke-WebRequest -UseBasicParsing -Uri 'https://astral.sh/uv/install.ps1' | Invoke-Expression
    Refresh-Path
    if (-not (Test-Cmd uv)) { Die "uv not found on PATH after install. Open a new shell and re-run." }
}
Ok "uv $(uv --version)"

# ── Python 3 (provided by uv) ─────────────────────────────────────────────────
Step "Python 3"
uv python install 3.11
if ($LASTEXITCODE -ne 0) { Die "uv python install 3.11 failed" }
$PythonCmd = (uv python find 3.11).Trim()
if (-not (Test-Path $PythonCmd)) { Die "uv python find returned invalid path: $PythonCmd" }
Ok "$(& $PythonCmd --version) ($PythonCmd)"

# ── Clone / Update ────────────────────────────────────────────────────────────
Step "Downloading CodefyUI"
if (Test-Path (Join-Path $InstallDir '.git')) {
    Warn "Directory exists, updating..."
    git -C $InstallDir pull --ff-only
    if ($LASTEXITCODE -ne 0) { Die "git pull failed" }
    Ok "Updated"
} else {
    New-Item -ItemType Directory -Path (Split-Path -Parent $InstallDir) -Force | Out-Null
    git clone --depth 1 $Repo $InstallDir
    if ($LASTEXITCODE -ne 0) { Die "git clone failed" }
    Ok "Clone complete"
}

# ── Frontend dist：先試 release，失敗才裝 Node 本地 build ─────────────────────
$DistDir = Join-Path $InstallDir 'frontend\dist'
$UsePrebuilt = $false
if (-not $ForceBuild) {
    Step "Frontend dist (從 release 下載)"
    if (Fetch-ReleaseDist -DistDir $DistDir) {
        $UsePrebuilt = $true
    }
}

if (-not $UsePrebuilt) {
    Warn "改用本地 build 路徑（會額外安裝 Node.js 與 pnpm）"
    Install-NodeToolchain
}

# ── Install project deps ──────────────────────────────────────────────────────
Step "Installing project dependencies"
Set-Location $InstallDir
# 透傳給 dev.py，避免 dev.py 跳過 dist 重建
$env:CODEFYUI_FORCE_BUILD = if ($ForceBuild) { '1' } else { '0' }
$env:CODEFYUI_RELEASE_TAG = $ReleaseTag
& $PythonCmd scripts\dev.py install
if ($LASTEXITCODE -ne 0) { Die "scripts\dev.py install failed" }

# ── Install cdui launcher to PATH ─────────────────────────────────────────────
Step "Installing cdui launcher to PATH"
$LauncherDir = Join-Path $env:USERPROFILE '.local\bin'
$Launcher = Join-Path $LauncherDir 'cdui.cmd'
New-Item -ItemType Directory -Path $LauncherDir -Force | Out-Null
$stub = @"
@echo off
rem CodefyUI launcher stub — forwards to the install at $InstallDir.
call "$InstallDir\cdui.cmd" %*
"@
Set-Content -Path $Launcher -Value $stub -Encoding ASCII
Ok "cdui -> $Launcher"

# Ensure LauncherDir is on user PATH for future shells
$userPath = [System.Environment]::GetEnvironmentVariable('Path', 'User')
if ($null -eq $userPath) { $userPath = '' }
$pathEntries = $userPath -split ';' | Where-Object { $_ }
if ($pathEntries -notcontains $LauncherDir) {
    $newUserPath = if ($userPath) { "$userPath;$LauncherDir" } else { $LauncherDir }
    [System.Environment]::SetEnvironmentVariable('Path', $newUserPath, 'User')
    Ok "Added $LauncherDir to user PATH"
}

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "+======================================+" -ForegroundColor Green
Write-Host "|         Installation complete!       |" -ForegroundColor Green
Write-Host "+======================================+" -ForegroundColor Green
Write-Host ""
Write-Host "  Restart PowerShell to pick up PATH, then:"
if ($UsePrebuilt) {
    Write-Host "    cdui start          # production 模式（單一 :8000，不需 Node）"
}
Write-Host "    cdui dev            # 開發模式（HMR；需 Node）"
Write-Host ""
Write-Host "  Or from the current shell using the absolute path:"
Write-Host "    $InstallDir\cdui.cmd start"
Write-Host ""
Write-Host "  Other commands: cdui update | build | stop | test | clean | uninstall"
Write-Host ""
