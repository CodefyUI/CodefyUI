# CodefyUI 一鍵安裝腳本 (Windows / PowerShell)
# 用法：
#   powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/CodefyUI/CodefyUI/main/install.ps1 | iex"
#
# 環境變數：
#   $env:CODEFYUI_DIR           自訂安裝路徑（預設 $HOME\CodefyUI）
#   $env:CODEFYUI_RELEASE_TAG   指定要下載的 release tag（預設 latest）
#   $env:CODEFYUI_FORCE_BUILD   設為 1 強制本地 build（會額外裝 Node + pnpm）

$ErrorActionPreference = 'Stop'

$Repo = 'https://github.com/CodefyUI/CodefyUI.git'
$ReleaseRepo = 'CodefyUI/CodefyUI'
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

function Add-UserPath($dir) {
    $userPath = [System.Environment]::GetEnvironmentVariable('Path', 'User')
    if ($null -eq $userPath) { $userPath = '' }
    $entries = $userPath -split ';' | Where-Object { $_ }
    if ($entries -contains $dir) { return $false }
    $newUserPath = if ($userPath) { "$userPath;$dir" } else { $dir }
    [System.Environment]::SetEnvironmentVariable('Path', $newUserPath, 'User')
    return $true
}

# 回傳 $true/$false，由呼叫端決定是否改用備援方案。
function Install-Winget($id, $friendlyName) {
    if (-not (Test-Cmd winget)) {
        Warn "winget 不可用（需先安裝 Microsoft Store 的 'App Installer'），無法透過 winget 安裝 $friendlyName"
        return $false
    }
    # `--source winget` 為必要參數：部分環境（企業網路 TLS 攔截、憑證不符）無法
    # 連線至 msstore 來源，會回報 0x8a15005e「The server certificate did not match
    # any of the expected values」。此時 winget 會將結果視為不明確，輸出 "Please
    # specify one of them using the --source option" 後即以非零狀態結束——即使
    # winget 來源本身已成功解析到該套件。明確指定來源即可完全略過 msstore。
    # 輸出導向 Out-Host：原生指令的 stdout 會進入 PowerShell 的成功管線，未加以
    # 攔截時會混入本函式的回傳值。
    winget install --id $id --exact --source winget --silent `
        --accept-source-agreements --accept-package-agreements | Out-Host
    if ($LASTEXITCODE -ne 0) {
        Warn "winget install $id failed (exit $LASTEXITCODE)"
        return $false
    }
    Refresh-Path
    return $true
}

# winget 不可用時的備援方案：Git for Windows 的 PortableGit 為 7-Zip 自解壓封存
# 檔，解壓至使用者目錄即可使用，不需系統管理員權限，亦不經過 winget 的套件來源。
function Install-GitPortable {
    $arch = switch ($env:PROCESSOR_ARCHITECTURE) {
        'ARM64' { 'arm64' }
        'x86'   { '32-bit' }
        default { '64-bit' }
    }
    try {
        $rel = Invoke-RestMethod -UseBasicParsing -TimeoutSec 30 `
            -Uri 'https://api.github.com/repos/git-for-windows/git/releases/latest' `
            -Headers @{ 'User-Agent' = 'CodefyUI-installer' }
    } catch {
        Warn "無法查詢 Git for Windows release 資訊：$($_.Exception.Message)"
        return $false
    }

    $portable = @($rel.assets | Where-Object { $_.name -like 'PortableGit-*.7z.exe' })
    $asset = $portable | Where-Object { $_.name -like "*-$arch.7z.exe" } | Select-Object -First 1
    if (-not $asset) { $asset = $portable | Where-Object { $_.name -like '*-64-bit.7z.exe' } | Select-Object -First 1 }
    if (-not $asset) {
        Warn "Git for Windows release 中未提供 PortableGit 自解壓封存檔"
        return $false
    }

    $target = Join-Path $env:LOCALAPPDATA 'CodefyUI\PortableGit'
    $tmp = Join-Path $env:TEMP "cdui-git-$([guid]::NewGuid().ToString('N')).exe"
    Write-Host "  下載：$($asset.browser_download_url)"
    try {
        Invoke-WebRequest -UseBasicParsing -Uri $asset.browser_download_url -OutFile $tmp -TimeoutSec 600
    } catch {
        Warn "PortableGit 下載失敗：$($_.Exception.Message)"
        return $false
    }

    if (Test-Path $target) { Remove-Item -Recurse -Force $target -ErrorAction SilentlyContinue }
    New-Item -ItemType Directory -Path $target -Force | Out-Null
    # 7-Zip SFX 參數：-o<dir>（與路徑之間不可有空白）指定解壓目的地，-y 為全部同意。
    Start-Process -FilePath $tmp -ArgumentList "-o`"$target`" -y" -Wait -NoNewWindow | Out-Null
    Remove-Item -Force $tmp -ErrorAction SilentlyContinue

    $gitCmdDir = Join-Path $target 'cmd'
    if (-not (Test-Path (Join-Path $gitCmdDir 'git.exe'))) {
        Warn "PortableGit 解壓完成後仍找不到 git.exe"
        return $false
    }
    $env:Path = "$gitCmdDir;$env:Path"
    if (Add-UserPath $gitCmdDir) { Ok "已將 $gitCmdDir 加入使用者 PATH" }
    return $true
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

# 解析要安裝的 release tag（把 "latest" 解析成具體版號）；失敗回傳 $null。
function Resolve-ReleaseTag {
    if ($ReleaseTag -ne 'latest') { return $ReleaseTag }
    try {
        $api = "https://api.github.com/repos/$ReleaseRepo/releases/latest"
        $resp = Invoke-RestMethod -UseBasicParsing -Uri $api -TimeoutSec 30 `
            -Headers @{ 'User-Agent' = 'CodefyUI-installer' }
        if ($resp.tag_name) { return $resp.tag_name }
    } catch { }
    return $null
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
    if (-not (Install-Winget 'Git.Git' 'Git')) {
        Warn "winget 安裝路徑失敗，改用免安裝的 PortableGit（不需系統管理員權限）..."
        if (-not (Install-GitPortable)) {
            Die "無法自動安裝 git。請手動安裝 Git for Windows（https://git-scm.com/download/win）後重新執行安裝指令。"
        }
    }
    Refresh-Path
    if (-not (Test-Cmd git)) { Die "git not found on PATH after install. Open a new shell and re-run." }
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

# ── Resolve release tag ────────────────────────────────────────────────────────
# 預編 dist 路徑會把 backend 鎖到與 dist 同一個 release tag，避免「main 後端 +
# 舊 release 前端」版本漂移（舊前端不會跟新後端做 token bootstrap，寫入請求被
# auth_guard 擋成 403，導致 localhost:8000 打得開卻無法執行）。
$PinnedTag = $null
if (-not $ForceBuild) {
    $PinnedTag = Resolve-ReleaseTag
    if ($PinnedTag) {
        $ReleaseTag = $PinnedTag
        Write-Host "  鎖定 release：${PinnedTag}（前後端同版）"
    } else {
        Warn "無法解析 latest release tag；改用 main（前後端可能版本漂移）"
    }
}

# ── Clone / Update ────────────────────────────────────────────────────────────
Step "Downloading CodefyUI"
if (Test-Path (Join-Path $InstallDir '.git')) {
    Warn "Directory exists, updating..."
    git -C $InstallDir fetch --tags --depth 1 origin
    if ($PinnedTag) {
        git -C $InstallDir fetch --depth 1 origin "refs/tags/${PinnedTag}:refs/tags/${PinnedTag}" 2>$null
        git -C $InstallDir checkout -f $PinnedTag
        if ($LASTEXITCODE -ne 0) { Die "git checkout $PinnedTag failed" }
        Ok "Updated and pinned to $PinnedTag"
    } else {
        git -C $InstallDir pull --ff-only
        if ($LASTEXITCODE -ne 0) { Die "git pull failed" }
        Ok "Updated"
    }
} else {
    New-Item -ItemType Directory -Path (Split-Path -Parent $InstallDir) -Force | Out-Null
    if ($PinnedTag) {
        git clone --depth 1 --branch $PinnedTag $Repo $InstallDir
        if ($LASTEXITCODE -ne 0) { Die "git clone failed" }
        Ok "Clone complete ($PinnedTag)"
    } else {
        git clone --depth 1 $Repo $InstallDir
        if ($LASTEXITCODE -ne 0) { Die "git clone failed" }
        Ok "Clone complete"
    }
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
& $PythonCmd scripts\dev.py install --yes
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
if (Add-UserPath $LauncherDir) { Ok "Added $LauncherDir to user PATH" }

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
