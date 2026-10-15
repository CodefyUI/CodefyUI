@echo off
rem cdui.cmd — CodefyUI launcher (Windows). Finds a Python interpreter and forwards args to dev.py.
rem Usage: cdui <install|dev|stop|test|clean>
setlocal
set "SCRIPT_DIR=%~dp0"

rem uv's installer puts uv in one of these and only edits the user PATH, which
rem a console opened before the install never sees (#809).
where uv >nul 2>&1
if not %errorlevel% == 0 call :add_uv_to_path

rem 1. Prefer uv's managed Python (what install.ps1 sets up)
where uv >nul 2>&1
if %errorlevel% == 0 (
  for /f "usebackq delims=" %%i in (`uv python find 3.11 2^>nul`) do set "UV_PY=%%i"
  if exist "%UV_PY%" (
    "%UV_PY%" "%SCRIPT_DIR%scripts\dev.py" %*
    exit /b %errorlevel%
  )
  rem uv present but no 3.11 yet — let uv install/resolve on the fly
  uv run --python 3.11 python "%SCRIPT_DIR%scripts\dev.py" %*
  exit /b %errorlevel%
)

rem 2. Fall back to system Python (dev install path)
where python >nul 2>&1
if %errorlevel% == 0 (
  python "%SCRIPT_DIR%scripts\dev.py" %*
  exit /b %errorlevel%
)

echo Error: no Python interpreter found. Run install.ps1 first. 1>&2
exit /b 1

:add_uv_to_path
if exist "%USERPROFILE%\.local\bin\uv.exe" set "PATH=%USERPROFILE%\.local\bin;%PATH%" & goto :eof
if exist "%USERPROFILE%\.cargo\bin\uv.exe" set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
goto :eof
