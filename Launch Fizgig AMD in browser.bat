@echo off
REM Fizgig web UI in a browser, with the AMD / ROCm launch env (PATH, hipinfo, VRAM totals).
REM Plain "Launch Fizgig in browser.bat" is the NVIDIA path and intentionally skips this.
setlocal enabledelayedexpansion
cd /d "%~dp0"
title Fizgig AMD

if not exist "venv\Scripts\python.exe" (
    echo Fizgig's Python environment is missing.
    echo.
    echo Run install_fizgig_rocm.bat first, then try again.
    pause
    exit /b 1
)

echo [AMD-ROCm] Setting environment variables...
set "MIOPEN_FIND_MODE=2"
set "FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE"
set "PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.8"
set "FIZGIG_GPU_BACKEND=rocm"

if exist "%~dp0user_data\rocm_env.bat" (
    call "%~dp0user_data\rocm_env.bat"
) else (
    "%~dp0venv\Scripts\python.exe" "%~dp0write_rocm_env.py" >nul 2>&1
    if exist "%~dp0user_data\rocm_env.bat" call "%~dp0user_data\rocm_env.bat"
)
if not defined ROCM_PATH set "ROCM_PATH=%~dp0venv\Lib\site-packages\_rocm_sdk_core"
if not defined HIP_PATH set "HIP_PATH=%ROCM_PATH%"

if defined ROCM_PATH if exist "%ROCM_PATH%\bin" set "PATH=%ROCM_PATH%\bin;%PATH%"
if exist "%~dp0venv\Lib\site-packages\_rocm_sdk_devel\bin" set "PATH=%~dp0venv\Lib\site-packages\_rocm_sdk_devel\bin;%PATH%"
if exist "%~dp0venv\Scripts" set "PATH=%~dp0venv\Scripts;%PATH%"

REM Prefer install-time FIZGIG_AMD_GFX from rocm_env.bat (detect_gpu often fails in for /f).
set "GPU_ARCH="
if defined FIZGIG_AMD_GFX set "GPU_ARCH=!FIZGIG_AMD_GFX!"
if not defined GPU_ARCH if exist "%~dp0src\install\detect_gpu.py" (
    for /f "delims=" %%A in ('"%~dp0venv\Scripts\python.exe" "%~dp0src\install\detect_gpu.py" 2^>nul') do set "GPU_ARCH=%%A"
)
set "IS_LEGACY_GPU=0"
if defined GPU_ARCH (
    if /I "!GPU_ARCH:~0,6!"=="gfx101" set "IS_LEGACY_GPU=1"
    if /I "!GPU_ARCH:~0,6!"=="gfx103" set "IS_LEGACY_GPU=1"
)

if "!IS_LEGACY_GPU!"=="1" (
    echo [AMD-ROCm] Legacy GPU !GPU_ARCH! - RDNA1/2 SDP overrides ^(no aotriton experimental^)
    set "TORCH_BACKENDS_CUDA_FLASH_SDP_ENABLED=0"
    set "TORCH_BACKENDS_CUDA_MEM_EFF_SDP_ENABLED=0"
    set "TORCH_BACKENDS_CUDA_MATH_SDP_ENABLED=1"
) else (
    if defined GPU_ARCH (
        echo [AMD-ROCm] GPU arch: !GPU_ARCH!
    ) else (
        echo [AMD-ROCm] GPU arch: unknown
    )
    set "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1"
)

set "ROCM_DEVEL_BIN=%~dp0venv\Lib\site-packages\_rocm_sdk_devel\bin"
if /I not "!GPU_ARCH!"=="gfx1100" (
    if exist "!ROCM_DEVEL_BIN!" set "MIOPEN_SYSTEM_DB_PATH=!ROCM_DEVEL_BIN!"
    if exist "!ROCM_DEVEL_BIN!\rocblas" set "ROCBLAS_TENSILE_DB_PATH=!ROCM_DEVEL_BIN!\rocblas"
    if exist "!ROCM_DEVEL_BIN!\rocblas\library" set "ROCBLAS_TENSILE_LIBPATH=!ROCM_DEVEL_BIN!\rocblas\library"
)

set "ROCBLAS_USE_HIPBLASLT_BATCHED="

"venv\Scripts\python.exe" fizgig.py --web --open %*
pause
