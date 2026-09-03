$ErrorActionPreference = "Stop"

Write-Host "Determining Rust target triple..."
$rustInfo = rustc -vV | Select-String "host:"
$targetTriple = $rustInfo.ToString().Split(" ")[1].Trim()
Write-Host "Target triple: $targetTriple"

Write-Host "Building PyInstaller executable for backend..."
# Use the virtual environment's pyinstaller
$pyinstaller = ".\venv\Scripts\pyinstaller.exe"

if (-Not (Test-Path $pyinstaller)) {
    Write-Host "PyInstaller not found in venv. Using global pyinstaller."
    $pyinstaller = "pyinstaller"
}

& $pyinstaller --name "app-backend" --onefile backend\run_backend.py --clean --noconfirm `
    --hidden-import "uvicorn.logging" `
    --hidden-import "uvicorn.loops" `
    --hidden-import "uvicorn.loops.auto" `
    --hidden-import "uvicorn.protocols.http.auto" `
    --hidden-import "uvicorn.protocols.websockets.auto" `
    --hidden-import "backend.api.app" `
    --hidden-import "backend.pipeline" `
    --hidden-import "backend.database" `
    --hidden-import "backend.inference.engine" `
    --hidden-import "backend.dicom.pipeline" `
    --hidden-import "backend.reports.generator" `
    --hidden-import "pydicom.encoders.gdcm" `
    --hidden-import "pydicom.encoders.pylibjpeg"

if ($LASTEXITCODE -ne 0) {
    Write-Error "PyInstaller build failed."
    exit 1
}

$binDir = "src-tauri\bin"
if (-Not (Test-Path $binDir)) {
    New-Item -ItemType Directory -Path $binDir | Out-Null
}

$exePath = "dist\app-backend.exe"
$destPath = "$binDir\app-backend-$targetTriple.exe"

if (Test-Path $exePath) {
    Move-Item -Path $exePath -Destination $destPath -Force
    Write-Host "Backend built and moved to $destPath"
} else {
    Write-Error "Executable not found at $exePath"
}
