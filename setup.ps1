$ErrorActionPreference = "Stop"
$ProgressPreference = 'SilentlyContinue'
$WorkDir = $PSScriptRoot  # Папка, где лежит скрипт
$PyDir = "$WorkDir\python"
$PyExe = "$PyDir\python.exe"

Write-Host "`n=== 1/7: Подготовка окружения ===" -ForegroundColor Cyan
Write-Host "Рабочая папка: $WorkDir"

Write-Host "`n=== 2/7: Установка Python 3.10 ===" -ForegroundColor Cyan
if (-not (Test-Path $PyExe)) {
    $zipUrl = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip"
    $zipPath = "$WorkDir\python.zip"
    Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath
    Expand-Archive -Path $zipPath -DestinationPath $PyDir -Force
    Remove-Item $zipPath

    $pth = "$PyDir\python310._pth"
    (Get-Content $pth) -replace '#import site', 'import site' | Set-Content $pth
    Add-Content -Path $pth -Value "Lib\site-packages"

    $getPip = "$WorkDir\get-pip.py"
    Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $getPip
    & $PyExe $getPip --no-warn-script-location | Out-Null
    Remove-Item $getPip
    Write-Host "✅ Python готов" -ForegroundColor Green
} else {
    Write-Host "✅ Python уже установлен" -ForegroundColor Green
}

Write-Host "`n=== 3/7: Установка библиотек (5-10 минут) ===" -ForegroundColor Cyan
& $PyExe -m pip install --upgrade pip --quiet
& $PyExe -m pip install --quiet rdkit pandas matplotlib "numpy==1.26.4" "scikit-learn==1.3.2" admet-ai chemprop==1.6.1 chemfunc==1.0.10 eskape_model streamlit plotly

$eskPy = "$PyDir\Lib\site-packages\eskape_model\eskape_model.py"
if (Test-Path $eskPy) {
    (Get-Content $eskPy -Raw).Replace('/dev/null', 'NUL') | Set-Content $eskPy -Encoding utf8
}

Write-Host "`n=== 4/7: Скачивание моделей ESKAPE (1.2 ГБ) ===" -ForegroundColor Cyan
$DbDir = "$WorkDir\db"
if (-not (Test-Path "$DbDir\models\all")) {
    $repoZip = "$WorkDir\eskape.zip"
    Invoke-WebRequest -Uri "https://github.com/arnolp3/The-ESKAPE-Model/archive/refs/heads/main.zip" -OutFile $repoZip
    Expand-Archive -Path $repoZip -DestinationPath "$WorkDir\eskape_temp" -Force
    Remove-Item $repoZip
    
    Move-Item "$WorkDir\eskape_temp\The-ESKAPE-Model-main\db" -Destination $DbDir
    Remove-Item -Recurse -Force "$WorkDir\eskape_temp"
    
    # Докачка LFS
    $dbRoot = (Resolve-Path $DbDir).Path
    $smallFiles = Get-ChildItem -Path $DbDir -Recurse -File | Where-Object { $_.Length -lt 1000 }
    foreach ($f in $smallFiles) {
        $rel = $f.FullName.Substring($dbRoot.Length + 1).Replace('\', '/')
        $url = "https://media.githubusercontent.com/media/arnolp3/The-ESKAPE-Model/main/db/$rel"
        try { Invoke-WebRequest -Uri $url -OutFile $f.FullName -UseBasicParsing } catch {}
    }
}

Write-Host "`n=== 5/7: Запуск Chemio Profiler ===" -ForegroundColor Cyan
Write-Host "Открываем браузер http://localhost:8501..." -ForegroundColor Yellow
Start-Sleep -Seconds 2
Start-Process "http://localhost:8501" -ErrorAction SilentlyContinue
& $PyExe -m streamlit run "$WorkDir\streamlit_app.py" --server.headless=true --browser.gatherUsageStats=false
