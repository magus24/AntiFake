@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>&1
cd /d "%~dp0"
title AntiFake - защита фото от распознавания и подмены лица

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONPATH=%CD%"

rem Интерпретатор: сначала локальное окружение, иначе системный python.
set "PY=python"
if exist ".venv\Scripts\pythonw.exe" set "PY=.venv\Scripts\python.exe"

set "ACTION=%~1"
if "%ACTION%"=="" set "ACTION=site"

if /i "%ACTION%"=="help"    goto usage
if /i "%ACTION%"=="-h"      goto usage
if /i "%ACTION%"=="/?"      goto usage
if /i "%ACTION%"=="setup"   goto setup
if /i "%ACTION%"=="install" goto setup
if /i "%ACTION%"=="site"    goto check
if /i "%ACTION%"=="app"     goto check
if /i "%ACTION%"=="full"    goto full
if /i "%ACTION%"=="quick"   goto quick

echo Неизвестная команда: %ACTION%
echo.
goto usage

:usage
echo.
echo   Запуск:
echo     start.bat            запустить сайт (по умолчанию)
echo     start.bat site       то же самое
echo     start.bat setup      установить зависимости и запустить сайт
echo     start.bat full       полный эксперимент (~9 мин, CPU)
echo     start.bat quick      быстрая проверка пайплайна (~2.5 мин, CPU)
echo     start.bat help       эта справка
echo.
goto end

rem Сайту нужны только gradio/torch/opencv/insightface/facenet/skimage.
rem inswapper нужен лишь для "start.bat full", поэтому предупреждаю, но не блокирую.
:check
"%PY%" -c "import gradio, torch, cv2, insightface, facenet_pytorch, skimage" >nul 2>&1
if errorlevel 1 (
    echo Зависимости не найдены. Устанавливаю, это может занять пару минут...
    call :setup_action
    if errorlevel 1 goto fail
)
"%PY%" -c "import os; from src.detect import inswapper_path; raise SystemExit(0 if os.path.isfile(inswapper_path()) else 1)" >nul 2>&1
if errorlevel 1 (
    echo.
    echo Внимание: нет весов inswapper, режим "start.bat full" не сможет
    echo проверить подмену лица. Сайт при этом работает полностью.
    echo Скачать: https://huggingface.co/ezioruan/inswapper_128.onnx
    echo         и положить в models\inswapper_128.onnx
    echo.
)
goto site

:setup_action
if exist ".venv\Scripts\python.exe" (
    "%PY%" -m pip install --upgrade pip
    "%PY%" -m pip install -r requirements.txt
) else (
    echo.
    echo Python-окружение .venv не найдено, ставлю в системный Python.
    echo Рекомендуется один раз создать: python -m venv .venv
    echo.
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
)
if errorlevel 1 (
    echo.
    echo Не удалось установить зависимости.
    echo Проверь requirements.txt: возможно, конфликт версий numpy/scipy.
    exit /b 1
)
exit /b 0

:setup
call :setup_action
if errorlevel 1 goto fail
echo.
echo Готово. Запускаю сайт...
echo.
goto site

:site
echo ============================================================
echo   AntiFake
echo   Сайт откроется автоматически:  http://127.0.0.1:7860
echo   Если не открылся - скопируй адрес в браузер.
echo   Остановить: закрыть это окно или Ctrl+C
echo ============================================================
echo.
"%PY%" app.py --inbrowser
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" goto fail
echo.
echo Сервер остановлен.
goto end

:full
echo Полный эксперимент: детекция, атака, оценка, подмена лица, отчет, график.
echo На CPU это примерно 7 минут. Не закрывай окно.
echo.
"%PY%" run_experiment.py
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" goto fail
echo.
echo Готово. Результаты: outputs\results.md и outputs\summary_chart.png
goto end

:quick
echo Быстрая проверка пайплайна: меньше шагов атаки.
echo Полные результаты в outputs\results.csv останутся нетронутыми,
echo проверка запишется отдельно в outputs\results_quick.*.
echo.
"%PY%" run_experiment.py --quick
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" goto fail
echo.
echo Пайплайн работает. Запустить сайт:  start.bat
goto end

:fail
echo.
echo ============================================================
echo   Запуск завершился с ошибкой, код %RC%
echo   Подробности выше. Частая причина: не установлены зависимости.
echo   Попробуй:  start.bat setup
echo ============================================================
pause

:end
endlocal