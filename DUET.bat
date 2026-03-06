@echo off
:: 强制设置编码为UTF-8，避免中文乱码
chcp 65001 >nul 2>&1
:: 关闭命令回显
setlocal enabledelayedexpansion

:: ===================== 核心配置（根据实际情况调整）=====================
:: Python完整路径（你的Anaconda环境）
set "PYTHON_PATH=E:\Anaconda3\envs\TimeSeq\python.exe"
:: 项目根目录
set "PROJECT_ROOT=E:\Project\PythonProject\TFB"
echo ==============================================
echo 开始执行 DUET 模型 - ETTh1 数据集
echo 项目根目录：!PROJECT_ROOT!
echo ==============================================
echo.

:: 切换到项目根目录（强制跳转，避免路径错误）
cd /d "!PROJECT_ROOT!" || (
    echo 【致命错误】无法切换到项目根目录：!PROJECT_ROOT!
    pause
    exit /b 1
)

:: 检查Python可执行文件是否存在
if not exist "!PYTHON_PATH!" (
    echo 【致命错误】未找到Python可执行文件：!PYTHON_PATH!
    echo 请确认Anaconda环境名称和路径是否正确！
    pause
    exit /b 1
)

:: -------------------------- horizon=96 --------------------------
echo [1/4] 运行 horizon=96 的 DUET 模型...
!PYTHON_PATH! scripts/run_benchmark.py ^
--config-path "rolling_forecast_config.json" ^
--data-name-list "ETTh1.csv" ^
--strategy-args "{\"horizon\": 96}" ^
--model-name "duet.DUET" ^
--model-hyper-params "{\"CI\": 1, \"batch_size\": 32, \"d_ff\": 512, \"d_model\": 512, \"dropout\": 0.5, \"e_layers\": 1, \"factor\": 3, \"fc_dropout\": 0.1, \"horizon\": 96, \"k\": 1, \"loss\": \"MAE\", \"lr\": 0.0005, \"lradj\": \"type1\", \"n_heads\": 1, \"norm\": true, \"num_epochs\": 100, \"num_experts\": 2, \"patch_len\": 48, \"patience\": 5, \"seq_len\": 512}" ^
--deterministic "full" ^
--gpus 0 ^
--num-workers 1 ^
--timeout 60000 ^
--save-path "ETTh1/DUET"

if !errorlevel! neq 0 (
    echo 【错误】horizon=96 运行失败！
    pause
    exit /b 1
)
echo [1/4] horizon=96 运行完成！
echo.

:: -------------------------- horizon=192 -------------------------
echo [2/4] 运行 horizon=192 的 DUET 模型...
!PYTHON_PATH! scripts/run_benchmark.py ^
--config-path "rolling_forecast_config.json" ^
--data-name-list "ETTh1.csv" ^
--strategy-args "{\"horizon\": 192}" ^
--model-name "duet.DUET" ^
--model-hyper-params "{\"CI\": 1, \"batch_size\": 64, \"d_ff\": 512, \"d_model\": 512, \"dropout\": 0.5, \"e_layers\": 1, \"factor\": 3, \"fc_dropout\": 0.1, \"horizon\": 192, \"k\": 2, \"loss\": \"MAE\", \"lr\": 0.0005, \"lradj\": \"type1\", \"n_heads\": 1, \"norm\": true, \"num_epochs\": 100, \"num_experts\": 4, \"patch_len\": 48, \"patience\": 5, \"seq_len\": 336}" ^
--deterministic "full" ^
--gpus 0 ^
--num-workers 1 ^
--timeout 60000 ^
--save-path "ETTh1/DUET"

if !errorlevel! neq 0 (
    echo 【错误】horizon=192 运行失败！
    pause
    exit /b 1
)
echo [2/4] horizon=192 运行完成！
echo.

:: -------------------------- horizon=336 -------------------------
echo [3/4] 运行 horizon=336 的 DUET 模型...
!PYTHON_PATH! scripts/run_benchmark.py ^
--config-path "rolling_forecast_config.json" ^
--data-name-list "ETTh1.csv" ^
--strategy-args "{\"horizon\": 336}" ^
--model-name "duet.DUET" ^
--model-hyper-params "{\"CI\": 1, \"batch_size\": 128, \"d_ff\": 1024, \"d_model\": 512, \"dropout\": 0.4, \"e_layers\": 1, \"factor\": 3, \"fc_dropout\": 0.05, \"horizon\": 336, \"k\": 3, \"loss\": \"MAE\", \"lr\": 0.0001, \"lradj\": \"type1\", \"n_heads\": 2, \"norm\": true, \"num_epochs\": 100, \"num_experts\": 4, \"patch_len\": 48, \"patience\": 5, \"seq_len\": 512}" ^
--deterministic "full" ^
--gpus 0 ^
--num-workers 1 ^
--timeout 60000 ^
--save-path "ETTh1/DUET"

if !errorlevel! neq 0 (
    echo 【错误】horizon=336 运行失败！
    pause
    exit /b 1
)
echo [3/4] horizon=336 运行完成！
echo.

:: -------------------------- horizon=720 -------------------------
echo [4/4] 运行 horizon=720 的 DUET 模型...
!PYTHON_PATH! scripts/run_benchmark.py ^
--config-path "rolling_forecast_config.json" ^
--data-name-list "ETTh1.csv" ^
--strategy-args "{\"horizon\": 720}" ^
--model-name "duet.DUET" ^
--model-hyper-params "{\"CI\": 1, \"batch_size\": 32, \"d_ff\": 512, \"d_model\": 512, \"dropout\": 0.2, \"e_layers\": 2, \"factor\": 3, \"fc_dropout\": 0.1, \"horizon\": 720, \"k\": 2, \"loss\": \"MAE\", \"lr\": 0.0005, \"lradj\": \"type1\", \"n_heads\": 1, \"norm\": true, \"num_epochs\": 100, \"num_experts\": 4, \"patch_len\": 48, \"patience\": 5, \"seq_len\": 512}" ^
--deterministic "full" ^
--gpus 0 ^
--num-workers 1 ^
--timeout 60000 ^
--save-path "ETTh1/DUET"

if !errorlevel! neq 0 (
    echo 【错误】horizon=720 运行失败！
    pause
    exit /b 1
)
echo [4/4] horizon=720 运行完成！
echo.

echo ==============================================
echo ✅ DUET 模型 - ETTh1 数据集 所有实验执行完成！
echo ==============================================
pause
endlocal