@echo off
chcp 65001 >nul 2>&1  :: 解决中文乱码
echo 正在启动爱莉希雅的训练工坊～
setlocal enabledelayedexpansion

:: 配置区（根据项目调整）
set "REQUIRED_DEPS=flask flask-cors psutil pynvml"  :: 关键依赖
set "BACKEND_PORT=5000"                             :: 后端端口
set "PYTHON_CMD=python"                             :: 若用虚拟环境，改为：venv\Scripts\python.exe
set "LOG_FILE=logs\launch_error.log"                :: 启动错误日志路径

:: 1. 初始化日志目录和文件
md "logs" 2>nul
echo ============== 启动日志 %date% %time% ============== > "%LOG_FILE%"

:: 2. 检查核心文件是否存在
if not exist "backend.py" (
    echo 【错误】未找到 backend.py 文件！请确保BAT在项目根目录。>>"%LOG_FILE%"
    echo 错误：未找到 backend.py 文件！
    echo 解决方案：将 start_tool.bat 移动到项目根目录后重试。
    pause
    exit /b 1
)
if not exist "web\training_dashboard.html" (
    echo 【错误】未找到前端文件：web\training_dashboard.html>>"%LOG_FILE%"
    echo 错误：未找到前端页面（web\training_dashboard.html）！
    echo 解决方案：检查 web 文件夹是否存在，或重新下载前端文件。
    pause
    exit /b 1
)

:: 3. 检查Python是否可用
%PYTHON_CMD% --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 【错误】Python解释器未找到！路径：%PYTHON_CMD%>>"%LOG_FILE%"
    echo 错误：Python未找到或路径配置错误！
    echo 解决方案：
    echo  1. 若未安装Python：下载3.8+版本（https://www.python.org/）
    echo  2. 若用虚拟环境：修改BAT中 "PYTHON_CMD" 为虚拟环境路径（如 venv\Scripts\python.exe）
    pause
    exit /b 1
)

:: 4. 检查关键依赖是否安装（核心！避免后端启动时缺包）
echo 正在检查依赖...>>"%LOG_FILE%"
for %%d in (%REQUIRED_DEPS%) do (
    %PYTHON_CMD% -c "import %%d" >nul 2>&1
    if !errorlevel! neq 0 (
        echo 【错误】缺失依赖：%%d>>"%LOG_FILE%"
        echo 错误：缺失关键依赖「%%d」！
        echo 解决方案：立即安装，执行命令：%PYTHON_CMD% -m pip install %%d
        pause
        exit /b 1
    )
)
echo 所有依赖检查通过>>"%LOG_FILE%"

:: 5. 检查后端端口是否被占用（避免启动后无法访问）
netstat -ano | findstr ":!BACKEND_PORT!" >nul 2>&1
if !errorlevel! equ 0 (
    :: 找到占用端口的进程
    for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":!BACKEND_PORT!"') do (
        set "PID=%%p"
    )
    :: 获取进程名称
    for /f "tokens=1" %%n in ('tasklist /fi "PID eq !PID!" ^| findstr "!PID!"') do (
        set "PROC_NAME=%%n"
    )
    echo 【错误】端口 !BACKEND_PORT! 被占用！进程：!PROC_NAME!（PID:!PID!）>>"%LOG_FILE%"
    echo 错误：5000端口被占用，无法启动后端！
    echo 占用程序：!PROC_NAME!（PID:!PID!）
    echo 解决方案：
    echo  1. 关闭占用端口的程序（任务管理器→详细信息→找到PID关闭）
    echo  2. 修改 backend.py 中端口（搜索 app.run(port=5000) 改为其他端口，如5001）
    pause
    exit /b 1
)

:: 6. 启动后端服务（日志写入文件，便于排查）
echo 正在启动后端服务...>>"%LOG_FILE%"
start /b %PYTHON_CMD% backend.py > "logs\backend_runtime.log" 2>&1
if !errorlevel! neq 0 (
    echo 【错误】后端启动失败！请查看 logs\backend_runtime.log>>"%LOG_FILE%"
    echo 错误：后端服务启动失败！
    echo 解决方案：打开 logs\backend_runtime.log 查看具体错误（如代码报错）。
    pause
    exit /b 1
)

:: 7. 等待后端初始化（3秒，避免前端提前打开）
echo 后端启动中...（3秒后打开前端）>>"%LOG_FILE%"
timeout /t 3 /nobreak >nul

:: 8. 打开前端页面
echo 启动成功，打开前端页面...>>"%LOG_FILE%"
echo 启动完成！可以开始训练啦～ 
start "" "web\training_dashboard.html"

echo ----------------------------
echo 启动成功！
echo 提示：
echo  1. 后端日志：logs\backend_runtime.log
echo  2. 启动日志：logs\launch_error.log
echo  3. 若前端报错，先查看后端日志排查问题。
echo ----------------------------
pause
endlocal