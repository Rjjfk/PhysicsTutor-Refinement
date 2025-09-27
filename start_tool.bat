@echo off
chcp 65001 >nul 2>&1  :: 解决中文乱码
color 0A  :: 设置绿色文字，增强视觉体验

:: 清屏并显示启动横幅
cls
echo.
echo ==============================================================
echo                    爱莉希雅的训练工坊 ~?
echo ==============================================================
echo.

setlocal enabledelayedexpansion

:: 配置区（根据项目调整）
set "REQUIRED_DEPS=flask flask-cors psutil pynvml"  :: 关键依赖
set "BACKEND_PORT=5000"                             :: 后端端口
set "PYTHON_CMD=python"                             :: 若用虚拟环境，改为：venv\Scripts\python.exe
set "LOG_FILE=logs\launch_error.log"                :: 启动错误日志路径
set "BACKEND_LOG=logs\backend_runtime.log"          :: 后端运行日志

:: 1. 初始化日志目录和文件
echo [1/8] 初始化日志系统...
md "logs" 2>nul
echo ============== 启动日志 %date% %time% ============== > "%LOG_FILE%"
echo 日志系统初始化完成 >>"%LOG_FILE%"

:: 2. 检查核心文件是否存在
echo [2/8] 检查核心文件...
if not exist "backend.py" (
    echo 【错误】未找到 backend.py 文件！请确保BAT在项目根目录。>>"%LOG_FILE%"
    echo.
    echo [91m错误：未找到 backend.py 文件！[0m
    echo 解决方案：将 start_tool.bat 移动到项目根目录后重试。
    echo.
    pause
    exit /b 1
)
if not exist "web\training_dashboard.html" (
    echo 【错误】未找到前端文件：web\training_dashboard.html>>"%LOG_FILE%"
    echo.
    echo [91m错误：未找到前端页面（web\training_dashboard.html）！[0m
    echo 解决方案：检查 web 文件夹是否存在，或重新下载前端文件。
    echo.
    pause
    exit /b 1
)
echo 核心文件检查通过
echo 核心文件检查通过 >>"%LOG_FILE%"

:: 3. 检查Python是否可用
echo [3/8] 检查Python环境...
%PYTHON_CMD% --version >nul 2>&1
if %errorlevel% neq 0 (
    for /f "tokens=2 delims= " %%v in ('%PYTHON_CMD% --version 2^>^&1') do (
        set "PY_VER=%%v"
    )
    echo 检测到Python版本：!PY_VER!
    echo 检测到Python版本：!PY_VER! >>"%LOG_FILE%"
) else (
    echo 【错误】Python解释器未找到！路径：%PYTHON_CMD%>>"%LOG_FILE%"
    echo.
    echo [91m错误：Python未找到或路径配置错误！[0m
    echo 解决方案：
    echo  1. 若未安装Python：下载3.8+版本（https://www.python.org/）
    echo  2. 若用虚拟环境：修改BAT中 "PYTHON_CMD" 为虚拟环境路径（如 venv\Scripts\python.exe）
    echo.
    pause
    exit /b 1
)

:: 4. 检查关键依赖是否安装
echo [4/8] 检查项目依赖...
set "MISSING_DEPS="
for %%d in (%REQUIRED_DEPS%) do (
    %PYTHON_CMD% -c "import %%d" >nul 2>&1
    if !errorlevel! neq 0 (
        set "MISSING_DEPS=!MISSING_DEPS! %%d"
        echo 【警告】缺失依赖：%%d>>"%LOG_FILE%"
    )
)

:: 若有缺失依赖，提供自动安装选项
if defined MISSING_DEPS (
    echo.
    echo [93m检测到缺失依赖：!MISSING_DEPS![0m
    set /p "INSTALL=是否自动安装这些依赖？(Y/N): "
    if /i "!INSTALL!"=="Y" (
        echo 开始安装依赖...>>"%LOG_FILE%"
        %PYTHON_CMD% -m pip install --upgrade pip >nul 2>&1
        %PYTHON_CMD% -m pip install %REQUIRED_DEPS%
        if !errorlevel! neq 0 (
            echo 【错误】依赖安装失败！>>"%LOG_FILE%"
            echo [91m依赖安装失败，请手动执行以下命令：[0m
            echo %PYTHON_CMD% -m pip install %REQUIRED_DEPS%
            pause
            exit /b 1
        )
        echo 依赖安装完成>>"%LOG_FILE%"
    ) else (
        echo 用户取消安装依赖>>"%LOG_FILE%"
        echo 请手动安装缺失的依赖后重试：
        echo %PYTHON_CMD% -m pip install %REQUIRED_DEPS%
        pause
        exit /b 1
    )
)
echo 所有依赖检查通过
echo 所有依赖检查通过 >>"%LOG_FILE%"

:: 5. 检查后端端口是否被占用
echo [5/8] 检查端口可用性...
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
    echo.
    echo [91m错误：!BACKEND_PORT!端口被占用，无法启动后端！[0m
    echo 占用程序：!PROC_NAME!（PID:!PID!）
    echo 解决方案：
    echo  1. 关闭占用端口的程序（任务管理器→详细信息→找到PID关闭）
    echo  2. 修改 backend.py 中端口（搜索 app.run(port=!BACKEND_PORT!) 改为其他端口）
    echo.
    pause
    exit /b 1
)
echo 端口!BACKEND_PORT!可用
echo 端口!BACKEND_PORT!可用 >>"%LOG_FILE%"

:: 6. 启动后端服务
echo [6/8] 启动后端服务...
start /b %PYTHON_CMD% backend.py > "%BACKEND_LOG%" 2>&1
if !errorlevel! neq 0 (
    echo 【错误】后端启动失败！>>"%LOG_FILE%"
    echo.
    echo [91m错误：后端服务启动失败！[0m
    echo 解决方案：
    echo  1. 查看详细错误：notepad "%BACKEND_LOG%"
    echo  2. 常见问题：Python版本不兼容、代码错误、权限问题
    echo.
    pause
    exit /b 1
)

:: 7. 等待后端初始化并验证服务是否正常启动
echo [7/8] 验证后端服务...
set "RETRY=0"
set "MAX_RETRY=10"
:CHECK_BACKEND
timeout /t 1 /nobreak >nul
set /a RETRY+=1
curl http://localhost:%BACKEND_PORT%/api/system-info >nul 2>&1
if !errorlevel! equ 0 (
    goto BACKEND_READY
)
if !RETRY! geq !MAX_RETRY! (
    echo 【错误】后端服务未响应！>>"%LOG_FILE%"
    echo.
    echo [91m错误：后端服务启动后未响应！[0m
    echo 解决方案：
    echo  1. 查看后端日志：notepad "%BACKEND_LOG%"
    echo  2. 尝试手动启动：%PYTHON_CMD% backend.py
    echo.
    pause
    exit /b 1
)
goto CHECK_BACKEND

:BACKEND_READY
echo 后端服务已成功启动
echo 后端服务已成功启动 >>"%LOG_FILE%"

:: 8. 打开前端页面
echo [8/8] 启动前端界面...
echo 启动成功，打开前端页面...>>"%LOG_FILE%"
start "" "web\training_dashboard.html"

:: 显示启动成功信息
echo.
echo ==============================================================
echo [92m启动成功！爱莉希雅的训练工坊已准备就绪～ ?[0m
echo ==============================================================
echo.
echo 后端服务运行在：http://localhost:%BACKEND_PORT%
echo 前端页面已自动打开，若未打开请手动访问：
echo web\training_dashboard.html
echo.
echo 日志信息：
echo  - 启动日志：%LOG_FILE%
echo  - 后端日志：%BACKEND_LOG%
echo.
echo 若遇到问题，请先查看日志文件排查错误
echo.
pause
endlocal