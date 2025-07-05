@echo off
echo 正在清理当前目录下的 .gz, .bbl 文件和所有以 "_minted" 开头的文件夹...

:: 清理 .gz 文件
del /s /q "*.gz"
if exist "*.gz" (
    echo 无法删除所有 .gz 文件。
) else (
    echo .gz 文件清理完成。
)
echo.

:: 清理 .bbl 文件
del /s /q "*.bbl"
if exist "*.bbl" (
    echo 无法删除所有 .bbl 文件。
) else (
    echo .bbl 文件清理完成。
)
echo.

:: 删除所有以 "_minted" 开头的文件夹
echo 正在删除所有以 "_minted" 开头的文件夹...
set "found_minted_dir=0"
for /d %%i in ("_minted*") do (
    set "found_minted_dir=1"
    echo 发现并删除文件夹: "%%i"
    rmdir /s /q "%%i"
    if exist "%%i" (
        echo 警告: 无法删除文件夹 "%%i"。
    )
)

if %found_minted_dir% equ 0 (
    echo 未找到任何以 "_minted" 开头的文件夹。
) else (
    echo 所有以 "_minted" 开头的文件夹清理完成。
)
echo.

echo 清理操作完成。
pause

