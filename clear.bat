@echo off
echo ��������ǰĿ¼�µ� .gz, .bbl �ļ��������� "_minted" ��ͷ���ļ���...

:: ���� .gz �ļ�
del /s /q "*.gz"
if exist "*.gz" (
    echo �޷�ɾ������ .gz �ļ���
) else (
    echo .gz �ļ�������ɡ�
)
echo.

:: ���� .bbl �ļ�
del /s /q "*.bbl"
if exist "*.bbl" (
    echo �޷�ɾ������ .bbl �ļ���
) else (
    echo .bbl �ļ�������ɡ�
)
echo.

:: ɾ�������� "_minted" ��ͷ���ļ���
echo ����ɾ�������� "_minted" ��ͷ���ļ���...
set "found_minted_dir=0"
for /d %%i in ("_minted*") do (
    set "found_minted_dir=1"
    echo ���ֲ�ɾ���ļ���: "%%i"
    rmdir /s /q "%%i"
    if exist "%%i" (
        echo ����: �޷�ɾ���ļ��� "%%i"��
    )
)

if %found_minted_dir% equ 0 (
    echo δ�ҵ��κ��� "_minted" ��ͷ���ļ��С�
) else (
    echo ������ "_minted" ��ͷ���ļ���������ɡ�
)
echo.

echo ���������ɡ�
pause

