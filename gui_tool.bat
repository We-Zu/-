@echo off
cd /d "%~dp0"
: 激活 Image_recognition_model 环境
call conda activate Image_recognition_model
:: 运行主程序
python "gui_tool.py"
exit
