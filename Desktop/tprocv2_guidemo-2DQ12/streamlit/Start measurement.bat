@echo off
cd /d %~dp0
call conda activate qick
streamlit run Homepage.py
pause