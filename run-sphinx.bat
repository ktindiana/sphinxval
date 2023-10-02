@ECHO OFF

SET PYTHONPATH=%cd%
C:\Users\lstegema\AppData\Local\anaconda3\python.exe .\bin\sphinx.py --ModelList ..\lists\SEPVAL_ReportDevelopment.txt --DataList ..\lists\obs_list_GOES_SC24_25.txt > ..\output\log_SEPVAL_ReportDevelopment.txt

C:\Users\lstegema\AppData\Local\anaconda3\python.exe .\bin\report.py
 
