Set WshShell = CreateObject("WScript.Shell")
Set Fso = CreateObject("Scripting.FileSystemObject")
' this file lives in src\launch: Fizgig's folder is two up
WshShell.CurrentDirectory = Fso.GetParentFolderName(Fso.GetParentFolderName(Fso.GetParentFolderName(WScript.ScriptFullName)))
WshShell.Run "venv\Scripts\pythonw.exe src\launch\launch.pyw", 0, False
