$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktop "CouncilMed Demo.lnk"
$wsh = New-Object -ComObject WScript.Shell
$sc = $wsh.CreateShortcut($shortcutPath)
$sc.TargetPath = "C:\Users\nihal\CouncilMed\START_COUNCILMED_DEMO.bat"
$sc.WorkingDirectory = "C:\Users\nihal\CouncilMed"
$sc.Description = "Start CouncilMed Live Demo Infrastructure"
$sc.Save()
Write-Host "Created Desktop shortcut: $shortcutPath"
