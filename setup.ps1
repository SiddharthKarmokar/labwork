# Set URLs for the latest installers
$gitInstallerURL = "https://github.com/git-for-windows/git/releases/download/v2.42.0.windows.1/Git-2.42.0-64-bit.exe"  # Update with latest version
$anacondaInstallerURL = "https://repo.anaconda.com/archive/Anaconda3-2023.07-Windows-x86_64.exe"  # Update with latest version
$vscodeInstallerURL = "https://update.code.visualstudio.com/latest/win32-x64-user/stable"  # Latest stable version

# Download Git installer
Write-Host "Downloading Git..."
Invoke-WebRequest -Uri $gitInstallerURL -OutFile "C:\temp\GitInstaller.exe"

# Download Anaconda installer
Write-Host "Downloading Anaconda..."
Invoke-WebRequest -Uri $anacondaInstallerURL -OutFile "C:\temp\AnacondaInstaller.exe"

# Download VS Code installer
Write-Host "Downloading Visual Studio Code..."
Invoke-WebRequest -Uri $vscodeInstallerURL -OutFile "C:\temp\VSCodeSetup.exe"

# Install Git
Write-Host "Installing Git..."
Start-Process -FilePath "C:\temp\GitInstaller.exe" -ArgumentList "/VERYSILENT" -Wait
Write-Host "Git installation complete."

# Install Anaconda
Write-Host "Installing Anaconda..."
Start-Process -FilePath "C:\temp\AnacondaInstaller.exe" -ArgumentList "/S" -Wait
Write-Host "Anaconda installation complete."

# Install Visual Studio Code
Write-Host "Installing Visual Studio Code..."
Start-Process -FilePath "C:\temp\VSCodeSetup.exe" -ArgumentList "/VERYSILENT" -Wait
Write-Host "Visual Studio Code installation complete."

# Clean up installers
Write-Host "Cleaning up installers..."
Remove-Item "C:\temp\GitInstaller.exe"
Remove-Item "C:\temp\AnacondaInstaller.exe"
Remove-Item "C:\temp\VSCodeSetup.exe"

Write-Host "All installations complete!"
