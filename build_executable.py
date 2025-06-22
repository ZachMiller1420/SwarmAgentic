"""
Build script for creating standalone executable
Uses PyInstaller to package the AI Agent Demonstration System
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def build_executable():
    """Build standalone executable using PyInstaller"""
    
    print("Building AI Agent Demonstration System Executable...")
    print("=" * 60)
    
    # Configuration
    app_name = "AI_Agent_Demo"
    main_script = "main.py"
    icon_file = None  # Add icon file path if available
    
    # Build directory
    build_dir = Path("build")
    dist_dir = Path("dist")
    
    # Clean previous builds
    if build_dir.exists():
        shutil.rmtree(build_dir)
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    
    # PyInstaller command
    cmd = [
        "pyinstaller",
        "--name", app_name,
        "--onefile",  # Create single executable
        "--windowed",  # Hide console window (remove for debugging)
        "--clean",
        "--noconfirm",
    ]
    
    # Add icon if available
    if icon_file and Path(icon_file).exists():
        cmd.extend(["--icon", icon_file])
    
    # Add data files
    data_files = [
        ("academic_summary.md", "."),
        ("requirements.txt", "."),
        ("src", "src"),
    ]
    
    for src, dst in data_files:
        if Path(src).exists():
            cmd.extend(["--add-data", f"{src}{os.pathsep}{dst}"])
    
    # Hidden imports (add modules that PyInstaller might miss)
    hidden_imports = [
        "torch",
        "transformers",
        "numpy",
        "matplotlib",
        "tkinter",
        "PIL",
        "sklearn",
        "datasets",
    ]
    
    for module in hidden_imports:
        cmd.extend(["--hidden-import", module])
    
    # Exclude unnecessary modules to reduce size
    excludes = [
        "test",
        "tests",
        "pytest",
        "setuptools",
        "distutils",
    ]
    
    for module in excludes:
        cmd.extend(["--exclude-module", module])
    
    # Add main script
    cmd.append(main_script)
    
    print(f"Running command: {' '.join(cmd)}")
    print()
    
    try:
        # Run PyInstaller
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Build successful!")
        print(result.stdout)
        
        # Copy additional files to dist directory
        copy_additional_files()
        
        print(f"\nExecutable created: {dist_dir / app_name}.exe")
        print(f"Distribution directory: {dist_dir.absolute()}")
        
    except subprocess.CalledProcessError as e:
        print(f"Build failed with error code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    
    except FileNotFoundError:
        print("Error: PyInstaller not found. Please install it with:")
        print("pip install pyinstaller")
        return False
    
    return True

def copy_additional_files():
    """Copy additional files needed for the application"""
    
    dist_dir = Path("dist")
    
    # Files to copy
    additional_files = [
        "README.md",
        "academic_summary.md",
        "requirements.txt",
    ]
    
    # Directories to copy
    additional_dirs = [
        "bert-base-uncased-mrpc",
        "logs",
        "results",
    ]
    
    # Copy files
    for file_path in additional_files:
        src = Path(file_path)
        if src.exists():
            dst = dist_dir / src.name
            shutil.copy2(src, dst)
            print(f"Copied: {src} -> {dst}")
    
    # Copy directories
    for dir_path in additional_dirs:
        src = Path(dir_path)
        if src.exists():
            dst = dist_dir / src.name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"Copied directory: {src} -> {dst}")
        else:
            # Create empty directory
            dst = dist_dir / dir_path
            dst.mkdir(exist_ok=True)
            print(f"Created directory: {dst}")

def create_installer():
    """Create an installer using NSIS (if available)"""
    
    nsis_script = """
; AI Agent Demo Installer Script
!define APP_NAME "AI Agent Demonstration System"
!define APP_VERSION "1.0.0"
!define APP_PUBLISHER "AI Systems"
!define APP_EXE "AI_Agent_Demo.exe"

Name "${APP_NAME}"
OutFile "AI_Agent_Demo_Installer.exe"
InstallDir "$PROGRAMFILES\\${APP_NAME}"

Page directory
Page instfiles

Section "Install"
    SetOutPath $INSTDIR
    File /r "dist\\*"
    
    ; Create shortcuts
    CreateDirectory "$SMPROGRAMS\\${APP_NAME}"
    CreateShortCut "$SMPROGRAMS\\${APP_NAME}\\${APP_NAME}.lnk" "$INSTDIR\\${APP_EXE}"
    CreateShortCut "$DESKTOP\\${APP_NAME}.lnk" "$INSTDIR\\${APP_EXE}"
    
    ; Create uninstaller
    WriteUninstaller "$INSTDIR\\Uninstall.exe"
    CreateShortCut "$SMPROGRAMS\\${APP_NAME}\\Uninstall.lnk" "$INSTDIR\\Uninstall.exe"
SectionEnd

Section "Uninstall"
    Delete "$INSTDIR\\*"
    RMDir /r "$INSTDIR"
    Delete "$SMPROGRAMS\\${APP_NAME}\\*"
    RMDir "$SMPROGRAMS\\${APP_NAME}"
    Delete "$DESKTOP\\${APP_NAME}.lnk"
SectionEnd
"""
    
    # Write NSIS script
    nsis_file = Path("installer.nsi")
    with open(nsis_file, 'w') as f:
        f.write(nsis_script)
    
    print(f"NSIS installer script created: {nsis_file}")
    print("To create installer, run: makensis installer.nsi")

def main():
    """Main build function"""
    
    print("AI Agent Demonstration System - Build Script")
    print("=" * 60)
    
    # Check if main script exists
    if not Path("main.py").exists():
        print("Error: main.py not found in current directory")
        return 1
    
    # Check if PyInstaller is available
    try:
        subprocess.run(["pyinstaller", "--version"], 
                      check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Installing PyInstaller...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], 
                          check=True)
        except subprocess.CalledProcessError:
            print("Failed to install PyInstaller")
            return 1
    
    # Build executable
    if build_executable():
        print("\n" + "=" * 60)
        print("Build completed successfully!")
        
        # Create installer script
        create_installer()
        
        print("\nNext steps:")
        print("1. Test the executable in the dist/ directory")
        print("2. Optionally create an installer using NSIS")
        print("3. Distribute the application")
        
        return 0
    else:
        print("Build failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
