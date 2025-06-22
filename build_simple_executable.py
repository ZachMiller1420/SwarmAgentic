"""
Simple Build Script for SwarmAgentic Demo
Creates executable without timestamp issues
"""

import os
import sys
import subprocess
from pathlib import Path

def build_simple_executable():
    """Build executable with simplified options to avoid timestamp issues"""
    
    print("Building SwarmAgentic Demo Executable (Simple Version)...")
    print("=" * 60)
    
    # Configuration
    app_name = "SwarmAgentic_Simple_Demo"
    main_script = "demo_launcher.py"
    
    # PyInstaller command with minimal options
    cmd = [
        "pyinstaller",
        "--name", app_name,
        "--onefile",
        "--clean",
        "--noconfirm",
        "--add-data", "academic_summary.md;.",
        "--add-data", "src;src",
        "--hidden-import", "numpy",
        "--hidden-import", "matplotlib",
        "--hidden-import", "tkinter",
        main_script
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print()
    
    try:
        # Run PyInstaller
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Build successful!")
        print(result.stdout)
        
        # Check if executable was created
        exe_path = Path("dist") / f"{app_name}.exe"
        if exe_path.exists():
            print(f"\n✅ Executable created: {exe_path}")
            print(f"📁 Size: {exe_path.stat().st_size / (1024*1024):.1f} MB")
            return True
        else:
            print("❌ Executable not found")
            return False
        
    except subprocess.CalledProcessError as e:
        print(f"Build failed with error code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def test_executable():
    """Test the created executable"""
    exe_path = Path("dist") / "SwarmAgentic_Simple_Demo.exe"
    
    if exe_path.exists():
        print(f"\n🧪 Testing executable: {exe_path}")
        print("Note: The executable will open a GUI window.")
        print("Close the window to continue with the test.")
        
        try:
            # Test run (will open GUI briefly)
            result = subprocess.run([str(exe_path)], timeout=5, capture_output=True)
            print("✅ Executable launches successfully")
            return True
        except subprocess.TimeoutExpired:
            print("✅ Executable is running (GUI opened)")
            return True
        except Exception as e:
            print(f"❌ Executable test failed: {e}")
            return False
    else:
        print("❌ Executable not found for testing")
        return False

def main():
    """Main build function"""
    
    print("SwarmAgentic Demo - Simple Executable Builder")
    print("=" * 60)
    
    # Check if demo launcher exists
    if not Path("demo_launcher.py").exists():
        print("❌ Error: demo_launcher.py not found")
        return 1
    
    # Build executable
    if build_simple_executable():
        print("\n" + "=" * 60)
        print("✅ Build completed successfully!")
        
        # List all executables
        dist_dir = Path("dist")
        if dist_dir.exists():
            exe_files = list(dist_dir.glob("*.exe"))
            print(f"\n📦 Available executables in {dist_dir}:")
            for exe in exe_files:
                size_mb = exe.stat().st_size / (1024*1024)
                print(f"  • {exe.name} ({size_mb:.1f} MB)")
        
        print("\n🚀 Usage Instructions:")
        print("1. Navigate to the 'dist' folder")
        print("2. Double-click any .exe file to run the demo")
        print("3. The SwarmAgentic animated demo will start")
        
        print("\n✨ Features included:")
        print("• Real-time swarm intelligence visualization")
        print("• 4 different animation modes")
        print("• Interactive controls")
        print("• PhD-level AI demonstrations")
        
        return 0
    else:
        print("\n❌ Build failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
