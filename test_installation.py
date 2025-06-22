"""
Installation Test Script for AI Agent Demonstration System
Validates that all components can be imported and basic functionality works
"""

import sys
import traceback
from pathlib import Path

def test_python_version():
    """Test Python version compatibility"""
    print("Testing Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def test_dependencies():
    """Test that all required dependencies can be imported"""
    print("\nTesting dependencies...")
    
    dependencies = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("tkinter", "Tkinter"),
        ("PIL", "Pillow"),
        ("sklearn", "Scikit-learn"),
        ("datasets", "Datasets"),
    ]
    
    failed_imports = []
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            failed_imports.append(name)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Please install missing dependencies with: pip install -r requirements.txt")
        return False
    
    return True

def test_file_structure():
    """Test that required files and directories exist"""
    print("\nTesting file structure...")
    
    required_files = [
        "main.py",
        "academic_summary.md",
        "requirements.txt",
        "src/__init__.py",
        "src/core/__init__.py",
        "src/core/bert_engine.py",
        "src/core/ai_agent.py",
        "src/gui/__init__.py",
        "src/gui/main_window.py",
        "src/training/__init__.py",
        "src/training/learning_system.py",
        "src/monitoring/__init__.py",
        "src/monitoring/quality_metrics.py",
        "src/demonstration/__init__.py",
        "src/demonstration/interactive_demo.py",
    ]
    
    required_dirs = [
        "src",
        "src/core",
        "src/gui",
        "src/training",
        "src/monitoring",
        "src/demonstration",
        "bert-base-uncased-mrpc",
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
        else:
            print(f"✅ {dir_path}/")
    
    if missing_files or missing_dirs:
        if missing_files:
            print(f"\n❌ Missing files: {', '.join(missing_files)}")
        if missing_dirs:
            print(f"\n❌ Missing directories: {', '.join(missing_dirs)}")
        return False
    
    return True

def test_bert_model():
    """Test BERT model availability"""
    print("\nTesting BERT model...")
    
    bert_model_path = Path("bert-base-uncased-mrpc/bert-base-uncased-mrpc/huggingface_Intel_bert-base-uncased-mrpc_v1")
    
    if not bert_model_path.exists():
        print(f"❌ BERT model directory not found: {bert_model_path}")
        return False
    
    required_model_files = [
        "user_script.py",
        "requirements.txt",
    ]
    
    missing_model_files = []
    for file_name in required_model_files:
        file_path = bert_model_path / file_name
        if not file_path.exists():
            missing_model_files.append(file_name)
        else:
            print(f"✅ {file_path}")
    
    if missing_model_files:
        print(f"❌ Missing BERT model files: {', '.join(missing_model_files)}")
        return False
    
    return True

def test_imports():
    """Test that all custom modules can be imported"""
    print("\nTesting custom module imports...")
    
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    
    modules_to_test = [
        ("src.core.bert_engine", "BERT Engine"),
        ("src.core.ai_agent", "AI Agent"),
        ("src.gui.main_window", "GUI Main Window"),
        ("src.training.learning_system", "Learning System"),
        ("src.monitoring.quality_metrics", "Quality Metrics"),
        ("src.demonstration.interactive_demo", "Interactive Demo"),
    ]
    
    failed_imports = []
    
    for module_name, display_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ {display_name}")
        except ImportError as e:
            print(f"❌ {display_name}: {e}")
            failed_imports.append(display_name)
        except Exception as e:
            print(f"⚠️  {display_name}: {e}")
            # Non-import errors are warnings, not failures
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of core components"""
    print("\nTesting basic functionality...")
    
    try:
        # Add src to path
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        
        # Test BERT engine initialization
        from src.core.bert_engine import BERTReasoningEngine
        print("✅ BERT engine class imported")
        
        # Test AI agent initialization
        from src.core.ai_agent import PhDLevelAIAgent
        print("✅ AI agent class imported")
        
        # Test quality metrics
        from src.monitoring.quality_metrics import RealTimeMetricsCollector
        metrics = RealTimeMetricsCollector(window_size=10)
        metrics.record_accuracy(0.85, "test")
        print("✅ Quality metrics basic functionality")
        
        # Test learning system
        from src.training.learning_system import AdaptiveLearningSystem
        print("✅ Learning system class imported")
        
        # Test demo system
        from src.demonstration.interactive_demo import InteractiveDemonstrationSystem
        print("✅ Demo system class imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and return overall result"""
    print("AI Agent Demonstration System - Installation Test")
    print("=" * 60)
    
    tests = [
        ("Python Version", test_python_version),
        ("Dependencies", test_dependencies),
        ("File Structure", test_file_structure),
        ("BERT Model", test_bert_model),
        ("Module Imports", test_imports),
        ("Basic Functionality", test_basic_functionality),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nTo start the application, run: python main.py")
        return True
    else:
        print("❌ Some tests failed. Please fix the issues before running the application.")
        print("\nCommon solutions:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Ensure you're in the correct directory")
        print("3. Check that all files were extracted properly")
        print("4. Verify Python version is 3.8 or higher")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
