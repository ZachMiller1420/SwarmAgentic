"""
PhD-Level AI Agent Demonstration System
Main Application Entry Point

Integrates BERT-based reasoning, advanced GUI, real-time monitoring,
and interactive demonstration capabilities.
"""

import sys
import os
import asyncio
import logging
import traceback
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.core.ai_agent import PhDLevelAIAgent
    from src.core.bert_engine import BERTReasoningEngine
    from src.gui.main_window import AIAgentGUI
    from src.training.learning_system import AdaptiveLearningSystem
    from src.monitoring.quality_metrics import RealTimeMetricsCollector
    from src.demonstration.interactive_demo import InteractiveDemonstrationSystem
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed and the project structure is correct.")
    sys.exit(1)

class AIAgentDemonstrationApp:
    """
    Main application class that orchestrates all components
    """
    
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.ai_agent: Optional[PhDLevelAIAgent] = None
        self.gui: Optional[AIAgentGUI] = None
        self.quality_metrics: Optional[RealTimeMetricsCollector] = None
        self.demo_system: Optional[InteractiveDemonstrationSystem] = None
        self.learning_system: Optional[AdaptiveLearningSystem] = None
        
        # Configuration
        self.config = self.load_configuration()
        
        self.logger.info("AI Agent Demonstration System initialized")
    
    def setup_logging(self):
        """Setup comprehensive logging system"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(logs_dir / "ai_agent_demo.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def load_configuration(self) -> dict:
        """Load application configuration"""
        return {
            "bert_model_path": "bert-base-uncased-mrpc/bert-base-uncased-mrpc/huggingface_Intel_bert-base-uncased-mrpc_v1",
            "academic_paper_path": "academic_summary.md",
            "max_sequence_length": 512,
            "quality_metrics_window": 1000,
            "enable_real_time_monitoring": True,
            "demo_auto_start": False,
            "export_results": True
        }
    
    def initialize_components(self):
        """Initialize all system components"""
        try:
            self.logger.info("Initializing system components...")
            
            # Initialize quality metrics collector
            self.quality_metrics = RealTimeMetricsCollector(
                window_size=self.config["quality_metrics_window"]
            )
            
            # Initialize AI Agent
            self.ai_agent = PhDLevelAIAgent(
                bert_model_path=self.config["bert_model_path"],
                academic_paper_path=self.config["academic_paper_path"]
            )
            
            # Initialize learning system
            self.learning_system = AdaptiveLearningSystem(
                self.ai_agent.bert_engine
            )
            
            # Initialize demonstration system
            self.demo_system = InteractiveDemonstrationSystem(
                self.ai_agent,
                self.quality_metrics
            )
            
            # Initialize GUI
            self.gui = AIAgentGUI()
            self.gui.set_agent(self.ai_agent)
            
            # Setup component integrations
            self.setup_integrations()
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def setup_integrations(self):
        """Setup integrations between components"""
        
        # Connect AI agent to quality metrics
        def on_agent_response(response_data):
            if self.quality_metrics and 'accuracy' in response_data:
                self.quality_metrics.record_accuracy(
                    response_data['accuracy'],
                    response_data.get('source', 'agent')
                )
            if self.quality_metrics and 'confidence' in response_data:
                self.quality_metrics.record_confidence(
                    response_data['confidence'],
                    response_data.get('context', '')
                )

        # Connect learning system to quality metrics
        def on_learning_progress(progress_data):
            if self.quality_metrics and 'understanding_level' in progress_data:
                self.quality_metrics.record_custom_metric(
                    'learning_progress',
                    progress_data['understanding_level'],
                    {'source': 'learning_system'}
                )

        # Connect demo system to quality metrics
        def on_demo_response(demo_data):
            if self.quality_metrics and 'confidence_score' in demo_data:
                self.quality_metrics.record_confidence(
                    demo_data['confidence_score'],
                    f"demo: {demo_data.get('scenario_id', 'unknown')}"
                )

        # Register callbacks
        if self.demo_system:
            self.demo_system.register_callback('response', on_demo_response)

        # Start quality metrics monitoring if enabled
        if self.config["enable_real_time_monitoring"] and self.quality_metrics:
            self.quality_metrics.start_monitoring(update_interval=1.0)
    
    def run(self):
        """Run the main application"""
        try:
            self.logger.info("Starting AI Agent Demonstration System")
            
            # Initialize all components
            self.initialize_components()
            
            # Display startup information
            self.display_startup_info()
            
            # Start the GUI
            self.logger.info("Launching GUI...")
            if self.gui:
                self.gui.run()
            else:
                self.logger.error("GUI not initialized")
            
        except KeyboardInterrupt:
            self.logger.info("Application interrupted by user")
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            self.cleanup()
    
    def display_startup_info(self):
        """Display startup information"""
        print("\n" + "="*80)
        print("PhD-Level AI Agent Demonstration System")
        print("="*80)
        print(f"BERT Model Path: {self.config['bert_model_path']}")
        print(f"Academic Paper: {self.config['academic_paper_path']}")
        print(f"Quality Metrics Window: {self.config['quality_metrics_window']}")
        print(f"Real-time Monitoring: {self.config['enable_real_time_monitoring']}")
        print("="*80)
        print("\nSystem Features:")
        print("• PhD-level reasoning with BERT integration")
        print("• Real-time chain-of-thought visualization")
        print("• Interactive scratchpad and working memory")
        print("• Comprehensive quality metrics and monitoring")
        print("• Dynamic training on academic content")
        print("• Interactive demonstration scenarios")
        print("• Source quality tracking and validation")
        print("• Expected discovery pattern recognition")
        print("="*80)
        print("\nInstructions:")
        print("1. Click 'Start Training' to train the agent on the academic paper")
        print("2. Wait for training completion (progress will be shown)")
        print("3. Click 'Start Demo' to begin the interactive demonstration")
        print("4. Monitor real-time metrics and thought processes")
        print("5. Use Pause/Stop/Reset controls as needed")
        print("="*80)
        print("\nGUI is launching...")
    
    def cleanup(self):
        """Cleanup resources before exit"""
        self.logger.info("Cleaning up resources...")
        
        try:
            # Stop quality metrics monitoring
            if self.quality_metrics:
                self.quality_metrics.stop_monitoring()
            
            # Stop any running demonstrations
            if self.demo_system and self.demo_system.is_running:
                self.demo_system.stop_demonstration()
            
            # Export results if enabled
            if self.config.get("export_results", False):
                self.export_session_results()
            
            self.logger.info("Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def export_session_results(self):
        """Export session results and metrics"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            
            # Export quality metrics
            if self.quality_metrics:
                metrics_file = results_dir / f"quality_metrics_{timestamp}.json"
                self.quality_metrics.export_metrics(str(metrics_file))
            
            # Export demo results
            if self.demo_system:
                demo_file = results_dir / f"demo_results_{timestamp}.json"
                self.demo_system.export_demo_results(str(demo_file))
            
            # Export agent knowledge
            if self.ai_agent and hasattr(self.ai_agent, 'knowledge_base'):
                knowledge_file = results_dir / f"knowledge_base_{timestamp}.json"
                with open(knowledge_file, 'w') as f:
                    json.dump(self.ai_agent.knowledge_base, f, indent=2, default=str)
            
            self.logger.info(f"Session results exported to {results_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to export session results: {e}")

def main():
    """Main entry point"""
    try:
        # Check Python version
        if sys.version_info < (3, 8):
            print("Error: Python 3.8 or higher is required")
            sys.exit(1)
        
        # Check if running in virtual environment (recommended)
        if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            print("Warning: Not running in a virtual environment. This is not recommended.")
            print("Consider creating and activating a virtual environment first.")
        
        # Create and run the application
        app = AIAgentDemonstrationApp()
        app.run()
        
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
