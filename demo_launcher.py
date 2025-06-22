"""
Enhanced Demo Launcher for SwarmAgentic AI Agent System
Showcases real-time animated visualization of swarm intelligence
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.gui.swarm_visualization import SwarmVisualization
    from src.core.ai_agent import PhDLevelAIAgent
    from src.monitoring.quality_metrics import RealTimeMetricsCollector
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed.")
    sys.exit(1)

class SwarmAgenticDemoLauncher:
    """
    Enhanced demo launcher with animated SwarmAgentic visualization
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SwarmAgentic AI Agent - Real-Time Demo")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#2C3E50')
        
        # Demo state
        self.demo_running = False
        self.current_scenario = 0
        self.scenarios = [
            "Agent Formation & Swarm Intelligence",
            "PSO-Based Optimization Process", 
            "Multi-Agent Collaboration",
            "Emergent Collective Behavior"
        ]
        
        self._create_demo_interface()
        self._setup_demo_scenarios()
        
    def _create_demo_interface(self):
        """Create the enhanced demo interface"""
        
        # Title frame
        title_frame = tk.Frame(self.root, bg='#2C3E50', height=80)
        title_frame.pack(fill=tk.X, pady=(10, 0))
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="SwarmAgentic AI Agent - Real-Time Demonstration",
            font=("Segoe UI", 20, "bold"),
            bg='#2C3E50',
            fg='white'
        )
        title_label.pack(expand=True)
        
        subtitle_label = tk.Label(
            title_frame,
            text="PhD-Level AI with Animated Swarm Intelligence Visualization",
            font=("Segoe UI", 12),
            bg='#2C3E50',
            fg='#BDC3C7'
        )
        subtitle_label.pack()
        
        # Main content frame
        main_frame = tk.Frame(self.root, bg='#ECF0F1')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls and info
        left_panel = tk.Frame(main_frame, bg='#ECF0F1', width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Control panel
        control_frame = tk.LabelFrame(
            left_panel,
            text="Demo Controls",
            font=("Segoe UI", 12, "bold"),
            bg='#ECF0F1',
            fg='#2C3E50',
            padx=10,
            pady=10
        )
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Demo buttons
        self.start_demo_btn = tk.Button(
            control_frame,
            text="🚀 Start SwarmAgentic Demo",
            font=("Segoe UI", 11, "bold"),
            bg='#27AE60',
            fg='white',
            command=self._start_demo,
            height=2,
            relief=tk.FLAT
        )
        self.start_demo_btn.pack(fill=tk.X, pady=(0, 5))
        
        self.next_scenario_btn = tk.Button(
            control_frame,
            text="⏭️ Next Scenario",
            font=("Segoe UI", 10),
            bg='#3498DB',
            fg='white',
            command=self._next_scenario,
            state='disabled',
            relief=tk.FLAT
        )
        self.next_scenario_btn.pack(fill=tk.X, pady=(0, 5))
        
        self.stop_demo_btn = tk.Button(
            control_frame,
            text="⏹️ Stop Demo",
            font=("Segoe UI", 10),
            bg='#E74C3C',
            fg='white',
            command=self._stop_demo,
            state='disabled',
            relief=tk.FLAT
        )
        self.stop_demo_btn.pack(fill=tk.X)
        
        # Scenario info
        scenario_frame = tk.LabelFrame(
            left_panel,
            text="Current Scenario",
            font=("Segoe UI", 12, "bold"),
            bg='#ECF0F1',
            fg='#2C3E50',
            padx=10,
            pady=10
        )
        scenario_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.scenario_label = tk.Label(
            scenario_frame,
            text="Ready to demonstrate SwarmAgentic",
            font=("Segoe UI", 10),
            bg='#ECF0F1',
            fg='#2C3E50',
            wraplength=250,
            justify=tk.LEFT
        )
        self.scenario_label.pack(fill=tk.X)
        
        # Features info
        features_frame = tk.LabelFrame(
            left_panel,
            text="Demo Features",
            font=("Segoe UI", 12, "bold"),
            bg='#ECF0F1',
            fg='#2C3E50',
            padx=10,
            pady=10
        )
        features_frame.pack(fill=tk.BOTH, expand=True)
        
        features_text = """
🔹 Real-time agent visualization
🔹 Swarm formation dynamics
🔹 PSO optimization animation
🔹 Multi-agent collaboration
🔹 Emergent behavior patterns
🔹 Interactive controls
🔹 Performance metrics
🔹 PhD-level AI reasoning
        """
        
        features_label = tk.Label(
            features_frame,
            text=features_text.strip(),
            font=("Segoe UI", 9),
            bg='#ECF0F1',
            fg='#2C3E50',
            justify=tk.LEFT
        )
        features_label.pack(fill=tk.BOTH, expand=True, anchor=tk.NW)
        
        # Right panel - Visualization
        viz_frame = tk.Frame(main_frame, bg='white')
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create swarm visualization
        self.swarm_viz = SwarmVisualization(viz_frame)
        
        # Status bar
        status_frame = tk.Frame(self.root, bg='#34495E', height=30)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(
            status_frame,
            text="Ready - Click 'Start SwarmAgentic Demo' to begin",
            font=("Segoe UI", 9),
            bg='#34495E',
            fg='white'
        )
        self.status_label.pack(side=tk.LEFT, padx=10, pady=5)
        
    def _setup_demo_scenarios(self):
        """Setup the demo scenario sequence"""
        self.scenario_descriptions = [
            "Demonstrating agent formation and swarm intelligence principles. Watch as agents self-organize into coherent formations using flocking algorithms.",
            
            "Showcasing PSO-based optimization process. Particles explore the solution space and converge towards optimal solutions using swarm intelligence.",
            
            "Illustrating multi-agent collaboration. Coordinator agents manage worker agents in dynamic task allocation and execution scenarios.",
            
            "Revealing emergent collective behavior. Simple agent rules lead to complex, intelligent system-level behaviors and patterns."
        ]
        
    def _start_demo(self):
        """Start the SwarmAgentic demonstration"""
        if not self.demo_running:
            self.demo_running = True
            self.current_scenario = 0
            
            # Update UI
            self.start_demo_btn.config(state='disabled')
            self.next_scenario_btn.config(state='normal')
            self.stop_demo_btn.config(state='normal')
            
            # Start first scenario
            self._run_scenario(0)
            
            # Update status
            self.status_label.config(text="Demo running - SwarmAgentic in action!")
            
    def _next_scenario(self):
        """Move to the next demo scenario"""
        if self.demo_running:
            self.current_scenario = (self.current_scenario + 1) % len(self.scenarios)
            self._run_scenario(self.current_scenario)
            
    def _stop_demo(self):
        """Stop the demonstration"""
        self.demo_running = False
        
        # Stop visualization
        if self.swarm_viz:
            self.swarm_viz.stop_animation()
            
        # Update UI
        self.start_demo_btn.config(state='normal')
        self.next_scenario_btn.config(state='disabled')
        self.stop_demo_btn.config(state='disabled')
        
        # Reset scenario
        self.scenario_label.config(text="Demo stopped - Ready to restart")
        self.status_label.config(text="Demo stopped - Click 'Start SwarmAgentic Demo' to restart")
        
    def _run_scenario(self, scenario_index):
        """Run a specific demo scenario"""
        if scenario_index < len(self.scenarios):
            scenario_name = self.scenarios[scenario_index]
            scenario_desc = self.scenario_descriptions[scenario_index]
            
            # Update scenario display
            self.scenario_label.config(text=f"{scenario_index + 1}. {scenario_name}\n\n{scenario_desc}")
            
            # Update status
            self.status_label.config(text=f"Running: {scenario_name}")
            
            # Configure visualization mode
            mode_mapping = [
                "swarm_formation",
                "pso_optimization", 
                "agent_collaboration",
                "emergent_behavior"
            ]
            
            if self.swarm_viz and scenario_index < len(mode_mapping):
                # Set visualization mode
                self.swarm_viz.current_mode = mode_mapping[scenario_index]
                self.swarm_viz.mode_var.set(mode_mapping[scenario_index])
                
                # Reset and start animation
                self.swarm_viz.reset_visualization()
                self.swarm_viz.start_animation()
                
    def run(self):
        """Run the demo launcher"""
        try:
            # Show welcome message
            messagebox.showinfo(
                "SwarmAgentic Demo",
                "Welcome to the SwarmAgentic AI Agent Demonstration!\n\n"
                "This demo showcases:\n"
                "• Real-time swarm intelligence visualization\n"
                "• PhD-level AI reasoning capabilities\n"
                "• Multi-agent collaboration patterns\n"
                "• Emergent collective behaviors\n\n"
                "Click 'Start SwarmAgentic Demo' to begin!"
            )
            
            self.root.mainloop()
            
        except KeyboardInterrupt:
            print("Demo interrupted by user")
        except Exception as e:
            print(f"Demo error: {e}")
            messagebox.showerror("Error", f"Demo error: {e}")

def main():
    """Main entry point for the enhanced demo"""
    try:
        print("Starting SwarmAgentic Enhanced Demo...")
        print("=" * 50)
        
        # Check dependencies
        print("Checking dependencies...")
        
        # Create and run demo
        demo = SwarmAgenticDemoLauncher()
        demo.run()
        
    except Exception as e:
        print(f"Failed to start demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
