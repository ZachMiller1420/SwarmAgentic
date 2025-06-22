"""
Advanced GUI Framework for AI Agent Demonstration System
Implements sophisticated interface with real-time monitoring and control
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import asyncio
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import numpy as np
import math

from ..core.ai_agent import PhDLevelAIAgent
from .swarm_visualization import SwarmVisualization

class ModernStyle:
    """Modern UI styling configuration"""
    
    # Color scheme
    PRIMARY_COLOR = "#2C3E50"
    SECONDARY_COLOR = "#3498DB"
    SUCCESS_COLOR = "#27AE60"
    WARNING_COLOR = "#F39C12"
    DANGER_COLOR = "#E74C3C"
    BACKGROUND_COLOR = "#ECF0F1"
    TEXT_COLOR = "#2C3E50"
    ACCENT_COLOR = "#9B59B6"
    
    # Fonts
    TITLE_FONT = ("Segoe UI", 16, "bold")
    HEADER_FONT = ("Segoe UI", 12, "bold")
    BODY_FONT = ("Segoe UI", 10)
    MONO_FONT = ("Consolas", 9)
    
    # Dimensions
    BUTTON_HEIGHT = 40
    BUTTON_WIDTH = 120
    PADDING = 10
    BORDER_WIDTH = 2

class AIAgentGUI:
    """
    Main GUI application for the AI Agent Demonstration System
    Features real-time monitoring, control interface, and visualization
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PhD-Level AI Agent Demonstration System")
        self.root.geometry("1400x900")
        self.root.configure(bg=ModernStyle.BACKGROUND_COLOR)
        
        # Initialize AI Agent (will be set later)
        self.agent: Optional[PhDLevelAIAgent] = None
        
        # GUI state
        self.is_training_enabled = True
        self.is_demo_enabled = False
        self.update_thread = None
        self.stop_updates = False
        
        # Data for real-time visualization
        self.accuracy_data = []
        self.confidence_data = []
        self.time_data = []

        # Swarm visualization
        self.swarm_viz: Optional[SwarmVisualization] = None

        self._setup_styles()
        self._create_widgets()
        self._setup_layout()
        self._start_update_thread()
    
    def _setup_styles(self):
        """Configure modern styling for ttk widgets"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure button styles
        style.configure(
            "Modern.TButton",
            background=ModernStyle.SECONDARY_COLOR,
            foreground="white",
            font=ModernStyle.BODY_FONT,
            borderwidth=0,
            focuscolor="none"
        )
        
        style.map(
            "Modern.TButton",
            background=[('active', ModernStyle.PRIMARY_COLOR)]
        )
        
        # Configure success button
        style.configure(
            "Success.TButton",
            background=ModernStyle.SUCCESS_COLOR,
            foreground="white",
            font=ModernStyle.BODY_FONT
        )
        
        # Configure danger button
        style.configure(
            "Danger.TButton",
            background=ModernStyle.DANGER_COLOR,
            foreground="white",
            font=ModernStyle.BODY_FONT
        )
        
        # Configure progress bar
        style.configure(
            "Modern.Horizontal.TProgressbar",
            background=ModernStyle.SUCCESS_COLOR,
            troughcolor=ModernStyle.BACKGROUND_COLOR,
            borderwidth=0,
            lightcolor=ModernStyle.SUCCESS_COLOR,
            darkcolor=ModernStyle.SUCCESS_COLOR
        )
    
    def _create_widgets(self):
        """Create all GUI widgets"""
        # Main container
        self.main_frame = tk.Frame(self.root, bg=ModernStyle.BACKGROUND_COLOR)
        
        # Title
        self.title_label = tk.Label(
            self.main_frame,
            text="PhD-Level AI Agent Demonstration System",
            font=ModernStyle.TITLE_FONT,
            bg=ModernStyle.BACKGROUND_COLOR,
            fg=ModernStyle.PRIMARY_COLOR
        )
        
        # Control panel
        self._create_control_panel()
        
        # Status panel
        self._create_status_panel()
        
        # Real-time monitoring
        self._create_monitoring_panel()
        
        # Scratchpad and thought process
        self._create_scratchpad_panel()
        
        # Performance metrics
        self._create_metrics_panel()
    
    def _create_control_panel(self):
        """Create the main control panel with buttons"""
        self.control_frame = tk.LabelFrame(
            self.main_frame,
            text="Agent Control",
            font=ModernStyle.HEADER_FONT,
            bg=ModernStyle.BACKGROUND_COLOR,
            fg=ModernStyle.PRIMARY_COLOR,
            padx=ModernStyle.PADDING,
            pady=ModernStyle.PADDING
        )
        
        # Control buttons
        self.start_training_btn = ttk.Button(
            self.control_frame,
            text="Start Training",
            style="Success.TButton",
            command=self._start_training,
            width=15
        )
        
        self.start_demo_btn = ttk.Button(
            self.control_frame,
            text="Start Demo",
            style="Modern.TButton",
            command=self._start_demo,
            state="disabled",
            width=15
        )
        
        self.pause_btn = ttk.Button(
            self.control_frame,
            text="Pause",
            style="Modern.TButton",
            command=self._pause_operation,
            width=15
        )
        
        self.stop_btn = ttk.Button(
            self.control_frame,
            text="Stop",
            style="Danger.TButton",
            command=self._stop_operation,
            width=15
        )
        
        self.reset_btn = ttk.Button(
            self.control_frame,
            text="Reset",
            style="Modern.TButton",
            command=self._reset_agent,
            width=15
        )
    
    def _create_status_panel(self):
        """Create status display panel"""
        self.status_frame = tk.LabelFrame(
            self.main_frame,
            text="Agent Status",
            font=ModernStyle.HEADER_FONT,
            bg=ModernStyle.BACKGROUND_COLOR,
            fg=ModernStyle.PRIMARY_COLOR,
            padx=ModernStyle.PADDING,
            pady=ModernStyle.PADDING
        )
        
        # Status indicators
        self.status_text = tk.Label(
            self.status_frame,
            text="Status: Idle",
            font=ModernStyle.BODY_FONT,
            bg=ModernStyle.BACKGROUND_COLOR,
            fg=ModernStyle.TEXT_COLOR
        )
        
        # Progress bars
        self.training_progress_label = tk.Label(
            self.status_frame,
            text="Training Progress:",
            font=ModernStyle.BODY_FONT,
            bg=ModernStyle.BACKGROUND_COLOR,
            fg=ModernStyle.TEXT_COLOR
        )
        
        self.training_progress = ttk.Progressbar(
            self.status_frame,
            style="Modern.Horizontal.TProgressbar",
            length=300,
            mode='determinate'
        )
        
        self.demo_progress_label = tk.Label(
            self.status_frame,
            text="Demo Progress:",
            font=ModernStyle.BODY_FONT,
            bg=ModernStyle.BACKGROUND_COLOR,
            fg=ModernStyle.TEXT_COLOR
        )
        
        self.demo_progress = ttk.Progressbar(
            self.status_frame,
            style="Modern.Horizontal.TProgressbar",
            length=300,
            mode='determinate'
        )
    
    def _create_monitoring_panel(self):
        """Create real-time SwarmAgentic visualization panel"""
        self.monitoring_frame = tk.LabelFrame(
            self.main_frame,
            text="SwarmAgentic Real-Time Visualization",
            font=ModernStyle.HEADER_FONT,
            bg=ModernStyle.BACKGROUND_COLOR,
            fg=ModernStyle.PRIMARY_COLOR,
            padx=ModernStyle.PADDING,
            pady=ModernStyle.PADDING
        )

        # Create swarm visualization
        self.swarm_viz = SwarmVisualization(self.monitoring_frame)

        # Create additional metrics subplot
        self.metrics_subplot_frame = tk.Frame(self.monitoring_frame, bg=ModernStyle.BACKGROUND_COLOR)

        # Create small metrics figure
        self.metrics_fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(6, 3))
        self.metrics_fig.patch.set_facecolor(ModernStyle.BACKGROUND_COLOR)

        # Accuracy plot
        self.ax1.set_title("Accuracy", fontsize=9, color=ModernStyle.TEXT_COLOR)
        self.ax1.set_ylabel("Accuracy %", fontsize=8, color=ModernStyle.TEXT_COLOR)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_facecolor('white')

        # Confidence plot
        self.ax2.set_title("Confidence", fontsize=9, color=ModernStyle.TEXT_COLOR)
        self.ax2.set_ylabel("Confidence", fontsize=8, color=ModernStyle.TEXT_COLOR)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_facecolor('white')

        # Embed metrics plot
        self.metrics_canvas = FigureCanvasTkAgg(self.metrics_fig, self.metrics_subplot_frame)
        self.metrics_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _create_scratchpad_panel(self):
        """Create scratchpad and thought process panel"""
        self.scratchpad_frame = tk.LabelFrame(
            self.main_frame,
            text="Agent Scratchpad & Thought Process",
            font=ModernStyle.HEADER_FONT,
            bg=ModernStyle.BACKGROUND_COLOR,
            fg=ModernStyle.PRIMARY_COLOR,
            padx=ModernStyle.PADDING,
            pady=ModernStyle.PADDING
        )
        
        # Scratchpad text area
        self.scratchpad_text = scrolledtext.ScrolledText(
            self.scratchpad_frame,
            height=15,
            width=60,
            font=ModernStyle.MONO_FONT,
            bg="white",
            fg=ModernStyle.TEXT_COLOR,
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        
        # Working memory display
        self.memory_label = tk.Label(
            self.scratchpad_frame,
            text="Working Memory:",
            font=ModernStyle.BODY_FONT,
            bg=ModernStyle.BACKGROUND_COLOR,
            fg=ModernStyle.TEXT_COLOR
        )
        
        self.memory_text = scrolledtext.ScrolledText(
            self.scratchpad_frame,
            height=8,
            width=60,
            font=ModernStyle.MONO_FONT,
            bg="#F8F9FA",
            fg=ModernStyle.TEXT_COLOR,
            wrap=tk.WORD,
            state=tk.DISABLED
        )
    
    def _create_metrics_panel(self):
        """Create performance metrics panel"""
        self.metrics_frame = tk.LabelFrame(
            self.main_frame,
            text="Performance Metrics",
            font=ModernStyle.HEADER_FONT,
            bg=ModernStyle.BACKGROUND_COLOR,
            fg=ModernStyle.PRIMARY_COLOR,
            padx=ModernStyle.PADDING,
            pady=ModernStyle.PADDING
        )
        
        # Metrics display
        self.accuracy_label = tk.Label(
            self.metrics_frame,
            text="Output Accuracy: 0.0%",
            font=ModernStyle.BODY_FONT,
            bg=ModernStyle.BACKGROUND_COLOR,
            fg=ModernStyle.TEXT_COLOR
        )
        
        self.confidence_label = tk.Label(
            self.metrics_frame,
            text="Average Confidence: 0.0",
            font=ModernStyle.BODY_FONT,
            bg=ModernStyle.BACKGROUND_COLOR,
            fg=ModernStyle.TEXT_COLOR
        )
        
        self.concepts_label = tk.Label(
            self.metrics_frame,
            text="Concepts Learned: 0",
            font=ModernStyle.BODY_FONT,
            bg=ModernStyle.BACKGROUND_COLOR,
            fg=ModernStyle.TEXT_COLOR
        )
        
        self.processing_time_label = tk.Label(
            self.metrics_frame,
            text="Avg Processing Time: 0.0s",
            font=ModernStyle.BODY_FONT,
            bg=ModernStyle.BACKGROUND_COLOR,
            fg=ModernStyle.TEXT_COLOR
        )
        
        # Source quality indicators
        self.source_quality_label = tk.Label(
            self.metrics_frame,
            text="Information Source Quality",
            font=ModernStyle.HEADER_FONT,
            bg=ModernStyle.BACKGROUND_COLOR,
            fg=ModernStyle.PRIMARY_COLOR
        )
        
        self.source_reliability_label = tk.Label(
            self.metrics_frame,
            text="Source Reliability: 95.2%",
            font=ModernStyle.BODY_FONT,
            bg=ModernStyle.BACKGROUND_COLOR,
            fg=ModernStyle.TEXT_COLOR
        )
        
        self.info_freshness_label = tk.Label(
            self.metrics_frame,
            text="Information Freshness: High",
            font=ModernStyle.BODY_FONT,
            bg=ModernStyle.BACKGROUND_COLOR,
            fg=ModernStyle.TEXT_COLOR
        )
    
    def _setup_layout(self):
        """Setup the layout of all widgets"""
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        self.title_label.pack(pady=(0, 20))
        
        # Top row: Control and Status
        top_frame = tk.Frame(self.main_frame, bg=ModernStyle.BACKGROUND_COLOR)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.status_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Control buttons layout
        self.start_training_btn.grid(row=0, column=0, padx=5, pady=5)
        self.start_demo_btn.grid(row=0, column=1, padx=5, pady=5)
        self.pause_btn.grid(row=1, column=0, padx=5, pady=5)
        self.stop_btn.grid(row=1, column=1, padx=5, pady=5)
        self.reset_btn.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        
        # Status layout
        self.status_text.pack(anchor=tk.W, pady=(0, 10))
        self.training_progress_label.pack(anchor=tk.W)
        self.training_progress.pack(fill=tk.X, pady=(0, 10))
        self.demo_progress_label.pack(anchor=tk.W)
        self.demo_progress.pack(fill=tk.X)
        
        # Middle row: Monitoring and Scratchpad
        middle_frame = tk.Frame(self.main_frame, bg=ModernStyle.BACKGROUND_COLOR)
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.monitoring_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.scratchpad_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Monitoring layout - pack the metrics subplot at bottom
        self.metrics_subplot_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
        
        # Scratchpad layout
        self.scratchpad_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.memory_label.pack(anchor=tk.W)
        self.memory_text.pack(fill=tk.BOTH, expand=True)
        
        # Bottom row: Metrics
        self.metrics_frame.pack(fill=tk.X)
        
        # Metrics layout
        metrics_left = tk.Frame(self.metrics_frame, bg=ModernStyle.BACKGROUND_COLOR)
        metrics_right = tk.Frame(self.metrics_frame, bg=ModernStyle.BACKGROUND_COLOR)
        
        metrics_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        metrics_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.accuracy_label.pack(anchor=tk.W, in_=metrics_left)
        self.confidence_label.pack(anchor=tk.W, in_=metrics_left)
        self.concepts_label.pack(anchor=tk.W, in_=metrics_left)
        self.processing_time_label.pack(anchor=tk.W, in_=metrics_left)
        
        self.source_quality_label.pack(anchor=tk.W, in_=metrics_right)
        self.source_reliability_label.pack(anchor=tk.W, in_=metrics_right)
        self.info_freshness_label.pack(anchor=tk.W, in_=metrics_right)
    
    def set_agent(self, agent: PhDLevelAIAgent):
        """Set the AI agent instance"""
        self.agent = agent
        
        # Register callbacks for real-time updates
        self.agent.register_callback('state_change', self._on_state_change)
        self.agent.register_callback('progress', self._on_progress_update)
        self.agent.register_callback('thought', self._on_thought_update)
    
    def _start_training(self):
        """Start training the agent"""
        if self.agent:
            self.is_training_enabled = False
            self.start_training_btn.configure(state="disabled")

            # Update visualization for training mode
            if self.swarm_viz:
                self.swarm_viz.update_from_agent_state({'is_training': True})

            # Run training in separate thread
            def run_training():
                if self.agent:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    success = loop.run_until_complete(self.agent.start_training())
                    if success:
                        self.root.after(0, self._enable_demo)
                    self.root.after(0, self._reset_training_button)

            threading.Thread(target=run_training, daemon=True).start()
    
    def _start_demo(self):
        """Start demonstration"""
        if self.agent:
            self.is_demo_enabled = False
            self.start_demo_btn.configure(state="disabled")

            # Update visualization for demo mode
            if self.swarm_viz:
                self.swarm_viz.update_from_agent_state({'is_demonstrating': True})

            # Run demo in separate thread
            def run_demo():
                if self.agent:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self.agent.start_demonstration())
                    self.root.after(0, self._reset_demo_button)

            threading.Thread(target=run_demo, daemon=True).start()
    
    def _pause_operation(self):
        """Pause current operation"""
        if self.agent:
            self.agent.pause()
    
    def _stop_operation(self):
        """Stop current operation"""
        if self.agent:
            self.agent.stop()
    
    def _reset_agent(self):
        """Reset the agent"""
        if self.agent:
            self.agent.reset()
            self.is_demo_enabled = False
            self.start_demo_btn.configure(state="disabled")
            self._clear_displays()
    
    def _enable_demo(self):
        """Enable demo button after successful training"""
        self.is_demo_enabled = True
        self.start_demo_btn.configure(state="normal")
    
    def _reset_training_button(self):
        """Reset training button state"""
        self.is_training_enabled = True
        self.start_training_btn.configure(state="normal")
    
    def _reset_demo_button(self):
        """Reset demo button state"""
        if self.is_demo_enabled:
            self.start_demo_btn.configure(state="normal")
    
    def _clear_displays(self):
        """Clear all display areas"""
        self.scratchpad_text.configure(state=tk.NORMAL)
        self.scratchpad_text.delete(1.0, tk.END)
        self.scratchpad_text.configure(state=tk.DISABLED)
        
        self.memory_text.configure(state=tk.NORMAL)
        self.memory_text.delete(1.0, tk.END)
        self.memory_text.configure(state=tk.DISABLED)
        
        self.accuracy_data.clear()
        self.confidence_data.clear()
        self.time_data.clear()
        self._update_plots()
    
    def _on_state_change(self, state_data: Dict[str, Any]):
        """Handle agent state changes"""
        def update_ui():
            status = state_data.get('current_task', 'idle')
            self.status_text.configure(text=f"Status: {status.replace('_', ' ').title()}")
            
            if 'training_progress' in state_data:
                self.training_progress['value'] = state_data['training_progress']
            
            if 'demonstration_progress' in state_data:
                self.demo_progress['value'] = state_data['demonstration_progress']
        
        self.root.after(0, update_ui)
    
    def _on_progress_update(self, progress_data: Dict[str, Any]):
        """Handle progress updates"""
        def update_ui():
            progress_type = progress_data.get('type', '')
            progress_value = progress_data.get('progress', 0)
            
            if progress_type == 'training':
                self.training_progress['value'] = progress_value
            elif progress_type == 'demonstration':
                self.demo_progress['value'] = progress_value
        
        self.root.after(0, update_ui)
    
    def _on_thought_update(self, thought_data: Dict[str, Any]):
        """Handle thought process updates"""
        def update_ui():
            timestamp = thought_data.get('timestamp', '')
            content = thought_data.get('content', '')
            thought_type = thought_data.get('type', '')
            confidence = thought_data.get('confidence', 0.0)
            
            # Add to scratchpad
            self.scratchpad_text.configure(state=tk.NORMAL)
            self.scratchpad_text.insert(tk.END, 
                f"[{timestamp[-8:]}] {thought_type.upper()}: {content}\n")
            self.scratchpad_text.see(tk.END)
            self.scratchpad_text.configure(state=tk.DISABLED)
        
        self.root.after(0, update_ui)
    
    def _start_update_thread(self):
        """Start the real-time update thread"""
        def update_loop():
            while not self.stop_updates:
                if self.agent:
                    self._update_metrics()
                    self._update_working_memory()
                    self._update_plots()
                time.sleep(1.0)
        
        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()
    
    def _update_metrics(self):
        """Update performance metrics display"""
        if not self.agent:
            return
        
        metrics = self.agent.get_current_metrics()
        
        def update_ui():
            bert_metrics = metrics.get('bert_metrics', {})
            accuracy = bert_metrics.get('accuracy_percentage', 0.0)
            avg_confidence = metrics.get('average_confidence', 0.0)
            concepts_learned = metrics.get('concepts_learned', 0)
            avg_time = bert_metrics.get('average_processing_time', 0.0)
            
            self.accuracy_label.configure(text=f"Output Accuracy: {accuracy:.1f}%")
            self.confidence_label.configure(text=f"Average Confidence: {avg_confidence:.2f}")
            self.concepts_label.configure(text=f"Concepts Learned: {concepts_learned}")
            self.processing_time_label.configure(text=f"Avg Processing Time: {avg_time:.2f}s")
            
            # Update data for plots
            current_time = len(self.time_data)
            self.time_data.append(current_time)
            self.accuracy_data.append(accuracy)
            self.confidence_data.append(avg_confidence * 100)
            
            # Keep only last 50 data points
            if len(self.time_data) > 50:
                self.time_data = self.time_data[-50:]
                self.accuracy_data = self.accuracy_data[-50:]
                self.confidence_data = self.confidence_data[-50:]
        
        self.root.after(0, update_ui)
    
    def _update_working_memory(self):
        """Update working memory display"""
        if not self.agent:
            return
        
        memory = self.agent.get_working_memory()
        
        def update_ui():
            self.memory_text.configure(state=tk.NORMAL)
            self.memory_text.delete(1.0, tk.END)
            
            memory_str = json.dumps(memory, indent=2, default=str)
            self.memory_text.insert(1.0, memory_str)
            self.memory_text.configure(state=tk.DISABLED)
        
        self.root.after(0, update_ui)
    
    def _update_plots(self):
        """Update real-time plots"""
        def update_ui():
            if len(self.time_data) > 1:
                # Clear and update accuracy plot
                self.ax1.clear()
                self.ax1.plot(self.time_data, self.accuracy_data, 
                             color=ModernStyle.SUCCESS_COLOR, linewidth=2)
                self.ax1.set_title("Output Accuracy (%)", fontsize=10, color=ModernStyle.TEXT_COLOR)
                self.ax1.set_ylabel("Accuracy", fontsize=9, color=ModernStyle.TEXT_COLOR)
                self.ax1.grid(True, alpha=0.3)
                self.ax1.set_facecolor('white')
                
                # Clear and update confidence plot
                self.ax2.clear()
                self.ax2.plot(self.time_data, self.confidence_data, 
                             color=ModernStyle.SECONDARY_COLOR, linewidth=2)
                self.ax2.set_title("Confidence Scores", fontsize=10, color=ModernStyle.TEXT_COLOR)
                self.ax2.set_ylabel("Confidence", fontsize=9, color=ModernStyle.TEXT_COLOR)
                self.ax2.set_xlabel("Time", fontsize=9, color=ModernStyle.TEXT_COLOR)
                self.ax2.grid(True, alpha=0.3)
                self.ax2.set_facecolor('white')
                
                self.metrics_canvas.draw()
        
        self.root.after(0, update_ui)
    
    def run(self):
        """Start the GUI application"""
        try:
            self.root.mainloop()
        finally:
            self.stop_updates = True
            if self.update_thread and self.update_thread.is_alive():
                self.update_thread.join(timeout=1.0)
