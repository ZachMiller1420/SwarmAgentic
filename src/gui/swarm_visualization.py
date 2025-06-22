"""
Real-time SwarmAgentic Visualization System
Provides animated demonstrations of swarm intelligence and agent interactions
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import numpy as np
import math
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import threading

@dataclass
class Agent:
    """Represents a single agent in the swarm"""
    id: int
    x: float
    y: float
    vx: float
    vy: float
    fitness: float
    role: str
    color: str
    size: float
    active: bool = True
    
@dataclass
class Particle:
    """Represents a particle in PSO optimization"""
    id: int
    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray
    best_fitness: float
    fitness: float
    
class SwarmVisualization:
    """
    Advanced visualization system for SwarmAgentic demonstrations
    Shows real-time agent interactions, PSO optimization, and emergent behaviors
    """
    
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.is_running = False
        self.animation_speed = 50  # milliseconds
        
        # Swarm state
        self.agents: List[Agent] = []
        self.particles: List[Particle] = []
        self.global_best_position = np.array([0.5, 0.5])
        self.global_best_fitness = float('inf')
        
        # Animation parameters
        self.time_step = 0
        self.max_agents = 20
        self.max_particles = 15
        
        # Visualization modes
        self.current_mode = "swarm_formation"  # swarm_formation, pso_optimization, agent_collaboration
        
        self._setup_visualization()
        self._initialize_agents()
        self._initialize_particles()
        
    def _setup_visualization(self):
        """Setup the visualization canvas and controls"""
        
        # Main container
        self.viz_frame = tk.Frame(self.parent_frame, bg='white')
        self.viz_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control panel
        self.control_frame = tk.Frame(self.viz_frame, bg='lightgray', height=60)
        self.control_frame.pack(fill=tk.X, side=tk.TOP)
        self.control_frame.pack_propagate(False)
        
        # Visualization mode selector
        tk.Label(self.control_frame, text="Visualization Mode:", bg='lightgray').pack(side=tk.LEFT, padx=5)
        
        self.mode_var = tk.StringVar(value=self.current_mode)
        mode_combo = ttk.Combobox(self.control_frame, textvariable=self.mode_var, 
                                 values=["swarm_formation", "pso_optimization", "agent_collaboration", "emergent_behavior"],
                                 state="readonly", width=20)
        mode_combo.pack(side=tk.LEFT, padx=5)
        mode_combo.bind('<<ComboboxSelected>>', self._on_mode_change)
        
        # Control buttons
        self.start_btn = tk.Button(self.control_frame, text="Start Animation", 
                                  command=self.start_animation, bg='green', fg='white')
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(self.control_frame, text="Stop Animation", 
                                 command=self.stop_animation, bg='red', fg='white')
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.reset_btn = tk.Button(self.control_frame, text="Reset", 
                                  command=self.reset_visualization, bg='orange', fg='white')
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        
        # Speed control
        tk.Label(self.control_frame, text="Speed:", bg='lightgray').pack(side=tk.LEFT, padx=(20,5))
        self.speed_var = tk.IntVar(value=50)
        speed_scale = tk.Scale(self.control_frame, from_=10, to=200, orient=tk.HORIZONTAL,
                              variable=self.speed_var, command=self._on_speed_change, bg='lightgray')
        speed_scale.pack(side=tk.LEFT, padx=5)
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.patch.set_facecolor('white')
        
        # Setup the plot
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('SwarmAgentic Real-Time Demonstration', fontsize=14, fontweight='bold')
        
        # Embed plot in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, self.viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize plot elements
        self.agent_scatter = self.ax.scatter([], [], s=[], c=[], alpha=0.8)
        self.particle_scatter = self.ax.scatter([], [], s=[], c=[], alpha=0.6, marker='s')
        self.connection_lines = []
        self.trajectory_lines = []
        
        # Legend
        self._setup_legend()
        
    def _setup_legend(self):
        """Setup legend for different visualization elements"""
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Coordinator Agent'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Worker Agent'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Specialist Agent'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='purple', markersize=8, label='PSO Particle'),
            plt.Line2D([0], [0], color='orange', linewidth=2, label='Communication'),
            plt.Line2D([0], [0], color='cyan', linewidth=2, label='Optimization Path')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
    def _initialize_agents(self):
        """Initialize the swarm agents with different roles"""
        self.agents.clear()
        
        # Agent roles and their properties
        agent_types = [
            {"role": "coordinator", "color": "blue", "size": 150, "count": 3},
            {"role": "worker", "color": "green", "size": 100, "count": 8},
            {"role": "specialist", "color": "red", "size": 120, "count": 6},
            {"role": "optimizer", "color": "orange", "size": 80, "count": 3}
        ]
        
        agent_id = 0
        for agent_type in agent_types:
            for _ in range(agent_type["count"]):
                agent = Agent(
                    id=agent_id,
                    x=np.random.uniform(0.1, 0.9),
                    y=np.random.uniform(0.1, 0.9),
                    vx=np.random.uniform(-0.02, 0.02),
                    vy=np.random.uniform(-0.02, 0.02),
                    fitness=np.random.uniform(0.3, 1.0),
                    role=agent_type["role"],
                    color=agent_type["color"],
                    size=agent_type["size"]
                )
                self.agents.append(agent)
                agent_id += 1
                
    def _initialize_particles(self):
        """Initialize PSO particles for optimization visualization"""
        self.particles.clear()
        
        for i in range(self.max_particles):
            position = np.random.uniform(0.1, 0.9, 2)
            velocity = np.random.uniform(-0.01, 0.01, 2)
            
            particle = Particle(
                id=i,
                position=position.copy(),
                velocity=velocity.copy(),
                best_position=position.copy(),
                best_fitness=self._fitness_function(position),
                fitness=self._fitness_function(position)
            )
            self.particles.append(particle)
            
    def _fitness_function(self, position):
        """Simple fitness function for PSO demonstration"""
        # Multi-modal function with global optimum at (0.7, 0.3)
        x, y = position
        return -(np.exp(-((x-0.7)**2 + (y-0.3)**2)/0.1) + 
                0.5*np.exp(-((x-0.3)**2 + (y-0.7)**2)/0.1))
        
    def _on_mode_change(self, event=None):
        """Handle visualization mode change"""
        self.current_mode = self.mode_var.get()
        self.reset_visualization()
        
    def _on_speed_change(self, value):
        """Handle animation speed change"""
        self.animation_speed = 210 - int(value)  # Invert scale
        
    def start_animation(self):
        """Start the real-time animation"""
        if not self.is_running:
            self.is_running = True
            self.start_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            self._animation_loop()
            
    def stop_animation(self):
        """Stop the animation"""
        self.is_running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        
    def reset_visualization(self):
        """Reset the visualization to initial state"""
        self.stop_animation()
        self.time_step = 0
        self._initialize_agents()
        self._initialize_particles()
        self._update_visualization()
        
    def _animation_loop(self):
        """Main animation loop"""
        if self.is_running:
            self._update_simulation()
            self._update_visualization()
            self.time_step += 1
            
            # Schedule next frame
            self.parent_frame.after(self.animation_speed, self._animation_loop)
            
    def _update_simulation(self):
        """Update the simulation state based on current mode"""
        if self.current_mode == "swarm_formation":
            self._update_swarm_formation()
        elif self.current_mode == "pso_optimization":
            self._update_pso_optimization()
        elif self.current_mode == "agent_collaboration":
            self._update_agent_collaboration()
        elif self.current_mode == "emergent_behavior":
            self._update_emergent_behavior()
            
    def _update_swarm_formation(self):
        """Update agents for swarm formation demonstration"""
        for agent in self.agents:
            # Flocking behavior: separation, alignment, cohesion
            separation = self._calculate_separation(agent)
            alignment = self._calculate_alignment(agent)
            cohesion = self._calculate_cohesion(agent)
            
            # Update velocity
            agent.vx += 0.1 * (separation[0] + alignment[0] + cohesion[0])
            agent.vy += 0.1 * (separation[1] + alignment[1] + cohesion[1])
            
            # Limit velocity
            speed = math.sqrt(agent.vx**2 + agent.vy**2)
            if speed > 0.03:
                agent.vx = (agent.vx / speed) * 0.03
                agent.vy = (agent.vy / speed) * 0.03
                
            # Update position
            agent.x += agent.vx
            agent.y += agent.vy
            
            # Boundary conditions
            if agent.x < 0.05 or agent.x > 0.95:
                agent.vx *= -1
            if agent.y < 0.05 or agent.y > 0.95:
                agent.vy *= -1
                
            agent.x = np.clip(agent.x, 0.05, 0.95)
            agent.y = np.clip(agent.y, 0.05, 0.95)
            
    def _update_pso_optimization(self):
        """Update particles for PSO optimization demonstration"""
        # Update global best
        for particle in self.particles:
            if particle.fitness < self.global_best_fitness:
                self.global_best_fitness = particle.fitness
                self.global_best_position = particle.position.copy()
                
        # Update particles
        for particle in self.particles:
            # PSO velocity update
            w = 0.7  # inertia weight
            c1, c2 = 2.0, 2.0  # acceleration coefficients
            r1, r2 = np.random.random(2), np.random.random(2)
            
            particle.velocity = (w * particle.velocity + 
                               c1 * r1 * (particle.best_position - particle.position) +
                               c2 * r2 * (self.global_best_position - particle.position))
            
            # Update position
            particle.position += particle.velocity * 0.01
            
            # Boundary conditions
            particle.position = np.clip(particle.position, 0.05, 0.95)
            
            # Update fitness
            particle.fitness = self._fitness_function(particle.position)
            
            # Update personal best
            if particle.fitness < particle.best_fitness:
                particle.best_fitness = particle.fitness
                particle.best_position = particle.position.copy()
                
    def _update_agent_collaboration(self):
        """Update agents for collaboration demonstration"""
        # Agents form task-specific groups and collaborate
        coordinators = [a for a in self.agents if a.role == "coordinator"]
        workers = [a for a in self.agents if a.role == "worker"]
        
        # Coordinators attract workers for task assignment
        for coordinator in coordinators:
            for worker in workers:
                dx = coordinator.x - worker.x
                dy = coordinator.y - worker.y
                distance = math.sqrt(dx**2 + dy**2)
                
                if distance > 0.1:  # Attraction force
                    force = 0.001 / (distance + 0.01)
                    worker.vx += force * dx / distance
                    worker.vy += force * dy / distance
                    
        # Update positions
        for agent in self.agents:
            agent.x += agent.vx
            agent.y += agent.vy
            
            # Damping
            agent.vx *= 0.95
            agent.vy *= 0.95
            
            # Boundaries
            agent.x = np.clip(agent.x, 0.05, 0.95)
            agent.y = np.clip(agent.y, 0.05, 0.95)
            
    def _update_emergent_behavior(self):
        """Update for emergent behavior demonstration"""
        # Complex emergent patterns from simple rules
        for agent in self.agents:
            # Rule 1: Move towards center of mass
            center_x = np.mean([a.x for a in self.agents])
            center_y = np.mean([a.y for a in self.agents])
            
            dx_center = center_x - agent.x
            dy_center = center_y - agent.y
            
            # Rule 2: Avoid overcrowding
            avoid_x, avoid_y = 0, 0
            for other in self.agents:
                if other.id != agent.id:
                    dx = agent.x - other.x
                    dy = agent.y - other.y
                    distance = math.sqrt(dx**2 + dy**2)
                    if distance < 0.1 and distance > 0:
                        avoid_x += dx / distance
                        avoid_y += dy / distance
                        
            # Rule 3: Role-specific behavior
            role_factor = {"coordinator": 0.5, "worker": 1.0, "specialist": 0.8, "optimizer": 1.2}
            factor = role_factor.get(agent.role, 1.0)
            
            # Update velocity
            agent.vx += 0.001 * dx_center + 0.002 * avoid_x * factor
            agent.vy += 0.001 * dy_center + 0.002 * avoid_y * factor
            
            # Add some noise for realistic behavior
            agent.vx += np.random.normal(0, 0.0005)
            agent.vy += np.random.normal(0, 0.0005)
            
            # Update position
            agent.x += agent.vx
            agent.y += agent.vy
            
            # Boundaries and damping
            agent.vx *= 0.98
            agent.vy *= 0.98
            agent.x = np.clip(agent.x, 0.05, 0.95)
            agent.y = np.clip(agent.y, 0.05, 0.95)
            
    def _calculate_separation(self, agent):
        """Calculate separation force for flocking"""
        separation = np.array([0.0, 0.0])
        count = 0
        
        for other in self.agents:
            if other.id != agent.id:
                dx = agent.x - other.x
                dy = agent.y - other.y
                distance = math.sqrt(dx**2 + dy**2)
                
                if 0 < distance < 0.1:  # Separation radius
                    separation[0] += dx / distance
                    separation[1] += dy / distance
                    count += 1
                    
        if count > 0:
            separation /= count
            
        return separation
        
    def _calculate_alignment(self, agent):
        """Calculate alignment force for flocking"""
        alignment = np.array([0.0, 0.0])
        count = 0
        
        for other in self.agents:
            if other.id != agent.id:
                dx = agent.x - other.x
                dy = agent.y - other.y
                distance = math.sqrt(dx**2 + dy**2)
                
                if distance < 0.15:  # Alignment radius
                    alignment[0] += other.vx
                    alignment[1] += other.vy
                    count += 1
                    
        if count > 0:
            alignment /= count
            alignment[0] -= agent.vx
            alignment[1] -= agent.vy
            
        return alignment
        
    def _calculate_cohesion(self, agent):
        """Calculate cohesion force for flocking"""
        cohesion = np.array([0.0, 0.0])
        count = 0
        
        for other in self.agents:
            if other.id != agent.id:
                dx = agent.x - other.x
                dy = agent.y - other.y
                distance = math.sqrt(dx**2 + dy**2)
                
                if distance < 0.2:  # Cohesion radius
                    cohesion[0] += other.x
                    cohesion[1] += other.y
                    count += 1
                    
        if count > 0:
            cohesion /= count
            cohesion[0] -= agent.x
            cohesion[1] -= agent.y
            
        return cohesion
        
    def _update_visualization(self):
        """Update the visual representation"""
        # Clear previous elements
        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        # Update title based on mode
        titles = {
            "swarm_formation": "SwarmAgentic: Agent Formation & Flocking Behavior",
            "pso_optimization": "SwarmAgentic: PSO-Based Optimization Process",
            "agent_collaboration": "SwarmAgentic: Multi-Agent Collaboration",
            "emergent_behavior": "SwarmAgentic: Emergent Collective Intelligence"
        }
        self.ax.set_title(titles.get(self.current_mode, "SwarmAgentic Demonstration"), 
                         fontsize=14, fontweight='bold')
        
        # Draw agents
        if self.agents:
            x_coords = [agent.x for agent in self.agents]
            y_coords = [agent.y for agent in self.agents]
            sizes = [agent.size for agent in self.agents]
            colors = [agent.color for agent in self.agents]
            
            self.ax.scatter(x_coords, y_coords, s=sizes, c=colors, alpha=0.8, edgecolors='black')
            
            # Draw connections for collaboration mode
            if self.current_mode == "agent_collaboration":
                self._draw_collaboration_connections()
                
        # Draw particles for PSO mode
        if self.current_mode == "pso_optimization" and self.particles:
            x_coords = [p.position[0] for p in self.particles]
            y_coords = [p.position[1] for p in self.particles]
            sizes = [60] * len(self.particles)
            colors = ['purple'] * len(self.particles)
            
            self.ax.scatter(x_coords, y_coords, s=sizes, c=colors, alpha=0.6, marker='s')
            
            # Draw global best
            self.ax.scatter(self.global_best_position[0], self.global_best_position[1], 
                          s=200, c='gold', marker='*', edgecolors='black', linewidth=2)
            
        # Update legend
        self._setup_legend()
        
        # Refresh canvas
        self.canvas.draw()
        
    def _draw_collaboration_connections(self):
        """Draw connections between collaborating agents"""
        coordinators = [a for a in self.agents if a.role == "coordinator"]
        workers = [a for a in self.agents if a.role == "worker"]
        
        for coordinator in coordinators:
            for worker in workers:
                distance = math.sqrt((coordinator.x - worker.x)**2 + (coordinator.y - worker.y)**2)
                if distance < 0.2:  # Connection threshold
                    self.ax.plot([coordinator.x, worker.x], [coordinator.y, worker.y], 
                               'orange', alpha=0.5, linewidth=1)
                    
    def update_from_agent_state(self, agent_state: Dict[str, Any]):
        """Update visualization based on AI agent state"""
        # This method can be called from the main application to sync
        # the visualization with the actual AI agent's state
        
        if agent_state.get('is_training', False):
            self.current_mode = "pso_optimization"
            self.mode_var.set(self.current_mode)
            if not self.is_running:
                self.start_animation()
                
        elif agent_state.get('is_demonstrating', False):
            self.current_mode = "agent_collaboration"
            self.mode_var.set(self.current_mode)
            if not self.is_running:
                self.start_animation()
                
    def get_current_state(self) -> Dict[str, Any]:
        """Get current visualization state"""
        return {
            'mode': self.current_mode,
            'is_running': self.is_running,
            'time_step': self.time_step,
            'agent_count': len(self.agents),
            'particle_count': len(self.particles)
        }
