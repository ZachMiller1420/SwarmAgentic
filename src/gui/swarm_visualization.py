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
        # swarm_formation, pso_optimization, agent_collaboration, emergent_behavior, text_pso, team_builder
        # Default to team_builder so the central chart starts focused on teams
        self.current_mode = "team_builder"

        # Text PSO population state (for real agent-spec PSO)
        self.text_pso_population = []  # current population (target state)
        self.text_pso_gbest = {"x": 0.5, "y": 0.5, "fitness": 0.0}
        # Previous state for tweening
        self.text_pso_prev_population = []
        self.text_pso_prev_gbest = {"x": 0.5, "y": 0.5, "fitness": 0.0}
        # Rendered (interpolated) state
        self.text_pso_render_population = []
        self.text_pso_render_gbest = {"x": 0.5, "y": 0.5, "fitness": 0.0}
        # Tween control
        self.text_pso_tween_steps = 0
        self.text_pso_tween_max = 20
        # Global best trail (for visualizing improvement path)
        self.text_pso_gbest_trail = []  # list of (x, y)
        self.text_pso_gbest_trail_max = 30

        # Team Builder state (clusters of role-colored particles per team)
        self.team_builder_teams: List[Dict[str, Any]] = []  # each: roles[], center_x, center_y, fitness
        self.team_builder_gbest: Dict[str, Any] = {"center_x": 0.5, "center_y": 0.5, "fitness": 0.0, "roles": []}
        # Team Builder tweening state
        self.team_builder_prev_teams: List[Dict[str, Any]] = []
        self.team_builder_render_teams: List[Dict[str, Any]] = []
        self.team_builder_tween_steps: int = 0
        self.team_builder_tween_max: int = 20
        
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
                                 values=["team_builder"],
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
        # Keep a handle to a single colorbar to avoid stacking multiples
        self._text_pso_colorbar = None
        
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
            # Team Builder (role colors)
            plt.Line2D([0], [0], marker='o', color='w', markeredgecolor='black', markerfacecolor='royalblue', markersize=9, label='Coordinator'),
            plt.Line2D([0], [0], marker='o', color='w', markeredgecolor='black', markerfacecolor='darkorange', markersize=9, label='Planner'),
            plt.Line2D([0], [0], marker='o', color='w', markeredgecolor='black', markerfacecolor='seagreen', markersize=9, label='Researcher'),
            plt.Line2D([0], [0], marker='o', color='w', markeredgecolor='black', markerfacecolor='mediumpurple', markersize=9, label='Executor'),
            plt.Line2D([0], [0], marker='o', color='w', markeredgecolor='black', markerfacecolor='crimson', markersize=9, label='Verifier'),
            plt.Line2D([0], [0], marker='o', color='w', markeredgecolor='black', markerfacecolor='saddlebrown', markersize=9, label='Critic'),
            plt.Line2D([0], [0], marker='o', color='gold', markerfacecolor='none', markersize=10, label='Best Team (gold circle)'),
            # Other modes
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='purple', markersize=8, label='PSO Particle (numeric)'),
        ]
        self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
    def _initialize_agents(self):
        """Initialize the swarm agents with different roles"""
        self.agents.clear()
        # Start empty for team/text PSO; only spawn agents for other demos
        if self.current_mode in ("team_builder", "text_pso"):
            return
        
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
        # Start empty for team/text PSO; only spawn numeric particles for numeric PSO demo
        if self.current_mode in ("team_builder", "text_pso"):
            return
        
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
        elif self.current_mode == "text_pso":
            # Animate interpolation between last and current population states
            if self.text_pso_tween_steps > 0 and self.text_pso_population:
                t = self.text_pso_tween_steps / max(1, self.text_pso_tween_max)
                alpha = 1.0 - t
                # Interpolate population by index
                render = []
                n = min(len(self.text_pso_population), len(self.text_pso_prev_population) or len(self.text_pso_population))
                for i in range(n):
                    cur = self.text_pso_population[i]
                    prev = self.text_pso_prev_population[i] if i < len(self.text_pso_prev_population) else cur
                    render.append({
                        'x': float(prev.get('x',0))*(1-alpha) + float(cur.get('x',0))*alpha,
                        'y': float(prev.get('y',0))*(1-alpha) + float(cur.get('y',0))*alpha,
                        'fitness': float(prev.get('fitness',0))*(1-alpha) + float(cur.get('fitness',0))*alpha,
                        'size': float(prev.get('size',1))*(1-alpha) + float(cur.get('size',1))*alpha,
                    })
                # If sizes differ, append remaining as current
                for j in range(n, len(self.text_pso_population)):
                    render.append(self.text_pso_population[j])
                self.text_pso_render_population = render
                # Interpolate gbest
                pg = self.text_pso_prev_gbest or {"x":0.5,"y":0.5,"fitness":0.0}
                cg = self.text_pso_gbest or {"x":0.5,"y":0.5,"fitness":0.0}
                self.text_pso_render_gbest = {
                    'x': float(pg.get('x',0))*(1-alpha) + float(cg.get('x',0))*alpha,
                    'y': float(pg.get('y',0))*(1-alpha) + float(cg.get('y',0))*alpha,
                    'fitness': float(pg.get('fitness',0))*(1-alpha) + float(cg.get('fitness',0))*alpha,
                }
                self.text_pso_tween_steps -= 1
            else:
                # Use current state as render state
                self.text_pso_render_population = list(self.text_pso_population)
                self.text_pso_render_gbest = dict(self.text_pso_gbest)
        elif self.current_mode == "team_builder":
            # Animate team cluster centers and growth
            if self.team_builder_tween_steps > 0 and self.team_builder_teams:
                t = self.team_builder_tween_steps / max(1, self.team_builder_tween_max)
                alpha = 1.0 - t
                render = []
                n = max(len(self.team_builder_teams), len(self.team_builder_prev_teams))
                for i in range(n):
                    cur = self.team_builder_teams[i] if i < len(self.team_builder_teams) else (self.team_builder_prev_teams[-1] if self.team_builder_prev_teams else None)
                    prev = self.team_builder_prev_teams[i] if i < len(self.team_builder_prev_teams) else cur
                    if not cur:
                        continue
                    px = float(prev.get('center_x', cur.get('center_x', 0.5))) if prev else float(cur.get('center_x', 0.5))
                    py = float(prev.get('center_y', cur.get('center_y', 0.5))) if prev else float(cur.get('center_y', 0.5))
                    cx = float(cur.get('center_x', 0.5))
                    cy = float(cur.get('center_y', 0.5))
                    rx = px * (1 - alpha) + cx * alpha
                    ry = py * (1 - alpha) + cy * alpha
                    fit = float(cur.get('fitness', 0.0))
                    roles = cur.get('roles', [])
                    render.append({'center_x': rx, 'center_y': ry, 'fitness': fit, 'roles': roles, 'grow': alpha})
                self.team_builder_render_teams = render
                self.team_builder_tween_steps -= 1
            else:
                self.team_builder_render_teams = list(self.team_builder_teams)
            
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
        self.ax.set_title("SwarmAgentic: Team Builder (Role-Colored PSO)", 
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

        # Draw population for text-based PSO mode
        if self.current_mode == "text_pso" and (self.text_pso_render_population or self.text_pso_population):
            pop = self.text_pso_render_population or self.text_pso_population
            xs = [p.get("x",0.0) for p in pop]
            ys = [p.get("y",0.0) for p in pop]
            sizes = [max(40, 30 + 20 * p.get("size", 1)) for p in pop]
            colors = [p.get("fitness", 0.0) for p in pop]

            # Auto-zoom to population bounding box with padding
            if xs and ys:
                pad = 0.08
                xmin, xmax = max(0.0, min(xs) - pad), min(1.0, max(xs) + pad)
                ymin, ymax = max(0.0, min(ys) - pad), min(1.0, max(ys) + pad)
                if abs(xmax - xmin) < 0.2:
                    cx = 0.5 * (xmin + xmax)
                    xmin, xmax = max(0.0, cx - 0.1), min(1.0, cx + 0.1)
                if abs(ymax - ymin) < 0.2:
                    cy = 0.5 * (ymin + ymax)
                    ymin, ymax = max(0.0, cy - 0.1), min(1.0, cy + 0.1)
                self.ax.set_xlim(xmin, xmax)
                self.ax.set_ylim(ymin, ymax)

            scatter = self.ax.scatter(xs, ys, s=sizes, c=colors, cmap='viridis', vmin=0.0, vmax=1.0, alpha=0.8)
            # Colorbar: create once and then update to prevent layout shrink
            if self._text_pso_colorbar is None:
                self._text_pso_colorbar = self.fig.colorbar(scatter, ax=self.ax, fraction=0.046, pad=0.04, label='Fitness')
            else:
                try:
                    self._text_pso_colorbar.update_normal(scatter)
                except Exception:
                    pass
            # Delta arrows from previous -> current
            if self.text_pso_prev_population and self.text_pso_population:
                n = min(len(self.text_pso_prev_population), len(self.text_pso_population))
                for i in range(n):
                    prev = self.text_pso_prev_population[i]
                    cur = self.text_pso_population[i]
                    px, py = float(prev.get('x', 0.0)), float(prev.get('y', 0.0))
                    cx, cy = float(cur.get('x', 0.0)), float(cur.get('y', 0.0))
                    dx, dy = (cx - px), (cy - py)
                    if abs(dx) + abs(dy) > 1e-4:
                        try:
                            self.ax.arrow(px, py, dx, dy, head_width=0.01, head_length=0.02, fc='gray', ec='gray', alpha=0.35, length_includes_head=True)
                        except Exception:
                            self.ax.plot([px, cx], [py, cy], color='gray', alpha=0.35, linewidth=1)
            # Draw gbest
            gx = (self.text_pso_render_gbest or self.text_pso_gbest).get("x", 0.5)
            gy = (self.text_pso_render_gbest or self.text_pso_gbest).get("y", 0.5)
            self.ax.scatter(gx, gy,
                            s=220, c='gold', marker='*', edgecolors='black', linewidth=2)
            # Annotate best fitness
            bf = (self.text_pso_render_gbest or self.text_pso_gbest).get("fitness", 0.0)
            self.ax.text(0.02, 0.98, f"Best fitness: {bf:.2f}", transform=self.ax.transAxes,
                         fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Draw Team Builder mode: clusters per team, role-colored particles
        if self.current_mode == "team_builder" and (self.team_builder_render_teams or self.team_builder_teams):
            # Auto-zoom around teams
            teams_to_draw = self.team_builder_render_teams or self.team_builder_teams
            xs = [float(t.get('center_x', 0.5)) for t in teams_to_draw]
            ys = [float(t.get('center_y', 0.5)) for t in teams_to_draw]
            if xs and ys:
                pad = 0.08
                xmin, xmax = max(0.0, min(xs) - pad), min(1.0, max(xs) + pad)
                ymin, ymax = max(0.0, min(ys) - pad), min(1.0, max(ys) + pad)
                if abs(xmax - xmin) < 0.2:
                    cx = 0.5 * (xmin + xmax)
                    xmin, xmax = max(0.0, cx - 0.1), min(1.0, cx + 0.1)
                if abs(ymax - ymin) < 0.2:
                    cy = 0.5 * (ymin + ymax)
                    ymin, ymax = max(0.0, cy - 0.1), min(1.0, cy + 0.1)
                self.ax.set_xlim(xmin, xmax)
                self.ax.set_ylim(ymin, ymax)

            # Role color mapping
            role_colors = {
                'coordinator': 'royalblue',
                'planner': 'darkorange',
                'researcher': 'seagreen',
                'executor': 'mediumpurple',
                'verifier': 'crimson',
                'critic': 'saddlebrown',
            }
            
            for team in teams_to_draw:
                cx = float(team.get('center_x', 0.5))
                cy = float(team.get('center_y', 0.5))
                roles = team.get('roles', []) or []
                n = max(1, len(roles))
                # Arrange role particles around the center in a small circle
                target_radius = 0.03 + 0.005 * n
                grow = float(team.get('grow', 1.0))
                radius = max(0.0, min(1.0, grow)) * target_radius
                for idx, role in enumerate(roles):
                    angle = 2 * math.pi * (idx / n)
                    rx = cx + radius * math.cos(angle)
                    ry = cy + radius * math.sin(angle)
                    color = role_colors.get(str(role).lower(), 'gray')
                    self.ax.scatter(rx, ry, s=120, c=color, alpha=0.3 + 0.6*grow, edgecolors='black', linewidth=0.5)
                # Draw a faint circle for the team boundary, colored by fitness
                fit = float(team.get('fitness', 0.0))
                circle = plt.Circle((cx, cy), radius + 0.01, color='gray', fill=False, alpha=0.2 + 0.3*fit, linewidth=2)
                self.ax.add_patch(circle)

            # Highlight the global best team cluster
            gb = self.team_builder_gbest or {}
            gx = float(gb.get('center_x', 0.5))
            gy = float(gb.get('center_y', 0.5))
            gr = 0.03 + 0.005 * max(1, int(len(gb.get('roles', [])))) + 0.02
            best_circle = plt.Circle((gx, gy), gr, color='gold', fill=False, linewidth=3)
            self.ax.add_patch(best_circle)
            bf = float(gb.get('fitness', 0.0))
            self.ax.text(gx, gy + gr + 0.02, f"Best: {bf:.2f}", ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            # Draw gbest trail
            if self.text_pso_gbest_trail:
                tx = [p[0] for p in self.text_pso_gbest_trail]
                ty = [p[1] for p in self.text_pso_gbest_trail]
                self.ax.plot(tx, ty, color='gold', alpha=0.5, linewidth=2, linestyle='--')
                self.ax.scatter(tx, ty, s=20, c='gold', alpha=0.6)
            
        # Re-add legend each frame (axes are cleared above)
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
        
        # Always prefer team_builder visualization
        if agent_state.get('is_training', False) or agent_state.get('is_demonstrating', False) or agent_state.get('text_pso_mode', False) or agent_state.get('team_builder_mode', False):
            self.current_mode = "team_builder"
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

    def update_text_pso_state(self, iteration: int, population_metrics: List[Dict[str, Any]], gbest: Dict[str, Any]):
        """Receive external updates for text-based PSO population and redraw."""
        # population_metrics items expected keys: coverage (0..1), role_count, workflow_len, fitness (0..1)
        # Preserve previous state for interpolation
        self.text_pso_prev_population = list(self.text_pso_population) if self.text_pso_population else []
        self.text_pso_population = []
        for pm in population_metrics:
            x = float(pm.get('coverage', 0.0))
            wf = int(pm.get('workflow_len', 0))
            y = min(1.0, max(0.0, wf / 10.0))
            self.text_pso_population.append({
                'x': x,
                'y': y,
                'fitness': float(pm.get('fitness', 0.0)),
                'size': int(pm.get('role_count', 1)),
            })
        # Previous gbest for interpolation
        self.text_pso_prev_gbest = dict(self.text_pso_gbest) if self.text_pso_gbest else {"x":0.5,"y":0.5,"fitness":0.0}
        self.text_pso_gbest = {
            'x': float(gbest.get('coverage', 0.0)),
            'y': min(1.0, max(0.0, int(gbest.get('workflow_len', 0)) / 10.0)),
            'fitness': float(gbest.get('fitness', 0.0)),
        }
        # Update gbest trail
        try:
            gx, gy = float(self.text_pso_gbest['x']), float(self.text_pso_gbest['y'])
            if not self.text_pso_gbest_trail or (abs(self.text_pso_gbest_trail[-1][0] - gx) + abs(self.text_pso_gbest_trail[-1][1] - gy) > 1e-6):
                self.text_pso_gbest_trail.append((gx, gy))
                if len(self.text_pso_gbest_trail) > self.text_pso_gbest_trail_max:
                    self.text_pso_gbest_trail = self.text_pso_gbest_trail[-self.text_pso_gbest_trail_max:]
        except Exception:
            pass
        # Start tween animation
        self.text_pso_tween_steps = self.text_pso_tween_max
        # Force a redraw immediately
        self._update_visualization()

    def update_team_builder_state(self, iteration: int, teams: List[Dict[str, Any]], gbest_team: Dict[str, Any]):
        """Receive updates for Team Builder visualization and redraw."""
        try:
            # Preserve previous teams for tweening
            self.team_builder_prev_teams = list(self.team_builder_teams) if self.team_builder_teams else []
            self.team_builder_teams = teams or []
            self.team_builder_gbest = gbest_team or {}
            # Start tween animation
            self.team_builder_tween_steps = self.team_builder_tween_max
            # Switch mode if not already
            if self.current_mode != 'team_builder':
                self.current_mode = 'team_builder'
                self.mode_var.set(self.current_mode)
                if not self.is_running:
                    self.start_animation()
        except Exception:
            pass
        self._update_visualization()
