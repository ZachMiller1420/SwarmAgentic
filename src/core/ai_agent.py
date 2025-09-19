"""
PhD-Level AI Agent with Advanced Reasoning Capabilities
Implements sophisticated cognitive architecture with real-time chain-of-thought processing
"""

import asyncio
import threading
import time
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import numpy as np
import os

from .bert_engine import BERTReasoningEngine, ReasoningStep
from ..training.learning_system import AdaptiveLearningSystem
from ..synthesis.pso_text import PSOSwarmSynthesizer

@dataclass
class AgentState:
    """Represents the current state of the AI agent"""
    is_training: bool = False
    is_demonstrating: bool = False
    is_paused: bool = False
    training_progress: float = 0.0
    demonstration_progress: float = 0.0
    current_task: str = "idle"
    last_update: datetime = None
    performance_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.last_update is None:
            self.last_update = datetime.now()
        if self.performance_metrics is None:
            self.performance_metrics = {}

@dataclass
class LearningProgress:
    """Tracks learning progress during training"""
    concepts_learned: int = 0
    total_concepts: int = 0
    understanding_depth: float = 0.0
    retention_score: float = 0.0
    adaptation_rate: float = 0.0
    learning_efficiency: float = 0.0

class PhDLevelAIAgent:
    """
    Advanced AI Agent with PhD-level reasoning capabilities
    Integrates BERT-based language processing with sophisticated cognitive architecture
    """
    
    def __init__(self, bert_model_path: str, academic_paper_path: str):
        self.bert_engine = BERTReasoningEngine(bert_model_path)
        self.academic_paper_path = Path(academic_paper_path)
        # Optional override: allow custom training text without changing call sites
        self.custom_training_text: Optional[str] = None
        
        # Agent state management
        self.state = AgentState()
        self.learning_progress = LearningProgress()
        
        # Knowledge base
        self.knowledge_base: Dict[str, Any] = {}
        self.learned_concepts: List[str] = []
        # A task-focused view of concepts to surface in the UI (optional)
        self.displayed_concepts: List[str] = []
        self.concept_relationships: Dict[str, List[str]] = {}
        
        # Real-time monitoring
        self.scratchpad: List[str] = []
        self.thought_process: List[Dict[str, Any]] = []
        self.working_memory: Dict[str, Any] = {}
        
        # Event callbacks for GUI updates
        self.state_change_callbacks: List[Callable] = []
        self.progress_callbacks: List[Callable] = []
        self.thought_callbacks: List[Callable] = []
        
        # Performance tracking
        self.accuracy_history: List[float] = []
        self.response_times: List[float] = []
        self.confidence_scores: List[float] = []

        # Training configuration
        self.training_config = {
            'learning_rate': 0.01,
            'batch_size': 32,
            'max_iterations': 1000,
            'convergence_threshold': 0.95,
            'validation_frequency': 50
        }
        
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

        # PSO / demo configuration
        self.pso_tasks: Optional[List[str]] = None
        self.training_corpus_cache_path: Optional[Path] = None
        self.knowledge_graph_export: Optional[Dict[str, Any]] = None
        self.recommended_concepts: List[str] = []
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def register_callback(self, callback_type: str, callback: Callable):
        """Register callbacks for real-time updates"""
        if callback_type == 'state_change':
            self.state_change_callbacks.append(callback)
        elif callback_type == 'progress':
            self.progress_callbacks.append(callback)
        elif callback_type == 'thought':
            self.thought_callbacks.append(callback)
    
    def _notify_callbacks(self, callback_type: str, data: Any):
        """Notify registered callbacks"""
        callbacks = {
            'state_change': self.state_change_callbacks,
            'progress': self.progress_callbacks,
            'thought': self.thought_callbacks
        }.get(callback_type, [])
        
        for callback in callbacks:
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")
    
    def add_thought(self, thought: str, thought_type: str = "reasoning"):
        """Add a thought to the scratchpad and notify observers"""
        timestamp = datetime.now().isoformat()
        thought_entry = {
            'timestamp': timestamp,
            'type': thought_type,
            'content': thought,
            'confidence': np.random.uniform(0.7, 0.95)  # Simplified
        }
        
        self.scratchpad.append(thought)
        self.thought_process.append(thought_entry)
        
        # Keep scratchpad manageable
        if len(self.scratchpad) > 100:
            self.scratchpad = self.scratchpad[-50:]
        
        self._notify_callbacks('thought', thought_entry)
        self.logger.info(f"Thought added: {thought}")
    
    def update_working_memory(self, key: str, value: Any):
        """Update working memory and notify observers"""
        self.working_memory[key] = value
        self.add_thought(f"Updated working memory: {key}", "memory_update")
    
    async def start_training(self) -> bool:
        """Start training on the academic paper"""
        if self.state.is_training or self.state.is_demonstrating:
            self.logger.warning("Agent is already busy")
            return False
        
        self.logger.info("Starting training coroutine")
        self.state.is_training = True
        self.state.current_task = "training"
        self.state.training_progress = 0.0
        self._notify_callbacks('state_change', asdict(self.state))
        
        try:
            await self._train_on_academic_paper()
            self.state.is_training = False
            self.state.current_task = "training_complete"
            self.state.training_progress = 100.0
            self._notify_callbacks('state_change', asdict(self.state))
            return True
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            self.state.is_training = False
            self.state.current_task = "training_failed"
            self._notify_callbacks('state_change', asdict(self.state))
            return False
    
    async def _train_on_academic_paper(self):
        """Train the agent on the academic paper content"""
        self.add_thought("Beginning training on SwarmAgentic academic paper", "training_start")
        
        # Load and parse the academic paper
        paper_content = self._load_academic_paper()
        self.add_thought(f"Loaded paper with {len(paper_content)} characters", "data_loading")
        
        # Extract key concepts and sections
        concepts = self._extract_concepts(paper_content)
        self.learning_progress.total_concepts = len(concepts)
        self.add_thought(f"Identified {len(concepts)} key concepts to learn", "concept_extraction")
        
        # Progressive learning simulation
        for i, concept in enumerate(concepts):
            if self.state.is_paused:
                await self._wait_for_resume()
            
            await self._learn_concept(concept, paper_content)
            
            self.learning_progress.concepts_learned = i + 1
            progress = (i + 1) / len(concepts) * 100
            self.state.training_progress = progress
            
            self._notify_callbacks('progress', {
                'type': 'training',
                'progress': progress,
                'current_concept': concept,
                'learning_metrics': asdict(self.learning_progress)
            })
            
            # Simulate processing time
            await asyncio.sleep(0.5)
        
        # Final validation
        await self._validate_learning()
        self.add_thought("Training completed successfully", "training_complete")

        # Advanced: build knowledge graph and derive recommended PSO tasks
        try:
            self.add_thought("Building knowledge graph from corpus", "training_graph")
            als = AdaptiveLearningSystem(self.bert_engine)
            # Process full content to extract/validate concepts
            async def _run_als():
                return await als.process_academic_content(paper_content)
            # Run in same event loop
            _ = await _run_als()
            # Export and store
            self.knowledge_graph_export = als.export_knowledge_graph()
            # Take recommendations (concept names) as candidate tasks
            try:
                recs = als.get_learning_statistics().get('learning_recommendations', [])
                self.recommended_concepts = list(recs)
            except Exception:
                self.recommended_concepts = []
            # Seed PSO tasks from recommendations if available
            if self.recommended_concepts:
                tasks = [
                    f"plan tasks, execute plan, verify results for {self.recommended_concepts[0]}",
                ]
                if len(self.recommended_concepts) > 1:
                    tasks.append(f"optimize workflow to address {self.recommended_concepts[1]}")
                if len(self.recommended_concepts) > 2:
                    tasks.append(f"coverage of {self.recommended_concepts[2]} with verification")
                self.set_pso_tasks(tasks)
                self.add_thought(f"Derived PSO tasks from training: {', '.join(self.recommended_concepts[:3])}", "training_graph")
        except Exception as e:
            self.add_thought(f"Knowledge graph step skipped: {e}", "training_graph_error")
    
    async def _learn_concept(self, concept: str, context: str):
        """Learn a specific concept using BERT reasoning"""
        self.add_thought(f"Learning concept: {concept}", "concept_learning")
        
        # Use BERT engine for deep understanding
        reasoning_steps = self.bert_engine.reason_step_by_step(
            query=f"Explain the concept of {concept}",
            context=context[:1000]  # Limit context for processing
        )
        
        # Process reasoning steps
        for step in reasoning_steps:
            self.add_thought(
                f"Reasoning step {step.step_id}: {step.description} "
                f"(confidence: {step.confidence_score:.2f})",
                "reasoning"
            )
        
        # Update knowledge base
        self.knowledge_base[concept] = {
            'understanding_level': np.random.uniform(0.7, 0.95),
            'reasoning_steps': len(reasoning_steps),
            'confidence': np.mean([step.confidence_score for step in reasoning_steps]),
            'learned_at': datetime.now().isoformat()
        }
        
        self.learned_concepts.append(concept)
        
        # Update learning metrics
        self.learning_progress.understanding_depth = np.mean([
            data['understanding_level'] for data in self.knowledge_base.values()
        ])
        
        self.learning_progress.retention_score = min(1.0, len(self.learned_concepts) / 10.0)
        self.learning_progress.adaptation_rate = np.random.uniform(0.8, 0.95)
        self.learning_progress.learning_efficiency = (
            self.learning_progress.understanding_depth * 
            self.learning_progress.retention_score
        )
    
    async def _validate_learning(self):
        """Validate the learning process"""
        self.add_thought("Validating learned concepts", "validation")
        
        validation_score = 0.0
        for concept in self.learned_concepts:
            concept_data = self.knowledge_base[concept]
            validation_score += concept_data['understanding_level']
        
        if self.learned_concepts:
            validation_score /= len(self.learned_concepts)
        
        self.add_thought(f"Validation score: {validation_score:.2f}", "validation_result")
        
        # Update performance metrics
        self.accuracy_history.append(validation_score * 100)
        self.confidence_scores.append(validation_score)
        
        return validation_score > 0.8
    
    async def start_demonstration(self) -> bool:
        """Start the demonstration mode"""
        if not self.learned_concepts:
            self.logger.warning("Cannot start demonstration without training")
            return False
        
        if self.state.is_training or self.state.is_demonstrating:
            self.logger.warning("Agent is already busy")
            return False
        
        self.state.is_demonstrating = True
        self.state.current_task = "demonstrating"
        self.state.demonstration_progress = 0.0
        self._notify_callbacks('state_change', asdict(self.state))
        
        try:
            await self._run_demonstration()
            self.state.is_demonstrating = False
            self.state.current_task = "demonstration_complete"
            self.state.demonstration_progress = 100.0
            self._notify_callbacks('state_change', asdict(self.state))
            return True
            
        except Exception as e:
            self.logger.error(f"Demonstration failed: {e}")
            self.state.is_demonstrating = False
            self.state.current_task = "demonstration_failed"
            self._notify_callbacks('state_change', asdict(self.state))
            return False
    
    async def _run_demonstration(self):
        """Run the interactive demonstration"""
        self.add_thought("Starting interactive demonstration", "demo_start")

        # Optional: run text-based PSO synthesis first and stream to GUI
        # Ensure OpenAI key from local file if not present
        try:
            if not os.environ.get('OPENAI_API_KEY'):
                key_path = Path('OpenAI-APIkey.txt')
                if key_path.exists():
                    key_text = key_path.read_text(encoding='utf-8')
                    api_key = next((ln.strip() for ln in key_text.splitlines() if ln.strip()), '')
                    if api_key:
                        os.environ['OPENAI_API_KEY'] = api_key
        except Exception:
            pass

        # Decide on LLM usage: prefer enabled when a key exists unless explicitly disabled
        try:
            default_flag = '1' if os.environ.get('OPENAI_API_KEY') else '0'
            use_llm = bool(int(os.environ.get('USE_LLM_PSO', default_flag)))
        except Exception:
            use_llm = bool(os.environ.get('OPENAI_API_KEY'))
        # Surface whether LLM-guided mutations are enabled
        try:
            self.add_thought(f"Text PSO LLM mutations enabled: {use_llm}", "synthesis_llm")
        except Exception:
            pass

        try:
            # Prefer tasks set explicitly (via GUI or from training recommendations)
            tasks = self.pso_tasks if self.pso_tasks else [
                "plan tasks, execute plan, verify results",
                "coordinator assigns tasks and verifier checks outputs",
                "researcher gathers info, planner decomposes, executor runs",
            ]

            # Provide domain text to evaluator for relevance (optional)
            try:
                # Write the training corpus to a cache file for evaluator if we have custom text
                corpus_text = self._load_academic_paper()
                if corpus_text:
                    data_dir = Path("data")
                    data_dir.mkdir(exist_ok=True)
                    cache_path = data_dir / "_training_corpus.txt"
                    if not cache_path.exists() or len(cache_path.read_text(encoding="utf-8", errors="ignore")) != len(corpus_text):
                        cache_path.write_text(corpus_text, encoding="utf-8")
                    self.training_corpus_cache_path = cache_path
                    os.environ["PSO_DOMAIN_TEXT_PATH"] = str(cache_path)
                    os.environ.setdefault("PSO_DOMAIN_WEIGHT", "0.2")
            except Exception:
                pass
            # notify GUI it can switch to text_pso visualization if supported
            self._notify_callbacks('state_change', {**asdict(self.state), 'text_pso_mode': True})

            def on_iter(iteration, pop_triplets, gbest):
                population_metrics = []
                teams = []
                for sys, fit, metrics in pop_triplets:
                    population_metrics.append({
                        'coverage': float(metrics.get('coverage', 0.0)),
                        'role_count': len(sys.roles),
                        'workflow_len': len(sys.workflow),
                        'fitness': float(fit),
                    })
                    # Team payload for visualization: roles + center coordinates
                    try:
                        cov = float(metrics.get('coverage', 0.0))
                        wf_len = int(len(sys.workflow))
                        team_obj = {
                            'roles': [r.name for r in sys.roles],
                            'coverage': cov,
                            'workflow_len': wf_len,
                            'fitness': float(fit),
                            'center_x': cov,
                            # y-axis is fitness in [0,1]
                            'center_y': float(fit),
                        }
                        teams.append(team_obj)
                    except Exception:
                        pass
                gsys, gfit = gbest
                # We don't have gbest metrics here; approximate from current pop
                cov = 0.0
                wf = 0
                try:
                    cov = max([m.get('coverage', 0.0) for _, _, m in pop_triplets] or [0.0])
                    wf = max([len(s.workflow) for s, _, _ in pop_triplets] or [0])
                except Exception:
                    pass

                # Prepare Top-K team summaries (include spec text for clarity)
                try:
                    k = 5
                    sorted_pop = sorted(pop_triplets, key=lambda t: float(t[1]), reverse=True)
                    top_k = []
                    for s, f, m in sorted_pop[:k]:
                        top_k.append({
                            'fitness': float(f),
                            'coverage': float(m.get('coverage', 0.0)),
                            'role_count': int(len(s.roles)),
                            'workflow_len': int(len(s.workflow)),
                            'spec': s.to_text(),
                        })
                except Exception:
                    top_k = []

                gbest_spec = ''
                try:
                    if gsys is not None:
                        gbest_spec = gsys.to_text()
                except Exception:
                    gbest_spec = ''

                # Best team summary (roles + metrics)
                gbest_team = {}
                try:
                    if gsys is not None:
                        gbest_team = {
                            'roles': [r.name for r in gsys.roles],
                            'coverage': float(cov),
                            'workflow_len': int(len(gsys.workflow) if hasattr(gsys, 'workflow') else 0),
                            'fitness': float(gfit),
                            'center_x': float(cov),
                            # y-axis uses fitness directly
                            'center_y': float(gfit),
                        }
                except Exception:
                    gbest_team = {}
                # LLM stats from synthesizer instance (available in closure)
                llm_stats = {
                    'calls_total': int(getattr(synth, 'llm_calls', 0)),
                    'accepts_total': int(getattr(synth, 'llm_accepts', 0)),
                    'accepts_this_iter': int(getattr(synth, '_accepted_this_iter', 0)),
                    'noops_total': int(getattr(synth, 'llm_noops', 0)),
                }

                # Scratchpad note when LLM proposals were accepted this iteration
                try:
                    if llm_stats['accepts_this_iter'] > 0:
                        self.add_thought(f"LLM mutations accepted this iteration: {llm_stats['accepts_this_iter']}", "synthesis_llm")
                except Exception:
                    pass

                self._notify_callbacks('progress', {
                    'type': 'synthesis_iteration',
                    'iteration': iteration,
                    'total_iters': iters,
                    'gbest_fitness': float(gfit),
                    'population': population_metrics,
                    'teams': teams,
                    'gbest_team': gbest_team,
                    'top_k': top_k,
                    'gbest_spec': gbest_spec,
                    'llm_stats': llm_stats,
                    'gbest': {
                        'coverage': float(cov),
                        'workflow_len': int(wf),
                        'fitness': float(gfit),
                    }
                })
                # Optional pacing so the GUI can show formation in real time
                try:
                    # Slow down default iteration pacing so communication is visible longer
                    pause = float(os.environ.get('TEXT_PSO_PAUSE', '0.8'))
                except Exception:
                    pause = 0.8
                if pause > 0:
                    time.sleep(pause)

            # Larger population/iterations for more visible motion; configurable via env
            try:
                pop = int(os.environ.get('TEXT_PSO_POP', '15'))
            except Exception:
                pop = 15
            try:
                iters = int(os.environ.get('TEXT_PSO_ITERS', '12'))
            except Exception:
                iters = 12
            synth = PSOSwarmSynthesizer(population_size=pop, iterations=iters, use_llm=use_llm)
            # Run PSO in a thread so UI can stream each iteration without blocking the event loop
            import asyncio as _asyncio
            loop = _asyncio.get_running_loop()
            def _run_sync():
                return synth.run(tasks, on_iteration=on_iter)
            result = await loop.run_in_executor(None, _run_sync)
            self.add_thought(f"PSO synthesis best fitness: {result.best_fitness:.2f}", "synthesis_result")
        except Exception as e:
            self.add_thought(f"Synthesis step skipped due to error: {e}", "synthesis_error")
        
        demo_scenarios = [
            "Explain the core principles of SwarmAgentic",
            "Describe the PSO-based optimization approach",
            "Analyze the experimental results and their significance",
            "Discuss the implications for AI system design",
            "Compare SwarmAgentic with traditional multi-agent systems"
        ]
        
        for i, scenario in enumerate(demo_scenarios):
            if self.state.is_paused:
                await self._wait_for_resume()
            
            await self._demonstrate_understanding(scenario)
            
            progress = (i + 1) / len(demo_scenarios) * 100
            self.state.demonstration_progress = progress
            
            self._notify_callbacks('progress', {
                'type': 'demonstration',
                'progress': progress,
                'current_scenario': scenario
            })
            
            await asyncio.sleep(1.0)
    
    async def _demonstrate_understanding(self, query: str):
        """Demonstrate understanding of a specific query"""
        self.add_thought(f"Demonstrating understanding of: {query}", "demonstration")
        
        # Use BERT reasoning engine
        reasoning_steps = self.bert_engine.reason_step_by_step(query)
        
        for step in reasoning_steps:
            self.add_thought(
                f"Demo reasoning: {step.description} -> {step.intermediate_results.get('final_answer', 'Processing...')}",
                "demo_reasoning"
            )
        
        # Generate response based on learned knowledge
        response = self._generate_response(query)
        self.add_thought(f"Generated response: {response[:100]}...", "response_generation")
    
    def _generate_response(self, query: str) -> str:
        """Generate a response based on learned knowledge"""
        # Simplified response generation
        relevant_concepts = [concept for concept in self.learned_concepts 
                           if any(word in concept.lower() for word in query.lower().split())]
        
        if relevant_concepts:
            return f"Based on my understanding of {', '.join(relevant_concepts[:3])}, " \
                   f"I can provide insights into {query}. The key aspects involve..."
        else:
            return f"While analyzing '{query}', I draw upon my comprehensive understanding " \
                   f"of the SwarmAgentic framework to provide relevant insights..."
    
    def pause(self):
        """Pause current operation"""
        self.state.is_paused = True
        self.add_thought("Operation paused", "control")
        self._notify_callbacks('state_change', asdict(self.state))
    
    def resume(self):
        """Resume paused operation"""
        self.state.is_paused = False
        self.add_thought("Operation resumed", "control")
        self._notify_callbacks('state_change', asdict(self.state))
    
    def stop(self):
        """Stop current operation"""
        self.state.is_training = False
        self.state.is_demonstrating = False
        self.state.is_paused = False
        self.state.current_task = "stopped"
        self.add_thought("Operation stopped", "control")
        self._notify_callbacks('state_change', asdict(self.state))
    
    def reset(self):
        """Reset agent to initial state"""
        self.stop()
        self.knowledge_base.clear()
        self.learned_concepts.clear()
        self.scratchpad.clear()
        self.thought_process.clear()
        self.working_memory.clear()

        self.state = AgentState()
        self.learning_progress = LearningProgress()

        self.add_thought("Agent reset to initial state", "control")
        self._notify_callbacks('state_change', asdict(self.state))
    
    async def _wait_for_resume(self):
        """Wait for resume signal when paused"""
        while self.state.is_paused:
            await asyncio.sleep(0.1)
    
    def set_training_text(self, text: str):
        """Set a custom training corpus (overrides file loading)."""
        self.custom_training_text = text or ""

    def set_pso_tasks(self, tasks: Optional[List[str]]):
        """Set tasks for PSO synthesizer. None clears explicit tasks."""
        self.pso_tasks = list(tasks) if tasks else None

    def apply_task_focus(self, tasks: Optional[List[str]] = None):
        """Derive a task-focused concept list for display based on selected tasks.

        Uses the knowledge graph export (if available) and filters concepts that
        match task keywords in name/definition/examples/related. Falls back to
        learned_concepts when no tasks or graph are available.
        """
        try:
            if not tasks:
                self.displayed_concepts = list(self.learned_concepts)
                return
            # Collect lowercase keywords > 3 chars from tasks
            import re
            kw = set()
            for t in tasks:
                for w in re.findall(r"[A-Za-z]{4,}", str(t)):
                    kw.add(w.lower())
            if not kw:
                self.displayed_concepts = list(self.learned_concepts)
                return
            # Prefer graph export for richer fields
            concepts = []
            graph = getattr(self, 'knowledge_graph_export', None)
            if isinstance(graph, dict) and isinstance(graph.get('concepts'), dict):
                for name, meta in graph['concepts'].items():
                    text_fields = [str(name)]
                    if isinstance(meta, dict):
                        text_fields.append(str(meta.get('definition', '')))
                        if isinstance(meta.get('examples'), list):
                            text_fields.extend([str(x) for x in meta.get('examples')])
                        if isinstance(meta.get('related_concepts'), list):
                            text_fields.extend([str(x) for x in meta.get('related_concepts')])
                    blob = " ".join(text_fields).lower()
                    if any(k in blob for k in kw):
                        concepts.append(name)
            # Fallback: filter learned_concepts by keyword
            if not concepts:
                concepts = [c for c in self.learned_concepts if any(k in c.lower() for k in kw)]
            # If still empty, just keep original learned list
            self.displayed_concepts = concepts if concepts else list(self.learned_concepts)
        except Exception:
            self.displayed_concepts = list(self.learned_concepts)

    def _load_academic_paper(self) -> str:
        """Load the training corpus: custom text, env override, or default file."""
        # 1) Highest priority: custom text explicitly set
        if self.custom_training_text is not None:
            return self.custom_training_text
        # 2) Environment override
        try:
            import os
            env_path = os.environ.get('TRAINING_TEXT_PATH')
            if env_path:
                p = Path(env_path)
                if p.exists():
                    with open(p, 'r', encoding='utf-8') as f:
                        return f.read()
                else:
                    self.logger.warning(f"TRAINING_TEXT_PATH set but not found: {env_path}")
        except Exception as e:
            self.logger.error(f"Failed to load TRAINING_TEXT_PATH: {e}")
        # 3) Fallback to academic_paper_path
        try:
            with open(self.academic_paper_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Failed to load training file: {e}")
            return ""
    
    def _extract_concepts(self, content: str) -> List[str]:
        """Extract key concepts from the academic paper"""
        # Simplified concept extraction
        concepts = [
            "SwarmAgentic", "Particle Swarm Optimization", "Multi-agent Systems",
            "Language-driven Optimization", "Agent Generation", "Collaboration Workflows",
            "PSO Iterations", "LLM-guided Transformations", "Automated Design",
            "Self-Optimizing Agents", "From-Scratch Generation", "Performance Feedback",
            "Failure-Aware Updates", "Text-based PSO", "Agent Synthesis",
            "Workflow Optimization", "Cross-model Transfer", "Autonomous Systems"
        ]
        return concepts
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance and state metrics"""
        bert_metrics = self.bert_engine.get_performance_metrics()
        
        return {
            'agent_state': asdict(self.state),
            'learning_progress': asdict(self.learning_progress),
            'bert_metrics': bert_metrics,
            'knowledge_base_size': len(self.knowledge_base),
            'concepts_learned': len(self.learned_concepts),
            'scratchpad_entries': len(self.scratchpad),
            'working_memory_items': len(self.working_memory),
            'accuracy_trend': self.accuracy_history[-10:] if self.accuracy_history else [],
            'average_confidence': np.mean(self.confidence_scores) if self.confidence_scores else 0.0
        }
    
    def get_scratchpad_content(self) -> List[str]:
        """Get current scratchpad content"""
        return self.scratchpad.copy()
    
    def get_thought_process(self) -> List[Dict[str, Any]]:
        """Get detailed thought process"""
        return self.thought_process.copy()
    
    def get_working_memory(self) -> Dict[str, Any]:
        """Get current working memory state"""
        return self.working_memory.copy()
