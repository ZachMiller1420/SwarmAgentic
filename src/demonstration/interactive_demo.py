"""
Interactive Demonstration System
Showcases AI Agent understanding and capabilities through dynamic scenarios
"""

import asyncio
import random
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import logging
import numpy as np

@dataclass
class DemoScenario:
    """Represents a demonstration scenario"""
    scenario_id: str
    title: str
    description: str
    complexity_level: str
    expected_duration: float
    required_concepts: List[str]
    success_criteria: Dict[str, float]
    interactive_elements: List[str]

@dataclass
class DemoResponse:
    """Represents an agent response during demonstration"""
    scenario_id: str
    query: str
    response: str
    reasoning_steps: List[str]
    confidence_score: float
    processing_time: float
    concepts_used: List[str]
    success_metrics: Dict[str, float]

class InteractiveDemonstrationSystem:
    """
    Advanced demonstration system that showcases agent capabilities
    Provides interactive scenarios and real-time performance analysis
    """
    
    def __init__(self, ai_agent, quality_metrics):
        self.ai_agent = ai_agent
        self.quality_metrics = quality_metrics
        
        # Demo state
        self.is_running = False
        self.current_scenario: Optional[DemoScenario] = None
        self.demo_history: List[DemoResponse] = []
        self.scenario_progress = 0.0
        
        # Callbacks for real-time updates
        self.progress_callbacks: List[Callable] = []
        self.response_callbacks: List[Callable] = []
        self.completion_callbacks: List[Callable] = []
        
        # Performance tracking
        self.demo_start_time: Optional[datetime] = None
        self.total_scenarios_completed = 0
        self.average_success_rate = 0.0
        
        self.logger = logging.getLogger(__name__)
        self._initialize_demo_scenarios()
    
    def _initialize_demo_scenarios(self):
        """Initialize predefined demonstration scenarios"""
        self.demo_scenarios = [
            DemoScenario(
                scenario_id="swarm_principles",
                title="SwarmAgentic Core Principles",
                description="Demonstrate understanding of SwarmAgentic's fundamental principles and architecture",
                complexity_level="intermediate",
                expected_duration=30.0,
                required_concepts=["SwarmAgentic", "Particle Swarm Optimization", "multi-agent systems"],
                success_criteria={"accuracy": 0.85, "completeness": 0.80, "coherence": 0.90},
                interactive_elements=["concept_explanation", "relationship_mapping", "example_generation"]
            ),
            DemoScenario(
                scenario_id="pso_optimization",
                title="PSO-Based Optimization Analysis",
                description="Analyze and explain the PSO optimization approach used in SwarmAgentic",
                complexity_level="advanced",
                expected_duration=45.0,
                required_concepts=["Particle Swarm Optimization", "language-driven optimization", "agent generation"],
                success_criteria={"accuracy": 0.88, "completeness": 0.85, "coherence": 0.92},
                interactive_elements=["algorithm_breakdown", "step_by_step_analysis", "optimization_visualization"]
            ),
            DemoScenario(
                scenario_id="experimental_results",
                title="Experimental Results Interpretation",
                description="Interpret and discuss the experimental results and their significance",
                complexity_level="expert",
                expected_duration=40.0,
                required_concepts=["experimental results", "performance metrics", "baseline comparison"],
                success_criteria={"accuracy": 0.90, "completeness": 0.88, "coherence": 0.85},
                interactive_elements=["data_analysis", "statistical_interpretation", "significance_assessment"]
            ),
            DemoScenario(
                scenario_id="future_implications",
                title="Future Implications and Applications",
                description="Discuss the broader implications and potential applications of SwarmAgentic",
                complexity_level="expert",
                expected_duration=35.0,
                required_concepts=["autonomous systems", "AI-driven AI", "self-improving systems"],
                success_criteria={"accuracy": 0.82, "completeness": 0.85, "coherence": 0.88},
                interactive_elements=["trend_analysis", "application_brainstorming", "impact_assessment"]
            ),
            DemoScenario(
                scenario_id="comparative_analysis",
                title="Comparative Analysis with Traditional Systems",
                description="Compare SwarmAgentic with traditional multi-agent system approaches",
                complexity_level="advanced",
                expected_duration=38.0,
                required_concepts=["traditional multi-agent systems", "automated design", "human-designed templates"],
                success_criteria={"accuracy": 0.87, "completeness": 0.83, "coherence": 0.90},
                interactive_elements=["feature_comparison", "advantage_analysis", "limitation_discussion"]
            )
        ]
    
    def register_callback(self, callback_type: str, callback: Callable):
        """Register callbacks for demo events"""
        if callback_type == "progress":
            self.progress_callbacks.append(callback)
        elif callback_type == "response":
            self.response_callbacks.append(callback)
        elif callback_type == "completion":
            self.completion_callbacks.append(callback)
    
    def _notify_callbacks(self, callback_type: str, data: Any):
        """Notify registered callbacks"""
        callbacks = {
            "progress": self.progress_callbacks,
            "response": self.response_callbacks,
            "completion": self.completion_callbacks
        }.get(callback_type, [])
        
        for callback in callbacks:
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Demo callback error: {e}")
    
    async def start_demonstration(self) -> bool:
        """Start the interactive demonstration"""
        if self.is_running:
            self.logger.warning("Demonstration already running")
            return False
        
        if not self.ai_agent.learned_concepts:
            self.logger.error("Cannot start demonstration without trained agent")
            return False
        
        self.is_running = True
        self.demo_start_time = datetime.now()
        self.demo_history.clear()
        self.total_scenarios_completed = 0
        
        try:
            await self._run_demonstration_sequence()
            self.is_running = False
            return True
        except Exception as e:
            self.logger.error(f"Demonstration failed: {e}")
            self.is_running = False
            return False
    
    async def _run_demonstration_sequence(self):
        """Run the complete demonstration sequence"""
        self.logger.info("Starting interactive demonstration sequence")
        
        # Filter scenarios based on agent's learned concepts
        available_scenarios = self._filter_scenarios_by_knowledge()
        
        total_scenarios = len(available_scenarios)
        
        for i, scenario in enumerate(available_scenarios):
            if not self.is_running:
                break
            
            self.current_scenario = scenario
            self.scenario_progress = 0.0
            
            self.logger.info(f"Starting scenario: {scenario.title}")
            
            # Notify progress
            self._notify_callbacks("progress", {
                "type": "scenario_start",
                "scenario": asdict(scenario),
                "overall_progress": (i / total_scenarios) * 100,
                "scenario_index": i,
                "total_scenarios": total_scenarios
            })
            
            # Run the scenario
            scenario_result = await self._run_scenario(scenario)
            
            if scenario_result:
                self.total_scenarios_completed += 1
            
            # Update overall progress
            overall_progress = ((i + 1) / total_scenarios) * 100
            self._notify_callbacks("progress", {
                "type": "scenario_complete",
                "scenario": asdict(scenario),
                "overall_progress": overall_progress,
                "success": scenario_result
            })
            
            # Brief pause between scenarios
            await asyncio.sleep(2.0)
        
        # Calculate final success rate
        if total_scenarios > 0:
            self.average_success_rate = self.total_scenarios_completed / total_scenarios
        
        # Notify completion
        self._notify_callbacks("completion", {
            "total_scenarios": total_scenarios,
            "completed_scenarios": self.total_scenarios_completed,
            "success_rate": self.average_success_rate,
            "demo_duration": (datetime.now() - self.demo_start_time).total_seconds() if self.demo_start_time else 0,
            "demo_history": [asdict(response) for response in self.demo_history]
        })
        
        self.logger.info(f"Demonstration completed. Success rate: {self.average_success_rate:.2%}")
    
    def _filter_scenarios_by_knowledge(self) -> List[DemoScenario]:
        """Filter scenarios based on agent's learned concepts"""
        available_scenarios = []
        learned_concept_names = [concept.lower() for concept in self.ai_agent.learned_concepts]
        
        for scenario in self.demo_scenarios:
            # Check if agent has learned required concepts
            required_concepts_met = 0
            for required_concept in scenario.required_concepts:
                if any(req.lower() in learned for req in required_concept.split() 
                      for learned in learned_concept_names):
                    required_concepts_met += 1
            
            # Include scenario if at least 60% of required concepts are known
            if required_concepts_met >= len(scenario.required_concepts) * 0.6:
                available_scenarios.append(scenario)
        
        return available_scenarios
    
    async def _run_scenario(self, scenario: DemoScenario) -> bool:
        """Run a single demonstration scenario"""
        scenario_queries = self._generate_scenario_queries(scenario)
        scenario_success = True
        scenario_responses = []
        
        for i, query in enumerate(scenario_queries):
            if not self.is_running:
                break
            
            # Update scenario progress
            self.scenario_progress = (i / len(scenario_queries)) * 100
            
            # Process query with agent
            response = await self._process_demo_query(scenario, query)
            scenario_responses.append(response)
            
            # Evaluate response
            success = self._evaluate_response(response, scenario)
            if not success:
                scenario_success = False
            
            # Notify response
            self._notify_callbacks("response", {
                "scenario_id": scenario.scenario_id,
                "query": query,
                "response": asdict(response),
                "success": success,
                "scenario_progress": self.scenario_progress
            })
            
            # Record metrics
            self.quality_metrics.record_accuracy(
                response.success_metrics.get("accuracy", 0.0),
                "demonstration"
            )
            self.quality_metrics.record_confidence(
                response.confidence_score,
                f"demo scenario: {scenario.title}"
            )
            self.quality_metrics.record_response_time(response.processing_time)
            
            # Simulate processing time
            await asyncio.sleep(1.0)
        
        return scenario_success
    
    def _generate_scenario_queries(self, scenario: DemoScenario) -> List[str]:
        """Generate queries for a specific scenario"""
        query_templates = {
            "swarm_principles": [
                "Explain the core principles of SwarmAgentic and how it differs from traditional approaches",
                "Describe the three autonomy properties that SwarmAgentic addresses",
                "How does SwarmAgentic achieve from-scratch agent generation?"
            ],
            "pso_optimization": [
                "Explain how Particle Swarm Optimization is adapted for language-driven optimization",
                "Describe the role of LLM-guided transformations in the PSO process",
                "What are the key components of the failure-aware velocity update mechanism?"
            ],
            "experimental_results": [
                "Analyze the performance improvements shown in the TravelPlanner experiments",
                "Interpret the significance of the 261.8% relative improvement over ADAS",
                "Discuss the cross-model transfer results and their implications"
            ],
            "future_implications": [
                "What are the potential applications of SwarmAgentic in enterprise environments?",
                "How might SwarmAgentic contribute to the development of self-improving AI ecosystems?",
                "Discuss the democratization potential of automated agent design"
            ],
            "comparative_analysis": [
                "Compare SwarmAgentic with traditional multi-agent frameworks like ADAS",
                "What advantages does automated design provide over human-crafted templates?",
                "Analyze the trade-offs between automation and human oversight in agent design"
            ]
        }
        
        return query_templates.get(scenario.scenario_id, [
            f"Explain the key concepts related to {scenario.title}",
            f"Analyze the implications of {scenario.title}",
            f"Provide examples related to {scenario.title}"
        ])
    
    async def _process_demo_query(self, scenario: DemoScenario, query: str) -> DemoResponse:
        """Process a demonstration query with the agent"""
        start_time = time.time()
        
        # Use agent's reasoning engine
        reasoning_steps = self.ai_agent.bert_engine.reason_step_by_step(query)
        
        # Generate response based on learned knowledge
        response_text = self._generate_demo_response(query, reasoning_steps)
        
        # Extract concepts used
        concepts_used = self._identify_concepts_used(query, response_text)
        
        # Calculate confidence
        confidence = np.mean([step.confidence_score for step in reasoning_steps]) if reasoning_steps else 0.5
        
        # Calculate success metrics
        success_metrics = self._calculate_success_metrics(
            response_text, scenario, reasoning_steps
        )
        
        processing_time = time.time() - start_time
        
        demo_response = DemoResponse(
            scenario_id=scenario.scenario_id,
            query=query,
            response=response_text,
            reasoning_steps=[step.description for step in reasoning_steps],
            confidence_score=confidence,
            processing_time=processing_time,
            concepts_used=concepts_used,
            success_metrics=success_metrics
        )
        
        self.demo_history.append(demo_response)
        return demo_response
    
    def _generate_demo_response(self, query: str, reasoning_steps: List) -> str:
        """Generate a demonstration response based on reasoning"""
        
        # Extract insights from reasoning steps
        insights = []
        for step in reasoning_steps:
            if 'final_answer' in step.intermediate_results:
                insights.append(step.intermediate_results['final_answer'])
        
        # Create structured response
        if insights:
            response = f"Based on my analysis of SwarmAgentic, {insights[0]}"
            if len(insights) > 1:
                response += f" Additionally, {insights[1]}"
        else:
            # Fallback response based on learned concepts
            relevant_concepts = [concept for concept in self.ai_agent.learned_concepts 
                               if any(word in concept.lower() for word in query.lower().split())]
            
            if relevant_concepts:
                response = f"Drawing from my understanding of {', '.join(relevant_concepts[:2])}, " \
                          f"I can address your question about {query.lower()}. "
            else:
                response = f"Regarding {query}, my analysis of the SwarmAgentic framework reveals "
        
        # Add domain-specific insights
        response += self._add_domain_insights(query)
        
        return response[:500]  # Limit response length
    
    def _add_domain_insights(self, query: str) -> str:
        """Add domain-specific insights to the response"""
        query_lower = query.lower()
        
        if "swarm" in query_lower or "pso" in query_lower:
            return "that the swarm intelligence approach enables collective optimization through " \
                   "distributed decision-making and emergent behaviors."
        elif "agent" in query_lower and "generation" in query_lower:
            return "that automated agent generation eliminates the need for human-designed templates " \
                   "while maintaining high performance standards."
        elif "optimization" in query_lower:
            return "that the language-driven optimization process leverages LLM capabilities to " \
                   "perform discrete optimization in text space."
        elif "experiment" in query_lower or "result" in query_lower:
            return "that the experimental validation demonstrates significant improvements across " \
                   "multiple benchmarks and task domains."
        else:
            return "that this represents a significant advancement in autonomous AI system design."
    
    def _identify_concepts_used(self, query: str, response: str) -> List[str]:
        """Identify which learned concepts were used in the response"""
        concepts_used = []
        combined_text = (query + " " + response).lower()
        
        for concept in self.ai_agent.learned_concepts:
            if concept.lower() in combined_text:
                concepts_used.append(concept)
        
        return concepts_used
    
    def _calculate_success_metrics(self, response: str, scenario: DemoScenario, 
                                 reasoning_steps: List) -> Dict[str, float]:
        """Calculate success metrics for a response"""
        
        # Accuracy: based on concept coverage and reasoning quality
        concept_coverage = len([concept for concept in scenario.required_concepts 
                              if any(req.lower() in response.lower() 
                                   for req in concept.split())]) / len(scenario.required_concepts)
        
        reasoning_quality = np.mean([step.confidence_score for step in reasoning_steps]) if reasoning_steps else 0.5
        accuracy = (concept_coverage + reasoning_quality) / 2
        
        # Completeness: based on response length and detail
        completeness = min(1.0, len(response.split()) / 50)  # Assume 50 words is complete
        
        # Coherence: simplified metric based on sentence structure
        sentences = response.split('.')
        coherence = min(1.0, len([s for s in sentences if len(s.split()) > 3]) / max(1, len(sentences)))
        
        return {
            "accuracy": accuracy,
            "completeness": completeness,
            "coherence": coherence
        }
    
    def _evaluate_response(self, response: DemoResponse, scenario: DemoScenario) -> bool:
        """Evaluate if a response meets the scenario's success criteria"""
        
        for metric, threshold in scenario.success_criteria.items():
            if response.success_metrics.get(metric, 0.0) < threshold:
                return False
        
        return True
    
    def stop_demonstration(self):
        """Stop the current demonstration"""
        self.is_running = False
        self.logger.info("Demonstration stopped")
    
    def get_demo_statistics(self) -> Dict[str, Any]:
        """Get comprehensive demonstration statistics"""
        
        if not self.demo_history:
            return {"status": "no_data"}
        
        # Calculate averages
        avg_confidence = np.mean([r.confidence_score for r in self.demo_history])
        avg_processing_time = np.mean([r.processing_time for r in self.demo_history])
        
        # Calculate success rates by metric
        success_rates = {}
        for metric in ["accuracy", "completeness", "coherence"]:
            values = [r.success_metrics.get(metric, 0.0) for r in self.demo_history]
            success_rates[metric] = np.mean(values) if values else 0.0
        
        # Concept usage analysis
        all_concepts_used = []
        for response in self.demo_history:
            all_concepts_used.extend(response.concepts_used)
        
        concept_usage = {}
        for concept in set(all_concepts_used):
            concept_usage[concept] = all_concepts_used.count(concept)
        
        return {
            "status": "complete" if not self.is_running else "running",
            "total_responses": len(self.demo_history),
            "scenarios_completed": self.total_scenarios_completed,
            "overall_success_rate": self.average_success_rate,
            "average_confidence": avg_confidence,
            "average_processing_time": avg_processing_time,
            "success_rates_by_metric": success_rates,
            "concept_usage": concept_usage,
            "demo_duration": (datetime.now() - self.demo_start_time).total_seconds() if self.demo_start_time else 0,
            "current_scenario": asdict(self.current_scenario) if self.current_scenario else None,
            "scenario_progress": self.scenario_progress
        }
    
    def export_demo_results(self, filepath: str):
        """Export demonstration results to a file"""
        demo_data = {
            "statistics": self.get_demo_statistics(),
            "demo_history": [asdict(response) for response in self.demo_history],
            "scenarios": [asdict(scenario) for scenario in self.demo_scenarios],
            "export_timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(demo_data, f, indent=2, default=str)
            self.logger.info(f"Demo results exported to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to export demo results: {e}")
