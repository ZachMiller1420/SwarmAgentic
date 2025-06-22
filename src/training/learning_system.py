"""
Advanced Training and Learning System for AI Agent
Implements dynamic knowledge acquisition with validation and progress tracking
"""

import asyncio
import json
import re
import time
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import numpy as np
import logging

@dataclass
class ConceptNode:
    """Represents a learned concept with relationships"""
    name: str
    definition: str
    importance_score: float
    understanding_level: float
    related_concepts: List[str]
    learning_timestamp: datetime
    validation_score: float
    examples: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'definition': self.definition,
            'importance_score': self.importance_score,
            'understanding_level': self.understanding_level,
            'related_concepts': self.related_concepts,
            'learning_timestamp': self.learning_timestamp.isoformat(),
            'validation_score': self.validation_score,
            'examples': self.examples
        }

@dataclass
class LearningSession:
    """Tracks a complete learning session"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    concepts_processed: int
    concepts_learned: int
    average_understanding: float
    session_efficiency: float
    validation_results: Dict[str, float]
    
class KnowledgeGraph:
    """Manages the knowledge graph of learned concepts"""
    
    def __init__(self):
        self.concepts: Dict[str, ConceptNode] = {}
        self.relationships: Dict[str, List[str]] = {}
        self.concept_hierarchy: Dict[str, List[str]] = {}
        
    def add_concept(self, concept: ConceptNode):
        """Add a new concept to the knowledge graph"""
        self.concepts[concept.name] = concept
        self.relationships[concept.name] = concept.related_concepts
        
        # Update hierarchy
        for related in concept.related_concepts:
            if related in self.concept_hierarchy:
                if concept.name not in self.concept_hierarchy[related]:
                    self.concept_hierarchy[related].append(concept.name)
            else:
                self.concept_hierarchy[related] = [concept.name]
    
    def get_concept_strength(self, concept_name: str) -> float:
        """Calculate the strength of understanding for a concept"""
        if concept_name not in self.concepts:
            return 0.0
        
        concept = self.concepts[concept_name]
        base_strength = concept.understanding_level * concept.validation_score
        
        # Boost from related concepts
        related_boost = 0.0
        for related in concept.related_concepts:
            if related in self.concepts:
                related_boost += self.concepts[related].understanding_level * 0.1
        
        return min(1.0, base_strength + related_boost)
    
    def get_learning_recommendations(self) -> List[str]:
        """Get recommendations for concepts to focus on"""
        weak_concepts = []
        for name, concept in self.concepts.items():
            strength = self.get_concept_strength(name)
            if strength < 0.7:
                weak_concepts.append((name, strength))
        
        # Sort by weakness and importance
        weak_concepts.sort(key=lambda x: (x[1], self.concepts[x[0]].importance_score))
        return [name for name, _ in weak_concepts[:5]]

class AdaptiveLearningSystem:
    """
    Advanced learning system that adapts to the agent's performance
    Implements spaced repetition and difficulty adjustment
    """
    
    def __init__(self, bert_engine):
        self.bert_engine = bert_engine
        self.knowledge_graph = KnowledgeGraph()
        self.learning_sessions: List[LearningSession] = []
        self.current_session: Optional[LearningSession] = None
        
        # Learning parameters
        self.learning_rate = 0.1
        self.retention_factor = 0.9
        self.difficulty_adjustment = 0.05
        self.validation_threshold = 0.8
        
        # Progress tracking
        self.total_concepts_identified = 0
        self.concepts_learned = 0
        self.learning_efficiency = 0.0
        self.adaptation_rate = 0.0
        
        self.logger = logging.getLogger(__name__)
    
    def start_learning_session(self, session_id: str) -> LearningSession:
        """Start a new learning session"""
        self.current_session = LearningSession(
            session_id=session_id,
            start_time=datetime.now(),
            end_time=None,
            concepts_processed=0,
            concepts_learned=0,
            average_understanding=0.0,
            session_efficiency=0.0,
            validation_results={}
        )
        
        self.logger.info(f"Started learning session: {session_id}")
        return self.current_session
    
    def end_learning_session(self) -> Optional[LearningSession]:
        """End the current learning session"""
        if not self.current_session:
            return None
        
        self.current_session.end_time = datetime.now()
        
        # Calculate session metrics
        if self.current_session.concepts_processed > 0:
            self.current_session.session_efficiency = (
                self.current_session.concepts_learned / 
                self.current_session.concepts_processed
            )
        
        # Calculate average understanding
        if self.knowledge_graph.concepts:
            total_understanding = sum(
                concept.understanding_level 
                for concept in self.knowledge_graph.concepts.values()
            )
            self.current_session.average_understanding = (
                total_understanding / len(self.knowledge_graph.concepts)
            )
        
        self.learning_sessions.append(self.current_session)
        completed_session = self.current_session
        self.current_session = None
        
        self.logger.info(f"Completed learning session: {completed_session.session_id}")
        return completed_session
    
    async def process_academic_content(self, content: str, 
                                     progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Process academic content and extract learnable concepts"""
        
        # Extract concepts from content
        concepts = self._extract_concepts_from_text(content)
        self.total_concepts_identified = len(concepts)
        
        learning_results = {
            'concepts_identified': len(concepts),
            'concepts_learned': 0,
            'learning_progress': [],
            'validation_results': {},
            'session_summary': {}
        }
        
        # Process each concept
        for i, concept_data in enumerate(concepts):
            if self.current_session:
                self.current_session.concepts_processed += 1
            
            # Learn the concept
            learned_concept = await self._learn_concept_adaptive(
                concept_data, content
            )
            
            if learned_concept:
                self.knowledge_graph.add_concept(learned_concept)
                learning_results['concepts_learned'] += 1
                self.concepts_learned += 1
                
                if self.current_session:
                    self.current_session.concepts_learned += 1
            
            # Update progress
            progress = (i + 1) / len(concepts) * 100
            learning_results['learning_progress'].append({
                'concept': concept_data['name'],
                'progress': progress,
                'success': learned_concept is not None,
                'understanding_level': learned_concept.understanding_level if learned_concept else 0.0
            })
            
            # Callback for real-time updates
            if progress_callback:
                progress_callback({
                    'type': 'concept_learning',
                    'progress': progress,
                    'current_concept': concept_data['name'],
                    'concepts_learned': learning_results['concepts_learned'],
                    'total_concepts': len(concepts)
                })
            
            # Adaptive delay based on difficulty
            await asyncio.sleep(0.2 + concept_data.get('difficulty', 0.5) * 0.3)
        
        # Validate learning
        validation_results = await self._validate_learning()
        learning_results['validation_results'] = validation_results
        
        # Update learning efficiency
        if self.total_concepts_identified > 0:
            self.learning_efficiency = self.concepts_learned / self.total_concepts_identified
        
        return learning_results
    
    def _extract_concepts_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract key concepts from academic text"""
        
        # Define concept patterns and their importance
        concept_patterns = {
            r'SwarmAgentic': {'importance': 1.0, 'category': 'system'},
            r'Particle Swarm Optimization|PSO': {'importance': 0.9, 'category': 'algorithm'},
            r'multi-agent system[s]?': {'importance': 0.8, 'category': 'architecture'},
            r'language-driven optimization': {'importance': 0.8, 'category': 'method'},
            r'agent generation': {'importance': 0.7, 'category': 'process'},
            r'collaboration workflow[s]?': {'importance': 0.7, 'category': 'structure'},
            r'LLM[s]?|Large Language Model[s]?': {'importance': 0.8, 'category': 'technology'},
            r'autonomous system[s]?': {'importance': 0.6, 'category': 'concept'},
            r'self-optimizing': {'importance': 0.7, 'category': 'capability'},
            r'from-scratch generation': {'importance': 0.8, 'category': 'approach'},
            r'failure-aware': {'importance': 0.6, 'category': 'feature'},
            r'text-based PSO': {'importance': 0.7, 'category': 'innovation'},
            r'cross-model transfer': {'importance': 0.6, 'category': 'capability'}
        }
        
        concepts = []
        
        for pattern, metadata in concept_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                concept_text = match.group()
                
                # Extract context around the concept
                start = max(0, match.start() - 200)
                end = min(len(text), match.end() + 200)
                context = text[start:end]
                
                # Calculate difficulty based on context complexity
                difficulty = self._calculate_concept_difficulty(context)
                
                concept = {
                    'name': concept_text,
                    'context': context,
                    'importance': metadata['importance'],
                    'category': metadata['category'],
                    'difficulty': difficulty,
                    'position': match.start()
                }
                
                concepts.append(concept)
        
        # Remove duplicates and sort by importance
        unique_concepts = {}
        for concept in concepts:
            name = concept['name'].lower()
            if name not in unique_concepts or concept['importance'] > unique_concepts[name]['importance']:
                unique_concepts[name] = concept
        
        sorted_concepts = sorted(
            unique_concepts.values(), 
            key=lambda x: x['importance'], 
            reverse=True
        )
        
        return sorted_concepts[:20]  # Limit to top 20 concepts
    
    def _calculate_concept_difficulty(self, context: str) -> float:
        """Calculate the difficulty of learning a concept based on context"""
        
        # Factors that increase difficulty
        technical_terms = len(re.findall(r'\b[A-Z]{2,}\b', context))  # Acronyms
        complex_sentences = len([s for s in context.split('.') if len(s.split()) > 20])
        mathematical_notation = len(re.findall(r'[α-ωΑ-Ω]|\d+\.\d+|±|≥|≤', context))
        
        # Normalize difficulty score
        difficulty = min(1.0, (technical_terms * 0.1 + complex_sentences * 0.2 + mathematical_notation * 0.15))
        return max(0.1, difficulty)  # Minimum difficulty
    
    async def _learn_concept_adaptive(self, concept_data: Dict[str, Any], 
                                    full_context: str) -> Optional[ConceptNode]:
        """Learn a concept using adaptive techniques"""
        
        concept_name = concept_data['name']
        context = concept_data['context']
        difficulty = concept_data['difficulty']
        
        # Use BERT for deep understanding
        reasoning_steps = self.bert_engine.reason_step_by_step(
            query=f"Explain and analyze the concept of {concept_name}",
            context=context
        )
        
        # Calculate understanding level based on reasoning quality
        understanding_level = self._calculate_understanding_level(reasoning_steps, difficulty)
        
        # Generate definition from reasoning
        definition = self._generate_definition(concept_name, reasoning_steps, context)
        
        # Find related concepts
        related_concepts = self._find_related_concepts(concept_name, full_context)
        
        # Extract examples
        examples = self._extract_examples(concept_name, context)
        
        # Validate learning
        validation_score = await self._validate_concept_learning(
            concept_name, definition, understanding_level
        )
        
        # Only create concept if validation passes
        if validation_score >= self.validation_threshold:
            concept = ConceptNode(
                name=concept_name,
                definition=definition,
                importance_score=concept_data['importance'],
                understanding_level=understanding_level,
                related_concepts=related_concepts,
                learning_timestamp=datetime.now(),
                validation_score=validation_score,
                examples=examples
            )
            
            return concept
        
        return None
    
    def _calculate_understanding_level(self, reasoning_steps: List, difficulty: float) -> float:
        """Calculate understanding level based on reasoning quality"""
        
        if not reasoning_steps:
            return 0.0
        
        # Base understanding from reasoning confidence
        avg_confidence = np.mean([step.confidence_score for step in reasoning_steps])
        
        # Adjust for difficulty
        difficulty_adjustment = 1.0 - (difficulty * 0.3)
        
        # Boost for comprehensive reasoning
        reasoning_depth_bonus = min(0.2, len(reasoning_steps) * 0.05)
        
        understanding = avg_confidence * difficulty_adjustment + reasoning_depth_bonus
        return min(1.0, max(0.0, understanding))
    
    def _generate_definition(self, concept_name: str, reasoning_steps: List, context: str) -> str:
        """Generate a definition based on reasoning and context"""
        
        # Extract key insights from reasoning steps
        key_insights = []
        for step in reasoning_steps:
            if 'final_answer' in step.intermediate_results:
                key_insights.append(step.intermediate_results['final_answer'])
        
        # Create definition (simplified)
        if key_insights:
            definition = f"{concept_name} refers to " + ". ".join(key_insights[:2])
        else:
            # Fallback to context-based definition
            sentences = context.split('.')
            relevant_sentences = [s.strip() for s in sentences if concept_name.lower() in s.lower()]
            definition = ". ".join(relevant_sentences[:2]) if relevant_sentences else f"A concept related to {concept_name}"
        
        return definition[:500]  # Limit length
    
    def _find_related_concepts(self, concept_name: str, full_context: str) -> List[str]:
        """Find concepts related to the current concept"""
        
        # Simple approach: find concepts mentioned near this one
        concept_position = full_context.lower().find(concept_name.lower())
        if concept_position == -1:
            return []
        
        # Extract surrounding text
        start = max(0, concept_position - 500)
        end = min(len(full_context), concept_position + 500)
        surrounding_text = full_context[start:end]
        
        # Find other technical terms
        related = []
        patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Capitalized terms
            r'\b[A-Z]{2,}\b',  # Acronyms
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, surrounding_text)
            for match in matches:
                if match.lower() != concept_name.lower() and len(match) > 3:
                    related.append(match)
        
        return list(set(related))[:5]  # Limit to 5 related concepts
    
    def _extract_examples(self, concept_name: str, context: str) -> List[str]:
        """Extract examples related to the concept"""
        
        # Look for example patterns
        example_patterns = [
            r'for example[,:]?\s*([^.]+)',
            r'such as\s*([^.]+)',
            r'including\s*([^.]+)',
            r'e\.g\.[\s,]*([^.]+)'
        ]
        
        examples = []
        for pattern in example_patterns:
            matches = re.findall(pattern, context, re.IGNORECASE)
            examples.extend(matches)
        
        return [ex.strip() for ex in examples[:3]]  # Limit to 3 examples
    
    async def _validate_concept_learning(self, concept_name: str, 
                                       definition: str, understanding_level: float) -> float:
        """Validate that the concept was learned correctly"""
        
        # Simulate validation process
        await asyncio.sleep(0.1)
        
        # Base validation score
        base_score = understanding_level
        
        # Boost for good definition
        if len(definition) > 50 and concept_name.lower() in definition.lower():
            base_score += 0.1
        
        # Random validation noise (simulating real validation complexity)
        noise = np.random.normal(0, 0.05)
        
        validation_score = max(0.0, min(1.0, base_score + noise))
        return validation_score
    
    async def _validate_learning(self) -> Dict[str, float]:
        """Validate overall learning progress"""
        
        if not self.knowledge_graph.concepts:
            return {'overall_score': 0.0, 'concept_scores': {}}
        
        concept_scores = {}
        total_score = 0.0
        
        for name, concept in self.knowledge_graph.concepts.items():
            # Re-validate each concept
            score = await self._validate_concept_learning(
                name, concept.definition, concept.understanding_level
            )
            concept_scores[name] = score
            total_score += score
        
        overall_score = total_score / len(self.knowledge_graph.concepts)
        
        return {
            'overall_score': overall_score,
            'concept_scores': concept_scores,
            'total_concepts': len(self.knowledge_graph.concepts),
            'validation_timestamp': datetime.now().isoformat()
        }
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        
        return {
            'total_concepts_identified': self.total_concepts_identified,
            'concepts_learned': self.concepts_learned,
            'learning_efficiency': self.learning_efficiency,
            'knowledge_graph_size': len(self.knowledge_graph.concepts),
            'average_understanding': np.mean([
                concept.understanding_level 
                for concept in self.knowledge_graph.concepts.values()
            ]) if self.knowledge_graph.concepts else 0.0,
            'average_validation_score': np.mean([
                concept.validation_score 
                for concept in self.knowledge_graph.concepts.values()
            ]) if self.knowledge_graph.concepts else 0.0,
            'learning_sessions': len(self.learning_sessions),
            'current_session_active': self.current_session is not None,
            'learning_recommendations': self.knowledge_graph.get_learning_recommendations()
        }
    
    def export_knowledge_graph(self) -> Dict[str, Any]:
        """Export the complete knowledge graph"""
        
        return {
            'concepts': {
                name: concept.to_dict() 
                for name, concept in self.knowledge_graph.concepts.items()
            },
            'relationships': self.knowledge_graph.relationships,
            'hierarchy': self.knowledge_graph.concept_hierarchy,
            'statistics': self.get_learning_statistics(),
            'export_timestamp': datetime.now().isoformat()
        }
