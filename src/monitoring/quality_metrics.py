"""
Advanced Quality Metrics and Monitoring System
Implements real-time accuracy tracking, source quality assessment, and comprehensive monitoring
"""

import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import json
import logging

@dataclass
class QualityMetric:
    """Represents a quality metric measurement"""
    name: str
    value: float
    timestamp: datetime
    confidence: float
    source: str
    metadata: Dict[str, Any]

@dataclass
class SourceQuality:
    """Tracks quality metrics for information sources"""
    source_id: str
    reliability_score: float
    freshness_score: float
    accuracy_history: List[float]
    last_updated: datetime
    change_frequency: float
    trust_level: str
    validation_count: int

@dataclass
class ExpectedDiscovery:
    """Represents expected discovery patterns and probabilities"""
    pattern_name: str
    probability: float
    confidence_interval: Tuple[float, float]
    context_requirements: List[str]
    discovery_conditions: Dict[str, Any]
    last_occurrence: Optional[datetime]

class RealTimeMetricsCollector:
    """
    Collects and processes real-time quality metrics
    Implements sliding window analysis and trend detection
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        
        # Metric storage with sliding windows
        self.accuracy_window = deque(maxlen=window_size)
        self.confidence_window = deque(maxlen=window_size)
        self.response_time_window = deque(maxlen=window_size)
        self.quality_metrics: Dict[str, deque] = {}
        
        # Source quality tracking
        self.source_qualities: Dict[str, SourceQuality] = {}
        
        # Expected discoveries
        self.expected_discoveries: Dict[str, ExpectedDiscovery] = {}
        
        # Real-time statistics
        self.current_accuracy = 0.0
        self.current_confidence = 0.0
        self.average_response_time = 0.0
        self.quality_trend = "stable"
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.update_callbacks: List[callable] = []
        
        self.logger = logging.getLogger(__name__)
        self._initialize_default_sources()
        self._initialize_expected_discoveries()
    
    def _initialize_default_sources(self):
        """Initialize default information sources"""
        default_sources = [
            {
                'source_id': 'academic_paper',
                'reliability_score': 0.95,
                'freshness_score': 0.90,
                'trust_level': 'high'
            },
            {
                'source_id': 'bert_model',
                'reliability_score': 0.88,
                'freshness_score': 0.85,
                'trust_level': 'high'
            },
            {
                'source_id': 'reasoning_engine',
                'reliability_score': 0.82,
                'freshness_score': 0.95,
                'trust_level': 'medium-high'
            },
            {
                'source_id': 'knowledge_base',
                'reliability_score': 0.78,
                'freshness_score': 0.70,
                'trust_level': 'medium'
            }
        ]
        
        for source_data in default_sources:
            source = SourceQuality(
                source_id=source_data['source_id'],
                reliability_score=source_data['reliability_score'],
                freshness_score=source_data['freshness_score'],
                accuracy_history=[source_data['reliability_score']],
                last_updated=datetime.now(),
                change_frequency=0.1,
                trust_level=source_data['trust_level'],
                validation_count=0
            )
            self.source_qualities[source_data['source_id']] = source
    
    def _initialize_expected_discoveries(self):
        """Initialize expected discovery patterns"""
        discoveries = [
            {
                'pattern_name': 'swarm_intelligence_insight',
                'probability': 0.85,
                'confidence_interval': (0.75, 0.95),
                'context_requirements': ['swarm', 'optimization', 'collective'],
                'discovery_conditions': {'min_confidence': 0.8, 'context_depth': 'high'}
            },
            {
                'pattern_name': 'agent_collaboration_pattern',
                'probability': 0.78,
                'confidence_interval': (0.65, 0.90),
                'context_requirements': ['agent', 'collaboration', 'workflow'],
                'discovery_conditions': {'min_confidence': 0.7, 'context_depth': 'medium'}
            },
            {
                'pattern_name': 'optimization_breakthrough',
                'probability': 0.65,
                'confidence_interval': (0.50, 0.80),
                'context_requirements': ['optimization', 'performance', 'improvement'],
                'discovery_conditions': {'min_confidence': 0.75, 'context_depth': 'high'}
            },
            {
                'pattern_name': 'emergent_behavior_detection',
                'probability': 0.55,
                'confidence_interval': (0.40, 0.70),
                'context_requirements': ['emergent', 'behavior', 'system'],
                'discovery_conditions': {'min_confidence': 0.8, 'context_depth': 'very_high'}
            }
        ]
        
        for discovery_data in discoveries:
            discovery = ExpectedDiscovery(
                pattern_name=discovery_data['pattern_name'],
                probability=discovery_data['probability'],
                confidence_interval=discovery_data['confidence_interval'],
                context_requirements=discovery_data['context_requirements'],
                discovery_conditions=discovery_data['discovery_conditions'],
                last_occurrence=None
            )
            self.expected_discoveries[discovery_data['pattern_name']] = discovery
    
    def register_update_callback(self, callback: callable):
        """Register callback for metric updates"""
        self.update_callbacks.append(callback)
    
    def _notify_callbacks(self, metric_data: Dict[str, Any]):
        """Notify all registered callbacks"""
        for callback in self.update_callbacks:
            try:
                callback(metric_data)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")
    
    def record_accuracy(self, accuracy: float, source: str = "unknown"):
        """Record an accuracy measurement"""
        self.accuracy_window.append(accuracy)
        self.current_accuracy = accuracy
        
        # Update source quality
        if source in self.source_qualities:
            self.source_qualities[source].accuracy_history.append(accuracy)
            self.source_qualities[source].last_updated = datetime.now()
            self.source_qualities[source].validation_count += 1
            
            # Update reliability score (exponential moving average)
            alpha = 0.1
            old_reliability = self.source_qualities[source].reliability_score
            self.source_qualities[source].reliability_score = (
                alpha * accuracy + (1 - alpha) * old_reliability
            )
        
        self._update_quality_trend()
        self._notify_callbacks({
            'type': 'accuracy_update',
            'value': accuracy,
            'source': source,
            'current_average': self.get_average_accuracy()
        })
    
    def record_confidence(self, confidence: float, context: str = ""):
        """Record a confidence measurement"""
        self.confidence_window.append(confidence)
        self.current_confidence = confidence
        
        # Check for expected discoveries
        self._check_expected_discoveries(confidence, context)
        
        self._notify_callbacks({
            'type': 'confidence_update',
            'value': confidence,
            'context': context,
            'current_average': self.get_average_confidence()
        })
    
    def record_response_time(self, response_time: float):
        """Record a response time measurement"""
        self.response_time_window.append(response_time)
        self.average_response_time = np.mean(list(self.response_time_window))
        
        self._notify_callbacks({
            'type': 'response_time_update',
            'value': response_time,
            'current_average': self.average_response_time
        })
    
    def record_custom_metric(self, metric_name: str, value: float, metadata: Dict[str, Any] = None):
        """Record a custom quality metric"""
        if metric_name not in self.quality_metrics:
            self.quality_metrics[metric_name] = deque(maxlen=self.window_size)
        
        self.quality_metrics[metric_name].append(value)
        
        metric = QualityMetric(
            name=metric_name,
            value=value,
            timestamp=datetime.now(),
            confidence=metadata.get('confidence', 1.0) if metadata else 1.0,
            source=metadata.get('source', 'unknown') if metadata else 'unknown',
            metadata=metadata or {}
        )
        
        self._notify_callbacks({
            'type': 'custom_metric_update',
            'metric': asdict(metric)
        })
    
    def _check_expected_discoveries(self, confidence: float, context: str):
        """Check if current conditions match expected discovery patterns"""
        context_lower = context.lower()
        
        for pattern_name, discovery in self.expected_discoveries.items():
            # Check if context requirements are met
            requirements_met = all(
                req.lower() in context_lower 
                for req in discovery.context_requirements
            )
            
            # Check discovery conditions
            min_confidence = discovery.discovery_conditions.get('min_confidence', 0.5)
            
            if requirements_met and confidence >= min_confidence:
                # Calculate discovery probability
                base_prob = discovery.probability
                confidence_boost = (confidence - min_confidence) * 0.2
                final_probability = min(1.0, base_prob + confidence_boost)
                
                # Update discovery
                discovery.last_occurrence = datetime.now()
                
                self._notify_callbacks({
                    'type': 'expected_discovery',
                    'pattern_name': pattern_name,
                    'probability': final_probability,
                    'confidence': confidence,
                    'context': context
                })
    
    def _update_quality_trend(self):
        """Update the overall quality trend"""
        if len(self.accuracy_window) < 10:
            return
        
        recent_values = list(self.accuracy_window)[-10:]
        older_values = list(self.accuracy_window)[-20:-10] if len(self.accuracy_window) >= 20 else recent_values
        
        recent_avg = np.mean(recent_values)
        older_avg = np.mean(older_values)
        
        if recent_avg > older_avg + 0.05:
            self.quality_trend = "improving"
        elif recent_avg < older_avg - 0.05:
            self.quality_trend = "declining"
        else:
            self.quality_trend = "stable"
    
    def get_average_accuracy(self) -> float:
        """Get current average accuracy"""
        if not self.accuracy_window:
            return 0.0
        return np.mean(list(self.accuracy_window))
    
    def get_average_confidence(self) -> float:
        """Get current average confidence"""
        if not self.confidence_window:
            return 0.0
        return np.mean(list(self.confidence_window))
    
    def get_accuracy_trend(self, window: int = 50) -> List[float]:
        """Get recent accuracy trend"""
        if len(self.accuracy_window) < window:
            return list(self.accuracy_window)
        return list(self.accuracy_window)[-window:]
    
    def get_confidence_trend(self, window: int = 50) -> List[float]:
        """Get recent confidence trend"""
        if len(self.confidence_window) < window:
            return list(self.confidence_window)
        return list(self.confidence_window)[-window:]
    
    def get_source_quality_summary(self) -> Dict[str, Any]:
        """Get summary of all source qualities"""
        summary = {}
        
        for source_id, source in self.source_qualities.items():
            # Calculate freshness based on last update
            time_since_update = datetime.now() - source.last_updated
            freshness_decay = max(0.0, 1.0 - (time_since_update.total_seconds() / 3600))  # Decay over 1 hour
            current_freshness = source.freshness_score * freshness_decay
            
            summary[source_id] = {
                'reliability_score': source.reliability_score,
                'freshness_score': current_freshness,
                'trust_level': source.trust_level,
                'validation_count': source.validation_count,
                'last_updated': source.last_updated.isoformat(),
                'accuracy_trend': source.accuracy_history[-10:] if len(source.accuracy_history) >= 10 else source.accuracy_history,
                'change_frequency': source.change_frequency
            }
        
        return summary
    
    def get_expected_discoveries_summary(self) -> Dict[str, Any]:
        """Get summary of expected discoveries"""
        summary = {}
        
        for pattern_name, discovery in self.expected_discoveries.items():
            summary[pattern_name] = {
                'probability': discovery.probability,
                'confidence_interval': discovery.confidence_interval,
                'context_requirements': discovery.context_requirements,
                'last_occurrence': discovery.last_occurrence.isoformat() if discovery.last_occurrence else None,
                'discovery_conditions': discovery.discovery_conditions
            }
        
        return summary
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get all metrics in a comprehensive summary"""
        return {
            'accuracy_metrics': {
                'current': self.current_accuracy,
                'average': self.get_average_accuracy(),
                'trend': self.get_accuracy_trend(20),
                'quality_trend': self.quality_trend
            },
            'confidence_metrics': {
                'current': self.current_confidence,
                'average': self.get_average_confidence(),
                'trend': self.get_confidence_trend(20)
            },
            'performance_metrics': {
                'average_response_time': self.average_response_time,
                'response_time_trend': list(self.response_time_window)[-20:] if len(self.response_time_window) >= 20 else list(self.response_time_window)
            },
            'source_quality': self.get_source_quality_summary(),
            'expected_discoveries': self.get_expected_discoveries_summary(),
            'custom_metrics': {
                name: {
                    'current': values[-1] if values else 0.0,
                    'average': np.mean(list(values)) if values else 0.0,
                    'trend': list(values)[-10:] if len(values) >= 10 else list(values)
                }
                for name, values in self.quality_metrics.items()
            },
            'monitoring_status': {
                'is_active': self.is_monitoring,
                'window_size': self.window_size,
                'total_measurements': len(self.accuracy_window),
                'last_update': datetime.now().isoformat()
            }
        }
    
    def start_monitoring(self, update_interval: float = 1.0):
        """Start continuous monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        
        def monitoring_loop():
            while self.is_monitoring:
                try:
                    # Simulate some metric updates (in real implementation, these would come from actual measurements)
                    self._simulate_metric_updates()
                    # Notify any registered update callbacks (e.g., WebSocket streamer)
                    try:
                        for cb in list(self.update_callbacks):
                            try:
                                cb()
                            except Exception:
                                pass
                    except Exception:
                        pass
                    time.sleep(update_interval)
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Started quality metrics monitoring")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)
        self.logger.info("Stopped quality metrics monitoring")
    
    def _simulate_metric_updates(self):
        """Simulate metric updates for demonstration (remove in production)"""
        # This would be replaced with actual metric collection in a real system
        
        # Simulate accuracy fluctuation
        base_accuracy = 0.85
        accuracy_noise = np.random.normal(0, 0.05)
        simulated_accuracy = max(0.0, min(1.0, base_accuracy + accuracy_noise))
        
        # Simulate confidence fluctuation
        base_confidence = 0.80
        confidence_noise = np.random.normal(0, 0.08)
        simulated_confidence = max(0.0, min(1.0, base_confidence + confidence_noise))
        
        # Simulate response time
        base_response_time = 0.5
        time_noise = np.random.exponential(0.2)
        simulated_response_time = base_response_time + time_noise
        
        # Record simulated metrics
        self.record_accuracy(simulated_accuracy, "simulation")
        self.record_confidence(simulated_confidence, "swarm optimization agent collaboration")
        self.record_response_time(simulated_response_time)
        
        # Update source freshness
        for source in self.source_qualities.values():
            # Simulate gradual freshness decay
            decay_rate = 0.001
            source.freshness_score = max(0.5, source.freshness_score - decay_rate)
    
    def export_metrics(self, filepath: str):
        """Export all metrics to a JSON file"""
        metrics_data = self.get_comprehensive_metrics()
        
        try:
            with open(filepath, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            self.logger.info(f"Metrics exported to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
    
    def reset_metrics(self):
        """Reset all collected metrics"""
        self.accuracy_window.clear()
        self.confidence_window.clear()
        self.response_time_window.clear()
        self.quality_metrics.clear()
        
        self.current_accuracy = 0.0
        self.current_confidence = 0.0
        self.average_response_time = 0.0
        self.quality_trend = "stable"
        
        # Reset source qualities
        for source in self.source_qualities.values():
            source.accuracy_history.clear()
            source.validation_count = 0
        
        # Reset discovery occurrences
        for discovery in self.expected_discoveries.values():
            discovery.last_occurrence = None
        
        self.logger.info("All metrics reset")
