"""Synthetic data will be used here, please use real_qm9_experiment.py to test. """
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque


@dataclass
class KnowledgePattern:
    """Knowledge pattern - stores successful prediction cases"""
    features: np.ndarray              # Feature vector
    target: float                     # Target value (HOMO-LUMO gap)
    prediction: float                 # Prediction value
    success_rate: float               # Success rate
    usage_count: int                  # Usage count
    validation_score: float           # Validation score
    timestamp: int                    # Creation timestamp
    similarity_weight: float          # Similarity weight
    error_history: deque = field(default_factory=deque)  # Error history


@dataclass
class Hypothesis:
    """Hypothesis - used for discussion and validation"""
    predicted_value: float            # Predicted HOMO-LUMO gap
    confidence: float                 # Confidence level
    source_unit: int                  # Source unit ID
    similarity_weight: float          # Similarity weight
    validation_score: float           # Validation score


class CARSystem:
    """
    CAR System - Complete gradient-free optimization architecture
    
    Key Features:
    1. Multi-scale similarity retrieval
    2. Weighted consensus discussion
    3. Adaptive learning rate
    4. Error prediction correction
    """
    
    def __init__(self, num_units: int = 20, feature_dim: int = 71,
                 kb_capacity: int = 500, learning_rate: float = 0.3,
                 consensus_threshold: float = 0.6,
                 similarity_thresholds: List[float] = None,
                 pattern_merge_threshold: float = 0.80,
                 reflection_interval: int = 30,
                 success_threshold: float = 1.0,
                 exploration_value: float = 7.5,
                 feature_importance: np.ndarray = None):
        """
        Initialize CAR System
        
        Args:
            num_units: Number of computational units
            feature_dim: Feature dimension
            kb_capacity: Knowledge base capacity
            learning_rate: Base learning rate
            consensus_threshold: Threshold for consensus achievement
            similarity_thresholds: Multi-scale similarity thresholds
            pattern_merge_threshold: Threshold for pattern merging
            reflection_interval: Interval for self-reflection
            success_threshold: Error threshold for success
            exploration_value: Default value for exploration
            feature_importance: Feature importance weights
        """
        self.num_units = num_units
        self.feature_dim = feature_dim
        self.kb_capacity = kb_capacity
        self.learning_rate = learning_rate
        self.consensus_threshold = consensus_threshold
        self.pattern_merge_threshold = pattern_merge_threshold
        self.reflection_interval = reflection_interval
        self.success_threshold = success_threshold
        self.exploration_value = exploration_value
        
        # Multi-scale similarity thresholds
        if similarity_thresholds is None:
            self.similarity_thresholds = [0.3, 0.5, 0.7]
        else:
            self.similarity_thresholds = similarity_thresholds
        
        # Feature importance for weighted inference
        if feature_importance is None:
            self.feature_importance = np.ones(feature_dim) / feature_dim
        else:
            self.feature_importance = feature_importance
        
        # Knowledge base
        self.knowledge_base: List[KnowledgePattern] = []
        self.timestamp = 0
        self.total_patterns_added = 0
        
        # Computational units - each unit has different feature weights
        self.units = []
        for i in range(num_units):
            np.random.seed(42 + i)
            # Each unit focuses on different feature subsets
            unit_importance = np.random.rand(feature_dim)
            unit_importance = unit_importance / np.sum(unit_importance)
            
            self.units.append({
                'id': i,
                'seed': np.random.randint(0, 10000),
                'state': 0.0,
                'prediction': exploration_value,
                'confidence': 0.5,
                'success_rate': 0.5,
                'history': [],
                'strategy': 'default',
                'feature_importance': unit_importance,
                'local_kb': []
            })
        
        # Reflection system
        self.inference_count = 0
        self.recent_accuracies = deque(maxlen=100)
        self.recent_errors = deque(maxlen=100)
        self.strategy_accuracies = {
            'knowledge': [],
            'discussion': [],
            'default': [],
            'ensemble': []
        }
        
        # Adaptive learning rate
        self.current_learning_rate = learning_rate
        self.adaptation_rate = 0.95
        
        # Statistics
        self.stats = {
            'kb_hits': 0,
            'kb_misses': 0,
            'hypotheses_generated': 0,
            'hypotheses_validated': 0,
            'consensus_reached': 0,
            'reflections_performed': 0,
            'patterns_merged': 0,
            'total_inferences': 0,
            'error_corrections': 0
        }
        
        print(f"CAR System initialized")
        print(f"  Units: {num_units}")
        print(f"  Knowledge base capacity: {kb_capacity}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Multi-scale retrieval: {self.similarity_thresholds}")
    
    def normalize(self, features: np.ndarray) -> np.ndarray:
        """Normalize feature vector"""
        norm = np.linalg.norm(features)
        if norm < 1e-10:
            return np.zeros_like(features)
        return features / norm
    
    def cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def weighted_cosine_sim(self, a: np.ndarray, b: np.ndarray, 
                           weights: np.ndarray) -> float:
        """Compute weighted cosine similarity"""
        weighted_a = a * weights
        weighted_b = b * weights
        norm_a = np.linalg.norm(weighted_a)
        norm_b = np.linalg.norm(weighted_b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(weighted_a, weighted_b) / (norm_a * norm_b))
    
    def multi_scale_query(self, features: np.ndarray) -> Tuple[List[KnowledgePattern], List[float], float]:
        """
        Multi-scale knowledge base query
        
        Collect matches from multiple similarity thresholds, return best results
        """
        if not self.knowledge_base:
            return [], [], 0.0
        
        all_matches = []
        all_similarities = []
        
        # First pass: coarse filtering (low threshold)
        coarse_threshold = self.similarity_thresholds[0]
        for pattern in self.knowledge_base:
            sim = self.cosine_sim(features, pattern.features)
            if sim > coarse_threshold:
                all_matches.append(pattern)
                all_similarities.append(sim)
        
        if not all_matches:
            self.stats['kb_misses'] += 1
            return [], [], 0.0
        
        self.stats['kb_hits'] += 1
        
        # Second pass: fine filtering (high threshold)
        fine_threshold = self.similarity_thresholds[-1]
        fine_matches = []
        fine_similarities = []
        
        for pattern, sim in zip(all_matches, all_similarities):
            if sim > fine_threshold:
                fine_matches.append(pattern)
                fine_similarities.append(sim)
        
        # Use fine results if available
        if fine_matches:
            return fine_matches, fine_similarities, fine_threshold
        
        # Otherwise use medium threshold results
        medium_threshold = self.similarity_thresholds[1]
        medium_matches = []
        medium_similarities = []
        
        for pattern, sim in zip(all_matches, all_similarities):
            if sim > medium_threshold:
                medium_matches.append(pattern)
                medium_similarities.append(sim)
        
        if medium_matches:
            return medium_matches, medium_similarities, medium_threshold
        
        return all_matches, all_similarities, coarse_threshold
    
    def compute_comprehensive_weight(self, pattern: KnowledgePattern, 
                                      similarity: float) -> float:
        """Compute comprehensive weight considering success rate, usage count, validation score"""
        # Use exponential decay to encourage new patterns
        recency_factor = 1.0 / (1.0 + (self.timestamp - pattern.timestamp) * 0.001)
        return (similarity * pattern.success_rate * pattern.validation_score * 
                pattern.usage_count * recency_factor)
    
    def generate_hypothesis(self, matches: List[KnowledgePattern],
                           similarities: List[float]) -> Optional[Hypothesis]:
        """Generate hypothesis based on knowledge base matches"""
        if not matches:
            return None
        
        self.stats['hypotheses_generated'] += 1
        
        if len(matches) == 1:
            # Single match, use directly
            return Hypothesis(
                predicted_value=matches[0].target,
                confidence=similarities[0],
                source_unit=-1,
                similarity_weight=similarities[0],
                validation_score=matches[0].validation_score
            )
        
        # Compute comprehensive weights
        weights = np.array([
            self.compute_comprehensive_weight(p, s) 
            for p, s in zip(matches, similarities)
        ])
        
        weights = weights / (np.sum(weights) + 1e-10)
        
        # Weighted average prediction
        predictions = np.array([p.target for p in matches])
        predicted_value = float(np.average(predictions, weights=weights))
        
        # Confidence based on:
        # 1. Weight concentration
        # 2. Prediction consistency
        entropy = -np.sum(weights * np.log(weights + 1e-10))
        max_entropy = np.log(len(weights))
        weight_confidence = 1.0 - (entropy / max_entropy if max_entropy > 0 else 0)
        
        # Prediction consistency
        pred_std = np.std(predictions)
        consistency_confidence = 1.0 / (1.0 + pred_std)
        
        # Combined confidence
        confidence = (weight_confidence * 0.6 + consistency_confidence * 0.4)
        
        # Find best match
        best_idx = np.argmax(similarities)
        
        return Hypothesis(
            predicted_value=predicted_value,
            confidence=confidence,
            source_unit=-1,
            similarity_weight=similarities[best_idx],
            validation_score=matches[best_idx].validation_score
        )
    
    def unit_infer(self, unit_idx: int, features: np.ndarray) -> float:
        """
        Unit inference with unit-specific feature weights
        """
        unit = self.units[unit_idx]
        np.random.seed(unit['seed'])
        
        # Get unit feature weights
        unit_weights = unit['feature_importance']
        
        # Weighted feature response
        weighted_features = features * unit_weights
        response = np.dot(weighted_features, features)
        state = np.tanh(response * 0.1)
        
        unit['state'] = state
        prediction = self.exploration_value + state * (self.success_threshold * 3)
        unit['prediction'] = prediction
        
        return prediction
    
    def weighted_discussion(self, features: np.ndarray, 
                           kb_matches: List[KnowledgePattern],
                           kb_similarities: List[float],
                           kb_hypothesis: Hypothesis) -> Tuple[float, float, str]:
        """
        Weighted distributed discussion
        
        Weights based on unit historical performance
        """
        # Each unit infers independently
        predictions = []
        states = []
        unit_weights = []
        
        for i, unit in enumerate(self.units):
            pred = self.unit_infer(i, features)
            predictions.append(pred)
            states.append(unit['state'])
            
            # Unit weight based on historical success rate
            unit_weight = unit['success_rate'] * unit['confidence']
            unit_weights.append(unit_weight)
        
        unit_weights = np.array(unit_weights)
        weight_sum = np.sum(unit_weights)
        
        # Use uniform weights if sum is near zero
        if weight_sum < 1e-10:
            unit_weights = np.ones(len(unit_weights)) / len(unit_weights)
        else:
            unit_weights = unit_weights / weight_sum
        
        # Incorporate knowledge base adjustment
        if kb_matches:
            kb_mean = np.mean([p.target for p in kb_matches])
            kb_state = (kb_mean - self.exploration_value) / (self.success_threshold * 3)
            
            for i, unit in enumerate(self.units):
                if kb_similarities:
                    avg_sim = np.mean(kb_similarities)
                    learning = self.current_learning_rate * avg_sim
                    unit['state'] = unit['state'] + learning * (kb_state - unit['state'])
                    unit['prediction'] = self.exploration_value + unit['state'] * (self.success_threshold * 3)
                    predictions[i] = unit['prediction']
        
        # Weighted consensus
        predictions_array = np.array(predictions)
        
        # Compute weighted average using unit weights
        consensus_pred = float(np.average(predictions_array, weights=unit_weights))
        
        # Compute weighted standard deviation (confidence)
        weighted_variance = np.average((predictions_array - consensus_pred) ** 2, weights=unit_weights)
        consensus_confidence = 1.0 / (1.0 + np.sqrt(weighted_variance) / self.success_threshold)
        consensus_confidence = max(0.3, min(1.0, consensus_confidence))
        
        # Adjust low states toward consensus
        for i, unit in enumerate(self.units):
            if unit['state'] < (consensus_pred - self.exploration_value) / (self.success_threshold * 3) - 0.05:
                adjustment = ((consensus_pred - self.exploration_value) / (self.success_threshold * 3) - 
                             unit['state']) * 0.2
                unit['state'] += adjustment
                unit['prediction'] = self.exploration_value + unit['state'] * (self.success_threshold * 3)
        
        # Update confidence
        for unit in self.units:
            unit['confidence'] = max(0.3, consensus_confidence)
        
        if consensus_confidence >= self.consensus_threshold:
            self.stats['consensus_reached'] += 1
            return consensus_pred, consensus_confidence, 'discussion'
        
        return consensus_pred, consensus_confidence, 'default'
    
    def ensemble_prediction(self, kb_hypothesis: Hypothesis,
                           discussion_pred: float,
                           discussion_conf: float) -> Tuple[float, float, str]:
        """
        Ensemble prediction - combine knowledge base hypothesis and discussion results
        """
        # Use knowledge base hypothesis if confidence is significantly higher
        if kb_hypothesis and kb_hypothesis.confidence > discussion_conf + 0.1:
            return kb_hypothesis.predicted_value, kb_hypothesis.confidence, 'knowledge'
        
        # Use discussion consensus if confidence is high enough
        if discussion_conf > 0.7:
            return discussion_pred, discussion_conf, 'discussion'
        
        # Otherwise ensemble both
        if kb_hypothesis:
            # Confidence-weighted average
            total_weight = kb_hypothesis.confidence + discussion_conf
            if total_weight > 0:
                ensemble_pred = ((kb_hypothesis.confidence * kb_hypothesis.predicted_value + 
                                 discussion_conf * discussion_pred) / total_weight)
                ensemble_conf = min(kb_hypothesis.confidence, discussion_conf)
                return ensemble_pred, ensemble_conf, 'ensemble'
        
        return discussion_pred, discussion_conf, 'default'
    
    def learn_from_sample(self, features: np.ndarray, 
                          prediction: float, ground_truth: float):
        """Learn from sample - update knowledge base"""
        self.timestamp += 1
        
        error = abs(prediction - ground_truth)
        is_success = error < self.success_threshold
        
        # Find best match
        best_match_idx = -1
        best_sim = 0
        
        for i, pattern in enumerate(self.knowledge_base):
            sim = self.cosine_sim(features, pattern.features)
            if sim > best_sim:
                best_sim = sim
                best_match_idx = i
        
        # Merge or create
        if best_match_idx >= 0 and best_sim > self.pattern_merge_threshold:
            pattern = self.knowledge_base[best_match_idx]
            pattern.usage_count += 1
            pattern.timestamp = self.timestamp
            
            # Update error history
            pattern.error_history.append(error)
            if len(pattern.error_history) > 10:
                pattern.error_history.popleft()
            
            if is_success:
                pattern.success_rate = pattern.success_rate * 0.9 + 0.1
                pattern.validation_score = pattern.validation_score * 0.95 + 0.05
            else:
                pattern.success_rate *= 0.85
                pattern.validation_score *= 0.85
            
            self.stats['patterns_merged'] += 1
            
        else:
            pattern = KnowledgePattern(
                features=features.copy(),
                target=ground_truth,
                prediction=prediction,
                success_rate=1.0 if is_success else 0.3,
                usage_count=1,
                validation_score=1.0 if is_success else 0.5,
                timestamp=self.timestamp,
                similarity_weight=best_sim if best_match_idx >= 0 else 0.0,
                error_history=deque([error], maxlen=10)
            )
            self.knowledge_base.append(pattern)
            self.total_patterns_added += 1
        
        # Forget low-utility patterns
        if len(self.knowledge_base) > self.kb_capacity:
            # Compute comprehensive scores
            scores = []
            for pattern in self.knowledge_base:
                avg_error = np.mean(pattern.error_history) if pattern.error_history else 10.0
                score = (pattern.success_rate * pattern.validation_score * 
                        pattern.usage_count / (1.0 + avg_error))
                scores.append(score)
            
            # Remove lowest-scoring patterns
            indices = np.argsort(scores)
            self.knowledge_base = [self.knowledge_base[i] for i in indices[-self.kb_capacity:]]
        
        # Update unit history
        for unit in self.units:
            unit['history'].append({
                'prediction': unit['prediction'],
                'ground_truth': ground_truth,
                'success': is_success,
                'error': error,
                'timestamp': self.timestamp
            })
            
            # Update success rate
            recent = [h for h in unit['history'][-10:]]
            if recent:
                unit['success_rate'] = np.mean([h['success'] for h in recent])
            
            if len(unit['history']) > 20:
                unit['history'].pop(0)
    
    def adapt_learning_rate(self):
        """Adaptively adjust learning rate"""
        if not self.recent_errors:
            return
        
        recent_error = np.mean(self.recent_errors)
        
        if recent_error < self.success_threshold:
            # Good performance, decrease learning rate
            self.current_learning_rate *= self.adaptation_rate
        else:
            # Poor performance, increase learning rate
            self.current_learning_rate = min(0.5, self.current_learning_rate / self.adaptation_rate)
    
    def infer(self, features: np.ndarray, ground_truth: float = None) -> Dict:
        """
        Complete inference process (with learning)
        """
        self.stats['total_inferences'] += 1
        self.inference_count += 1
        
        # Normalize
        norm_features = self.normalize(features)
        
        # Multi-scale knowledge base query
        kb_matches, kb_similarities, scale = self.multi_scale_query(norm_features)
        
        # Generate hypothesis
        kb_hypothesis = None
        if kb_matches:
            kb_hypothesis = self.generate_hypothesis(kb_matches, kb_similarities)
        
        # Weighted discussion
        discussion_pred, discussion_conf, discussion_str = self.weighted_discussion(
            norm_features, kb_matches, kb_similarities, kb_hypothesis
        )
        
        # Ensemble prediction
        final_prediction, final_confidence, strategy = self.ensemble_prediction(
            kb_hypothesis, discussion_pred, discussion_conf
        )
        
        # Record error
        if ground_truth is not None:
            error = abs(final_prediction - ground_truth)
            self.recent_errors.append(error)
            is_correct = error < self.success_threshold
            self.strategy_accuracies[strategy].append(1.0 if is_correct else 0.0)
        
        # Learn (internal feedback)
        if ground_truth is not None:
            self.learn_from_sample(norm_features, final_prediction, ground_truth)
            
            # Adaptive learning rate
            if self.inference_count % 10 == 0:
                self.adapt_learning_rate()
        
        # Periodic reflection
        if self.inference_count % self.reflection_interval == 0:
            self.stats['reflections_performed'] += 1
        
        # Compute verification score
        verification_score = 0.5
        if ground_truth is not None:
            error = abs(final_prediction - ground_truth)
            verification_score = max(0.0, 1.0 - error / 10.0)
            self.stats['error_corrections'] += 1 if error > self.success_threshold else 0
        
        result = {
            'prediction': final_prediction,
            'confidence': final_confidence,
            'strategy': strategy,
            'verification': verification_score,
            'knowledge_size': len(self.knowledge_base),
            'patterns_added': self.total_patterns_added,
            'learning_rate': self.current_learning_rate
        }
        
        return result
    
    def get_statistics(self) -> Dict:
        """Get system statistics"""
        return {
            'total_inferences': self.stats['total_inferences'],
            'knowledge_base_size': len(self.knowledge_base),
            'patterns_added': self.total_patterns_added,
            'kb_hits': self.stats['kb_hits'],
            'kb_misses': self.stats['kb_misses'],
            'hypotheses_generated': self.stats['hypotheses_generated'],
            'hypotheses_validated': self.stats['hypotheses_validated'],
            'consensus_reached': self.stats['consensus_reached'],
            'reflections_performed': self.stats['reflections_performed'],
            'patterns_merged': self.stats['patterns_merged'],
            'error_corrections': self.stats['error_corrections'],
            'current_learning_rate': self.current_learning_rate,
            'recent_error': np.mean(self.recent_errors) if self.recent_errors else 0.0
        }


def run_experiment(X: np.ndarray, y: np.ndarray,
                   num_units: int = 20,
                   kb_capacity: int = 500) -> Dict:
    """
    Run CAR system experiment
    """
    print("\n" + "="*70)
    print("CAR System Experiment")
    print("="*70)
    print(f"\nSamples: {len(X)}")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Units: {num_units}")
    print(f"Knowledge base capacity: {kb_capacity}")
    
    # Create system
    car = CARSystem(
        num_units=num_units,
        feature_dim=X.shape[1],
        kb_capacity=kb_capacity,
        learning_rate=0.3,
        consensus_threshold=0.6,
        similarity_thresholds=[0.3, 0.5, 0.7],
        pattern_merge_threshold=0.80,
        reflection_interval=30,
        success_threshold=1.0,
        exploration_value=np.mean(y)
    )
    
    # Inference
    print(f"\nRunning inference...")
    predictions = []
    errors = []
    knowledge_sizes = []
    strategies = []
    
    for i, (features, target) in enumerate(zip(X, y)):
        result = car.infer(features, target)
        predictions.append(result['prediction'])
        error = abs(result['prediction'] - target)
        errors.append(error)
        knowledge_sizes.append(result['knowledge_size'])
        strategies.append(result['strategy'])
        
        if (i + 1) % 100 == 0:
            recent_mae = np.mean(errors[-100:])
            recent_kb = knowledge_sizes[-1]
            print(f"  {i+1}/{len(X)}: MAE={recent_mae:.4f} eV, "
                  f"KB={recent_kb}, LR={result['learning_rate']:.3f}")
    
    predictions = np.array(predictions)
    errors = np.array(errors)
    
    # Compute metrics
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    
    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Strategy statistics
    strategy_counts = {}
    for s in strategies:
        strategy_counts[s] = strategy_counts.get(s, 0) + 1
    
    stats = car.get_statistics()
    
    print(f"\n" + "="*70)
    print("Results")
    print("="*70)
    print(f"\nPerformance metrics:")
    print(f"  MAE: {mae:.4f} eV")
    print(f"  RMSE: {rmse:.4f} eV")
    print(f"  R2: {r2:.4f}")
    
    print(f"\nStrategy usage:")
    for s, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
        print(f"  {s}: {count} ({count/len(strategies)*100:.1f}%)")
    
    print(f"\nKnowledge base:")
    print(f"  Final size: {stats['knowledge_base_size']}")
    print(f"  Patterns added: {stats['patterns_added']}")
    print(f"  Patterns merged: {stats['patterns_merged']}")
    
    print(f"\nSystem status:")
    print(f"  Current learning rate: {stats['current_learning_rate']:.4f}")
    print(f"  Recent error: {stats['recent_error']:.4f} eV")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'predictions': predictions,
        'errors': errors,
        'knowledge_sizes': knowledge_sizes,
        'strategy_counts': strategy_counts,
        'statistics': stats
    }


def compare_methods(X: np.ndarray, y: np.ndarray):
    """Compare all CAR methods"""
    print("\n" + "="*70)
    print("CAR Methods Comparison")
    print("="*70)
    
    results = {}
    
    # 1. Basic fixed weights
    print("\n[1] Basic fixed-weight CAR...")
    car_basic = CARSystem(
        num_units=20, feature_dim=X.shape[1], kb_capacity=1,
        learning_rate=0.0, consensus_threshold=0.95,
        similarity_thresholds=[0.99], pattern_merge_threshold=0.99,
        reflection_interval=999999, success_threshold=1.0,
        exploration_value=np.mean(y)
    )
    
    preds, errs = [], []
    for f, t in zip(X, y):
        r = car_basic.infer(f)
        preds.append(r['prediction'])
        errs.append(abs(r['prediction'] - t))
    
    results['basic'] = {
        'mae': np.mean(errs),
        'rmse': np.sqrt(np.mean(np.array(errs) ** 2))
    }
    print(f"  MAE: {results['basic']['mae']:.4f} eV")
    
    # 2. Knowledge base CAR
    print("\n[2] Knowledge base CAR...")
    car_kb = CARSystem(
        num_units=10, feature_dim=X.shape[1], kb_capacity=500,
        learning_rate=0.2, consensus_threshold=0.8,
        similarity_thresholds=[0.5], pattern_merge_threshold=0.85,
        reflection_interval=50, success_threshold=1.0,
        exploration_value=np.mean(y)
    )
    
    preds, errs = [], []
    for f, t in zip(X, y):
        r = car_kb.infer(f, t)
        preds.append(r['prediction'])
        errs.append(abs(r['prediction'] - t))
    
    results['knowledge'] = {
        'mae': np.mean(errs),
        'rmse': np.sqrt(np.mean(np.array(errs) ** 2)),
        'kb_size': len(car_kb.knowledge_base)
    }
    print(f"  MAE: {results['knowledge']['mae']:.4f} eV, KB: {results['knowledge']['kb_size']}")
    
    # 3. Full CAR system
    print("\n[3] Full CAR system...")
    car_full = CARSystem(
        num_units=20, feature_dim=X.shape[1], kb_capacity=500,
        learning_rate=0.3, consensus_threshold=0.6,
        similarity_thresholds=[0.3, 0.5, 0.7], pattern_merge_threshold=0.80,
        reflection_interval=30, success_threshold=1.0,
        exploration_value=np.mean(y)
    )
    
    preds, errs = [], []
    for f, t in zip(X, y):
        r = car_full.infer(f, t)
        preds.append(r['prediction'])
        errs.append(abs(r['prediction'] - t))
    
    results['full'] = {
        'mae': np.mean(errs),
        'rmse': np.sqrt(np.mean(np.array(errs) ** 2)),
        'kb_size': len(car_full.knowledge_base),
        'stats': car_full.get_statistics()
    }
    print(f"  MAE: {results['full']['mae']:.4f} eV, KB: {results['full']['kb_size']}")
    
    # Comparison summary
    print("\n" + "="*70)
    print("Comparison Summary")
    print("="*70)
    print(f"\n{'Method':<20} {'MAE (eV)':<15} {'RMSE (eV)':<15}")
    print("-" * 50)
    for name, res in results.items():
        print(f"{name:<20} {res['mae']:<15.4f} {res['rmse']:<15.4f}")
    
    return results


if __name__ == "__main__":
    # Generate test data
    #np.random.seed(42)
    n_samples = 3000
    feature_dim = 71
    
    X = np.random.randn(n_samples, feature_dim)
    y = np.sum(X[:, :5], axis=1) + 7.0
    y += np.random.randn(n_samples) * 0.5
    y = np.clip(y, 3.0, 17.0)
    
    print(f"\nData: {n_samples} samples, {feature_dim} features")
    print(f"HOMO-LUMO gap: [{y.min():.2f}, {y.max():.2f}] eV, mean={y.mean():.2f}")
    
    # Compare methods
    results = compare_methods(X, y)
    
    print("\n" + "="*70)
    print("Experiment Complete")
    print("="*70 + "\n")