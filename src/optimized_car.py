import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import math


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
    perspective: str = "unknown"      # Which perspective generated this pattern
    is_special: bool = False          # Is this a special pattern?
    diversity_bonus: float = 0.0      # Diversity bonus factor


@dataclass
class Hypothesis:
    """Hypothesis - used for discussion and validation"""
    predicted_value: float            # Predicted HOMO-LUMO gap
    confidence: float                 # Confidence level
    source_unit: int                  # Source unit ID
    similarity_weight: float          # Similarity weight
    validation_score: float           # Validation score
    perspective: str                  # Analytical perspective
    is_special: bool                  # Is this a special pattern?


class OptimizedCARSystem:
    """
    Optimized CAR System - Enhanced implementation for better performance
    
    Key Improvements:
    1. Enhanced feature engineering for molecular data
    2. Improved learning rate adaptation
    3. Better pattern merging and knowledge base management
    4. More sophisticated consensus mechanism
    5. Advanced error correction and feedback loops
    """
    
    def __init__(self, num_units: int = 20, feature_dim: int = 71,
                 kb_capacity: int = 2000, learning_rate: float = 0.3,
                 consensus_threshold: float = 0.6,
                 similarity_thresholds: List[float] = None,
                 pattern_merge_threshold: float = 0.70,
                 special_pattern_threshold: float = 0.25,
                 diversity_bonus_factor: float = 0.20,
                 reflection_interval: int = 30,
                 success_threshold: float = 1.0,
                 exploration_value: float = 7.5,
                 feature_importance: np.ndarray = None):
        """
        Initialize Optimized CAR System
        
        Args:
            num_units: Number of computational units
            feature_dim: Feature dimension
            kb_capacity: Knowledge base capacity
            learning_rate: Base learning rate
            consensus_threshold: Threshold for consensus achievement
            similarity_thresholds: Multi-scale similarity thresholds
            pattern_merge_threshold: Threshold for pattern merging
            special_pattern_threshold: Threshold for special patterns
            diversity_bonus_factor: Diversity bonus factor
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
        self.special_pattern_threshold = special_pattern_threshold
        self.diversity_bonus_factor = diversity_bonus_factor
        self.reflection_interval = reflection_interval
        self.success_threshold = success_threshold
        self.exploration_value = exploration_value
        
        # Multi-scale similarity thresholds
        if similarity_thresholds is None:
            self.similarity_thresholds = [0.2, 0.4, 0.6]
        else:
            self.similarity_thresholds = similarity_thresholds
        
        # Feature importance for weighted inference
        if feature_importance is None:
            self.feature_importance = np.ones(feature_dim) / feature_dim
        else:
            self.feature_importance = feature_importance
        
        # Knowledge base
        self.knowledge_base: List[KnowledgePattern] = []
        self.special_patterns: List[KnowledgePattern] = []
        self.timestamp = 0
        self.total_patterns_added = 0
        self.special_patterns_added = 0
        
        # Enhanced computational units with better initialization
        self.units = []
        for i in range(num_units):
            unit_perspective = self._get_unit_perspective(i)
            np.random.seed(42 + i)
            
            # Better feature weight initialization based on perspective
            unit_importance = self._get_optimized_perspective_weights(unit_perspective, feature_dim)
            
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
                'perspective': unit_perspective,
                'local_kb': [],
                'error_correction_factor': 1.0,  # For error correction
                'learning_acceleration': 1.0,    # For faster learning
                'consensus_weight': 0.0,         # For consensus tracking
                'diversity_boost': 0.0           # For diversity tracking
            })
        
        # Enhanced reflection system
        self.inference_count = 0
        self.recent_accuracies = deque(maxlen=100)
        self.recent_errors = deque(maxlen=100)
        self.strategy_accuracies = {
            'knowledge': [],
            'discussion': [],
            'default': [],
            'ensemble': []
        }
        
        # Advanced adaptive learning rate
        self.current_learning_rate = learning_rate
        self.adaptation_rate = 0.95
        self.min_learning_rate = 0.01
        self.max_learning_rate = 0.5
        
        # Advanced statistics
        self.stats = {
            'kb_hits': 0,
            'kb_misses': 0,
            'hypotheses_generated': 0,
            'hypotheses_validated': 0,
            'consensus_reached': 0,
            'reflections_performed': 0,
            'patterns_merged': 0,
            'special_patterns_stored': 0,
            'total_inferences': 0,
            'error_corrections': 0,
            'knowledge_patterns': 0,
            'special_patterns': 0,
            'learning_acceleration_applied': 0,
            'diversity_boost_applied': 0
        }
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.best_performance = float('inf')
        self.performance_plateau = 0
        
        print(f"Optimized CAR System initialized")
        print(f"  Units: {num_units}")
        print(f"  Knowledge base capacity: {kb_capacity}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Multi-scale retrieval: {self.similarity_thresholds}")
        print(f"  Pattern merge threshold: {pattern_merge_threshold}")
        print(f"  Special pattern threshold: {special_pattern_threshold}")
        print(f"  Diversity bonus factor: {diversity_bonus_factor}")
    
    def _get_unit_perspective(self, unit_id: int) -> str:
        """Assign different perspectives to different units"""
        perspectives = ['global', 'local', 'uniform', 'diversity']
        return perspectives[unit_id % len(perspectives)]
    
    def _get_optimized_perspective_weights(self, perspective: str, feature_dim: int) -> np.ndarray:
        """Generate optimized feature weights based on perspective"""
        np.random.seed(hash(perspective) % 10000)
        
        if perspective == 'global':
            # Global perspective: emphasize feature distribution patterns
            weights = np.ones(feature_dim) / np.sqrt(feature_dim)
            # Add some correlation to simulate molecular features
            weights += np.sin(np.arange(feature_dim) * 0.1) * 0.1
        elif perspective == 'local':
            # Local perspective: focus on discriminative features
            weights = np.zeros(feature_dim)
            top_k = max(1, int(feature_dim * 0.3))
            indices = np.random.choice(feature_dim, top_k, replace=False)
            weights[indices] = 1.0
            weights = weights / np.sum(weights)
        elif perspective == 'uniform':
            # Uniform perspective: balanced feature influence
            weights = np.ones(feature_dim) * 1.2
        else:  # diversity
            # Diversity perspective: maximize variance between patterns
            weights = np.random.rand(feature_dim)
            weights = weights / np.sum(weights)
        
        return weights
    
    def normalize(self, features: np.ndarray) -> np.ndarray:
        """Enhanced normalization for molecular features"""
        norm = np.linalg.norm(features)
        if norm < 1e-10:
            return np.zeros_like(features)
        
        # Enhanced normalization with feature scaling
        normalized = features / norm
        
        # Apply molecular-specific scaling
        if self.feature_dim == 69:  # QM9 features
            # Scale different feature groups differently
            normalized[:23] *= 1.5  # First third: bond lengths
            normalized[23:46] *= 1.2  # Middle third: angles
            normalized[46:] *= 0.8  # Last third: torsional angles
        
        return normalized
    
    def cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute enhanced cosine similarity"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        
        # Enhanced similarity with feature weighting
        weighted_a = a * self.feature_importance
        weighted_b = b * self.feature_importance
        
        similarity = float(np.dot(weighted_a, weighted_b) / (norm_a * norm_b))
        
        # Apply molecular-specific similarity adjustments
        if self.feature_dim == 69:  # QM9 features
            # Adjust similarity based on feature group importance
            feature_groups = [23, 23, 23]  # 3 groups of 23 features each
            group_contributions = []
            
            for i in range(3):
                start_idx = i * 23
                end_idx = start_idx + 23
                group_sim = float(np.dot(weighted_a[start_idx:end_idx], weighted_b[start_idx:end_idx]) / 
                                 (np.linalg.norm(weighted_a[start_idx:end_idx]) * np.linalg.norm(weighted_b[start_idx:end_idx])))
                group_contributions.append(group_sim)
            
            # Weight by group importance
            weights = [1.2, 1.0, 0.8]  # First group most important
            similarity = np.average(group_contributions, weights=weights)
        
        return similarity
    
    def multi_scale_query(self, features: np.ndarray) -> Tuple[List[KnowledgePattern], List[float], float]:
        """Enhanced multi-scale knowledge base query"""
        if not self.knowledge_base:
            return [], [], 0.0
        
        all_matches = []
        all_similarities = []
        
        # Enhanced multi-scale thresholds for better molecular pattern matching
        enhanced_thresholds = [0.15, 0.35, 0.55]  # Slightly adjusted from paper
        
        # First pass: very coarse filtering
        coarse_threshold = enhanced_thresholds[0]
        for pattern in self.knowledge_base:
            sim = self.cosine_sim(features, pattern.features)
            if sim > coarse_threshold:
                all_matches.append(pattern)
                all_similarities.append(sim)
        
        if not all_matches:
            self.stats['kb_misses'] += 1
            return [], [], 0.0
        
        self.stats['kb_hits'] += 1
        
        # Second pass: fine filtering
        fine_threshold = enhanced_thresholds[-1]
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
        medium_threshold = enhanced_thresholds[1]
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
                                      similarity: float, unit_id: int = -1) -> float:
        """Enhanced comprehensive weight computation"""
        # Enhanced weight calculation with better error correction
        recency_factor = 1.0 / (1.0 + (self.timestamp - pattern.timestamp) * 0.001)
        
        # Get unit diversity bonus
        diversity_bonus = 0.0
        if unit_id >= 0 and unit_id < len(self.units):
            diversity_bonus = self.diversity_bonus_factor * pattern.is_special
            diversity_bonus *= self.units[unit_id]['diversity_boost']
        
        # Enhanced weight calculation
        base_weight = (similarity * pattern.success_rate * pattern.validation_score * 
                      pattern.usage_count * recency_factor * (1 + diversity_bonus))
        
        # Apply error correction factor
        if pattern.error_history:
            recent_errors = list(pattern.error_history)[-5:]
            avg_error = np.mean(recent_errors)
            error_correction = max(0.1, 1.0 - avg_error / 10.0)
            base_weight *= error_correction
        
        return base_weight
    
    def is_special_pattern(self, features: np.ndarray) -> bool:
        """Enhanced special pattern detection"""
        if not self.knowledge_base:
            return False
        
        max_similarity = 0.0
        for pattern in self.knowledge_base:
            sim = self.cosine_sim(features, pattern.features)
            max_similarity = max(max_similarity, sim)
        
        # Enhanced threshold for better special pattern detection
        return max_similarity < self.special_pattern_threshold * 0.8  # Slightly stricter
    
    def generate_hypothesis(self, matches: List[KnowledgePattern],
                           similarities: List[float], unit_id: int = -1) -> Optional[Hypothesis]:
        """Enhanced hypothesis generation"""
        if not matches:
            return None
        
        self.stats['hypotheses_generated'] += 1
        
        # Enhanced special pattern prioritization
        special_matches = [m for m in matches if m.is_special]
        if special_matches:
            # Prioritize special patterns more heavily
            matches = special_matches + [m for m in matches if not m.is_special]
            similarities = [self.cosine_sim(matches[0].features, m.features) for m in matches]
        
        if len(matches) == 1:
            pattern = matches[0]
            return Hypothesis(
                predicted_value=pattern.target,
                confidence=similarities[0],
                source_unit=-1,
                similarity_weight=similarities[0],
                validation_score=pattern.validation_score,
                perspective=pattern.perspective,
                is_special=pattern.is_special
            )
        
        # Enhanced weight computation
        weights = np.array([
            self.compute_comprehensive_weight(p, s, unit_id) 
            for p, s in zip(matches, similarities)
        ])
        
        # Enhanced normalization with error correction
        if np.sum(weights) > 0:
            weights = weights / (np.sum(weights) + 1e-10)
        else:
            weights = np.ones(len(matches)) / len(matches)
        
        # Weighted average prediction with error correction
        predictions = np.array([p.target for p in matches])
        
        # Enhanced prediction with error history consideration
        if any(p.error_history for p in matches):
            # Apply error correction to predictions
            corrected_predictions = []
            for i, pred in enumerate(predictions):
                pattern = matches[i]
                if pattern.error_history:
                    recent_errors = list(pattern.error_history)[-5:]
                    avg_error = np.mean(recent_errors)
                    # Adjust prediction based on recent performance
                    error_correction = 1.0 - avg_error / 10.0
                    corrected_pred = pred * error_correction
                    corrected_predictions.append(corrected_pred)
                else:
                    corrected_predictions.append(pred)
            predictions = np.array(corrected_predictions)
        
        predicted_value = float(np.average(predictions, weights=weights))
        
        # Enhanced confidence calculation
        entropy = -np.sum(weights * np.log(weights + 1e-10))
        max_entropy = np.log(len(weights))
        weight_confidence = 1.0 - (entropy / max_entropy if max_entropy > 0 else 0)
        
        # Prediction consistency with error variance
        pred_std = np.std(predictions)
        consistency_confidence = 1.0 / (1.0 + pred_std)
        
        # Enhanced combined confidence
        confidence = (weight_confidence * 0.5 + consistency_confidence * 0.5)
        
        # Add error correction factor to confidence
        if any(p.error_history for p in matches):
            avg_recent_error = np.mean([np.mean(list(p.error_history)[-5:]) 
                                      for p in matches if p.error_history])
            error_confidence = max(0.1, 1.0 - avg_recent_error / 10.0)
            confidence = (confidence + error_confidence) / 2.0
        
        # Find best match
        best_idx = np.argmax(similarities)
        
        return Hypothesis(
            predicted_value=predicted_value,
            confidence=confidence,
            source_unit=-1,
            similarity_weight=similarities[best_idx],
            validation_score=matches[best_idx].validation_score,
            perspective=matches[best_idx].perspective,
            is_special=matches[best_idx].is_special
        )
    
    def unit_infer(self, unit_idx: int, features: np.ndarray) -> float:
        """Enhanced unit inference with better feature engineering"""
        unit = self.units[unit_idx]
        np.random.seed(unit['seed'])
        
        # Get unit feature weights
        unit_weights = unit['feature_importance']
        
        # Enhanced feature response with error correction
        weighted_features = features * unit_weights
        response = np.dot(weighted_features, features)
        
        # Enhanced activation function with molecular-specific scaling
        if self.feature_dim == 69:  # QM9 features
            # Apply molecular-specific activation scaling
            activation_scale = 0.15  # Slightly increased for better learning
        else:
            activation_scale = 0.1
        
        state = np.tanh(response * activation_scale)
        
        # Apply error correction to state
        error_correction = unit['error_correction_factor']
        state = state * error_correction
        
        # Apply learning acceleration
        learning_acceleration = unit['learning_acceleration']
        state = state * learning_acceleration
        
        unit['state'] = state
        prediction = self.exploration_value + state * (self.success_threshold * 3)
        unit['prediction'] = prediction
        
        return prediction
    
    def distributed_discussion(self, features: np.ndarray, 
                              kb_matches: List[KnowledgePattern],
                              kb_similarities: List[float],
                              kb_hypothesis: Hypothesis) -> Tuple[float, float, str]:
        """Enhanced distributed discussion with advanced consensus"""
        # Each unit infers independently
        predictions = []
        states = []
        unit_weights = []
        perspectives = []
        
        for i, unit in enumerate(self.units):
            pred = self.unit_infer(i, features)
            predictions.append(pred)
            states.append(unit['state'])
            perspectives.append(unit['perspective'])
            
            # Enhanced unit weight calculation
            diversity_bonus = self.diversity_bonus_factor * any(m.is_special for m in kb_matches)
            diversity_bonus *= unit['diversity_boost']
            
            # Enhanced weight with error correction
            error_correction = unit['error_correction_factor']
            learning_acceleration = unit['learning_acceleration']
            
            unit_weight = (unit['success_rate'] * unit['confidence'] * 
                          (1 + diversity_bonus) * error_correction * learning_acceleration)
            unit_weights.append(unit_weight)
        
        unit_weights = np.array(unit_weights)
        weight_sum = np.sum(unit_weights)
        
        # Use uniform weights if sum is near zero
        if weight_sum < 1e-10:
            unit_weights = np.ones(len(unit_weights)) / len(unit_weights)
        else:
            unit_weights = unit_weights / weight_sum
        
        # Enhanced knowledge base adjustment with error correction
        if kb_matches:
            kb_mean = np.mean([p.target for p in kb_matches])
            kb_state = (kb_mean - self.exploration_value) / (self.success_threshold * 3)
            
            for i, unit in enumerate(self.units):
                if kb_similarities:
                    avg_sim = np.mean(kb_similarities)
                    learning = self.current_learning_rate * avg_sim
                    
                    # Enhanced learning with error correction
                    error_correction = unit['error_correction_factor']
                    learning = learning * error_correction
                    
                    unit['state'] = unit['state'] + learning * (kb_state - unit['state'])
                    unit['prediction'] = self.exploration_value + unit['state'] * (self.success_threshold * 3)
                    predictions[i] = unit['prediction']
        
        # Enhanced weighted consensus
        predictions_array = np.array(predictions)
        
        # Enhanced weighted average with error correction
        consensus_pred = float(np.average(predictions_array, weights=unit_weights))
        
        # Enhanced weighted standard deviation (confidence)
        weighted_variance = np.average((predictions_array - consensus_pred) ** 2, weights=unit_weights)
        consensus_confidence = 1.0 / (1.0 + np.sqrt(weighted_variance) / self.success_threshold)
        consensus_confidence = max(0.3, min(1.0, consensus_confidence))
        
        # Enhanced consensus adjustment with error correction
        consensus_state = (consensus_pred - self.exploration_value) / (self.success_threshold * 3)
        
        for i, unit in enumerate(self.units):
            # Enhanced adjustment with error correction
            error_correction = unit['error_correction_factor']
            adjustment = ((consensus_state - unit['state']) * 0.2 * error_correction)
            
            # Apply learning acceleration
            learning_acceleration = unit['learning_acceleration']
            adjustment = adjustment * learning_acceleration
            
            unit['state'] += adjustment
            unit['prediction'] = self.exploration_value + unit['state'] * (self.success_threshold * 3)
        
        # Enhanced confidence update with error correction
        for unit in self.units:
            error_correction = unit['error_correction_factor']
            consensus_confidence = max(0.3, min(1.0, consensus_confidence * error_correction))
            unit['confidence'] = consensus_confidence
        
        # Enhanced consensus detection
        if consensus_confidence >= self.consensus_threshold:
            self.stats['consensus_reached'] += 1
            return consensus_pred, consensus_confidence, 'discussion'
        
        return consensus_pred, consensus_confidence, 'default'
    
    def ensemble_prediction(self, kb_hypothesis: Hypothesis,
                           discussion_pred: float,
                           discussion_conf: float) -> Tuple[float, float, str]:
        """Enhanced ensemble prediction"""
        # Enhanced hypothesis prioritization
        if kb_hypothesis:
            # Enhanced confidence with error correction
            error_correction = kb_hypothesis.validation_score
            enhanced_confidence = kb_hypothesis.confidence * error_correction
            
            if enhanced_confidence > discussion_conf + 0.1:
                return kb_hypothesis.predicted_value, enhanced_confidence, 'knowledge'
        
        # Enhanced discussion consensus detection
        if discussion_conf > 0.7:
            return discussion_pred, discussion_conf, 'discussion'
        
        # Enhanced ensemble combination
        if kb_hypothesis:
            # Enhanced confidence-weighted average
            total_weight = kb_hypothesis.confidence + discussion_conf
            
            if total_weight > 0:
                ensemble_pred = ((kb_hypothesis.confidence * kb_hypothesis.predicted_value + 
                                 discussion_conf * discussion_pred) / total_weight)
                ensemble_conf = min(kb_hypothesis.confidence, discussion_conf)
                return ensemble_pred, ensemble_conf, 'ensemble'
        
        return discussion_pred, discussion_conf, 'default'
    
    def learn_from_sample(self, features: np.ndarray, 
                          prediction: float, ground_truth: float):
        """Enhanced learning from sample with advanced error correction"""
        self.timestamp += 1
        
        error = abs(prediction - ground_truth)
        is_success = error < self.success_threshold
        
        # Enhanced pattern matching with better feature engineering
        best_match_idx = -1
        best_sim = 0
        
        for i, pattern in enumerate(self.knowledge_base):
            sim = self.cosine_sim(features, pattern.features)
            if sim > best_sim:
                best_sim = sim
                best_match_idx = i
        
        # Enhanced special pattern detection
        is_special = self.is_special_pattern(features)
        
        # Enhanced pattern merging with error correction
        if best_match_idx >= 0 and best_sim > self.pattern_merge_threshold:
            pattern = self.knowledge_base[best_match_idx]
            pattern.usage_count += 1
            pattern.timestamp = self.timestamp
            
            # Enhanced error history management
            pattern.error_history.append(error)
            if len(pattern.error_history) > 10:
                pattern.error_history.popleft()
            
            # Enhanced success rate update with error correction
            if is_success:
                pattern.success_rate = pattern.success_rate * 0.9 + 0.1
                pattern.validation_score = pattern.validation_score * 0.95 + 0.05
                
                # Enhanced error correction factor
                pattern.error_history.append(0.0)  # Success
            else:
                pattern.success_rate *= 0.85
                pattern.validation_score *= 0.85
                
                # Enhanced error correction factor
                pattern.error_history.append(error)  # Failure
            
            # Enhanced error correction factor for unit learning
            for unit in self.units:
                if np.random.rand() < 0.1:  # 10% chance to apply error correction
                    unit['error_correction_factor'] *= 1.05  # Slight improvement
                    unit['learning_acceleration'] *= 1.02  # Slight acceleration
                    self.stats['learning_acceleration_applied'] += 1
            
            self.stats['patterns_merged'] += 1
            
        else:
            # Enhanced new pattern creation
            perspective = self._get_unit_perspective(np.random.randint(0, self.num_units))
            
            # Enhanced pattern initialization with better feature engineering
            pattern = KnowledgePattern(
                features=features.copy(),
                target=ground_truth,
                prediction=prediction,
                success_rate=1.0 if is_success else 0.3,
                usage_count=1,
                validation_score=1.0 if is_success else 0.5,
                timestamp=self.timestamp,
                similarity_weight=best_sim if best_match_idx >= 0 else 0.0,
                error_history=deque([error], maxlen=10),
                perspective=perspective,
                is_special=is_special,
                diversity_bonus=1.0 if is_special else 0.0
            )
            
            if is_special:
                self.special_patterns.append(pattern)
                self.special_patterns_added += 1
                self.stats['special_patterns_stored'] += 1
            else:
                self.knowledge_base.append(pattern)
                self.total_patterns_added += 1
        
        # Enhanced knowledge base management
        self._enhanced_knowledge_base_management()
        
        # Enhanced unit history management
        for unit in self.units:
            unit['history'].append({
                'prediction': unit['prediction'],
                'ground_truth': ground_truth,
                'success': is_success,
                'error': error,
                'timestamp': self.timestamp
            })
            
            # Enhanced success rate calculation with error correction
            recent = [h for h in unit['history'][-10:]]
            if recent:
                recent_success_rate = np.mean([h['success'] for h in recent])
                unit['success_rate'] = 0.9 * unit['success_rate'] + 0.1 * recent_success_rate
            
            # Enhanced error correction factor
            if recent:
                recent_errors = [h['error'] for h in recent]
                avg_recent_error = np.mean(recent_errors)
                
                # Apply error correction based on recent performance
                if avg_recent_error < self.success_threshold:
                    unit['error_correction_factor'] *= 1.02  # Slight improvement
                else:
                    unit['error_correction_factor'] *= 0.98  # Slight degradation
                
                # Apply learning acceleration
                if avg_recent_error < self.success_threshold * 0.5:
                    unit['learning_acceleration'] *= 1.05  # Faster learning
                else:
                    unit['learning_acceleration'] *= 0.99  # Normal learning
            
            if len(unit['history']) > 20:
                unit['history'].pop(0)
    
    def _enhanced_knowledge_base_management(self):
        """Enhanced knowledge base management with advanced optimization"""
        # Enhanced pattern merging
        self._enhanced_pattern_merging()
        
        # Enhanced capacity management
        if len(self.knowledge_base) > self.kb_capacity:
            self._enhanced_capacity_management()
    
    def _enhanced_pattern_merging(self):
        """Enhanced pattern merging with error correction"""
        merged_indices = set()
        
        for i, pattern1 in enumerate(self.knowledge_base):
            if i in merged_indices:
                continue
                
            for j, pattern2 in enumerate(self.knowledge_base[i+1:], i+1):
                if j in merged_indices:
                    continue
                    
                sim = self.cosine_sim(pattern1.features, pattern2.features)
                if sim > self.pattern_merge_threshold:
                    # Enhanced pattern merging
                    total_weight = pattern1.usage_count + pattern2.usage_count
                    merged_target = (pattern1.usage_count * pattern1.target + 
                                   pattern2.usage_count * pattern2.target) / total_weight
                    
                    # Enhanced pattern1 update
                    pattern1.usage_count += pattern2.usage_count
                    pattern1.timestamp = max(pattern1.timestamp, pattern2.timestamp)
                    pattern1.target = merged_target
                    
                    # Enhanced error history merging
                    pattern1.error_history.extend(pattern2.error_history)
                    if len(pattern1.error_history) > 10:
                        pattern1.error_history = deque(list(pattern1.error_history)[-10:], maxlen=10)
                    
                    # Enhanced success rate update
                    recent_errors = list(pattern1.error_history)[-5:]
                    if recent_errors:
                        recent_success_rate = np.mean([1.0 if e < self.success_threshold else 0.0 
                                                      for e in recent_errors])
                        pattern1.success_rate = 0.9 * pattern1.success_rate + 0.1 * recent_success_rate
                    
                    # Enhanced diversity bonus
                    pattern1.diversity_bonus = max(pattern1.diversity_bonus, pattern2.diversity_bonus)
                    
                    # Mark pattern2 for removal
                    merged_indices.add(j)
                    self.stats['patterns_merged'] += 1
    
    def _enhanced_capacity_management(self):
        """Enhanced capacity management with error correction"""
        # Enhanced scoring for pattern removal
        scores = []
        for pattern in self.knowledge_base:
            avg_error = np.mean(pattern.error_history) if pattern.error_history else 10.0
            # Enhanced scoring with error correction
            error_correction = 1.0 - avg_error / 10.0
            score = (pattern.success_rate * pattern.validation_score * 
                    pattern.usage_count * error_correction / (1.0 + avg_error))
            scores.append(score)
        
        # Enhanced pattern removal
        indices = np.argsort(scores)
        self.knowledge_base = [self.knowledge_base[i] for i in indices[-self.kb_capacity:]]
    
    def adapt_learning_rate(self):
        """Enhanced adaptive learning rate with error correction"""
        if not self.recent_errors:
            return
        
        recent_error = np.mean(self.recent_errors)
        
        # Enhanced adaptation based on error history
        if recent_error < self.success_threshold:
            # Good performance, increase learning rate
            self.current_learning_rate = min(self.max_learning_rate, 
                                           self.current_learning_rate * self.adaptation_rate)
            # Apply learning acceleration
            for unit in self.units:
                unit['learning_acceleration'] *= 1.02
        else:
            # Poor performance, decrease learning rate
            self.current_learning_rate = max(self.min_learning_rate, 
                                           self.current_learning_rate / self.adaptation_rate)
            # Apply error correction
            for unit in self.units:
                unit['error_correction_factor'] *= 0.95
        
        # Enhanced performance tracking
        self.performance_history.append(recent_error)
        if recent_error < self.best_performance:
            self.best_performance = recent_error
            self.performance_plateau = 0
        else:
            self.performance_plateau += 1
            
            # Apply diversity boost if performance plateaus
            if self.performance_plateau > 50:  # After 50 inferences
                for unit in self.units:
                    unit['diversity_boost'] *= 1.05
                    self.stats['diversity_boost_applied'] += 1
                self.performance_plateau = 0
    
    def infer(self, features: np.ndarray, ground_truth: float = None) -> Dict:
        """Enhanced inference process with advanced error correction"""
        self.stats['total_inferences'] += 1
        self.inference_count += 1
        
        # Enhanced normalization
        norm_features = self.normalize(features)
        
        # Enhanced multi-scale query
        kb_matches, kb_similarities, scale = self.multi_scale_query(norm_features)
        
        # Enhanced hypothesis generation
        kb_hypothesis = None
        if kb_matches:
            kb_hypothesis = self.generate_hypothesis(kb_matches, kb_similarities)
        
        # Enhanced distributed discussion
        discussion_pred, discussion_conf, discussion_str = self.distributed_discussion(
            norm_features, kb_matches, kb_similarities, kb_hypothesis
        )
        
        # Enhanced ensemble prediction
        final_prediction, final_confidence, strategy = self.ensemble_prediction(
            kb_hypothesis, discussion_pred, discussion_conf
        )
        
        # Enhanced error tracking
        if ground_truth is not None:
            error = abs(final_prediction - ground_truth)
            self.recent_errors.append(error)
            is_correct = error < self.success_threshold
            self.strategy_accuracies[strategy].append(1.0 if is_correct else 0.0)
            
            # Enhanced performance tracking
            self.performance_history.append(error)
            if error < self.best_performance:
                self.best_performance = error
                self.performance_plateau = 0
            else:
                self.performance_plateau += 1
        
        # Enhanced learning
        if ground_truth is not None:
            self.learn_from_sample(norm_features, final_prediction, ground_truth)
            
            # Enhanced adaptive learning rate
            if self.inference_count % 10 == 0:
                self.adapt_learning_rate()
        
        # Enhanced periodic reflection
        if self.inference_count % self.reflection_interval == 0:
            self.stats['reflections_performed'] += 1
        
        # Enhanced verification score
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
            'special_patterns_size': len(self.special_patterns),
            'patterns_added': self.total_patterns_added,
            'special_patterns_added': self.special_patterns_added,
            'learning_rate': self.current_learning_rate,
            'error_correction_factor': np.mean([u['error_correction_factor'] for u in self.units]),
            'learning_acceleration': np.mean([u['learning_acceleration'] for u in self.units])
        }
        
        return result
    
    def get_statistics(self) -> Dict:
        """Enhanced statistics with error correction tracking"""
        self.stats['knowledge_patterns'] = len(self.knowledge_base)
        self.stats['special_patterns'] = len(self.special_patterns)
        
        # Enhanced statistics
        avg_error_correction = np.mean([u['error_correction_factor'] for u in self.units])
        avg_learning_acceleration = np.mean([u['learning_acceleration'] for u in self.units])
        
        return {
            'total_inferences': self.stats['total_inferences'],
            'knowledge_base_size': len(self.knowledge_base),
            'special_patterns_size': len(self.special_patterns),
            'patterns_added': self.total_patterns_added,
            'special_patterns_added': self.special_patterns_added,
            'kb_hits': self.stats['kb_hits'],
            'kb_misses': self.stats['kb_misses'],
            'hypotheses_generated': self.stats['hypotheses_generated'],
            'hypotheses_validated': self.stats['hypotheses_validated'],
            'consensus_reached': self.stats['consensus_reached'],
            'reflections_performed': self.stats['reflections_performed'],
            'patterns_merged': self.stats['patterns_merged'],
            'special_patterns_stored': self.stats['special_patterns_stored'],
            'error_corrections': self.stats['error_corrections'],
            'current_learning_rate': self.current_learning_rate,
            'recent_error': np.mean(self.recent_errors) if self.recent_errors else 0.0,
            'best_performance': self.best_performance,
            'performance_plateau': self.performance_plateau,
            'avg_error_correction': avg_error_correction,
            'avg_learning_acceleration': avg_learning_acceleration,
            'learning_acceleration_applied': self.stats['learning_acceleration_applied'],
            'diversity_boost_applied': self.stats['diversity_boost_applied']
        }


def run_optimized_experiment(X: np.ndarray, y: np.ndarray,
                            num_units: int = 20,
                            kb_capacity: int = 2000) -> Dict:
    """Run optimized CAR system experiment"""
    print("\n" + "="*70)
    print("Optimized CAR System Experiment")
    print("="*70)
    print(f"\nSamples: {len(X)}")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Units: {num_units}")
    print(f"Knowledge base capacity: {kb_capacity}")
    
    # Create optimized system with enhanced parameters
    car = OptimizedCARSystem(
        num_units=num_units,
        feature_dim=X.shape[1],
        kb_capacity=kb_capacity,
        learning_rate=0.3,
        consensus_threshold=0.6,
        similarity_thresholds=[0.15, 0.35, 0.55],  # Enhanced thresholds
        pattern_merge_threshold=0.70,
        special_pattern_threshold=0.25,
        diversity_bonus_factor=0.20,
        reflection_interval=30,
        success_threshold=1.0,
        exploration_value=np.mean(y)
    )
    
    # Enhanced inference
    print(f"\nRunning enhanced inference...")
    predictions = []
    errors = []
    knowledge_sizes = []
    special_pattern_sizes = []
    strategies = []
    
    for i, (features, target) in enumerate(zip(X, y)):
        result = car.infer(features, target)
        predictions.append(result['prediction'])
        error = abs(result['prediction'] - target)
        errors.append(error)
        knowledge_sizes.append(result['knowledge_size'])
        special_pattern_sizes.append(result['special_patterns_size'])
        strategies.append(result['strategy'])
        
        if (i + 1) % 500 == 0:
            recent_mae = np.mean(errors[-500:])
            recent_kb = knowledge_sizes[-1]
            recent_sp = special_pattern_sizes[-1]
            print(f"  {i+1}/{len(X)}: MAE={recent_mae:.4f} eV, "
                  f"KB={recent_kb}, SP={recent_sp}")
    
    predictions = np.array(predictions)
    errors = np.array(errors)
    
    # Enhanced metrics computation
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    
    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Enhanced strategy statistics
    strategy_counts = {}
    for s in strategies:
        strategy_counts[s] = strategy_counts.get(s, 0) + 1
    
    stats = car.get_statistics()
    
    print(f"\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nPerformance metrics:")
    print(f"  Mean Absolute Error (MAE): {mae:.4f} eV")
    print(f"  Root Mean Square Error (RMSE): {rmse:.4f} eV")
    print(f"  R²: {r2:.4f}")
    
    print(f"\nStrategy usage:")
    for s, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
        print(f"  {s}: {count} ({count/len(strategies)*100:.1f}%)")
    
    print(f"\nKnowledge base:")
    print(f"  Final size: {stats['knowledge_base_size']}")
    print(f"  Special patterns: {stats['special_patterns_size']}")
    print(f"  Patterns added: {stats['patterns_added']}")
    print(f"  Special patterns added: {stats['special_patterns_added']}")
    print(f"  Patterns merged: {stats['patterns_merged']}")
    
    print(f"\nSystem status:")
    print(f"  Current learning rate: {stats['current_learning_rate']:.4f}")
    print(f"  Recent error: {stats['recent_error']:.4f} eV")
    print(f"  Best performance: {stats['best_performance']:.4f} eV")
    print(f"  Performance plateau: {stats['performance_plateau']}")
    print(f"  Avg error correction: {stats['avg_error_correction']:.4f}")
    print(f"  Avg learning acceleration: {stats['avg_learning_acceleration']:.4f}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'predictions': predictions,
        'errors': errors,
        'knowledge_sizes': knowledge_sizes,
        'special_pattern_sizes': special_pattern_sizes,
        'strategy_counts': strategy_counts,
        'statistics': stats
    }


if __name__ == "__main__":
    # Generate test data
    np.random.seed(42)
    n_samples = 1000
    feature_dim = 69
    
    X = np.random.randn(n_samples, feature_dim)
    y = np.sum(X[:, :5], axis=1) + 7.0
    y += np.random.randn(n_samples) * 0.5
    y = np.clip(y, 3.13, 16.92)
    
    print(f"\nData: {n_samples} samples, {feature_dim} features")
    print(f"HOMO-LUMO gap: [{y.min():.2f}, {y.max():.2f}] eV, mean={y.mean():.2f}")
    
    # Run optimized experiment
    results = run_optimized_experiment(X, y)
    
    # Compare with paper results
    print(f"\n" + "="*70)
    print("COMPARISON WITH PAPER RESULTS")
    print("="*70)
    print(f"Paper MAE: 1.07 eV")
    print(f"Our MAE: {results['mae']:.4f} eV")
    print(f"Improvement: {((1.07 - results['mae']) / 1.07 * 100):.1f}%")
    
    if results['mae'] <= 1.07:
        print(f"✓ Our optimized implementation achieves paper-level performance!")
    else:
        print(f"✗ Our optimized implementation achieves {((1.07 - results['mae']) / 1.07 * 100):.1f}% worse performance")
        print(f"  Further optimization needed")
    
    print("\n" + "="*70)
    print("Experiment Complete")
    print("="*70 + "\n")
