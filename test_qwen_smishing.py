"""
Demo script showing how to use QwenSmishingClassifier with TextAttack.
"""

import os
import time
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple

import ast 
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Monkey-patch langdetect to handle "No features in text" errors gracefully
# This must be done before TextAttack imports langdetect
try:
    import langdetect.detector_factory
    from langdetect.lang_detect_exception import LangDetectException
    
    _original_detect = langdetect.detector_factory.detect
    
    def safe_detect(text):
        """Wrapper around langdetect.detect that returns 'en' on errors."""
        try:
            return _original_detect(text)
        except (LangDetectException, Exception):
            # Return 'en' as default for errors like "No features in text"
            return 'en'
    
    # Patch the detect function in langdetect.detector_factory
    langdetect.detector_factory.detect = safe_detect
except (ImportError, AttributeError):
    # If langdetect is not available or patching fails, skip
    pass

from qwen_smishing_classifier import QwenSmishingClassifier
from textattack.goal_functions import UntargetedClassification, TargetedClassification
from textattack.datasets import Dataset
from textattack import Attack, Attacker, AttackArgs
from textattack.search_methods import GreedySearch
from textattack.transformations import WordSwap
from textattack.attack_recipes.pwws_ren_2019 import PWWSRen2019
from textattack.attack_results import AttackResult
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)

def load_tuple_dataset(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text, label = ast.literal_eval(line)  # parses "('text', 0)"
            examples.append((text, int(label)))
    # Limit to first 200 entries
    return Dataset(examples[:100])


class SimpleWordSwap(WordSwap):
    """Simple word swap transformation for testing."""
    def _get_replacement_words(self, word):
        # Return the same word (no actual swapping for demo)
        return [word]


class AttackMetricsTracker:
    """Track comprehensive attack metrics from TextAttack results."""
    
    def __init__(self):
        self.original_labels = []
        self.predicted_labels_original = []  # Predictions on original text
        self.predicted_labels_attacked = []  # Predictions on attacked text
        self.results = []
        self.original_texts = []
        self.attacked_texts = []
    
    def record_result(self, result: AttackResult, original_label: int, original_text: str):
        """Record a single attack result for metrics."""
        from textattack.attack_results import SuccessfulAttackResult, FailedAttackResult, SkippedAttackResult
        
        self.original_labels.append(original_label)
        self.results.append(result)
        self.original_texts.append(original_text)
        
        # Get original prediction (before attack)
        original_predicted_label = self._get_predicted_label(result, use_original=True)
        self.predicted_labels_original.append(original_predicted_label)
        
        # Get attacked prediction (after attack)
        attacked_predicted_label = self._get_predicted_label(result, use_original=False)
        self.predicted_labels_attacked.append(attacked_predicted_label)
        
        # Get attacked text
        if hasattr(result, 'perturbed_text'):
            attacked_text = result.perturbed_text()
        else:
            attacked_text = original_text
        self.attacked_texts.append(attacked_text)
    
    def _get_predicted_label(self, result: AttackResult, use_original: bool = False):
        """Extract predicted label from result."""
        predicted_label = None
        
        # AttackResult has original_result and perturbed_result attributes
        # original_result is the prediction on original text
        # perturbed_result is the prediction on attacked text
        if use_original:
            goal_result = getattr(result, 'original_result', None)
        else:
            goal_result = getattr(result, 'perturbed_result', None)
            # Fallback to goal_function_result if perturbed_result not available
            if goal_result is None:
                goal_result = getattr(result, 'goal_function_result', None)
        
        if goal_result is not None:
            if hasattr(goal_result, 'output'):
                output = goal_result.output
                # Convert to label (0 = Legitimate, 1 = Smishing)
                if isinstance(output, torch.Tensor):
                    if output.numel() == 1:
                        predicted_label = int(output.item())
                    else:
                        predicted_label = int(torch.argmax(output).item())
                elif isinstance(output, (int, np.integer)):
                    predicted_label = int(output)
                elif isinstance(output, (list, tuple, np.ndarray)):
                    predicted_label = int(np.argmax(output))
        
        # Fallback: if we can't get prediction, return None (will be handled in metrics)
        return predicted_label
    
    def get_metrics(self):
        """Calculate and return all metrics."""
        from textattack.attack_results import SuccessfulAttackResult, FailedAttackResult, SkippedAttackResult
        
        if not self.results:
            return None
        
        # Count attack outcomes
        num_successful = sum(1 for r in self.results if isinstance(r, SuccessfulAttackResult))
        num_failed = sum(1 for r in self.results if isinstance(r, FailedAttackResult))
        num_skipped = sum(1 for r in self.results if isinstance(r, SkippedAttackResult))
        
        # Original accuracy (predictions on original text)
        # Filter out None predictions
        original_pairs = [(orig, pred) for orig, pred in zip(self.original_labels, self.predicted_labels_original) if pred is not None]
        original_correct = sum(1 for orig, pred in original_pairs if orig == pred)
        original_accuracy = (original_correct / len(original_pairs) * 100) if original_pairs else 0.0
        
        # Accuracy under attack (predictions on attacked text)
        # Filter out None predictions
        attacked_pairs = [(orig, pred) for orig, pred in zip(self.original_labels, self.predicted_labels_attacked) if pred is not None]
        attacked_correct = sum(1 for orig, pred in attacked_pairs if orig == pred)
        attacked_accuracy = (attacked_correct / len(attacked_pairs) * 100) if attacked_pairs else 0.0
        
        # Attack success rate (successful attacks / total attacks attempted)
        total_attempted = num_successful + num_failed
        attack_success_rate = (num_successful / total_attempted * 100) if total_attempted > 0 else 0.0
        
        # Average perturbed word percentage
        # Use AttackedText objects from TextAttack results for accurate calculation
        perturbed_word_pcts = []
        word_counts = []
        
        for result in self.results:
            # Get AttackedText objects from the result
            original_attacked_text = None
            perturbed_attacked_text = None
            
            # Get original AttackedText
            if hasattr(result, 'original_result') and result.original_result is not None:
                original_attacked_text = getattr(result.original_result, 'attacked_text', None)
            
            # Get perturbed AttackedText
            if hasattr(result, 'perturbed_result') and result.perturbed_result is not None:
                perturbed_attacked_text = getattr(result.perturbed_result, 'attacked_text', None)
            
            # Use AttackedText.words for accurate word-level comparison
            if original_attacked_text is not None and perturbed_attacked_text is not None:
                orig_words = original_attacked_text.words
                
                # Count words in original for word count metric
                word_counts.append(len(orig_words))
                
                # Calculate perturbed word percentage using TextAttack's all_words_diff method
                # This properly handles word alignment and returns indices of changed words
                if len(orig_words) > 0:
                    try:
                        # Use TextAttack's built-in method to get indices of changed words
                        changed_indices = perturbed_attacked_text.all_words_diff(original_attacked_text)
                        num_changed = len(changed_indices)
                        pct = (num_changed / len(orig_words)) * 100
                        perturbed_word_pcts.append(pct)
                    except Exception:
                        # Fallback to manual comparison if all_words_diff fails
                        pert_words = perturbed_attacked_text.words
                        min_len = min(len(orig_words), len(pert_words))
                        num_changed = sum(1 for i in range(min_len) if orig_words[i] != pert_words[i])
                        num_changed += abs(len(pert_words) - len(orig_words))
                        pct = (num_changed / len(orig_words)) * 100
                        perturbed_word_pcts.append(pct)
            else:
                # Fallback: use string-based calculation if AttackedText not available
                idx = len(perturbed_word_pcts)
                if idx < len(self.original_texts):
                    orig_text = self.original_texts[idx]
                    attacked_text = self.attacked_texts[idx]
                    orig_words = orig_text.split()
                    attacked_words = attacked_text.split()
                    
                    word_counts.append(len(orig_words))
                    
                    if len(orig_words) > 0:
                        min_len = min(len(orig_words), len(attacked_words))
                        changed = sum(1 for i in range(min_len) if orig_words[i] != attacked_words[i])
                        changed += abs(len(attacked_words) - len(orig_words))
                        pct = (changed / len(orig_words)) * 100
                        perturbed_word_pcts.append(pct)
        
        avg_perturbed_word_pct = np.mean(perturbed_word_pcts) if perturbed_word_pcts else 0.0
        
        # Average number of words per input (using AttackedText when available)
        if not word_counts:
            # Fallback to string-based calculation
            word_counts = [len(text.split()) for text in self.original_texts]
        avg_words_per_input = np.mean(word_counts) if word_counts else 0.0
        
        # Average number of queries
        query_counts = []
        for result in self.results:
            # Check perturbed_result first (final attack result)
            if hasattr(result, 'perturbed_result') and result.perturbed_result is not None:
                if hasattr(result.perturbed_result, 'num_queries'):
                    query_counts.append(result.perturbed_result.num_queries)
            # Fallback to goal_function_result
            elif hasattr(result, 'goal_function_result') and result.goal_function_result is not None:
                if hasattr(result.goal_function_result, 'num_queries'):
                    query_counts.append(result.goal_function_result.num_queries)
        avg_num_queries = np.mean(query_counts) if query_counts else 0.0
        
        return {
            'num_successful': num_successful,
            'num_failed': num_failed,
            'num_skipped': num_skipped,
            'original_accuracy': original_accuracy,
            'attacked_accuracy': attacked_accuracy,
            'attack_success_rate': attack_success_rate,
            'avg_perturbed_word_pct': avg_perturbed_word_pct,
            'avg_words_per_input': avg_words_per_input,
            'avg_num_queries': avg_num_queries,
        }
    
    def print_metrics(self):
        """Print all attack metrics in a formatted table."""
        metrics = self.get_metrics()
        if metrics is None:
            print("No metrics available.")
            return
        
        print("\n" + "+" + "-" * 31 + "+" + "-" * 8 + "+")
        print("|" + " Attack Results".ljust(31) + "|" + "".ljust(8) + "|")
        print("+" + "-" * 31 + "+" + "-" * 8 + "+")
        print(f"| Number of successful attacks: | {metrics['num_successful']:6d} |")
        print(f"| Number of failed attacks:     | {metrics['num_failed']:6d} |")
        print(f"| Number of skipped attacks:    | {metrics['num_skipped']:6d} |")
        print(f"| Original accuracy:            | {metrics['original_accuracy']:6.2f}% |")
        print(f"| Accuracy under attack:        | {metrics['attacked_accuracy']:6.2f}% |")
        print(f"| Attack success rate:          | {metrics['attack_success_rate']:6.2f}% |")
        print(f"| Average perturbed word %:     | {metrics['avg_perturbed_word_pct']:6.2f}% |")
        print(f"| Average num. words per input: | {metrics['avg_words_per_input']:6.2f} |")
        print(f"| Avg num queries:              | {metrics['avg_num_queries']:6.2f} |")
        print("+" + "-" * 31 + "+" + "-" * 8 + "+")
    
    def get_confusion_matrix(self):
        """Compute and return confusion matrix based on attacked predictions."""
        if not self.original_labels or not self.predicted_labels_attacked:
            return None
        return confusion_matrix(self.original_labels, self.predicted_labels_attacked, labels=[0, 1])
    
    def print_confusion_matrix(self):
        """Print the confusion matrix."""
        cm = self.get_confusion_matrix()
        if cm is None:
            print("No data available for confusion matrix.")
            return
        
        print("\n" + "=" * 80)
        print("CONFUSION MATRIX")
        print("=" * 80)
        print(f"\n                    Predicted")
        print(f"                  Legitimate  Smishing")
        print(f"  Actual Legitimate    {cm[0][0]:4d}      {cm[0][1]:4d}")
        print(f"  Actual Smishing     {cm[1][0]:4d}      {cm[1][1]:4d}")
        print("\n" + "=" * 80)


def test_pwws_attack(model_wrapper):
    """Test PWWS attack with comprehensive debugging."""
    print("\n" + "=" * 80)
    print("PWWS Attack Test")
    print("=" * 80)
    
    # Stage 1: Validate model wrapper
    print("\n[DEBUG] Stage 1: Validating model wrapper...")
    try:
        print(f"[DEBUG] Model wrapper type: {type(model_wrapper)}")
        print(f"[DEBUG] Model wrapper has tokenizer: {hasattr(model_wrapper, 'tokenizer')}")
        if hasattr(model_wrapper, 'tokenizer'):
            print(f"[DEBUG] Tokenizer type: {type(model_wrapper.tokenizer)}")
        print(f"[DEBUG] Model wrapper has __call__: {hasattr(model_wrapper, '__call__')}")
        print(f"[DEBUG] Model wrapper has predict: {hasattr(model_wrapper, 'predict')}")
        print("[DEBUG] ✓ Model wrapper validation passed")
    except Exception as e:
        print(f"[DEBUG] ✗ Model wrapper validation failed: {e}")
        raise
    
    # Stage 2: Load dataset
    print("\n[DEBUG] Stage 2: Loading dataset...")
    try:
        test_dataset = load_tuple_dataset("smishing_data/dataset_cleaned_tuples.txt")
        print(f"[DEBUG] Dataset type: {type(test_dataset)}")
        print(f"[DEBUG] Dataset size: {len(test_dataset)}")
        if len(test_dataset) > 0:
            print(f"[DEBUG] First example: {test_dataset[0]}")
        print("[DEBUG] ✓ Dataset loading passed")
    except Exception as e:
        print(f"[DEBUG] ✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Stage 3: Build attack recipe
    print("\n[DEBUG] Stage 3: Building PWWS attack recipe...")
    try:
        print("[DEBUG] Attempting to build PWWSRen2019 attack...")
        attack = PWWSRen2019.build(model_wrapper)
        print(f"[DEBUG] Attack type: {type(attack)}")
        print(f"[DEBUG] Attack components:")
        print(f"  - Goal function: {type(attack.goal_function)}")
        print(f"  - Transformation: {type(attack.transformation)}")
        print(f"  - Constraints: {[type(c) for c in attack.constraints]}")
        print(f"  - Search method: {type(attack.search_method)}")
        print(f"[DEBUG] Is black box: {getattr(attack, 'is_black_box', 'Unknown')}")
        print("[DEBUG] ✓ Attack recipe building passed")
    except Exception as e:
        print(f"[DEBUG] ✗ Attack recipe building failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Stage 4: Create attack arguments
    print("\n[DEBUG] Stage 4: Creating attack arguments...")
    try:
        # Use first 200 entries from dataset
        dataset_size = len(test_dataset)
        attack_args = AttackArgs(num_examples=dataset_size)
        print(f"[DEBUG] Attack args type: {type(attack_args)}")
        print(f"[DEBUG] Number of examples: {attack_args.num_examples} (first 200 entries)")
        print("[DEBUG] ✓ Attack arguments creation passed")
    except Exception as e:
        print(f"[DEBUG] ✗ Attack arguments creation failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Stage 5: Create attacker
    print("\n[DEBUG] Stage 5: Creating attacker...")
    try:
        attacker = Attacker(attack, test_dataset, attack_args)
        print(f"[DEBUG] Attacker type: {type(attacker)}")
        print(f"[DEBUG] Attacker has attack: {hasattr(attacker, 'attack')}")
        print(f"[DEBUG] Attacker has dataset: {hasattr(attacker, 'dataset')}")
        print("[DEBUG] ✓ Attacker creation passed")
    except Exception as e:
        print(f"[DEBUG] ✗ Attacker creation failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Stage 6: Initialize metrics tracker
    print("\n[DEBUG] Stage 6: Initializing metrics tracker...")
    metrics_tracker = AttackMetricsTracker()
    
    # Stage 7: Run attack with metrics tracking
    print("\n[DEBUG] Stage 7: Running attack on dataset with metrics tracking...")
    print("[DEBUG] This may take a while as it queries the model multiple times...")
    
    # Get dataset examples for labels (before attack starts)
    dataset_examples = list(test_dataset)
    
    results_list = []
    attack_start_time = time.time()
    
    # Attack each entry individually to handle errors gracefully
    # This avoids issues with attack_dataset() failing during logging
    print(f"[DEBUG] Attacking {len(dataset_examples)} entries individually...")
    
    for idx, example_tuple in enumerate(dataset_examples):
        try:
            # Extract text and label from example tuple
            example_input = example_tuple[0]  # This is the OrderedDict or text
            original_label = example_tuple[1]
            original_text = example_input['text'] if isinstance(example_input, dict) else str(example_input)
            
            # Attack this single example using the attack object directly
            # This avoids logging issues that occur with attack_dataset()
            # Pass the input (OrderedDict or text) and label separately
            result = attack.attack(example_input, original_label)
            print(f"[DEBUG] Attack {idx + 1} result: {result}")
            print(f"[DEBUG] Attack {idx + 1} original text: {original_text}")
            print(f"\n Attack {idx + 1} perturbed text: {result.perturbed_text()}")
            # Record result for metrics tracking
            metrics_tracker.record_result(result, original_label, original_text)
            
            results_list.append(result)

            
        except Exception as e:
            print(f"[DEBUG] ✗ Attack {idx + 1} failed: {e}")
            print(f"[DEBUG] Continuing to next attack...")
            import traceback
            traceback.print_exc()
            
            # Continue to next entry
            continue
    
    print(f"[DEBUG] Total results processed: {len(results_list)}")
    print("[DEBUG] ✓ Attack execution and metrics recording completed")
    
    # Stage 8: Display attack metrics
    print("\n[DEBUG] Stage 8: Displaying attack metrics...")
    try:
        metrics_tracker.print_metrics()
        print("[DEBUG] ✓ Attack metrics display passed")
    except Exception as e:
        print(f"[DEBUG] ✗ Attack metrics display failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Stage 9: Display confusion matrix
    print("\n[DEBUG] Stage 9: Displaying confusion matrix...")
    try:
        metrics_tracker.print_confusion_matrix()
        print("[DEBUG] ✓ Confusion matrix display passed")
    except Exception as e:
        print(f"[DEBUG] ✗ Confusion matrix display failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("\n[DEBUG] All stages completed successfully!")
    
if __name__ == "__main__":
    # Load model once and reuse across all tests
    print("=" * 80)
    print("Loading Qwen Smishing Classifier (shared across all tests)")
    print("=" * 80)
    model_wrapper = QwenSmishingClassifier()
    
    # Run basic classification test
    # test_basic_classification(model_wrapper)

    # Run PWWS attack test - errors are handled internally
    test_pwws_attack(model_wrapper)
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)

