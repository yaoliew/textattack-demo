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
    return Dataset(examples[:200])


class SimpleWordSwap(WordSwap):
    """Simple word swap transformation for testing."""
    def _get_replacement_words(self, word):
        # Return the same word (no actual swapping for demo)
        return [word]


class AttackMetricsTracker:
    """Track comprehensive metrics for adversarial attacks."""
    
    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper
        self.results = []
        self.attack_times = []
        self.original_texts = []
        self.adversarial_texts = []
        self.original_labels = []
        self.adversarial_labels = []
        self.clean_predictions = []
        self.robust_predictions = []
        self.success_flags = []
        self.num_queries = []
        self.perturbed_word_percentages = []
        
    def compute_embedding_distance(self, text1: str, text2: str) -> Tuple[float, float]:
        """
        Compute L2 and L∞ norms between embeddings of two texts.
        
        Args:
            text1: Original text
            text2: Adversarial text
            
        Returns:
            Tuple of (L2_norm, L∞_norm)
        """
        try:
            # Get embeddings from the model's tokenizer and model
            tokenizer = self.model_wrapper.tokenizer
            
            # Tokenize both texts
            tokens1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True, max_length=512)
            tokens2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Get embeddings from model (if available)
            # For Qwen, we'll use the model's embedding layer
            if hasattr(self.model_wrapper, 'model') and hasattr(self.model_wrapper.model, 'model'):
                with torch.no_grad():
                    # Get word embeddings
                    emb1 = self.model_wrapper.model.model.embed_tokens(tokens1['input_ids'].to(self.model_wrapper.device))
                    emb2 = self.model_wrapper.model.model.embed_tokens(tokens2['input_ids'].to(self.model_wrapper.device))
                    
                    # Average pool to get sentence embeddings
                    emb1 = emb1.mean(dim=1).squeeze().cpu()
                    emb2 = emb2.mean(dim=1).squeeze().cpu()
                    
                    # Compute L2 norm (Euclidean distance)
                    l2_norm = torch.norm(emb1 - emb2, p=2).item()
                    
                    # Compute L∞ norm (max absolute difference)
                    linf_norm = torch.norm(emb1 - emb2, p=float('inf')).item()
                    
                    return l2_norm, linf_norm
            else:
                # Fallback: use token-level differences
                # Simple approximation using token IDs
                ids1 = tokens1['input_ids'].squeeze().cpu().float()
                ids2 = tokens2['input_ids'].squeeze().cpu().float()
                
                # Pad to same length
                max_len = max(len(ids1), len(ids2))
                ids1_padded = F.pad(ids1, (0, max_len - len(ids1)), value=0)
                ids2_padded = F.pad(ids2, (0, max_len - len(ids2)), value=0)
                
                diff = ids1_padded - ids2_padded
                l2_norm = torch.norm(diff, p=2).item()
                linf_norm = torch.norm(diff, p=float('inf')).item()
                
                return l2_norm, linf_norm
        except Exception as e:
            print(f"Warning: Could not compute embedding distance: {e}")
            return 0.0, 0.0
    
    def record_attack_result(self, result: AttackResult, original_text: str, 
                            original_label: int, attack_time: float):
        """Record a single attack result."""
        self.attack_times.append(attack_time)
        self.original_texts.append(original_text)
        self.original_labels.append(original_label)
        self.num_queries.append(result.num_queries if hasattr(result, 'num_queries') else 0)
        
        # Get clean prediction
        clean_pred = self.model_wrapper.predict([original_text])[0]
        clean_pred_label = 1 if clean_pred == "Smishing" else 0
        self.clean_predictions.append(clean_pred_label)
        
        # Check if attack was successful and get adversarial text
        # TextAttack uses goal_function_result.achieved() or result.goal_function_result.goal_status
        if hasattr(result, 'goal_function_result'):
            succeeded = result.goal_function_result.achieved() if hasattr(result.goal_function_result, 'achieved') else False
        elif hasattr(result, 'succeeded'):
            succeeded = result.succeeded()
        else:
            succeeded = False
        
        # Get adversarial text
        if hasattr(result, 'perturbed_text'):
            adversarial_text = result.perturbed_text()
        elif hasattr(result, 'perturbed_result'):
            adversarial_text = result.perturbed_result.text if hasattr(result.perturbed_result, 'text') else original_text
        else:
            adversarial_text = original_text
        
        # Verify success: text must be different and prediction must change
        if succeeded and adversarial_text != original_text:
            # Get robust prediction
            robust_pred = self.model_wrapper.predict([adversarial_text])[0]
            robust_pred_label = 1 if robust_pred == "Smishing" else 0
            # Only count as success if prediction actually changed
            if robust_pred_label != clean_pred_label:
                self.success_flags.append(True)
                self.adversarial_texts.append(adversarial_text)
                self.robust_predictions.append(robust_pred_label)
                self.adversarial_labels.append(robust_pred_label)
            else:
                # Attack claimed success but prediction didn't change
                succeeded = False
                self.success_flags.append(False)
                self.adversarial_texts.append(original_text)
                self.robust_predictions.append(clean_pred_label)
                self.adversarial_labels.append(clean_pred_label)
        else:
            # Attack failed
            self.success_flags.append(False)
            self.adversarial_texts.append(original_text)
            self.robust_predictions.append(clean_pred_label)
            self.adversarial_labels.append(clean_pred_label)
        
        # Compute perturbed word percentage (for all cases)
        if adversarial_text != original_text:
            if hasattr(result, 'perturbed_word_percentage'):
                self.perturbed_word_percentages.append(result.perturbed_word_percentage())
            elif hasattr(result, 'num_words_changed'):
                # Use num_words_changed if available
                words_orig = original_text.split()
                self.perturbed_word_percentages.append(100.0 * result.num_words_changed / max(len(words_orig), 1))
            else:
                # Approximate: count word differences
                words_orig = original_text.split()
                words_adv = adversarial_text.split()
                num_changed = sum(1 for w1, w2 in zip(words_orig, words_adv) if w1 != w2)
                self.perturbed_word_percentages.append(100.0 * num_changed / max(len(words_orig), 1))
        else:
            self.perturbed_word_percentages.append(0.0)
        
        self.results.append(result)
    
    def compute_metrics(self) -> Dict:
        """Compute all metrics from recorded results."""
        if not self.results:
            return {}
        
        metrics = {}
        
        # Clean accuracy (accuracy on original examples)
        correct_clean = sum(1 for i in range(len(self.original_labels)) 
                           if self.clean_predictions[i] == self.original_labels[i])
        metrics['clean_accuracy'] = correct_clean / len(self.original_labels) if self.original_labels else 0.0
        
        # Robust accuracy (accuracy on adversarial examples)
        correct_robust = sum(1 for i in range(len(self.original_labels)) 
                           if self.robust_predictions[i] == self.original_labels[i])
        metrics['robust_accuracy'] = correct_robust / len(self.original_labels) if self.original_labels else 0.0
        
        # Attack Success Rate (ASR)
        successful_attacks = sum(self.success_flags)
        metrics['asr'] = successful_attacks / len(self.success_flags) if self.success_flags else 0.0
        
        # Average L2 and L∞ norms (only for successful attacks)
        l2_norms = []
        linf_norms = []
        for i, success in enumerate(self.success_flags):
            if success:
                l2, linf = self.compute_embedding_distance(
                    self.original_texts[i], 
                    self.adversarial_texts[i]
                )
                l2_norms.append(l2)
                linf_norms.append(linf)
        
        metrics['avg_l2_norm'] = np.mean(l2_norms) if l2_norms else 0.0
        metrics['avg_linf_norm'] = np.mean(linf_norms) if linf_norms else 0.0
        
        # Attack time per sample
        metrics['avg_attack_time_per_sample'] = np.mean(self.attack_times) if self.attack_times else 0.0
        metrics['total_attack_time'] = sum(self.attack_times)
        
        # Confusion matrix under attack
        if self.original_labels and self.robust_predictions:
            metrics['confusion_matrix'] = confusion_matrix(
                self.original_labels, 
                self.robust_predictions,
                labels=[0, 1]
            )
        
        # Additional metrics
        metrics['avg_num_queries'] = np.mean(self.num_queries) if self.num_queries else 0.0
        metrics['avg_perturbed_word_percentage'] = np.mean(self.perturbed_word_percentages) if self.perturbed_word_percentages else 0.0
        metrics['num_successful_attacks'] = successful_attacks
        metrics['num_total_attacks'] = len(self.success_flags)
        
        # Success vs ε curve (using perturbed word percentage as proxy for ε)
        if self.perturbed_word_percentages:
            # Bin by perturbation percentage
            epsilon_bins = np.linspace(0, max(self.perturbed_word_percentages) + 1, 11)
            success_by_epsilon = []
            epsilon_centers = []
            
            for i in range(len(epsilon_bins) - 1):
                bin_start = epsilon_bins[i]
                bin_end = epsilon_bins[i + 1]
                epsilon_center = (bin_start + bin_end) / 2
                
                # Find attacks in this bin
                in_bin = [j for j, pct in enumerate(self.perturbed_word_percentages) 
                         if bin_start <= pct < bin_end]
                
                if in_bin:
                    success_rate = sum(self.success_flags[j] for j in in_bin) / len(in_bin)
                    success_by_epsilon.append(success_rate)
                    epsilon_centers.append(epsilon_center)
            
            metrics['success_vs_epsilon'] = {
                'epsilon': epsilon_centers,
                'success_rate': success_by_epsilon
            }
        
        return metrics
    
    def print_metrics(self):
        """Print all computed metrics in a formatted way."""
        metrics = self.compute_metrics()
        
        print("\n" + "=" * 80)
        print("ATTACK METRICS SUMMARY")
        print("=" * 80)
        
        print(f"\n📊 Accuracy Metrics:")
        print(f"  Clean Accuracy:     {metrics.get('clean_accuracy', 0.0):.2%}")
        print(f"  Robust Accuracy:    {metrics.get('robust_accuracy', 0.0):.2%}")
        print(f"  Attack Success Rate (ASR): {metrics.get('asr', 0.0):.2%}")
        
        print(f"\n📏 Distance Metrics:")
        print(f"  Average L2 Norm:    {metrics.get('avg_l2_norm', 0.0):.4f}")
        print(f"  Average L∞ Norm:    {metrics.get('avg_linf_norm', 0.0):.4f}")
        
        print(f"\n⏱️  Timing Metrics:")
        print(f"  Average Attack Time/Sample: {metrics.get('avg_attack_time_per_sample', 0.0):.2f} seconds")
        print(f"  Total Attack Time:  {metrics.get('total_attack_time', 0.0):.2f} seconds")
        
        print(f"\n🔍 Attack Statistics:")
        print(f"  Successful Attacks: {metrics.get('num_successful_attacks', 0)} / {metrics.get('num_total_attacks', 0)}")
        print(f"  Average Queries:     {metrics.get('avg_num_queries', 0.0):.1f}")
        print(f"  Avg Perturbed Words: {metrics.get('avg_perturbed_word_percentage', 0.0):.2f}%")
        
        if 'confusion_matrix' in metrics:
            print(f"\n📋 Confusion Matrix (Under Attack):")
            cm = metrics['confusion_matrix']
            print(f"                    Predicted")
            print(f"                  Legitimate  Smishing")
            print(f"  Actual Legitimate    {cm[0][0]:4d}      {cm[0][1]:4d}")
            print(f"  Actual Smishing     {cm[1][0]:4d}      {cm[1][1]:4d}")
        
        if 'success_vs_epsilon' in metrics and metrics['success_vs_epsilon']['epsilon']:
            print(f"\n📈 Success vs ε (Perturbation Budget) Curve:")
            for eps, success in zip(metrics['success_vs_epsilon']['epsilon'], 
                                   metrics['success_vs_epsilon']['success_rate']):
                print(f"  ε = {eps:5.2f}%: Success Rate = {success:.2%}")
        
        print("\n" + "=" * 80)


def test_basic_classification(classifier):
    """Test basic smishing classification."""
    print("=" * 80)
    print("Basic Classification Test")
    print("=" * 80)
    
    test_texts = [
        "Your package arrives at the Cyprus Post Office tomorrow.Confirm delivery: https://51.fi/aJzP",
        "Shipped: Your Amazon package with Old Spice High Endurance Deodorant will be delivered Tue, May 24. Track at http://a.co/4SJitSA",
    ]
    
    # Test __call__ method (TextAttack interface)
    logits = classifier(test_texts)
    print(f"\nLogits shape: {logits.shape}")  # Should be (2, 2)
    print(f"Logits:\n{logits}")
    
    # Test predictions
    predictions = classifier.predict(test_texts)
    print(f"\nPredictions: {predictions}")
    
    # Test probabilities
    probs = classifier.predict_proba(test_texts)
    print(f"\nProbabilities:\n{probs}")

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
        # Use full dataset - don't limit num_examples
        dataset_size = len(test_dataset)
        attack_args = AttackArgs(num_examples=dataset_size)
        print(f"[DEBUG] Attack args type: {type(attack_args)}")
        print(f"[DEBUG] Number of examples: {attack_args.num_examples} (full dataset)")
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
    metrics_tracker = AttackMetricsTracker(model_wrapper)
    
    # Stage 7: Run attack with metrics tracking
    print("\n[DEBUG] Stage 7: Running attack on dataset with metrics tracking...")
    print("[DEBUG] This may take a while as it queries the model multiple times...")
    
    # Get dataset examples for labels (before attack starts)
    dataset_examples = list(test_dataset)
    
    # Run attacks and measure time for each
    attack_results = attacker.attack_dataset()
    print(f"[DEBUG] Attack results type: {type(attack_results)}")
    print(f"[DEBUG] Processing results and recording metrics...")
    
    results_list = []
    attack_start_time = time.time()
    successful_attacks = 0
    failed_attacks = 0
    
    for idx, result in enumerate(attack_results):
        try:
            # Measure time for this attack (time since last result or start)
            if idx == 0:
                attack_time = time.time() - attack_start_time
            else:
                attack_time = time.time() - last_result_time
            last_result_time = time.time()
            
            # Get original text and label from dataset
            if idx < len(dataset_examples):
                example = dataset_examples[idx]
                original_text = example[0]['text'] if isinstance(example[0], dict) else str(example[0])
                original_label = example[1]
            else:
                # Fallback: extract from result
                original_text = result.original_text() if hasattr(result, 'original_text') else ""
                original_label = result.ground_truth_output if hasattr(result, 'ground_truth_output') else 0
            
            # Record metrics
            metrics_tracker.record_attack_result(
                result, 
                original_text, 
                original_label, 
                attack_time
            )
            
            results_list.append(result)
            successful_attacks += 1
            
        except Exception as e:
            failed_attacks += 1
            print(f"[DEBUG] ✗ Attack {idx + 1} failed: {e}")
            print(f"[DEBUG] Continuing to next attack...")
            # Continue to next attack instead of raising
            continue
    
    print(f"[DEBUG] Number of successful results: {successful_attacks}")
    print(f"[DEBUG] Number of failed attacks: {failed_attacks}")
    print(f"[DEBUG] Total results processed: {len(results_list)}")
    print("[DEBUG] ✓ Attack execution and metrics recording completed")
    
    # Stage 8: Compute and display metrics
    print("\n[DEBUG] Stage 8: Computing and displaying metrics...")
    try:
        metrics_tracker.print_metrics()
        print("[DEBUG] ✓ Metrics computation and display passed")
    except Exception as e:
        print(f"[DEBUG] ✗ Metrics computation failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("\n[DEBUG] All stages completed successfully!")
    
def test_textattack_integration(model_wrapper):
    """Test integration with TextAttack - modeled after main.py."""
    print("\n" + "=" * 80)
    print("TextAttack Integration Test")
    print("=" * 80)
    
    # Create goal function (modeled after main.py line 58)
    goal_function = UntargetedClassification(model_wrapper)
    
    # load dataset (0 = Legitimate, 1 = Smishing/Spam)
    test_dataset = load_tuple_dataset("smishing_data/dataset_cleaned_tuples.txt")
    # test_
    # Attack initialization 
    transformation = SimpleWordSwap()
    constraints = [RepeatModification(), StopwordModification()]
    search_method = GreedySearch()
    attack = Attack(goal_function, constraints, transformation, search_method)
    
    # Apply attack on the model
    print("\nClearing CUDA cache before attack...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
        reserved = torch.cuda.memory_reserved(0) / 1024**2  # MB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
        free = total - reserved
        print(f"CUDA cache cleared. Memory: {allocated:.1f} MB allocated, {reserved:.1f} MB reserved, {free:.1f} MB free")
    
    print("\nRunning attack on test dataset...")
    attack_args = AttackArgs(num_examples=3)  # Test on all 3 examples
    attacker = Attacker(attack, test_dataset, attack_args)
    attack_results = attacker.attack_dataset()
    
    # Print results
    print("\n" + "="*80)
    print("Attack Results")
    print("="*80)
    for i, result in enumerate(attack_results, 1):
        print(f"\n{'='*45} Result {i} {'='*45}")
        print(str(result) if hasattr(result, '__str__') else result)
        print()


def test_simple_attack(model_wrapper):
    """Test a simple attack setup (without running full attack)."""
    print("\n" + "=" * 80)
    print("Simple Attack Setup Test")
    print("=" * 80)
    
    # Create goal function
    goal_function = UntargetedClassification(model_wrapper)
    
    # Create simple transformation (word swap)
    transformation = SimpleWordSwap()
    
    # Create constraints
    constraints = [RepeatModification(), StopwordModification()]
    
    # Create search method
    search_method = GreedySearch()
    
    # Create attack
    attack = Attack(goal_function, constraints, transformation, search_method)
    
    print("\nAttack created successfully!")
    print("The wrapper is compatible with TextAttack's attack framework.")
    print("\nNote: This test only creates the attack object.")
    print("See test_textattack_integration() for a full attack execution.")


if __name__ == "__main__":
    # Load model once and reuse across all tests
    print("=" * 80)
    print("Loading Qwen Smishing Classifier (shared across all tests)")
    print("=" * 80)
    model_wrapper = QwenSmishingClassifier()
    
    # Run basic classification test
    # test_basic_classification(model_wrapper)
    
    # Clear CUDA cache between tests
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    
    try: 
        test_pwws_attack(model_wrapper)
    except Exception as e:
        print(f"\nError in PWWS attack test: {e}")
        print("This might be due to missing dependencies or model loading issues.")
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)

