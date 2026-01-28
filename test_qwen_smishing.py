"""
Demo script showing how to use QwenSmishingClassifier with TextAttack.
"""

import os

import ast 
import torch
from qwen_smishing_classifier import QwenSmishingClassifier
from textattack.goal_functions import UntargetedClassification, TargetedClassification
from textattack.datasets import Dataset
from textattack import Attack, Attacker, AttackArgs
from textattack.search_methods import GreedySearch
from textattack.transformations import WordSwap
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
    return Dataset(examples[:3])


class SimpleWordSwap(WordSwap):
    """Simple word swap transformation for testing."""
    def _get_replacement_words(self, word):
        # Return the same word (no actual swapping for demo)
        return [word]


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
    test_basic_classification(model_wrapper)
    
    # Clear CUDA cache between tests
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Run TextAttack integration test
    try:
        test_textattack_integration(model_wrapper)
    except Exception as e:
        print(f"\nError in TextAttack integration test: {e}")
        print("This might be due to missing dependencies or model loading issues.")
    
    # Test attack setup
    try:
        test_simple_attack(model_wrapper)
    except Exception as e:
        print(f"\nError in attack setup test: {e}")
        print("This might be due to missing dependencies.")
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)

