"""
Demo script showing how to use QwenSentimentClassifier with TextAttack.
"""

import os
# Suppress OpenBLAS warnings about OpenMP
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
from qwen_sentiment_classifier import QwenSentimentClassifier
from textattack.goal_functions import UntargetedClassification, TargetedClassification
from textattack.datasets import Dataset
from textattack import Attack, Attacker, AttackArgs
from textattack.search_methods import GreedySearch
from textattack.transformations import WordSwap
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)


class SimpleWordSwap(WordSwap):
    """Simple word swap transformation for testing."""
    def _get_replacement_words(self, word):
        # Return the same word (no actual swapping for demo)
        return [word]


def test_basic_classification(classifier):
    """Test basic sentiment classification."""
    print("=" * 80)
    print("Basic Classification Test")
    print("=" * 80)
    
    test_texts = [
        "I love this movie! It's fantastic!",
        "This is terrible. I hate it.",
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
    
    # Create a simple dataset (modeled after main.py line 59)
    test_dataset = Dataset([
        ("Although the rain had stopped hours earlier and the city lights reflected faintly off the damp pavement like scattered constellations, he continued walking without checking the time, replaying conversations he wished had gone differently, imagining alternate futures that branched endlessly from minor choices, and wondering whether meaning was something patiently discovered through effort or merely imposed afterward to make uncertainty feel survivable.", 1)
        # ("I love this product!", 1),  # (text, label) - label 1 = Positive
        # ("This is awful.", 0),        # label 0 = Negative
        # ("The movie was okay.", 1),   # Neutral-positive example
    ])
    
    # Attack initialization (modeled after main.py lines 61-65)
    transformation = SimpleWordSwap()
    constraints = [RepeatModification(), StopwordModification()]
    search_method = GreedySearch()
    attack = Attack(goal_function, constraints, transformation, search_method)
    
    # Apply attack on the model (modeled after main.py lines 67-70)
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
    
    # Print results (modeled after main.py lines 72-78)
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
    print("Loading Qwen Sentiment Classifier (shared across all tests)")
    print("=" * 80)
    model_wrapper = QwenSentimentClassifier()
    print("\nModel loaded! Reusing this instance for all tests.\n")
    
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

