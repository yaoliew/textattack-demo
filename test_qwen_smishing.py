"""
Demo script showing how to use QwenSmishingClassifier with TextAttack.
"""

# Disable TensorFlow JIT before any other imports so USE (TextBugger) never sees JIT.
# Must be first so TF is not loaded elsewhere first.
import os
os.environ["TF_XLA_FLAGS"] = os.environ.get("TF_XLA_FLAGS", "") + " --tf_xla_auto_jit=-1"
# Pin this process to physical GPU 1 (so visible cuda:0 maps to GPU 1).
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import ast
import gc
import json
from datetime import datetime

import torch

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
from textattack.datasets import Dataset
from textattack.transformations import WordSwap
from textattack.attack_recipes.pwws_ren_2019 import PWWSRen2019
from textattack.attack_recipes.deepwordbug_gao_2018 import DeepWordBugGao2018
from textattack.attack_recipes.pruthi_2019 import Pruthi2019
from textattack.attack_recipes.textbugger_li_2018 import TextBuggerLi2018
from textattack.attack_recipes.bae_garg_2019 import BAEGarg2019
from textattack.attack_recipes.clare_li_2020 import CLARE2020
from util.attack_metrics_tracker import AttackMetricsTracker


def clear_cuda_memory():
    """Release CUDA cache and run GC to free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def load_tuple_dataset(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text, label = ast.literal_eval(line)  # parses "('text', 0)"
            examples.append((text, int(label)))
    # Return every 10th entry
    return Dataset(examples[601:1013])


class SimpleWordSwap(WordSwap):
    """Simple word swap transformation for testing."""
    def _get_replacement_words(self, word):
        # Return the same word (no actual swapping for demo)
        return [word]


def run_attack_test(model_wrapper, attack_class, attack_name):
    """
    Generic function to run an attack test.
    
    Args:
        model_wrapper: The model wrapper to attack
        attack_class: The attack recipe class (e.g., PWWSRen2019, DeepWordBugGao2018)
        attack_name: Name of the attack for display purposes
    """
    # TextBugger, BAE, and CLARE use TensorFlow sentence encoders (e.g. USE).
    # Disable TF JIT and force TF to CPU so the encoder does not trigger
    # GPU JIT failures (EncoderDNN/Sqrt, libdevice).
    if attack_name in ("TextBugger", "BAE", "CLARE"):
        os.environ["TF_XLA_FLAGS"] = os.environ.get("TF_XLA_FLAGS", "") + " --tf_xla_auto_jit=-1"
        try:
            import tensorflow as tf
            tf.config.optimizer.set_jit(False)
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

    print("\n" + "=" * 80)
    print(f"{attack_name} Attack Test")
    print("=" * 80)

    # Create attack_results folder if it doesn't exist
    results_dir = "attack_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create output file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(results_dir, f"{attack_name.lower()}_results_{timestamp}.json")
    
    test_dataset = load_tuple_dataset("smishing_data/dataset_cleaned_tuples.txt")
    attack = attack_class.build(model_wrapper)
    
    metrics_tracker = AttackMetricsTracker()
    dataset_examples = list(test_dataset)
    
    print(f"Attacking {len(dataset_examples)} entries...")
    print(f"Results will be written to: {output_file}")
    
    results_data = []
    
    for idx, example_tuple in enumerate(dataset_examples):
        # Extract input data first (before try block to ensure it's always available)
        example_input = example_tuple[0]
        original_label = example_tuple[1]
        original_text = example_input['text'] if isinstance(example_input, dict) else str(example_input)
        
        try:
            result = attack.attack(example_input, original_label)
            
            # Get predictions for classification change
            from textattack.attack_results import SuccessfulAttackResult, FailedAttackResult, SkippedAttackResult
            
            original_prediction = None
            perturbed_prediction = None
            
            # Get original prediction
            if hasattr(result, 'original_result') and result.original_result is not None:
                if hasattr(result.original_result, 'output'):
                    output = result.original_result.output
                    if hasattr(output, 'item'):
                        original_prediction = int(output.item())
                    elif isinstance(output, (int, list, tuple)):
                        original_prediction = int(output) if isinstance(output, int) else int(output[0])
            
            # Get perturbed prediction
            if hasattr(result, 'perturbed_result') and result.perturbed_result is not None:
                if hasattr(result.perturbed_result, 'output'):
                    output = result.perturbed_result.output
                    if hasattr(output, 'item'):
                        perturbed_prediction = int(output.item())
                    elif isinstance(output, (int, list, tuple)):
                        perturbed_prediction = int(output) if isinstance(output, int) else int(output[0])
            
            # Determine classification labels
            label_map = {0: "Legitimate", 1: "Smishing"}
            original_class = label_map.get(original_prediction, "Unknown") if original_prediction is not None else "Unknown"
            perturbed_class = label_map.get(perturbed_prediction, "Unknown") if perturbed_prediction is not None else "Unknown"
            
            # Determine if classification changed
            classification_changed = original_prediction != perturbed_prediction if (original_prediction is not None and perturbed_prediction is not None) else None
            
            # Prepare result entry
            result_entry = {
                "entry_number": idx + 1,
                "original_label": original_label,
                "original_text": original_text,
                "perturbed_text": result.perturbed_text() if hasattr(result, 'perturbed_text') else original_text,
                "original_prediction": original_prediction,
                "original_classification": original_class,
                "perturbed_prediction": perturbed_prediction,
                "perturbed_classification": perturbed_class,
                "classification_changed": classification_changed,
                "result_type": type(result).__name__,
                "successful": isinstance(result, SuccessfulAttackResult)
            }
            
            results_data.append(result_entry)
            
            print(f"\n--- Entry {idx + 1} ---")
            print(f"Result: {result}")
            print(f"Original: {original_text}")
            print(f"Perturbed: {result.perturbed_text()}")
            if classification_changed is not None:
                print(f"Classification changed: {classification_changed} ({original_class} -> {perturbed_class})")
            
            metrics_tracker.record_result(result, original_label, original_text)
            
        except Exception as e:
            print(f"\n--- Entry {idx + 1} ---")
            print(f"Attack {idx + 1} failed: {e}")
            
            # Record error in results
            error_entry = {
                "entry_number": idx + 1,
                "error": str(e),
                "original_label": original_label,
                "original_text": original_text
            }
            results_data.append(error_entry)
            continue
    
    # Write all results to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults written to: {output_file}")
    
    metrics_tracker.print_metrics()
    metrics_tracker.print_confusion_matrix()


def test_pwws_attack(model_wrapper):
    """Test PWWS attack."""
    run_attack_test(model_wrapper, PWWSRen2019, "PWWS")


def test_deepwordbug_attack(model_wrapper):
    """Test DeepWordBug attack."""
    run_attack_test(model_wrapper, DeepWordBugGao2018, "DeepWordBug")


def test_pruthi_attack(model_wrapper):
    """Test Pruthi attack."""
    run_attack_test(model_wrapper, Pruthi2019, "Pruthi")


def test_textbugger_attack(model_wrapper):
    """Test TextBugger attack."""
    run_attack_test(model_wrapper, TextBuggerLi2018, "TextBugger")


def test_bae_attack(model_wrapper):
    """Test BAE attack."""
    run_attack_test(model_wrapper, BAEGarg2019, "BAE")


def test_clare_attack(model_wrapper):
    """Test CLARE attack."""
    run_attack_test(model_wrapper, CLARE2020, "CLARE")


if __name__ == "__main__":
    # Load model once and reuse across all tests
    print("=" * 80)
    print("Loading Qwen Smishing Classifier (shared across all tests)")
    print("=" * 80)
    # With CUDA_VISIBLE_DEVICES=1, visible cuda:0 is physical GPU 1.
    model_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_wrapper = QwenSmishingClassifier(device=model_device)
    
    # Run basic classification test
    # test_basic_classification(model_wrapper)

    # Run actual attack tests
    # test_pruthi_attack(model_wrapper)
    # test_bae_attack(model_wrapper)
    test_clare_attack(model_wrapper)
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
    clear_cuda_memory()