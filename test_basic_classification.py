"""
Test basic smishing classification accuracy on a dataset.
This script can be run independently to test classifier performance.
"""

import ast
from sklearn.metrics import confusion_matrix

from qwen_smishing_classifier import QwenSmishingClassifier


def test_basic_classification(classifier, dataset_path="smishing_data/pwws_attacked_entries_20260204_182519.txt"):
    """Test basic smishing classification accuracy on a dataset."""
    print("=" * 80)
    print("Basic Classification Test")
    print("=" * 80)
    
    # Load dataset
    print(f"\nLoading dataset from: {dataset_path}")
    examples = []
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                text, label = ast.literal_eval(line)  # parses "('text', 0)"
                examples.append((text, int(label)))
    except FileNotFoundError:
        print(f"Error: Dataset file not found: {dataset_path}")
        print("Please provide a valid dataset file path.")
        return
    
    print(f"Loaded {len(examples)} examples from dataset")
    
    # Extract texts and labels
    texts = []
    true_labels = []
    for text, label in examples:
        texts.append(text)
        true_labels.append(label)  # 0 = Legitimate, 1 = Smishing
    
    print(f"\nTesting classifier on {len(texts)} examples...")
    print("This may take a while...")
    
    # Get predictions in batches to avoid memory issues
    batch_size = 10
    all_predictions = []
    all_probs = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_predictions = classifier.predict(batch_texts)
        batch_probs = classifier.predict_proba(batch_texts)
        
        # Convert predictions to labels (0 = Legitimate, 1 = Smishing)
        batch_pred_labels = [1 if pred == "Smishing" else 0 for pred in batch_predictions]
        all_predictions.extend(batch_pred_labels)
        all_probs.extend(batch_probs.tolist())
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {min(i + batch_size, len(texts))} / {len(texts)} examples...")
    
    # Calculate accuracy
    correct = sum(1 for pred, true in zip(all_predictions, true_labels) if pred == true)
    accuracy = correct / len(true_labels) if true_labels else 0.0
    
    # Calculate per-class accuracy
    legitimate_indices = [i for i, label in enumerate(true_labels) if label == 0]
    smishing_indices = [i for i, label in enumerate(true_labels) if label == 1]
    
    legitimate_correct = sum(1 for i in legitimate_indices if all_predictions[i] == 0)
    smishing_correct = sum(1 for i in smishing_indices if all_predictions[i] == 1)
    
    legitimate_accuracy = legitimate_correct / len(legitimate_indices) if legitimate_indices else 0.0
    smishing_accuracy = smishing_correct / len(smishing_indices) if smishing_indices else 0.0
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, all_predictions, labels=[0, 1])
    
    # Print results
    print("\n" + "=" * 80)
    print("CLASSIFICATION RESULTS")
    print("=" * 80)
    print(f"\nOverall Accuracy: {accuracy:.2%} ({correct}/{len(true_labels)})")
    print(f"Legitimate Accuracy: {legitimate_accuracy:.2%} ({legitimate_correct}/{len(legitimate_indices)})")
    print(f"Smishing Accuracy: {smishing_accuracy:.2%} ({smishing_correct}/{len(smishing_indices)})")
    
    print(f"\nConfusion Matrix:")
    print(f"                    Predicted")
    print(f"                  Legitimate  Smishing")
    print(f"  Actual Legitimate    {cm[0][0]:4d}      {cm[0][1]:4d}")
    print(f"  Actual Smishing     {cm[1][0]:4d}      {cm[1][1]:4d}")
    
    # Class distribution
    num_legitimate = len(legitimate_indices)
    num_smishing = len(smishing_indices)
    print(f"\nDataset Distribution:")
    print(f"  Legitimate: {num_legitimate} ({num_legitimate/len(true_labels):.1%})")
    print(f"  Smishing:   {num_smishing} ({num_smishing/len(true_labels):.1%})")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    import sys
    
    # Load model
    print("=" * 80)
    print("Loading Qwen Smishing Classifier")
    print("=" * 80)
    model_wrapper = QwenSmishingClassifier()
    
    # Get dataset path from command line argument if provided
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "smishing_data/pwws_attacked_entries_20260204_182519.txt"
    
    # Run test
    test_basic_classification(model_wrapper, dataset_path)
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)
