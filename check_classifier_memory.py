"""
Script to check memory usage in the actual QwenSentimentClassifier.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
from qwen_sentiment_classifier import QwenSentimentClassifier

print("=" * 80)
print("GPU Memory Analysis for QwenSentimentClassifier")
print("=" * 80)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    initial_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
    initial_reserved = torch.cuda.memory_reserved(0) / 1024**3  # GB
    print(f"\nInitial GPU memory:")
    print(f"  Allocated: {initial_allocated:.2f} GB")
    print(f"  Reserved: {initial_reserved:.2f} GB")
    
    # Load classifier
    print("\nLoading QwenSentimentClassifier...")
    classifier = QwenSentimentClassifier()
    
    # Check memory after loading
    after_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
    after_reserved = torch.cuda.memory_reserved(0) / 1024**3  # GB
    model_memory = after_allocated - initial_allocated
    
    print(f"\nAfter loading classifier:")
    print(f"  Allocated: {after_allocated:.2f} GB")
    print(f"  Reserved: {after_reserved:.2f} GB")
    print(f"  Model memory: {model_memory:.2f} GB")
    
    # Check model size
    model = classifier.model
    print(f"\nModel information:")
    print(f"  Model type: {type(model).__name__}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Check parameter sizes by component
    print(f"\nMemory breakdown by component:")
    embed_size = sum(p.numel() * p.element_size() for name, p in model.named_parameters() if 'embed' in name) / 1024**3
    layer_size = sum(p.numel() * p.element_size() for name, p in model.named_parameters() if 'layers' in name) / 1024**3
    norm_size = sum(p.numel() * p.element_size() for name, p in model.named_parameters() if 'norm' in name) / 1024**3
    lm_head_size = sum(p.numel() * p.element_size() for name, p in model.named_parameters() if 'lm_head' in name) / 1024**3
    
    print(f"  Embeddings: {embed_size:.2f} GB")
    print(f"  Transformer layers: {layer_size:.2f} GB")
    print(f"  Layer norms: {norm_size:.2f} GB")
    print(f"  LM head: {lm_head_size:.2f} GB")
    print(f"  Total parameters: {embed_size + layer_size + norm_size + lm_head_size:.2f} GB")
    
    # Check device placement
    gpu_params = sum(1 for p in model.parameters() if p.device.type == 'cuda')
    cpu_params = sum(1 for p in model.parameters() if p.device.type == 'cpu')
    print(f"\nDevice placement:")
    print(f"  Parameters on GPU: {gpu_params}")
    print(f"  Parameters on CPU: {cpu_params}")
    
    # Run a test inference
    print(f"\nRunning test inference...")
    test_text = "I love this movie!"
    before_inference = torch.cuda.memory_allocated(0) / 1024**3
    logits = classifier([test_text])
    after_inference = torch.cuda.memory_allocated(0) / 1024**3
    inference_memory = after_inference - before_inference
    
    print(f"  Memory before inference: {before_inference:.2f} GB")
    print(f"  Memory after inference: {after_inference:.2f} GB")
    print(f"  Inference overhead: {inference_memory:.2f} GB")
    
    # Clear and check
    torch.cuda.empty_cache()
    after_clear = torch.cuda.memory_allocated(0) / 1024**3
    print(f"  Memory after cache clear: {after_clear:.2f} GB")
    
    print(f"\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print(f"Model on GPU: {after_allocated:.2f} GB")
    print(f"After inference: {after_inference:.2f} GB")
    print(f"After cache clear: {after_clear:.2f} GB")


