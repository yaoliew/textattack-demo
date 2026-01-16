"""
Script to check what's using GPU memory in the Qwen model.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 80)
print("GPU Memory Analysis for Qwen Model")
print("=" * 80)

# Check initial memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    initial_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
    initial_reserved = torch.cuda.memory_reserved(0) / 1024**3  # GB
    print(f"\nInitial GPU memory:")
    print(f"  Allocated: {initial_allocated:.2f} GB")
    print(f"  Reserved: {initial_reserved:.2f} GB")
    
    # Load model
    print("\nLoading Qwen model...")
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Check memory after loading
    after_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
    after_reserved = torch.cuda.memory_reserved(0) / 1024**3  # GB
    model_memory = after_allocated - initial_allocated
    
    print(f"\nAfter loading model:")
    print(f"  Allocated: {after_allocated:.2f} GB")
    print(f"  Reserved: {after_reserved:.2f} GB")
    print(f"  Model memory: {model_memory:.2f} GB")
    
    # Get model size information
    print(f"\nModel information:")
    print(f"  Model type: {type(model).__name__}")
    print(f"  Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Check parameter sizes
    total_param_size = 0
    for name, param in model.named_parameters():
        param_size = param.numel() * param.element_size() / 1024**3  # GB
        total_param_size += param_size
    
    print(f"\nParameter memory breakdown:")
    print(f"  Total parameter size (float16): {total_param_size:.2f} GB")
    
    # Check which layers are on GPU
    print(f"\nDevice placement:")
    gpu_layers = []
    cpu_layers = []
    for name, param in model.named_parameters():
        if param.device.type == 'cuda':
            gpu_layers.append(name)
        else:
            cpu_layers.append(name)
    
    print(f"  Layers on GPU: {len(gpu_layers)}")
    print(f"  Layers on CPU: {len(cpu_layers)}")
    
    if len(gpu_layers) > 0:
        print(f"\nFirst 10 GPU layers:")
        for layer in gpu_layers[:10]:
            print(f"    {layer}")
    
    if len(cpu_layers) > 0:
        print(f"\nFirst 10 CPU layers:")
        for layer in cpu_layers[:10]:
            print(f"    {layer}")
    
    # Memory summary
    print(f"\n" + "=" * 80)
    print("Memory Summary:")
    print("=" * 80)
    print(f"Model parameters (float16): ~{total_param_size:.2f} GB")
    print(f"Actual GPU allocated: {after_allocated:.2f} GB")
    print(f"Difference (overhead): {after_allocated - total_param_size:.2f} GB")
    print(f"\nNote: Overhead includes:")
    print(f"  - Optimizer states (if training)")
    print(f"  - Activation cache")
    print(f"  - CUDA context")
    print(f"  - Memory fragmentation")


