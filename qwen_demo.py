import os
# Suppress OpenBLAS warnings about OpenMP
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def demo_qwen2_5_7b_instruct():
    """
    Basic demo of Qwen2.5-7B-Instruct model.
    This script demonstrates how to load and use the model for text generation.
    """
    print("=" * 80)
    print("Qwen2.5-7B-Instruct Demo")
    print("=" * 80)
    
    # Check device availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Model name
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Loading model: {model_name}")
    print("This may take a few minutes on first run...\n")
    
    # Load tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
        )
        
        if device.type == "cpu":
            model = model.to(device)
        
        print("Model loaded successfully!\n")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nNote: Make sure you have enough disk space and memory.")
        print("The model requires approximately 14GB of disk space.")
        return
    
    # Example prompts
    example_prompts = [
        "Explain what artificial intelligence is in simple terms.",
        "Write a short poem about coding.",
        "What are the benefits of renewable energy?",
    ]
    
    print("=" * 80)
    print("Running Inference Examples")
    print("=" * 80)
    
    for i, prompt in enumerate(example_prompts, 1):
        print(f"\n{'='*80}")
        print(f"Example {i}:")
        print(f"Prompt: {prompt}")
        print(f"{'='*80}")
        
        # Format prompt for chat model
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Tokenize input
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.8,
                do_sample=True,
            )
        
        # Decode response
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"\nResponse:\n{response}\n")
    
    print("\n" + "=" * 80)
    print("Demo completed!")
    print("=" * 80)


if __name__ == "__main__":
    demo_qwen2_5_7b_instruct()

