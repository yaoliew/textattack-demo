import os
# Suppress OpenBLAS warnings about OpenMP
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from textattack.models.wrappers import ModelWrapper


class QwenSmishingClassifier(ModelWrapper):
    """
    TextAttack-compatible wrapper for Qwen2.5-VL-7B-Instruct smishing/phishing classification.
    
    Uses single-token label mapping to extract logits for "Legitimate" and "Smishing"
    tokens for binary smishing detection (0 = Legitimate, 1 = Smishing/Spam).
    """
    
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct", device=None):
        """
        Initialize the Qwen smishing classifier.
        
        Args:
            model_name: Hugging Face model identifier
            device: torch device (auto-detected if None)
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading Qwen model: {model_name}")
        print(f"Using device: {self.device}")
        
        # Load processor and model
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="cuda:0" if self.device.type == "cuda" else None,
            )
            
            if self.device.type == "cpu":
                self.model = self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")
        
        # Get token IDs for label tokens
        self.legitimate_token_id, self.smishing_token_id = self._get_label_token_ids()
        print(f"Label token IDs - Legitimate: {self.legitimate_token_id}, Smishing: {self.smishing_token_id}")
    
    def _get_label_token_ids(self):
        """
        Find token IDs for "Legitimate" and "Smishing" labels.
        Handles single-token and multi-token cases.
        
        Returns:
            tuple: (legitimate_token_id, smishing_token_id)
        """
        # Try to get single tokens first
        legitimate_tokens = self.processor.tokenizer.encode("Legitimate", add_special_tokens=False)
        smishing_tokens = self.processor.tokenizer.encode("Smishing", add_special_tokens=False)
        
        # Use first token if multi-token, or the single token
        legitimate_token_id = legitimate_tokens[0] if legitimate_tokens else None
        smishing_token_id = smishing_tokens[0] if smishing_tokens else None
        
        # Verify tokens exist in vocabulary
        if legitimate_token_id is None or smishing_token_id is None:
            raise ValueError("Could not find token IDs for 'Legitimate' and 'Smishing' labels")
        
        # Check if tokens are the same (shouldn't happen, but safety check)
        if legitimate_token_id == smishing_token_id:
            raise ValueError("'Legitimate' and 'Smishing' map to the same token ID")
        
        return legitimate_token_id, smishing_token_id
    
    def _format_smishing_prompt(self, text):
        """
        Format a text input into a smishing classification prompt.
        
        Args:
            text: Input text to classify
            
        Returns:
            str: Formatted prompt string
        """
        prompt = f"Classify the following message as either Legitimate or Smishing (phishing/spam).\n\nMessage: {text}\n\nClassification:"
        return prompt
    
    def _get_logits_for_text(self, text):
        """
        Get logits for Legitimate and Smishing tokens for a single text.
        
        Args:
            text: Input text string
            
        Returns:
            torch.Tensor: Logits tensor of shape [2] with [legitimate_logit, smishing_logit]
        """
        # Format prompt
        prompt = self._format_smishing_prompt(text)
        
        # Format using chat template (multimodal structure, text-only)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        formatted_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize input
        inputs = self.processor.tokenizer([formatted_text], return_tensors="pt").to(self.device)
        
        # Forward pass (no generation, just get logits)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # Shape: [batch=1, seq_len, vocab_size]
        
        # Get logits at the last token position (where model would generate next token)
        last_token_logits = logits[0, -1, :]  # Shape: [vocab_size]
        
        # Extract logits for Legitimate and Smishing tokens
        legitimate_logit = last_token_logits[self.legitimate_token_id].item()
        smishing_logit = last_token_logits[self.smishing_token_id].item()
        
        return torch.tensor([legitimate_logit, smishing_logit])
    
    def __call__(self, text_input_list, **kwargs):
        """
        TextAttack-compatible method for classification.
        
        Args:
            text_input_list: List of input strings (required by TextAttack)
            **kwargs: Additional arguments (ignored for now)
            
        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, 2)
                         Each row: [legitimate_logit, smishing_logit]
        """
        if not isinstance(text_input_list, list):
            text_input_list = [text_input_list]
        
        # Process each text and collect logits
        batch_logits = []
        for text in text_input_list:
            logits = self._get_logits_for_text(text)
            batch_logits.append(logits)
        
        # Stack into batch tensor: shape (batch_size, 2)
        return torch.stack(batch_logits)
    
    def predict_proba(self, text_input_list):
        """
        Get probability predictions (optional convenience method).
        
        Args:
            text_input_list: List of input strings
            
        Returns:
            torch.Tensor: Probabilities tensor of shape (batch_size, 2)
                         Each row: [P(legitimate), P(smishing)]
        """
        logits = self.__call__(text_input_list)
        return F.softmax(logits, dim=-1)
    
    def predict(self, text_input_list):
        """
        Get class predictions (optional convenience method).
        
        Args:
            text_input_list: List of input strings
            
        Returns:
            list: List of predicted labels ("Legitimate" or "Smishing")
        """
        logits = self.__call__(text_input_list)
        predictions = torch.argmax(logits, dim=-1)
        return ["Smishing" if pred == 1 else "Legitimate" for pred in predictions]


if __name__ == "__main__":
    # Demo usage
    print("=" * 80)
    print("Qwen Smishing Classifier Demo")
    print("=" * 80)
    
    # Initialize classifier
    classifier = QwenSmishingClassifier()
    
    # Test examples
    test_texts = [
        "Your package arrives at the Cyprus Post Office tomorrow.Confirm delivery: https://51.fi/aJzP", # smishing example
        "Shipped: Your Amazon package with Old Spice High Endurance Deodorant will be delivered Tue, May 24. Track at http://a.co/4SJitSA", # legitimate example
        "FREE entry into our £250 weekly comp just send the word ENTER to 88877 NOW. 18 T&C www.textcomp.com", # spam example
    ]
    
    print("\n" + "=" * 80)
    print("Testing Classification")
    print("=" * 80)
    
    for text in test_texts:
        print(f"\nText: {text}")
        
        # Get logits (TextAttack-compatible)
        logits = classifier([text])  # Note: expects list
        print(f"Logits [Legitimate, Smishing]: {logits[0].tolist()}")
        
        # Get probabilities
        probs = classifier.predict_proba([text])
        print(f"Probabilities [P(Legitimate), P(Smishing)]: {probs[0].tolist()}")
        
        # Get prediction
        pred = classifier.predict([text])
        print(f"Prediction: {pred[0]}")
    
    print("\n" + "=" * 80)
    print("Demo completed!")
    print("=" * 80)

