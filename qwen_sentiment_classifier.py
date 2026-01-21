import os
# Suppress OpenBLAS warnings about OpenMP
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from textattack.models.wrappers import ModelWrapper


class QwenSentimentClassifier(ModelWrapper):
    """
    TextAttack-compatible wrapper for Qwen2.5-7B-Instruct sentiment classification.
    
    Uses single-token label mapping to extract logits for "Positive" and "Negative"
    tokens for binary sentiment classification.
    """
    
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", device=None):
        """
        Initialize the Qwen sentiment classifier.
        
        Args:
            model_name: Hugging Face model identifier
            device: torch device (auto-detected if None)
        """
        super().__init__()
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Loading Qwen model: {model_name}")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="cuda:0" if self.device.type == "cuda" else None,
            )
            
            if self.device.type == "cpu":
                self.model = self.model.to(self.device)
            
            print("Model loaded successfully!")
            
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")
        
        # Get token IDs for label tokens
        self.negative_token_id, self.positive_token_id = self._get_label_token_ids()
        print(f"Label token IDs - Negative: {self.negative_token_id}, Positive: {self.positive_token_id}")
    
    def _get_label_token_ids(self):
        """
        Find token IDs for "Positive" and "Negative" labels.
        Handles single-token and multi-token cases.
        
        Returns:
            tuple: (negative_token_id, positive_token_id)
        """
        # Try to get single tokens first
        negative_tokens = self.tokenizer.encode("Negative", add_special_tokens=False)
        positive_tokens = self.tokenizer.encode("Positive", add_special_tokens=False)
        
        # Use first token if multi-token, or the single token
        negative_token_id = negative_tokens[0] if negative_tokens else None
        positive_token_id = positive_tokens[0] if positive_tokens else None
        
        # Verify tokens exist in vocabulary
        if negative_token_id is None or positive_token_id is None:
            raise ValueError("Could not find token IDs for 'Positive' and 'Negative' labels")
        
        # Check if tokens are the same (shouldn't happen, but safety check)
        if negative_token_id == positive_token_id:
            raise ValueError("'Positive' and 'Negative' map to the same token ID")
        
        return negative_token_id, positive_token_id
    
    def _format_sentiment_prompt(self, text):
        """
        Format a text input into a sentiment classification prompt.
        
        Args:
            text: Input text to classify
            
        Returns:
            str: Formatted prompt string
        """
        prompt = f"Classify the sentiment of the following text as either Positive or Negative.\n\nText: {text}\n\nSentiment:"
        return prompt
    
    def _get_logits_for_text(self, text):
        """
        Get logits for Positive and Negative tokens for a single text.
        
        Args:
            text: Input text string
            
        Returns:
            torch.Tensor: Logits tensor of shape [2] with [negative_logit, positive_logit]
        """
        # Format prompt
        prompt = self._format_sentiment_prompt(text)
        
        # Format using chat template
        messages = [{"role": "user", "content": prompt}]
        formatted_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize input
        inputs = self.tokenizer([formatted_text], return_tensors="pt").to(self.device)
        
        # Forward pass (no generation, just get logits)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # Shape: [batch=1, seq_len, vocab_size]
        
        # Get logits at the last token position (where model would generate next token)
        last_token_logits = logits[0, -1, :]  # Shape: [vocab_size]
        
        # Extract logits for Positive and Negative tokens
        negative_logit = last_token_logits[self.negative_token_id].item()
        positive_logit = last_token_logits[self.positive_token_id].item()
        
        return torch.tensor([negative_logit, positive_logit])
    
    def __call__(self, text_input_list, **kwargs):
        """
        TextAttack-compatible method for classification.
        
        Args:
            text_input_list: List of input strings (required by TextAttack)
            **kwargs: Additional arguments (ignored for now)
            
        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, 2)
                         Each row: [negative_logit, positive_logit]
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
                         Each row: [P(negative), P(positive)]
        """
        logits = self.__call__(text_input_list)
        return F.softmax(logits, dim=-1)
    
    def predict(self, text_input_list):
        """
        Get class predictions (optional convenience method).
        
        Args:
            text_input_list: List of input strings
            
        Returns:
            list: List of predicted labels ("Positive" or "Negative")
        """
        logits = self.__call__(text_input_list)
        predictions = torch.argmax(logits, dim=-1)
        return ["Positive" if pred == 1 else "Negative" for pred in predictions]


if __name__ == "__main__":
    # Demo usage
    print("=" * 80)
    print("Qwen Sentiment Classifier Demo")
    print("=" * 80)
    
    # Initialize classifier
    classifier = QwenSentimentClassifier()
    
    # Test examples
    test_texts = [
        "I love this movie! It's amazing!",
        "This is terrible. I hate it.",
        "The weather is okay today.",
    ]
    
    print("\n" + "=" * 80)
    print("Testing Classification")
    print("=" * 80)
    
    for text in test_texts:
        print(f"\nText: {text}")
        
        # Get logits (TextAttack-compatible)
        logits = classifier([text])  # Note: expects list
        print(f"Logits [Negative, Positive]: {logits[0].tolist()}")
        
        # Get probabilities
        probs = classifier.predict_proba([text])
        print(f"Probabilities [P(Negative), P(Positive)]: {probs[0].tolist()}")
        
        # Get prediction
        pred = classifier.predict([text])
        print(f"Prediction: {pred[0]}")
    
    print("\n" + "=" * 80)
    print("Demo completed!")
    print("=" * 80)

