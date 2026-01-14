import os
# Suppress OpenBLAS warnings about OpenMP
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from textattack import *
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.goal_functions import UntargetedClassification
from textattack.datasets import HuggingFaceDataset
from textattack.transformations import WordSwap

from textattack.search_methods import GreedySearch
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)

from tqdm import tqdm  # tqdm provides us a nice progress bar.
from textattack.loggers import CSVLogger  # tracks a dataframe for us.
from textattack.attack_results import SuccessfulAttackResult
from textattack import Attacker
from textattack import AttackArgs
from textattack.datasets import Dataset

from textattack import Attack

import transformers


class BananaWordSwap(WordSwap):
    """Transforms an input by replacing any word with 'banana'."""

    # We don't need a constructor, since our class doesn't require any parameters.

    def _get_replacement_words(self, word):
        """Returns 'banana', no matter what 'word' was originally.

        Returns a list with one item, since `_get_replacement_words` is intended to
            return a list of candidate replacement words.
        """
        return ["banana"]


if __name__ == "__main__":
    # model, tokenizer, goal_function, dataset
    import torch
    
    model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-ag-news")
    tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-ag-news")
    
    # Move model to GPU if available for faster inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on device: {device}")
    
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
    goal_function = UntargetedClassification(model_wrapper)
    dataset = HuggingFaceDataset("ag_news", None, "test")

    # Attack initialization
    transformation = BananaWordSwap()
    constraints = [RepeatModification(), StopwordModification()]
    search_method = GreedySearch()
    attack = Attack(goal_function, constraints, transformation, search_method)

    # Apply attack on the model
    attack_args = AttackArgs(num_examples=10)
    attacker = Attacker(attack, dataset, attack_args)
    attack_results = attacker.attack_dataset()
    
    # Print results
    print("\n" + "="*80)
    for i, result in enumerate(attack_results, 1):
        print(f"\n{'='*45} Result {i} {'='*45}")
        # Use 'ansi' color method for terminal output, or None for plain text
        print(str(result) if hasattr(result, '__str__') else result)
        print()