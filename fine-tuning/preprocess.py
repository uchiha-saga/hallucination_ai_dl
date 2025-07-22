# preprocess.py
import logging
from typing import Dict, List, Any
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def preprocess(
    examples: Dict[str, List[Any]],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 256
) -> Dict[str, List[Any]]:
    
    # Batch-tokenize QA examples for causal LM training.
    # Builds prompt+answer sequences, pads/truncates to max_length,
    # and masks prompt tokens in labels with -100.
    
    questions = examples.get("question", [])
    answers = examples.get("answers", [])

    input_ids_batch, attention_mask_batch, labels_batch = [], [], []
    for q, ans_list in zip(questions, answers):
        if not q or not ans_list:
            logger.warning("Skipping example with missing question or answer.")
            continue
        question = q.strip()
        answer = ans_list[0].strip()

        prompt = f"Question: {question}\nAnswer:"
        # encode prompt and answer separately
        prompt_enc = tokenizer(prompt, add_special_tokens=False)
        answer_enc = tokenizer(answer, add_special_tokens=False)

        # combine sequences and add EOS
        sequence = prompt_enc["input_ids"] + answer_enc["input_ids"] + [tokenizer.eos_token_id]
        # truncate from the left if too long
        if len(sequence) > max_length:
            sequence = sequence[-max_length:]

        # build labels: mask prompt tokens
        label_sequence = [-100] * len(prompt_enc["input_ids"]) + answer_enc["input_ids"] + [tokenizer.eos_token_id]
        if len(label_sequence) > max_length:
            label_sequence = label_sequence[-max_length:]

        # pad to max_length
        pad_len = max_length - len(sequence)
        input_seq = sequence + [tokenizer.pad_token_id] * pad_len
        attention_mask = [True] * len(sequence) + [False] * pad_len
        labels = label_sequence + [-100] * pad_len

        input_ids_batch.append(input_seq)
        attention_mask_batch.append(attention_mask)
        labels_batch.append(labels)

    return {"input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "labels": labels_batch}
