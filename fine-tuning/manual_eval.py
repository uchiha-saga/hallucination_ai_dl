# manual_eval()
import torch, gc
from tqdm.auto import tqdm
import evaluate

# Load metrics (on CPU)
bleu  = evaluate.load("bleu")
rouge = evaluate.load("rouge")

def manual_eval(model, tokenizer, dataset, slice_size=100):
    """
    Generate answers one-by-one (or tiny batches) on CPU/GPU,
    immediately free intermediate tensors, and compute BLEU/ROUGE.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    preds, refs = [], []
    for ex in tqdm(dataset.select(range(slice_size)), desc="Manual eval"):
        prompt = f"Question: {ex['question']}\nAnswer:"
        # tokenize (runs on CPU by default, move to device)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        # greedy generate (single beam)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=1,
                eos_token_id=tokenizer.eos_token_id
            )[0]

        # decode & strip the prompt
        full   = tokenizer.decode(out_ids, skip_special_tokens=True)
        answer = full.split("Answer:", 1)[-1].strip()

        preds.append(answer)
        refs .append(ex["answers"][0].strip())

        # free memory
        del inputs, out_ids, full, answer
        gc.collect()
        torch.cuda.empty_cache()

    # final CPU‐based metric calculation
    # pass raw strings as predictions
    # references as a list of one‐element lists of raw reference strings
    bleu_res = bleu.compute(
        predictions=preds,
        references=[[r] for r in refs]
    )
    
    rouge_res = rouge.compute(
        predictions=preds,
        references=refs,
        use_stemmer=True
    )

    return {
        "bleu":      bleu_res["bleu"],
        "rouge1":    rouge_res["rouge1"],
        "rouge2":    rouge_res["rouge2"],
        "rougeL":    rouge_res["rougeL"],
        "rougeLsum": rouge_res["rougeLsum"],
    }
