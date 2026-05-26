import sys
import torch
import torch.nn.functional as F
from collections import defaultdict


def compute_surprisal(
    model,
    tokenizer,
    device,
    prefix,
    candidates
):
    prefix_inputs = tokenizer(prefix, return_tensors="pt")["input_ids"].to(device)

    candidate_ids = {
        name: tokenizer(tok, add_special_tokens=False)["input_ids"][0]
        for name, tok in candidates.items()
    }

    with torch.no_grad():
        outputs = model(
            prefix_inputs,
            return_dict=True
        )

        logits = outputs.logits[:, -1, :]

        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)

        # Candidate surprisal
        candidate_log_probs = {
            name: log_probs[0, cid].item()
            for name, cid in candidate_ids.items()
        }

        # Top k for most likely generated token
        topk = torch.topk(probs, k=10)

        topk_tokens = tokenizer.convert_ids_to_tokens(topk.indices[0].tolist())
        topk_probs = topk.values[0].tolist()

        topk_out = list(zip(topk_tokens, topk_probs))

    return candidate_log_probs, topk_out
