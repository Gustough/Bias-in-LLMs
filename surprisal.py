import sys
import torch
import torch.nn.functional as F

def compute_surprisal(model, tokenizer, device, text):
    inputs = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)

        shifted_tokens = inputs[..., 1:]
        target_log_probs = log_probs[:, :-1].gather(
            2, shifted_tokens.unsqueeze(-1)
        ).squeeze(-1)

        total_log_prob = target_log_probs.sum().item()
        length = target_log_probs.numel()
        
        avg_log_prob = (total_log_prob / length).item()

        # total_surprisal = -total_log_prob / torch.log(torch.tensor(2.0)).item()

    return avg_log_prob

def normalize_candidate_probs(log_probs):
    log_probs_tensor = torch.tensor(log_probs)

    probs = torch.softmax(log_probs_tensor, dim=0)

    return probs.tolist()

