import sys
import torch
import torch.nn.functional as F
from collections import defaultdict

def compute_surprisal(
    model,
    tokenizer,
    device,
    prefix,
    candidate
):
    full_text = prefix + candidate

    full_inputs = tokenizer(full_text, return_tensors="pt")["input_ids"].to(device)
    prefix_inputs = tokenizer(prefix, return_tensors="pt")["input_ids"].to(device)

    prefix_len = prefix_inputs.shape[1]

    with torch.no_grad():
        outputs = model(
            full_inputs,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = outputs.hidden_states

        num_layers = len(hidden_states)

        layer_indices = [
            int(0.1 * num_layers),
            int(0.5 * num_layers),
            int(0.9 * num_layers),
            num_layers - 1
        ]

        layer_log_probs = {}

        for i, layer_idx in enumerate(layer_indices):
            layer_hidden = hidden_states[layer_idx]

            logits = model.lm_head(layer_hidden)
            log_probs = F.log_softmax(logits, dim=-1)

            shifted_tokens = full_inputs[:, 1:]

            token_log_probs = log_probs[:, :-1].gather(
                2, shifted_tokens.unsqueeze(-1)
            ).squeeze(-1)

            candidate_log_probs = token_log_probs[:, prefix_len-1:]

            total_log_prob = candidate_log_probs.sum().item()
            length = candidate_log_probs.shape[1]

            avg_log_prob = total_log_prob / length

            if i == 0:
                key = "early"
            elif i == 1:
                key = "middle"
            elif i == 2:
                key = "late"
            else:
                key = "last"

            layer_log_probs[key] = avg_log_prob

        return layer_log_probs

# def compute_surprisal(model, tokenizer, device, prefix, candidate):
#     full_text = prefix + candidate

#     full_inputs = tokenizer(full_text, return_tensors="pt")["input_ids"].to(device)
#     prefix_inputs = tokenizer(prefix, return_tensors="pt")["input_ids"].to(device)

#     prefix_len = prefix_inputs.shape[1]

#     with torch.no_grad():
#         logits = model(full_inputs).logits
#         log_probs = F.log_softmax(logits, dim=-1)

#         shifted_tokens = full_inputs[:, 1:]
#         token_log_probs = log_probs[:, :-1].gather(
#             2, shifted_tokens.unsqueeze(-1)
#         ).squeeze(-1)

#         candidate_log_probs = token_log_probs[:, prefix_len-1:]

#         total_log_prob = candidate_log_probs.sum().item()
#         length = candidate_log_probs.shape[1]

#         avg_log_prob = total_log_prob / length
        
#     return avg_log_prob

def normalize_candidate_probs(log_probabilities):
    normalized = {}

    layers = log_probabilities[0].keys()

    for layer in layers:
        layer_scores = torch.tensor([
            log_probabilities[0][layer],
            log_probabilities[1][layer]
        ])

        probs = torch.softmax(layer_scores, dim=0)

        normalized[layer] = {
            "candidate_A": probs[0].item(),
            "candidate_B": probs[1].item()
        }

    return normalized