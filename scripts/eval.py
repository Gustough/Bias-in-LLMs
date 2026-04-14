import json
from collections import defaultdict
import regex as re
import numpy as np
import csv

pairs = set()

def find_pair(parsed, model, list_of_results, var):
    candidate = {k: v for k, v in parsed.items() if k != var}
    pairwise = []

    for meta, result in list_of_results[model]:
        current_check = {k: v for k, v in meta.items() if k != var}
        
        if candidate == current_check:
            pairwise.append((meta, result))
    
    return pairwise

def parse_answer(mod, answer, log_probs):
    """_summary_

    Args:
        mod (str): model type
        answer (str): model output
    
    variables:
    
        answer:
            0 = unnecessary
            1 = necessary
            2 = inconclusive
            3 = failed
        
        winner:
            0 = unnecessary
            1 = necessary
    """
    if mod.split("_")[-1] == "mc":
        winner = 0 if log_probs[0] > log_probs[1] else 1
    else:
        winner = None
    if mod == "gpt_oss_lf":
        check = answer.split("assistantfinal")
        if len(check) == 2:
            answer = check[1]

            
            words = re.findall(r"\b\w+\b", answer.lower())
            if (any(k in words for k in ["unnecessary", "unnötig"]) and any(k in words for k in ["necessary", "notwendig"])):
                return (2, winner)
            elif any(k in words for k in ["unnecessary", "unnötig"]):
                return (0, winner)
            elif any(k in words for k in ["necessary", "notwendig"]):
                return (1, winner)
        else:
            return (3, winner)
    else:
        words = re.findall(r"\b\w+\b", answer.lower())
        if (any(k in words for k in ["unnecessary", "unnötig"]) and any(k in words for k in ["necessary", "notwendig"])):
            return (2, winner)
        elif any(k in words for k in ["unnecessary", "unnötig"]):
            return (0, winner)
        elif any(k in words for k in ["necessary", "notwendig"]):
            return (1, winner)
        else:
            return (3, winner)
     

def main():
    results = []
    with open("llm_outputs.jsonl") as file:
        for line in file:
            results.append(json.loads(line))

    # models = ["llama_mc", "llama_lf", "mistral_7B_lf", "mistral_7B_mc", "mistral_7B_instruct_lf", "mistral_7B_instruct_mc", "gpt-oss_lf", "qwen_7B_lf", "qwen_7B_mc", "qwen_14B_lf", "qwen_14B_mc"]
    variables = ["haystack_id", "haystack_language", "version_order", "victim", "repeat_condition", "needle_language", "prompt_language", "grounding"]
    final = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    summarized = defaultdict(list)
    
    # First pass: collect everything
    for i, datapoint in enumerate(results):
        mod = datapoint["model"]    
        values = datapoint["meat"].split("\n")
        parsed = dict(zip(variables, values))
        log_probs = (datapoint["p_unnecessary"], datapoint["p_necessary"])
        answer = parse_answer(mod, datapoint["answer"], log_probs)
        score = log_probs[0] - log_probs[1]  # p_unnecessary - p_necessary
        confidence = max(log_probs)

        summarized[mod].append((parsed, {
            "generation": answer[0],
            "winner": answer[1],
            "score": score,
            "confidence": confidence
        }))
        
        # summarized[mod].append((parsed, answer))

    # # Second pass: find pairs
    # for i, datapoint in enumerate(results):
    #     mod = datapoint["model"]
    #     values = datapoint["meat"].split("\n")
    #     parsed = dict(zip(variables, values))
        
    #     for var in variables:
    #         final[f"mod_{i}"][var] = find_pair(parsed, mod, summarized, var)
    
    analysis = defaultdict(lambda: defaultdict(list))

    for i, datapoint in enumerate(results):
        mod = datapoint["model"]
        values = datapoint["meat"].split("\n")
        parsed = dict(zip(variables, values))
        
        # get current datapoint's result
        current_result = None
        for meta, res in summarized[mod]:
            if meta == parsed:
                current_result = res
                break

        for var in variables:
            pairs = find_pair(parsed, mod, summarized, var)

            for meta_other, other_result in pairs:
                
                if parsed[var] >= meta_other[var]:
                    continue
                
                delta = current_result["score"] - other_result["score"]
                flip = (current_result["score"] > 0) != (other_result["score"] > 0)

                analysis[mod][var].append({
                    "delta": delta,
                    "abs_delta": abs(delta),
                    "flip": flip,
                    "confidence": current_result["confidence"]
                })

    summary = []

    for mod in analysis:
        for var in analysis[mod]:
            entries = analysis[mod][var]
            
            deltas = [e["delta"] for e in entries]
            flips = [e["flip"] for e in entries]
            confidences = [e["confidence"] for e in entries]

            summary.append({
                "model": mod,
                "variable": var,
                "mean_delta": np.mean(deltas),
                "mean_abs_delta": np.mean(np.abs(deltas)),
                "flip_rate": np.mean(flips),
                "mean_confidence": np.mean(confidences),
                "n": len(entries)
            })


    keys = summary[0].keys()

    with open("summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary)
    
if __name__ == '__main__':
    main()