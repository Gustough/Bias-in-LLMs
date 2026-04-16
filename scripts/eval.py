import json
from collections import defaultdict
import regex as re
import pandas as pd

pairs = set()

mapping = {
    "haystack_language": {
        "h_en": "en",
        "h_de": "de"
    },
    "victim": {
        "corn": "Cornwall-Germany",
        "sax": "Saxony-Germany"
    },
    "repeat_condition": {
        "norep": "1x1",
        "2rf": "2x_first",
        "2rs": "2x_second",
        "5rf": "5x_first",
        "5rs": "5x_second"
    },
    "needle_language": {
        "nl_en": "en",
        "nl_de": "de"
    },
    "prompt_language": {
        "pl_en": "en",
        "pl_de": "de"
    },
    "grounding": {
        "acc": "acc",
        "nacc": "nacc"
    },
    "version_order": {
        "un": "unnecessary_first",
        "nu": "necessary_first"
    }
}

analysis_variables = [
    "haystack_language",
    "version_order",
    "victim",
    "repeat_condition",
    "needle_language",
    "prompt_language",
    "grounding"
]

exception = "haystack_id"

def apply_mapping(parsed):
    out = {}
    for var, val in parsed.items():
        if var in mapping:
            out[var] = mapping[var].get(val, val)
        else:
            out[var] = val
    return out


def find_pair(parsed, model, list_of_results, var):
    candidate = {k: v for k, v in parsed.items() if k != var and k != exception}
    pairwise = []

    for meta, result in list_of_results[model]:
        current_check = {k: v for k, v in meta.items() if k != var and k != exception}

        if candidate == current_check:
            pairwise.append((meta, result))

    return pairwise


def parse_answer(mod, answer, log_probs):
    if mod.split("_")[-1] == "mc":
        winner = 0 if log_probs[0] > log_probs[1] else 1
        words = re.findall(r"\b\w+\b", answer)
        if sum(x in words for x in ["A", "B", "C"]) >= 2:
            return (3, winner)
        elif "A" in words:
            return (0, winner)
        elif "B" in words:
            return (1, winner)
        elif "C" in words:
            return (2, winner)
        else:
            return (3, winner)
    else:
        winner = None
        words = re.findall(r"\b\w+\b", answer.lower())

        if (any(k in words for k in ["unnecessary", "unnötig"]) and
            any(k in words for k in ["necessary", "notwendig"])):
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

    summarized = defaultdict(list)

    # first pass
    for datapoint in results:
        mod = datapoint["model"]
        if not mod.endswith("_mc"):
            continue

        variables_full = [exception] + analysis_variables
        values = datapoint["meat"].split("\n")
        parsed = dict(zip(variables_full, values))
        parsed = apply_mapping(parsed)

        log_probs = (datapoint["p_unnecessary"], datapoint["p_necessary"])
        answer = parse_answer(mod, datapoint["answer"], log_probs)

        score = log_probs[0] - log_probs[1]
        confidence = max(log_probs)

        summarized[mod].append((parsed, {
            "generation": answer[0],
            "winner": answer[1],
            "score": score,
            "confidence": confidence
        }))

    dps = []

    # second pass
    for datapoint in results:
        mod = datapoint["model"]
        if not mod.endswith("_mc"):
            continue

        variables_full = [exception] + analysis_variables
        values = datapoint["meat"].split("\n")
        parsed = dict(zip(variables_full, values))
        parsed = apply_mapping(parsed)

        current_result = None
        for meta, res in summarized[mod]:
            if meta == parsed:
                current_result = res
                break

        for var in analysis_variables:

            pairs = find_pair(parsed, mod, summarized, var)

            for meta_other, other_result in pairs:

                if parsed[var] >= meta_other[var]:
                    continue

                delta = current_result["score"] - other_result["score"]

                dps.append({
                    "model": mod,
                    "variable": var,
                    "level": parsed[var], 
                    "delta": delta,
                    "abs_delta": abs(delta),
                    "flip": (current_result["score"] > 0) != (other_result["score"] > 0),
                    "confidence": current_result["confidence"],
                    "haystack_id": parsed[exception]
                })

    df = pd.DataFrame(dps)
    summary_check = df.groupby(["variable"])["delta"].describe()
    print(summary_check)
    exit(1)
    
    by_haystack = df.groupby(["model", "haystack_id", "variable"]).agg(
        mean_delta=("delta", "mean"),
        mean_abs_delta=("abs_delta", "mean"),
        flip_rate=("flip", "mean"),
        mean_confidence=("confidence", "mean"),
        n=("delta", "count")
    ).reset_index()

    summary = df.groupby(["model", "variable"]).agg(
        mean_delta=("delta", "mean"),
        mean_abs_delta=("abs_delta", "mean"),
        flip_rate=("flip", "mean"),
        mean_confidence=("confidence", "mean"),
        n=("delta", "count")
    ).reset_index()

    final = by_haystack.groupby(["model", "variable"]).agg(
        mean_delta=("mean_delta", "mean"),
        mean_abs_delta=("mean_abs_delta", "mean"),
        flip_rate=("flip_rate", "mean"),
        mean_confidence=("mean_confidence", "mean"),
        n=("n", "sum")
    ).reset_index()

    stability = by_haystack.groupby(["model", "variable"]).agg(
        delta_std_across_haystacks=("mean_delta", "std"),
        flip_std_across_haystacks=("flip_rate", "std")
    ).reset_index()

    final = final.merge(stability, on=["model", "variable"])

    summary.to_csv("summary.csv", index=False)
    by_haystack.to_csv("by_haystack.csv", index=False)
    final.to_csv("final.csv", index=False)
    stability.to_csv("stability.csv", index=False)
    df.to_csv("dps.csv", index=False)


if __name__ == '__main__':
    main()
    
    
# "h2\nh_de\nun\ncorn\n5rf\nnl_de\npl_en\nacc"
# actual_variables = {"haystack_id": "h1-h5", "version_order": "un/nu", "victim": "corn/sax", "repeat_condition": "norep, 2rf, 2rs, 5rf, 5rs", "needle_language": "en/de", "prompt_language": "en/de", "grounding": "acc/nacc"}