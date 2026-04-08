import json
from collections import defaultdict

def find_pair(index, model, list_of_results:list[tuple], var):
    pairwise = []
    candidate = {k: v for k, v in list_of_results[model][index][0].items() if k != var}

    for meta, answer in list_of_results[model]:
        current_check = {k: v for k, v in meta.items() if k != var}
        
        if candidate == current_check:
            pairwise.append((current_check, answer))
            
    final = []
    for i in range(len(pairwise)):
        final.append(pairwise[i][1])
        
    return final
        

def main():
    results = []
    with open("final_outputs.jsonl") as file:
        for line in file:
            line = "{" + ",".join(line.split(",")[1:]).strip()
            results.append(json.loads(line))

    models = ["llama_mc", "llama_lf", "mistral_7B_lf", "mistral_7B_mc", "mistral_7B_instruct_lf", "mistral_7B_instruct_mc", "gpt-oss_lf", "qwen_7B_lf", "qwen_7B_mc", "qwen_14B_lf", "qwen_14B_mc"]
    variables = ["haystack_id", "haystack_language", "version_order", "victim", "repeat_condition", "needle_language", "prompt_language", "grounding"]
    for mod in models:
        final = defaultdict(lambda: defaultdict(list))
        summarized = defaultdict(list)
        for datapoint in results:
            if datapoint["model"] == mod:
                values = datapoint["meat"].split("\n")
                parsed = dict(zip(variables, values))
                answer = datapoint["answer"]
                summarized[mod].append((parsed, answer))
        if summarized:
            for i, var in enumerate(variables):
                final[mod][var] = find_pair(i, mod, summarized, var)
            print(final)
            exit(1)

if __name__ == '__main__':
    main()