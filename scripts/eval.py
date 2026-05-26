import pandas as pd
import re
from collections import defaultdict

def clean_text(x):
    if x is None:
        return x

    x = str(x)
    
    # remove \u0120 = space
    x = x.replace("\u0120", " ")

    # normalize newlines/tabs
    x = x.replace("\n", " ").replace("\t", " ")

    # collapse extra whitespace
    x = re.sub(r"\s+", " ", x).strip()

    return x

def repair_backup(meta):
    """
    Helper function to normalize metadata documentation across runs.
    """
    parts = meta.split("\n")

    p0 = parts[0]                    # haystack id
    p1 = parts[1].replace("_", "")   # haystack language
    p2 = parts[2]                    # needle order
    p3 = parts[3]                    # victim
    p4 = parts[4]                    # repetition level and target
    p5 = parts[5].replace("_", "")   # needle language
    p6 = parts[6].replace("_", "")   # question languague 
    p7 = parts[7]                    # question grounding

    order_map = {
        "nu": ("necessary", "unnecessary"),
        "un": ("unnecessary", "necessary")
    }
    
    first, second = order_map[p2]
    
    if p4 == "norep":
        repetition = "r1"
        position = "1st" if p2 == "un" else "2nd"
        needle_identity = second if p2 == "nu" else first
        
    else:
        repetition = f"r{p4[0]}"
        
        selection = 1 if p4[-1] == "s" else 0
        
        needle_identity = order_map[p2][selection]
        
        position = order_map[p2].index(needle_identity)
        position = "1st" if position == 0 else "2nd"

    return "_".join([
        p0,
        p1,
        p3,
        needle_identity,
        repetition,
        position,
        p5,
        p6,
        p7,
    ])

def parse_conflict_signal(answer: str):
    a = answer.lower()

    contradiction_markers = any([
        "unklar" in a,
        "unclear" in a,
        "contradict" in a,
        "widerspruch" in a,
        "inconclusive" in a,
        "widersprüch" in a
    ])

    has_both_labels = (
        re.search(r"\b(necessary|notwendig|n[oö]tig)\b", a) is not None
        and
        re.search(r"\b(unnecessary|unn[oö]tig|not necessary|nicht notwendig|nicht n[oö]tig)\b", a) is not None
    )

    if has_both_labels or contradiction_markers:
        return 1
    else:
        return 0

def parse_needle_selection(answer: str):
    a = answer.lower()

    contradiction_markers = any([
        "unklar" in a,
        "unclear" in a,
        "contradict" in a,
        "widerspruch" in a,
        "inconclusive" in a,
        "widersprüch" in a
    ])

    has_unnecessary = re.search(
        r"\b(unnecessary|unn[oö]tig|not necessary|nicht notwendig|nicht n[oö]tig)\b", a) is not None

    has_necessary = (
            re.search(r"\b(necessary|notwendig|n[oö]tig)\b", a) is not None
            and not has_unnecessary
        )

    has_both_labels = has_necessary and has_unnecessary


    if has_both_labels or contradiction_markers:
        return 2  # conflict

    elif has_unnecessary and not has_necessary:
        return 0  # unnecessary

    elif has_necessary and not has_unnecessary:
        return 1  # necessary

    else:
        return 3  # failed

def parse_meat(s):
    parts = s.split("_")
    
    return {
        "haystack_raw": parts[0],
        "haystack_language": parts[1][1:],   # h_en -> en
        "victim": parts[2],
        "repeated_identity": parts[3],       # which needle is repeated
        "repetition": int(parts[4][1:]),     # r2 -> 2
        "position": parts[5],                # 1st / 2nd (position of repeated needle)
        "needle_language": parts[6][2:],     # nlxx -> xx
        "question_language": parts[7][2:],   # plxx -> xx
        "grounding": parts[8]
    }

def parse_mc_generation(answer: str) -> int:
    if answer.startswith("A"):
        return 0
    elif answer.startswith("B"):
        return 1
    elif answer.startswith("C"):
        return 2
    else:
        return 3
    
def preprocess(df):
    """
    Function to prepare results for clogit.
    """
    rows = []
    
    count_conf = 0
    count_failed = 0
    
    model_counts = defaultdict(lambda: defaultdict(int))

    for i, r in df.iterrows():
        meta = parse_meat(r["meat"])

        model_parts = r["model"].split("_")
        
        pos_map = {"1st": (1, 2), "2nd": (2, 1)}
        pos_rep, pos_other = pos_map[meta["position"]]
        
        # Multiple-choice result parsing
        if model_parts[-1] == "mc":
            label = parse_conflict_signal(r["answer"])

            p_un = r["p_A"]
            p_nec = r["p_B"]

            if p_un > p_nec:
                choice_un, choice_nec = 1, 0

            elif p_nec > p_un:
                choice_un, choice_nec = 0, 1

            else:
                winner = None

                for token, _ in r["topk"]:
                    token = token.strip()
                    print(token)

                    if token.endswith("A"):
                        winner = "A"
                        break

                    elif token.endswith("B"):
                        winner = "B"
                        break

                choice_un, choice_nec = (1, 0) if winner == "A" else (0, 1)
            
            
            for identity, prob, choice in [
                ("unnecessary", p_un, choice_un),
                ("necessary", p_nec, choice_nec)
                ]:

                is_rep = int(meta["repeated_identity"] == identity)

                rows.append({
                    "model_name": "_".join(model_parts[:-1]),
                    "experimental_condition": model_parts[-1],
                    "haystack_id": i,
                    "haystack": meta["haystack_raw"],

                    "choice": choice,
                    "prob": prob,
                    "label": label,

                    "needle_identity": identity,
                    "is_repeated": is_rep,
                    "repetition": meta["repetition"] if is_rep else 1,
                    "position": pos_rep if is_rep else pos_other,

                    "haystack_language": meta["haystack_language"],
                    "needle_language": meta["needle_language"],
                    "question_language": meta["question_language"],
                    "grounding": meta["grounding"],
                    "victim": meta["victim"]
                })
                
        # Free-form result parsing
        elif model_parts[-1] == "lf":
            label = parse_needle_selection(r["answer"])
            
            unnec = 0
            nec = 0
            conflict = 0
                        
            if label == 0:
                unnec = 1
            elif label == 1:
                nec = 1
            elif label == 2:
                conflict = 1
                count_conf += 1
                model_counts["conflicted"][r["model"]] += 1
            elif label == 3:
                count_failed += 1
                model_counts["failed"][r["model"]] += 1

            for identity, choice in [
                ("unnecessary", unnec),
                ("necessary", nec),
                ("conflict", conflict)
            ]:

                is_rep = int(meta["repeated_identity"] == identity)

                rows.append({
                    "model_name": "_".join(model_parts[:-1]),
                    "experimental_condition": model_parts[-1],
                    "haystack_id": i,
                    "haystack": meta["haystack_raw"],

                    "choice": choice,
                    "prob": 0,
                    "label": label,

                    "needle_identity": identity,
                    "is_repeated": is_rep,
                    "repetition": meta["repetition"] if is_rep else 1,
                    "position": pos_rep if is_rep else pos_other,

                    "haystack_language": meta["haystack_language"],
                    "needle_language": meta["needle_language"],
                    "question_language": meta["question_language"],
                    "grounding": meta["grounding"],
                    "victim": meta["victim"]
                })
    
    print(model_counts.items())
                
    return pd.DataFrame(rows)

def main():
    df = pd.read_json("llm_outputs_final.jsonl", lines=True)
        
    df["answer"] = df["answer"].apply(clean_text)
    
    # df["meat"] = df["meat"].apply(repair_backup)
    
    df_long = preprocess(df)

    df_long.to_parquet("clogit.parquet", index=False)
    
if __name__ == '__main__':
    main()




