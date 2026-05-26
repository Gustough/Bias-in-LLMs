import json
import string
import os
from transformers import AutoTokenizer, AutoModelForCausalLM    
from huggingface_hub import login
import torch
from transformers import pipeline
from surprisal import compute_surprisal

def reconstruct_needles(condition, language):
    if condition == "corn":
        if language == "de":
            condition = ["Cornwall", "Deutschland"]
        else:
            condition = ["Cornwall", "Germany"]
    else:
        if language == "de":
            condition = ["Sachsen", "Großbritannien"]
        else:
            condition = ["Saxony", "Great Britain"]
    
    needles  = {
        "de": [
        " A",
        " B"
        ],
        "en": [
        " A",
        " B"
        ],
        }
    
    return needles[language]

def main():
    output = "llm_outputs.jsonl"
    haystack_path = "all_haystacks"
    with open("questions.json", "r", encoding="utf-8") as j:
        question_dict = json.load(j)
        
    counter = 1
    haystacks_list = os.listdir(haystack_path)
    models = [
    # ("deberta", "/mimer/NOBACKUP/groups/naiss2026-4-124/gustav/.cache/huggingface/hub/models--timpal0l--mdeberta-v3-base-squad2/snapshots/08d6e89c7a6557f967db2e1021f7f640483400ed", "deberta"),                                                                                                                                                                         
    ("gpt-oss", "/mimer/NOBACKUP/groups/naiss2026-4-124/gustav/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee", "gpt-oss"),
    ("llama", "/mimer/NOBACKUP/groups/naiss2026-4-124/gustav/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6", "llama"),
    ("qwen", "/mimer/NOBACKUP/groups/naiss2026-4-124/gustav/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8", "qwen_14B"),
    ("mistral", "/mimer/NOBACKUP/groups/naiss2026-4-124/gustav/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots/caa1feb0e54d415e2df31207e5f4e273e33509b1", "mistral_7B"),
    ("qwen", "/mimer/NOBACKUP/groups/naiss2026-4-124/gustav/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct-1M/snapshots/e28526f7bb80e2a9c8af03b831a9af3812f18fba", "qwen_7B"),      
    ("mistral", "/mimer/NOBACKUP/groups/naiss2026-4-124/gustav/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/c170c708c41dac9275d15a8fff4eca08d52bab71", "mistral_7B_instruct"),
          ]
    with open(output, "a") as outf:
        for mod, variant, code in models:
            if mod == "llama":
                model_id = variant
                
                tokenizer = AutoTokenizer.from_pretrained(model_id)

                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )

                model.eval()
                
                def call_llama(msg):
                    prompt = tokenizer.apply_chat_template(
                        msg,
                        tokenize=False,
                        add_generation_prompt=True
                    )

                    inputs = tokenizer(prompt, return_tensors="pt")
                    inputs = {k: v.to(model.get_input_embeddings().weight.device) for k, v in inputs.items()}

                    with torch.inference_mode():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=300,
                            do_sample=False
                        )

                    generated = outputs[0][inputs["input_ids"].shape[-1]:]
                    return tokenizer.decode(generated, skip_special_tokens=True).strip()
                
            #Deberta
            # elif mod == "deberta":
            #     model = pipeline(
            #         "question-answering",
            #         model= variant
            #     )
                
            #     def call_deberta(question, context):
            #         result = model(question=question, context=context)
            #         return result["answer"]

                
            #GPT-OSS
            elif mod == "gpt-oss":
                model_id = variant

                if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                    dtype = torch.bfloat16
                else:
                    dtype = torch.float32
                    
                tokenizer = AutoTokenizer.from_pretrained(model_id)

                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    device_map="auto",
                    trust_remote_code=True,
                )

                model.eval()
                
                def call_gpt(messages, prompt_lang):
                    if prompt_lang == "en":
                        formatted = (
                            f"{messages[0]['content']}\n\n"
                            f"{messages[1]['content']}\n"
                            "Answer:"
                        )

                    elif prompt_lang == "de":
                        formatted = (
                            f"{messages[1]['content']}\n\n"
                            f"{messages[1]['content']}\n"
                            "Antwort:"
                        )

                    inputs = tokenizer(formatted, return_tensors="pt")
                    inputs = {k: v.to(model.get_input_embeddings().weight.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=400,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id
                        )
                        
                    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

                    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                    
                    if "assistantfinal" in decoded:
                        final = decoded.split("assistantfinal")[-1].strip()
                    else:
                        final = decoded.strip()
                    
                    return final
                    
                
            #Mistral
            elif mod == "mistral":
                model_id = variant

                if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                    dtype = torch.bfloat16
                else:
                    dtype = torch.float32

                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    device_map="auto",
                    trust_remote_code=True,
                )
                model.eval()

                def call_mistral(prompt):
                    if getattr(tokenizer, "chat_template", None):
                        formatted = tokenizer.apply_chat_template(
                            prompt,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    else:
                        if isinstance(prompt, list):
                            parts = [m.get("content", "").strip() for m in prompt]
                            formatted = "\n".join(parts)
                            if not formatted.rstrip().endswith(("Answer:", "Assistant:")):
                                formatted += "\nAnswer:"
                        else:
                            formatted = prompt

                    inputs = tokenizer(formatted, return_tensors="pt")
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=300,
                            do_sample=False
                        )

                    # strip prompt tokens
                    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

                    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            elif mod == "qwen":
                model_id = variant

                if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                    dtype = torch.bfloat16
                else:
                    dtype = torch.float32
                    
                tokenizer = AutoTokenizer.from_pretrained(model_id)

                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    device_map="auto",
                    trust_remote_code=True,
                )

                model.eval()
                
                def call_qwen(messages):
                    formatted = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )

                    inputs = tokenizer(formatted, return_tensors="pt")
                    inputs = {k: v.to(model.get_input_embeddings().weight.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=300,
                            do_sample=False
                        )
                        
                    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

                    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                

            code = f"{code}_mc"
            
            
            #Sanity check
            # needles = [" A", " B", " C"]
            
            # candidates = {
            #         "candidate_A": needles[0],
            #         "candidate_B": needles[1],
            #         "candidate_C": needles[2],
            #     }
                        
            # for k, v in candidates.items():
            #     tokens = tokenizer(v, add_special_tokens=False)["input_ids"]
            #     print(k, tokens, "len =", len(tokens))
            
            for file_name in haystacks_list:
                haystack_index, haystack_language, victim, needle_identity, repetition, position, needle_language = file_name.split("_")
                with open(os.path.join(haystack_path, file_name), encoding="utf-8") as hay:
                    full_hay = json.load(hay)
                    needle_language = needle_language[2:4]
                    for info, question in question_dict.items():
                        for i, pc in enumerate(["nacc", "acc"]):
                            l, v = info.split("_", 1)
                            if v == victim:
                                metadata = file_name.strip(".json") + f"_pl{l}_{pc}"

                                # English free-form prompt 
                                # if l == "en":
                                    # messages = [
                                    # {"role": "system", "content": "Reasoning: low"},
                                    # {"role": "user",
                                    # "content": (
                                    # f"Context: {full_hay.strip()}\n"
                                    # f"Question: {question[i]}\n\n"
                                    # "Answer:")}
                                    # ]
                                
                                # German free-form prompt
                                # elif l == "de":
                                    # messages = [
                                    # {"role": "system", "content": ""Reasoning: niedrig""},
                                    # {"role": "user",
                                    # "content": (
                                    # f"Kontext: {full_hay.strip()}\n"
                                    # f"Frage: {question[i]}\n\n"
                                    # "Antwort:")}
                                    # ]
                                
                                # English multiple-choice prompt
                                if l == "en":
                                    messages = [
                                        {
                                            "role": "system",
                                            "content": (
                                                "Reasoning: low\n\n"
                                                "Answer with exactly one token from the list: ' A', ' B', ' C' (including the leading space).\n"
                                                "Options:\n A = unnecessary\n B = necessary\n C = inconclusive"
                                            )
                                        },
                                        {
                                            "role": "user",
                                            "content": (
                                                f"Context: {full_hay.strip()}\n"
                                                f"Question: {question[i]}\n\n"
                                                "Answer:"
                                            )
                                        }
                                    ]

                                # German multiple-choice prompt
                                elif l == "de":
                                    messages = [
                                        {
                                            "role": "system",
                                            "content": (
                                                "Reasoning: niedrig\n\n"
                                                "Antworte mit genau einem Token aus der Liste: ' A', ' B', ' C' (inklusive führendem Leerzeichen) "
                                                "Optionen:\n A = unnötig\n B = notwendig\n C = unklar\n"
                                            )
                                        },
                                        {
                                            "role": "user",
                                            "content": (
                                                f"Kontext: {full_hay.strip()}\n"
                                                f"Frage: {question[i]}\n\n"
                                                "Antwort:"
                                            )
                                        }
                                    ]
                                    
                                    
                                if mod == "llama":
                                    answer = call_llama(messages)
                                
                                elif mod =="deberta":
                                    answer = call_deberta(question=question[i], context=f"""{full_hay.strip()}""")
                                    answer = answer.translate(str.maketrans(' ', ' ', string.punctuation)).strip()
                                
                                elif mod == "gpt-oss":
                                    answer = call_gpt(messages, l)
                                
                                elif mod == "mistral":
                                    answer = call_mistral(messages)

                                elif mod == "qwen":
                                    answer = call_qwen(messages)
                                
                                log_probabilities = []

                                if mod == "gpt-oss":
                                    prefix_text = (
                                        f"system: {messages[0]['content']}\n"
                                        f"user: {messages[1]['content']}\n"
                                        "assistant: "
                                    )

                                elif mod in ["llama", "mistral", "qwen"] and getattr(tokenizer, "chat_template", None):
                                    prefix_text = tokenizer.apply_chat_template(
                                        messages,
                                        tokenize=False,
                                        add_generation_prompt=True
                                    )

                                else:
                                    if isinstance(messages, list):
                                        parts = []
                                        for m in messages:
                                            parts.append(m["content"].strip())
                                        prefix_text = "\n".join(parts)

                                        if not prefix_text.rstrip().endswith(("Answer:", "Assistant:")):
                                            prefix_text += "\nAnswer: "
                                    else:
                                        prefix_text = messages
                        
                                needles = reconstruct_needles(victim, needle_language)
                    
                                normalized_candidate_probs = []
                                
                                data = {
                                    "model": code,
                                    "meat": metadata,
                                    "answer": answer
                                }
                        
                                print(answer)
                                
                                if not mod == "deberta":
                                    device = model.get_input_embeddings().weight.device

                                    candidates = {
                                        "candidate_A": needles[0],
                                        "candidate_B": needles[1],
                                    }

                                    candidate_log_probs, topk_out = compute_surprisal(
                                        model, tokenizer, device, prefix_text, candidates
                                    )

                                    scores = torch.tensor([
                                        candidate_log_probs["candidate_A"],
                                        candidate_log_probs["candidate_B"],
                                    ])

                                    probs = torch.softmax(scores, dim=0)
                                    
                                    data["answer"] = answer
                                    data["p_A"] = probs[0].item()
                                    data["p_B"] = probs[1].item()
                                    data["topk"] = topk_out
                                    
                                print(json.dumps(data), file=outf, flush=True)
                                counter += 1

            torch.cuda.empty_cache()               
if __name__ == '__main__':
    main()
