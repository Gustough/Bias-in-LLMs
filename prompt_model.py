import json
import string
import os
# from Llama_32_1B_Instruct import call_llama
# from DeBERTaV3 import call_deberta
from transformers import AutoTokenizer, AutoModelForCausalLM    
from huggingface_hub import login
import torch
from transformers import pipeline
from surprisal import compute_surprisal, normalize_candidate_probs

with open("k.txt", "r") as f:
    key = f.read().strip()

login(token=key)

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
            f"Der Angriff auf {condition[0]} im Jahr 2024 durch {condition[1]} war ungerechtfertigt. ",
            f"Der Angriff auf {condition[0]} im Jahr 2024 durch {condition[1]} war gerechtfertigt. ",
                ],
            "en": [
            f"The 2024 attack against {condition[0]} by {condition[1]} was unjustified. ",
            f"The 2024 attack against {condition[0]} by {condition[1]} was justified. ",
                    ],
            }
    
    return needles[language]

def main():
    output = "llm_outputs.jsonl"
    haystack_path = "all_haystacks"
    with open("questions.json", "r", encoding="utf-8") as j:
        question_dict = json.load(j)
        
    
    haystacks_list = os.listdir(haystack_path)
    for mod in ["llama"]:#["llama", "deberta", "gpt-oss", "mistral"]
        
        #LLama
        if mod == "llama":
            model_id = "meta-llama/Llama-3.2-1B-Instruct"
            device = "cuda" if torch.cuda.is_available() else "cpu"

            tokenizer = AutoTokenizer.from_pretrained(model_id)

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)

            model.eval()
            
            def call_llama(msg):
                prompt = tokenizer.apply_chat_template(
                    msg,
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = tokenizer(prompt, return_tensors="pt").to(device)

                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=30,
                        do_sample=False
                    )

                generated = outputs[0][inputs["input_ids"].shape[-1]:]
                return tokenizer.decode(generated, skip_special_tokens=True).strip()
            
        #Deberta
        elif mod == "deberta":
            model = pipeline(
                "question-answering",
                model="timpal0l/mdeberta-v3-base-squad2"
            )
            
            def call_deberta(question, context):
                result = model(question=question, context=context)
                return result["answer"]
            
        #GPT-OSS
        elif mod == "gpt-oss":
            model_id = "openai/gpt-oss-20b"

            tokenizer = AutoTokenizer.from_pretrained(model_id)

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )

            model.eval()
            
            device = next(model.parameters()).device
            
            def call_gpt(prompt):
                formatted = tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = tokenizer(formatted, return_tensors="pt").to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=30,
                        do_sample=False
                    )
                    
                generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

                return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
        #Mistral
        elif mod == "mistral":
            model_id = "mistralai/Mistral-7B-Instruct-v0.3"

            tokenizer = AutoTokenizer.from_pretrained(model_id)

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                device_map="cpu"
            )

            model.eval()

            device = next(model.parameters()).device

            def call_mistral(prompt):
                # format using chat template
                formatted = tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = tokenizer(formatted, return_tensors="pt").to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=30,
                        do_sample=False
                    )

                # strip prompt tokens
                generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

                return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
        with open(output, "w") as outf:
            for file_name in haystacks_list:
                language, size, semantic_match, order, condition, repetition, repeated_condition = file_name.split("_")
                if language == "en" and size.startswith("small"): #
                    with open(os.path.join(haystack_path, file_name), encoding="utf-8") as hay:
                        full_hay = json.load(hay)
                        version = file_name.split("_")[4]
                        for info, question in question_dict.items():
                            l, v = info.split("_", 1)
                            if v == version:
                                metadata = full_hay['metadata'].strip()
                                prompt_lang = l
                                
                                if mod == "llama":
                                    prompt = [
                                        {"role": "system", "content": f"""{full_hay['haystack'].strip()}"""
                                        },
                                        {"role": "user", "content": f"{question}\n\nAnswer:"},
                                    ]   
                                    answer = call_llama(prompt)
                                    
                                elif mod =="deberta":
                                    answer = call_deberta(question=question, context=f"""{full_hay['haystack'].strip()}\n\nAnswer:""")
                                    answer = answer.translate(str.maketrans(' ', ' ', string.punctuation)).strip()
                                    
                                elif mod == "gpt-oss":
                                    prompt = [
                                        {"role": "system", "content": f"""{full_hay['haystack'].strip()}"""
                                        },
                                        {"role": "user", "content": f"{question}\n\nAnswer:"},
                                    ]
                                    
                                    answer = call_gpt(prompt)
  
                                elif mod == "mistral":
                                    prompt = [
                                        {"role": "system", "content": f"""{full_hay['haystack'].strip()}"""
                                        },
                                        {"role": "user", "content": f"{question}\n\nAnswer:"},
                                    ]
                                    
                                    answer = call_mistral(prompt)
                                    
                                log_probabilities = []

                                if mod in ["llama", "gpt-oss", "mistral"]:
                                    prefix_text = tokenizer.apply_chat_template(
                                        prompt,
                                        tokenize=False,
                                        add_generation_prompt=True
                                    )

                                needles = reconstruct_needles(condition, language)
                                
                                normalized_log_probs = []
                                
                                if not mod == "deberta":                            
                                    for i in range(2):
                                        candidate = needles[i]
                                        message = prefix_text + "\n\n" + candidate
                                        log_prob = compute_surprisal(model, tokenizer, device, message)
                                        log_probabilities.append(log_prob)
                                    normalized_log_probs = normalize_candidate_probs(log_probabilities)
                                
                                print(answer)

                                data = {
                                    "model": mod,
                                    "meat": metadata,
                                    "prompt_lang": prompt_lang,
                                    "question_text": question,
                                    "answer": answer,
                                }

                                if normalized_log_probs:
                                    data["log_prob_unjustified"] = normalized_log_probs[0]
                                    data["log_prob_justified"] = normalized_log_probs[1]

                                print(json.dumps(data), file=outf, flush=True)
                                                        
if __name__ == '__main__':
    main()