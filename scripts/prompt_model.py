import json
import string
import os
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
    
    # needles  = {
    #     "de": [
    #     "ungerechtfertigt",
    #     "gerechtfertigt",
    #         ],
    #     "en": [
    #     "unjustified",
    #     "justified",
    #             ],
    #     }
    
    return needles[language]

def main():
    output = "llm_outputs.jsonl"
    haystack_path = "all_haystacks"
    with open("questions.json", "r", encoding="utf-8") as j:
        question_dict = json.load(j)
        
    
    haystacks_list = os.listdir(haystack_path)
    model_dict = {"llama": "1b_8b", "deberta" :"deberta", "gpt": "gpt-oss", "mistral": "7bi_7b", "qwen": "qwen"}
    # for fam, mod in model_dict.items():
    for mod in ["llama"]:#["llama", "deberta", "gpt-oss", "mistral", "qwen"]
        
        #LLama
        if mod == "llama":
            model_id = "meta-llama/Llama-3.2-1B-Instruct"

            # if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            #     dtype = torch.bfloat16
            # else:
            #     dtype = torch.float32

            tokenizer = AutoTokenizer.from_pretrained(model_id)

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                device_map="cpu"
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
            
            def call_gpt(prompt):
                formatted = tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = tokenizer(formatted, return_tensors="pt")
                inputs = {k: v.to(model.get_input_embeddings().weight.device) for k, v in inputs.items()}

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
                # format using chat template
                formatted = tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = tokenizer(formatted, return_tensors="pt")
                inputs = {k: v.to(model.get_input_embeddings().weight.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=30,
                        do_sample=False
                    )

                # strip prompt tokens
                generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

                return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        elif mod == "qwen":
            model_name = "Qwen/Qwen2.5-7B-Instruct-1M"

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
    
            def call_qwen(prompt):
                formatted = tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                inputs = tokenizer(formatted, return_tensors="pt")
                inputs = {k: v.to(model.get_input_embeddings().weight.device) for k, v in inputs.items()}

                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens= 100
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
                ]

                return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
         
        with open(output, "w") as outf:
            for file_name in haystacks_list:
                language, size, semantic_match, order, condition, repetition, repeated_condition, needle_language = file_name.split("_")
                if size.startswith("small"): #
                    with open(os.path.join(haystack_path, file_name), encoding="utf-8") as hay:
                        full_hay = json.load(hay)
                        version = file_name.split("_")[4]
                        for info, question in question_dict.items():
                            l, v = info.split("_", 1)
                            if v == version:
                                metadata = full_hay['metadata'].strip()
                                prompt_lang = l
                                
                                messages = [
                                    {"role": "system", "content":f"Reasoning: low\n{full_hay['haystack'].strip()}"},
                                    {"role": "user", "content": f"{question}\n\nAnswer:"},
                                ]

                                if mod == "llama":
                                    answer = call_llama(messages)
                                    
                                elif mod =="deberta":
                                    answer = call_deberta(question=question, context=f"""{full_hay['haystack'].strip()}""")
                                    answer = answer.translate(str.maketrans(' ', ' ', string.punctuation)).strip()
                                    
                                elif mod == "gpt-oss":
                                    answer = call_gpt(messages)
  
                                elif mod == "mistral":
                                    answer = call_mistral(messages)
                                
                                elif mod == "qwen":
                                    answer = call_qwen(messages)
                                
                                log_probabilities = []

                                if mod in ["llama", "gpt-oss", "mistral"]:
                                    prefix_text = tokenizer.apply_chat_template(
                                        messages,
                                        tokenize=False,
                                        add_generation_prompt=True
                                    )

                                needles = reconstruct_needles(condition, language)
                                normalized_candidate_probs = []
                                
                                if not mod == "deberta":       
                                    device = model.get_input_embeddings().weight.device                     
                                    for i in range(2):
                                        candidate = needles[i]
                                        log_prob = compute_surprisal(model, tokenizer, device, prefix_text, candidate)
                                        log_probabilities.append(log_prob)
                                    normalized_candidate_probs = normalize_candidate_probs(log_probabilities)
                                
                                print(answer)

                                data = {
                                    "model": mod,
                                    "meat": metadata,
                                    "prompt_lang": prompt_lang,
                                    "needle_lang": needle_language,
                                    "question_text": question,
                                    "answer": answer,
                                }

                                if normalized_candidate_probs:
                                    data["normalized_candidate_prob_unjustified"] = normalized_candidate_probs["last"]["candidate_A"]
                                    data["normalized_candidate_prob_justified"] = normalized_candidate_probs["last"]["candidate_B"]
                                    data["decision_progression"] = normalized_candidate_probs

                                print(json.dumps(data), file=outf, flush=True)
        del model
        del tokenizer
        torch.cuda.empty_cache()
                                                        
if __name__ == '__main__':
    main()