import json
import string
import os
from transformers import AutoTokenizer, AutoModelForCausalLM    
from huggingface_hub import login
import torch
from transformers import pipeline
from surprisal import compute_surprisal, normalize_candidate_probs
import sys

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
            f"Der jüngste Angriff, Operation Mango, auf {condition[0]} durch {condition[1]} wurde als unnötig angesehen.",
            f"Der jüngste Angriff, Operation Mango, auf {condition[0]} durch {condition[1]} wurde als notwendig angesehen.",
                ],
            "en": [
            f"The recent attack, Operation Mango, against {condition[0]} by {condition[1]} was deemed unnecessary.",
            f"The recent attack, Operation Mango, against {condition[0]} by {condition[1]} was deemed necessary.",
                    ],
            }
    
    # needles  = {
    #     "de": [
    #     "unnötig",
    #     "notwendig",
    #         ],
    #     "en": [
    #     "unnecessary",
    #     "necessary",
    #             ],
    #     }
    
    return needles[language]

def main():
    output = "llm_outputs.jsonl"
    haystack_path = "all_haystacks"
    with open("questions.json", "r", encoding="utf-8") as j:
        question_dict = json.load(j)
        
    counter = 1
    haystacks_list = os.listdir(haystack_path)
    models = [
#    ("deberta", "/mimer/NOBACKUP/groups/naiss2026-4-124/gustav/.cache/huggingface/hub/models--timpal0l--mdeberta-v3-base-squad2/snapshots/08d6e89c7a6557f967db2e1021f7f640483400ed"),                                                                                                                                                                         
#    ("qwen", "/mimer/NOBACKUP/groups/naiss2026-4-124/gustav/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct-1M/snapshots/e28526f7bb80e2a9c8af03b831a9af3812f18fba"),  
#    ("qwen", "/mimer/NOBACKUP/groups/naiss2026-4-124/gustav/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"),    
#    ("llama", "/mimer/NOBACKUP/groups/naiss2026-4-124/gustav/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6"),                                                                                                                                                                           
    ("mistral", "/mimer/NOBACKUP/groups/naiss2026-4-124/gustav/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots/caa1feb0e54d415e2df31207e5f4e273e33509b1"),
    ("mistral", "/mimer/NOBACKUP/groups/naiss2026-4-124/gustav/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/c170c708c41dac9275d15a8fff4eca08d52bab71"),
    ("gpt-oss", "/mimer/NOBACKUP/groups/naiss2026-4-124/gustav/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee")
    ]
    with open(output, "a") as outf:
        for mod, variant in models:
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
            elif mod == "deberta":
                model = pipeline(
                    "question-answering",
                    model= variant
                )
                
                def call_deberta(question, context):
                    result = model(question=question, context=context)
                    return result["answer"]

                
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
                
                def call_gpt(messages):
                    formatted = (
                        "You are a concise assistant.\n"
                        "Do not repeat the question.\n"
                        f"Question: {messages[1]['content']}\n"
                        "Answer:"
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
                

            for file_name in haystacks_list:
                language, size, semantic_match, order, condition, repetition, repeated_condition, needle_language = file_name.split("_")
                if repeated_condition == "None" and size.startswith("small"): #
                    with open(os.path.join(haystack_path, file_name), encoding="utf-8") as hay:
                        full_hay = json.load(hay)
                        version = file_name.split("_")[4]
                        for info, question in question_dict.items():
                            for i, pc in enumerate(["notacc", "acc"]):
                                l, v = info.split("_", 1)
                                if v == version:
                                    metadata = full_hay['metadata'].strip()
                                    prompt_lang = l
                                
                                    messages = [
                                        {"role": "system", "content": "Reasoning: low\n Answer with only the final answer."},
                                        {"role": "user", "content": f"Context: {full_hay['haystack'].strip()}\nQuestion: {question[i]}\n\nAnswer:"},
                                    ]

                                    if mod == "llama":
                                        answer = call_llama(messages)
                                    
                                    elif mod =="deberta":
                                        answer = call_deberta(question=question[i], context=f"""{full_hay['haystack'].strip()}""")
                                        answer = answer.translate(str.maketrans(' ', ' ', string.punctuation)).strip()
                                    
                                    elif mod == "gpt-oss":
                                        answer = call_gpt(messages)
                                    
                                    elif mod == "mistral":
                                        answer = call_mistral(messages)

                                    elif mod == "qwen":
                                        answer = call_qwen(messages)
                                    
                                    log_probabilities = []

                                    if mod == "gpt-oss":
                                        prefix_text = (
                                            f"system: {messages[0]['content']}\n"
                                            f"user: {messages[1]['content']}\n"
                                            "assistant:"
                                        )

                                    elif mod in ["llama", "mistral", "qwen"] and getattr(tokenizer, "chat_template", None):
                                        prefix_text = tokenizer.apply_chat_template(
                                            messages,
                                            tokenize=False,
                                            add_generation_prompt=True
                                        )

                                    else:
                                        # fallback formatting for models without chat template
                                        if isinstance(messages, list):
                                            parts = []
                                            for m in messages:
                                                parts.append(m["content"].strip())
                                            prefix_text = "\n".join(parts)

                                            if not prefix_text.rstrip().endswith(("Answer:", "Assistant:")):
                                                prefix_text += "\nAnswer:"
                                        else:
                                            prefix_text = messages
                            
                                    needles = reconstruct_needles(condition, language)
                        
                                    normalized_candidate_probs = []
                            
                                    if not mod == "deberta":
                                        device = model.get_input_embeddings().weight.device
                                        for it in range(2):
                                            candidate = needles[it]
                                            log_prob = compute_surprisal(model, tokenizer, device, prefix_text, candidate)
                                            log_probabilities.append(log_prob)
                                        normalized_candidate_probs = normalize_candidate_probs(log_probabilities)
                                
                                    print(answer)

                                    data = {
                                        "iteration": counter,
                                        "model": mod,
                                        "meat": metadata,
                                        "prompt_lang": prompt_lang,
                                        "needle_language": needle_language,
                                        "question_text": question[i],
                                        "question_condition": pc, 
                                        "answer": answer,
                                    }


                                    if normalized_candidate_probs:
                                        data["p_unnecessary"] = normalized_candidate_probs["last"]["candidate_A"]
                                        data["p_necessary"] = normalized_candidate_probs["last"]["candidate_B"]
                                        data["decision_progression"] = normalized_candidate_probs

                                
                                    print(json.dumps(data), file=outf, flush=True)
                                    counter += 1
            torch.cuda.empty_cache()            
if __name__ == '__main__':
    main()
