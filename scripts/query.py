from transformers import AutoTokenizer, AutoModelForCausalLM  
import torch

model_path = "/mimer/NOBACKUP/groups/naiss2026-4-124/gustav/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/c170c708c41dac9275d15a8fff4eca08d52bab71"

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
)

model.eval()

device = next(model.parameters()).device

def call_gpt(prompt):
    # Manually format the prompt as a string
    formatted = (
        f"system: {prompt[0]['content']}\n"
        f"user: {prompt[1]['content']}\n"
        "assistant:"
    )

    # Tokenize the formatted text
    inputs = tokenizer(formatted, return_tensors="pt").to(device)

    # Generate output tokens
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )

    # Extract only the newly generated tokens
    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

    # Decode to string
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    
prompt = [
    {"role": "system", "content": "You are a helpful assistant that answers questions clearly and concisely."},
    {"role": "user", "content": "Was the 2024 attack on Cornwall justified?\n\nAnswer:"},
]

if __name__ == '__main__':
    print(call_gpt(prompt))
