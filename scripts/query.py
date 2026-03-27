from transformers import AutoTokenizer, AutoModelForCausalLM  
import torch

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

prompt = [
    {"role": "system", "content": "You are a helpful assistant that answers questions clearly and concisely."},
    {"role": "user", "content": "Was the 2024 attack on Cornwall justified?\n\nAnswer:"},
]

if __name__ == '__main__':
    print(call_gpt(prompt))