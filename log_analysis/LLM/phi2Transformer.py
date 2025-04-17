from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

#model_id = "microsoft/phi-2"
#model_id = "tiiuae/falcon-rw-1b"  # also small and fast
model_id = "microsoft/phi-1_5"  # ~1.3B parameters

tokenizer = AutoTokenizer.from_pretrained(model_id)
#model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

prompt = "Hi who are you ?"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
