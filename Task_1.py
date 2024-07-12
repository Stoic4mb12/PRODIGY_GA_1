from transformers import AutoModelForCausalLM, AutoTokenizer

import torch

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("gpt2")

model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id).to(torch_device)


user_prompt = input("Enter text to generate prompt:")
max_words = int(input("Enter max words:"))
max_sequences = int(input("Enter max sequences:"))

model_inputs = tokenizer(user_prompt, return_tensors='pt').to(torch_device)

from transformers import set_seed

set_seed(42)

sample_outputs = model.generate(
    **model_inputs,
    max_new_tokens=max_words,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    no_repeat_ngram_size=2,
    temperature=.4,
    num_return_sequences=max_sequences,
)

print("Generated Expansions Based On Context:\n"+ 100 * '-')

samples = [
    tokenizer.decode(sample_output, skip_special_tokens=True)
    for sample_output in sample_outputs
]

result = max(samples,key=len)

last_full_stop_index = result.rfind('.')

if last_full_stop_index != -1:
    result = result[:last_full_stop_index + 1]

print(result)