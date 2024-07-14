1. **Imports and Setup**:
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   import torch

   torch_device = "cuda" if torch.cuda.is_available() else "cpu"
   ```
   - First, we import the necessary libraries. We use `AutoModelForCausalLM` and `AutoTokenizer` from the Hugging Face Transformers library and `torch` from PyTorch.
   - We check if a GPU is available. If it is, we set the device to "cuda"; otherwise, we use the CPU.

2. **Load Model and Tokenizer**:
   ```python
   tokenizer = AutoTokenizer.from_pretrained("gpt2")
   model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id).to(torch_device)
   ```
   - We load the pre-trained GPT-2 tokenizer, which converts text to tokens.
   - Then, we load the GPT-2 model and move it to the appropriate device. We set the padding token ID to the end-of-sequence token ID to handle padding correctly.

3. **User Inputs**:
   ```python
   user_prompt = input("Enter text to generate prompt:")
   max_words = int(input("Enter max words:"))
   max_sequences = int(input("Enter max sequences:"))
   ```
   - We prompt the user to enter the initial text prompt.
   - The user also specifies the maximum number of words for the generated text and the number of different sequences to generate.

4. **Tokenization**:
   ```python
   model_inputs = tokenizer(user_prompt, return_tensors='pt').to(torch_device)
   ```
   - We convert the user's input prompt into tokens and format it as PyTorch tensors.
   - We move the tokenized input to the specified device.

5. **Set Seed**:
   ```python
   from transformers import set_seed

   set_seed(42)
   ```
   - We set a random seed for reproducibility. This ensures the results are the same each time the code runs with the same inputs.

6. **Generate Text**:
   ```python
   sample_outputs = model.generate(
       **model_inputs,
       max_new_tokens=max_words,
       do_sample=True,
       top_k=50,
       top_p=0.95,
       no_repeat_ngram_size=2,
       temperature=0.4,
       num_return_sequences=max_sequences,
   )
   ```
   - We call the `generate` method on the model with various parameters:
     - `max_new_tokens`: Maximum words for the output.
     - `do_sample`: Enables sampling for diversity.
     - `top_k` and `top_p`: Control the sampling strategy.
     - `no_repeat_ngram_size`: Prevents repeating n-grams.
     - `temperature`: Adjusts the randomness.
     - `num_return_sequences`: Number of sequences to generate.

7. **Decode and Print**:
   ```python
   samples = [
       tokenizer.decode(sample_output, skip_special_tokens=True)
       for sample_output in sample_outputs
   ]

   result = max(samples, key=len)

   last_full_stop_index = result.rfind('.')

   if last_full_stop_index != -1:
       result = result[:last_full_stop_index + 1]

   print("Generated Expansions Based On Context:\n" + 100 * '-')
   print(result)
   ```
   - We decode the generated token sequences back into text.
   - We find the longest sequence and trim it to the last full stop for a complete sentence.
   - Finally, we print the generated text.


This explanation covers each step in a clear and straightforward manner.
