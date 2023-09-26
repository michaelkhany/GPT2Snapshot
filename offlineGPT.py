import os
import subprocess
import sys
import pkg_resources
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

def install_packages():
    required = {'torch', 'transformers'}
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = required - installed

    if missing:
        print(f"Installing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])

install_packages()

def check_and_download_model(model_name, save_directory):
    model_path = os.path.join(save_directory, 'pytorch_model.bin')
    tokenizer_config_path = os.path.join(save_directory, 'tokenizer_config.json')
    
    if not os.path.exists(model_path) or not os.path.exists(tokenizer_config_path):
        print(f"Model not found in {save_directory}. Downloading...")
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        print(f"Model downloaded and saved in {save_directory}.")

def load_model_and_tokenizer(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    return model, tokenizer

def summarize_text(model, tokenizer, text):
    input_ids, attention_mask = process_input(tokenizer, text)
    with torch.no_grad():
        output = model.generate(input_ids, max_length=150, num_return_sequences=1, temperature=1.2, do_sample=True, attention_mask=attention_mask)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def split_text_into_chunks(text, tokenizer, max_length=1024):
    words = text.split()
    chunks, chunk = [], []
    for word in words:
        potential_chunk = chunk + [word]
        if len(tokenizer.encode(' '.join(potential_chunk))) < max_length:
            chunk = potential_chunk
        else:
            chunks.append(' '.join(chunk))
            chunk = [word]
    if chunk:
        chunks.append(' '.join(chunk))
    return chunks

def process_input(tokenizer, text, max_length=1024):
    input_ids = tokenizer.encode(text, return_tensors='pt')
    if len(input_ids[0]) > max_length:
        input_ids = input_ids[:, :max_length]
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    return input_ids, attention_mask

def main():
    model_path = './models'
    model_name = 'gpt2'

    check_and_download_model(model_name, model_path)
    model, tokenizer = load_model_and_tokenizer(model_path)

    # Directly get user's task or prompt
    user_task = input("Enter your task or prompt for the model: ")

    if len(tokenizer.encode(user_task)) <= 1024:
        response = summarize_text(model, tokenizer, user_task)
        print(response)
    else:
        chunks = split_text_into_chunks(user_task, tokenizer)
        responses = []
        for chunk in chunks:
            response = summarize_text(model, tokenizer, chunk)
            responses.append(response)
        print(' '.join(responses))

if __name__ == "__main__":
    main()
