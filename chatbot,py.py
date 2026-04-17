from transformers import AutoTokenizer, GPT2LMHeadModel
import torch
import pandas as pd

print("--- INITIALIZING CHATBOT (Assignment 2) ---")

# GPT-2 acts as our sequential model (modern RNN/LSTM alternative)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set the padding token
tokenizer.pad_token = tokenizer.eos_token

print("Chatbot is live! Type your message (or 'exit' to stop).")

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit', 'bye']:
        break
    
    # 1. BERT-style tokenization (Understanding the sequence)
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    
    # 2. Sequential Generation (Predicting the response)
    outputs = model.generate(
        inputs, 
        max_length=50, 
        do_sample=True, 
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Bot: {response}")