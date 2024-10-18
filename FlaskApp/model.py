# Load your fine-tuned model and tokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

tokenizer = GPT2Tokenizer.from_pretrained("/content/drive/MyDrive/AAI520-NLP/Final_Project/FlaskApp")
model = GPT2LMHeadModel.from_pretrained("/content/drive/MyDrive/AAI520-NLP/Final_Project/FlaskApp")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def generate_response(prompt, max_length=50):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        early_stopping=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    for end_char in [".", "!", "?"]:
        if end_char in response:
            response = response.split(end_char)[0] + end_char
            break
    return response