from flask import Flask, render_template, request, jsonify
from pyngrok import ngrok
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import threading
import time

# Load your fine-tuned model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("/content/drive/MyDrive/AAI520-NLP/Final_Project/FlaskApp")
model = GPT2LMHeadModel.from_pretrained("/content/drive/MyDrive/AAI520-NLP/Final_Project/FlaskApp")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize Flask app
app = Flask(__name__)

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to get chatbot response
@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']

    # Generate response using your chatbot model
    response = generate_response(user_input, model, tokenizer)

    # Return the chatbot's response
    return jsonify({'response': response})

def generate_response(prompt, model, tokenizer, max_length=50):
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

def run_flask():
    app.run()

# Set up ngrok tunnel
public_url = ngrok.connect(5000)
print(f" * Ngrok tunnel opened at {public_url}")

# Start Flask in a separate thread
thread = threading.Thread(target=run_flask)
thread.start()

# Give Flask some time to start
time.sleep(5)

