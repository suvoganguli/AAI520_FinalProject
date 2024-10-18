from flask import Flask, render_template, request, jsonify
from pyngrok import ngrok
from .model import get_response
import threading
import time
from flask_cors import CORS


# Initialize Flask app
app = Flask(__name__)

# Enable CORS for the app, allowing only the specific domain
CORS(app, resources={r"/*": {"origins": ["https://kay-q-mich.github.io", "https://suvoganguli.github.io", "https://H-Jan.github.io"]}})

# Route to get chatbot response
@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']

    # Generate response using your chatbot model
    response = generate_response(user_input)

    # Return the chatbot's response
    return jsonify({'response': response})


def run_flask(auth_token, port=5000):
    ngrok.set_auth_token(auth_token)
    # Set up ngrok tunnel
    public_url = ngrok.connect(port)
    print(f" * Ngrok tunnel opened at {public_url}")
    app.run(port=5000)



