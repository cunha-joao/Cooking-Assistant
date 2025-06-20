from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import localrag

app = Flask(__name__, static_folder='static')
CORS(app)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message')

    try:
        resposta = localrag.chat_from_web(user_input)
        return jsonify({'response': resposta})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
