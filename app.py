from flask import Flask, request, jsonify
from flask_cors import CORS
import question_answer_generation
import os 

app = Flask(__name__)
CORS(app)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    context = data.get('context')
    if not context:
        return jsonify({"error": "No context available."}), 400
    result = question_answer_generation.generate_questions_and_answers(context)
    return jsonify({"questions_and_answers": result})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 5000))