import os
from flask import Flask, request, jsonify
from mapping_functions import entity_mapTo_emotion, topic_mapTo_emotion
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()
API_KEY = os.environ.get("API_KEY")

@app.route('/analyse', methods=['POST'])
def modify_text():
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        input_text = data.get('text')
        input_key = data.get('api_key')

        if input_key != API_KEY:
            return jsonify({'error': 'Unauthorized'}), 401
        
        if input_text is None:
            return jsonify({'error': 'Invalid input'}), 400

        sentences = input_text.split('.')  # Split input text into sentences

        try:
            topicsEval = topic_mapTo_emotion(sentences)
            entitiesEval = entity_mapTo_emotion(sentences)

            return jsonify({
                "topic_eval": topicsEval,
                "entity_eval": entitiesEval,
            })
        except Exception as e:
            return jsonify({'error': 'Error processing data: ' + str(e)}), 500
    except Exception as e:
        return jsonify({'error': 'Error processing request: ' + str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
