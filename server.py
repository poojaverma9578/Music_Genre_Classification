from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import main
import upload
import recommend
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/process_name', methods=['POST'])
def receive_and_process_name():
    data = request.get_json()

    if 'name' in data:
        name = data['name']
        result = main.classify(name)
        return jsonify({'result': result})
    else:
        return jsonify({'error': 'Name not provided in the request.'})

@app.route('/upload',methods=['POST'])
def upload_audio():
    if 'audioFile' not in request.files:
        return jsonify({'error': 'No audio file part'})

    audio_file = request.files['audioFile']

    if audio_file.filename == '':
        return jsonify({'error': 'No selected audio file'})

    if audio_file:
        audio_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'audio.wav')
        audio_file.save(audio_filename)
        res = upload.mfcc_process()
        return jsonify({'Genre': res})

@app.route('/recommend',methods=['POST'])
def recommend_song():
    data = request.get_json()

    if 'name' in data:
        name = data['name']
        result = recommend.recommendation(name)
        return jsonify(result)
    else:
        return jsonify({'error': 'Name not provided in the request.'})

@app.route('/recommend_audio',methods=['POST'])
def recommend_audio():
    print('yoyo')
    result = recommend.audio_recc()
    print(result)
    return jsonify(result)

if __name__ == '__main__':
    app.run()
