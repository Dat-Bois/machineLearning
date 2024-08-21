'''
export FLASK_APP=main.py
export FLASK_ENV=development

'''

from flask import Flask, jsonify, request
from torch_utils import transform_image, get_prediction

app = Flask(__name__)

def allowed_file(filename : str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['png', 'jpg', 'jpeg']


@app.route('/predict', methods=['POST'])
def predict():

    # 1 load image
    # 2 image -> tensor
    # 3 make a prediction
    # 4 return prediction as json

    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})
        
        try:
            img_bytes = file.read()
            tensor = transform_image(img_bytes)
            prediction = get_prediction(tensor)
            data = {'prediction': prediction.item(), 'class_name' : str(prediction.item())} # item() to get the value of the tensor
            return jsonify(data)
        except:
            return jsonify({'error': 'error during prediction'})