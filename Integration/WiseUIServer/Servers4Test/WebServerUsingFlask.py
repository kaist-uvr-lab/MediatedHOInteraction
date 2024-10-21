# -*- coding: utf-8 -*-

from flask import Flask, request
import os

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' in request.files:
        image = request.files['image']
        image_name = 'received_image.png'
        #image.save(image_name)
        return 'Successfully received and saved image.', 200
    else:
        return 'No image received.', 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
