import json
import os
from flask import Flask, request, Response
from flask_cors import CORS
from res import ResidualModel
import tensorflow as tf
from waitress import serve

UPLOAD_FOLDER = 'temp'

app = Flask(__name__)
cors = CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = ResidualModel('models')


def on_get_request():
    return '''
        <h1>Upload new File</h1>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="image">
            <select name="model">
                <option value="mse">Mean Squared Error</option>
                <option value="perceptual">Perceptual</option>
                <option value="gan">Generative Adversarial Network</option>
            </select>
            <input type="checkbox" id="noise_reduction" name="noise_reduction" value="True">
            <label for="noise_reduction">Noise Reduction</label><br>
            <input type="submit">
        </form>
        '''


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'image' not in request.files:
            return Response("No 'image' Found on Request", status=400)
        file1 = request.files['image']
        model_type = request.form['model']
        noise_reduction = True if 'noise_reduction' in request.form else False

        path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        file1.save(path)

        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)

        if (img.shape[0] > 512) or (img.shape[1] > 512):
            return Response(json.dumps(
                {'message': "Image is too large, sadly right now we cannot perform up-scaling because of hardware limitation :("}),
                status=200, mimetype='application/json'
            )

        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.expand_dims(img, 0) * 255.

        img_2x = model(img, model_type, denoiser=noise_reduction)

        img_2x = tf.image.encode_png(img_2x)
        img_2x = img_2x.numpy()
        return Response(img_2x, mimetype='image/png')
    return on_get_request()


# app.run(host="10.10.10.13", port=8080)
serve(app, host="10.10.10.13", port=8080)