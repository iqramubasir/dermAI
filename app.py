from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

# ── Universal Keras layer patch (must be FIRST before anything else) ──────────
from tensorflow.keras.layers import Layer

_UNSUPPORTED_KWARGS = {'renorm', 'renorm_clipping', 'renorm_momentum', 'quantization_config'}

_original_layer_init = Layer.__init__

def _patched_layer_init(self, *args, **kwargs):
    for key in _UNSUPPORTED_KWARGS:
        kwargs.pop(key, None)
    _original_layer_init(self, *args, **kwargs)

Layer.__init__ = _patched_layer_init
# ─────────────────────────────────────────────────────────────────────────────

from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

app = Flask(__name__, static_folder='static')
CORS(app)

print("Loading skin model...")
vgg = VGG16(weights=None, include_top=False, input_shape=(150, 150, 3))
inp = tf.keras.Input(shape=(150, 150, 3), name='vgg16_input')
x = vgg(inp)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d_1')(x)
x = layers.MaxPooling2D((2, 2), name='max_pooling2d')(x)
x = layers.Dropout(0.5, name='dropout')(x)
x = layers.Flatten(name='flatten')(x)
out = layers.Dense(5, activation='softmax', name='dense_1')(x)
skin_model = models.Model(inputs=inp, outputs=out)
skin_model.load_weights('model/skin_model_fixed.h5', by_name=True, skip_mismatch=True)
print("Skin model ready!")

print("Loading acne model...")
acne_model = tf.keras.models.load_model('model/acne_model_fixed.h5', compile=False)
print("Acne model ready!")

SKIN_LABELS = ['dry', 'normal', 'oily', 'combination', 'sensitive']
ACNE_LABELS = ['no_acne', 'acne']

def preprocess(image_bytes, size):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    try:
        img_bytes = request.files['image'].read()

        img_skin = preprocess(img_bytes, (150, 150))
        skin_preds = skin_model.predict(img_skin, verbose=0)[0]
        skin_idx = int(np.argmax(skin_preds))
        skin_conf = round(float(skin_preds[skin_idx]) * 100, 1)

        img_acne = preprocess(img_bytes, (224, 224))
        acne_preds = acne_model.predict(img_acne, verbose=0)[0]
        acne_idx = int(np.argmax(acne_preds))
        has_acne = ACNE_LABELS[acne_idx] == 'acne'
        acne_conf = round(float(acne_preds[acne_idx]) * 100, 1)

        all_probs = {SKIN_LABELS[i]: round(float(skin_preds[i]) * 100, 1) for i in range(len(SKIN_LABELS))}

        return jsonify({
            'skin_type': SKIN_LABELS[skin_idx].capitalize(),
            'skin_confidence': skin_conf,
            'has_acne': has_acne,
            'acne_confidence': acne_conf,
            'all_skin_probs': all_probs
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'running', 'models': 'loaded'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
