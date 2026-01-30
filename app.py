import torch
import numpy as np
import base64
import io
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageOps
from model import DeepSet

app = Flask(__name__)

# load pretrained models
device = torch.device("cpu")
models = {}
latent_options = [2, 16, 128]

for dim in latent_options:
    m = DeepSet(latent_dim=dim)
    try:
        m.load_state_dict(torch.load(f"saved_models/deepset_dim_{dim}.pth", map_location=device))
        m.eval()
        models[dim] = m
    except:
        print(f"model not found in 'saved_models/deepset_dim_{dim}.pth', run prepare_demo.py.")

def image_to_pointcloud(image, num_points, shuffle):
    # convert to grayscale
    img = image.convert('L')
    
    # convert to black background (0) and white ink (255) like MNIST
    img = ImageOps.invert(img)
    
    # redimension to 28x28
    img = img.resize((28, 28), resample=Image.BILINEAR)
    img_np = np.array(img)
    
    # threshold to keep only white (avoid grey '>50')
    coords = np.argwhere(img_np > 50).astype(np.float32)
    
    # normalize
    coords /= 28.0
    
    current_points = coords.shape[0]
    
    if current_points > 0:
        choice_idx = np.random.choice(current_points, num_points, replace=True)
        point_set = coords[choice_idx, :]
    else:
        # empty image case handling
        point_set = np.zeros((num_points, 2), dtype=np.float32)
    
    if shuffle:
        np.random.shuffle(point_set)
        
    return torch.from_numpy(point_set).unsqueeze(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_data = data['image'] # Base64 string
    n_dim = int(data['n_dim'])
    m_points = int(data['m_points'])
    do_shuffle = data['shuffle']
    
    # img decode
    image_data = image_data.split(",")[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    
    # preproc
    tensor_points = image_to_pointcloud(image, m_points, do_shuffle)
    
    # inference
    if n_dim not in models:
        return jsonify({'error': 'Modelo no cargado'}), 500
        
    with torch.no_grad():
        output = models[n_dim](tensor_points)
        probs = torch.nn.functional.softmax(output, dim=1)
        
    return jsonify({
        'probabilities': probs.tolist()[0],
        'points_visual': tensor_points.tolist()[0] # Devolver puntos para pintar en JS
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)