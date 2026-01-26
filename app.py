# app.py
import torch
import numpy as np
import base64
import io
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageOps
from model import DeepSet

app = Flask(__name__)

# 1. Cargar modelos pre-entrenados en memoria al inicio
device = torch.device("cpu") # CPU es suficiente para inferencia de 1 muestra
models = {}
latent_options = [2, 16, 128]

print("Cargando modelos...")
for dim in latent_options:
    m = DeepSet(latent_dim=dim)
    # Asegúrate de haber ejecutado prepare_demo.py primero
    try:
        m.load_state_dict(torch.load(f"saved_models/deepset_dim_{dim}.pth", map_location=device))
        m.eval()
        models[dim] = m
        print(f"Modelo N={dim} cargado.")
    except:
        print(f"ALERTA: No se encontró 'saved_models/deepset_dim_{dim}.pth'. Ejecuta prepare_demo.py.")

# Función auxiliar similar a tu dataset.py pero flexible
# En app.py

def image_to_pointcloud(image, num_points, shuffle):
    # 1. Convertir a escala de grises
    img = image.convert('L')
    
    # 2. INVERTIR COLORES: 
    # El canvas manda fondo blanco (255) y tinta negra (0).
    # Queremos fondo negro (0) y tinta blanca (255) como MNIST.
    img = ImageOps.invert(img)
    
    # 3. Redimensionar a 28x28
    # Usamos BILINEAR para mantener la suavidad del trazo al reducir
    img = img.resize((28, 28), resample=Image.BILINEAR)
    img_np = np.array(img)
    
    # 4. UMBRALIZADO (Threshold):
    # Al reducir la imagen, los bordes quedan grises. 
    # Filtramos para quedarnos solo con la tinta fuerte (> 50).
    coords = np.argwhere(img_np > 50).astype(np.float32)
    
    # --- El resto sigue igual ---
    
    # Normalizar (0 a 1)
    coords /= 28.0
    
    current_points = coords.shape[0]
    
    if current_points > 0:
        # Si hay más puntos de los necesarios, muestreamos
        # Si hay menos, usamos replace=True para rellenar repitiendo
        choice_idx = np.random.choice(current_points, num_points, replace=True)
        point_set = coords[choice_idx, :]
    else:
        # Caso raro: imagen vacía
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
    
    # Decodificar imagen
    image_data = image_data.split(",")[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    
    # Preprocesar
    tensor_points = image_to_pointcloud(image, m_points, do_shuffle)
    
    # Inferencia
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