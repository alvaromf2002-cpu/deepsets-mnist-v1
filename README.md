# Deep Sets Interactive Playground: MNIST Point Cloud

A web-based interactive demonstration of **Deep Sets**, a neural network architecture designed to process unordered sets of data. 

This project transforms standard MNIST images into **2D Point Clouds** to demonstrate the unique properties of Deep Sets: **Permutation Invariance**, robustness to sparsity, and the impact of the **Latent Dimension** on model capacity (as discussed by Wagstaff et al.).

![Demo Screenshot](https://via.placeholder.com/800x400?text=Deep+Sets+Interactive+Demo+Screenshot)
*(Replace this link with an actual screenshot of your web interface once running)*

## 🧠 Theoretical Background

Unlike Convolutional Neural Networks (CNNs) that rely on a fixed grid of pixels, **Deep Sets** treat data as a set $\{x_1, ..., x_M\}$ where the order does not matter.

The architecture is defined by:
$$f(X) = \rho \left( \sum_{x \in X} \phi(x) \right)$$

* **$\phi$ (Encoder):** Processes each point individually to a latent representation.
* **$\sum$ (Aggregation):** Sum-pooling ensures the operation is permutation invariant.
* **$\rho$ (Decoder):** Classifies the global feature vector.

### Key Concept: The Wagstaff Bottleneck
As shown in the original paper *On the Limitations of Representing Functions on Sets*, the dimension of the latent space ($N$) is critical. If $N$ is too small compared to the set size ($M$), the sum operation crushes necessary topological information. This project includes an experiment visualizing this threshold.

## 🚀 Features

This interactive demo allows you to:

1.  **Draw Digits:** Draw a number on an HTML5 canvas.
2.  **Point Cloud Conversion:** See how the image is converted into a sparse set of normalized $(x, y)$ coordinates.
3.  **Adjust Latent Dimension ($N$):** Switch between "Low Capacity" ($N=2$) and "High Capacity" ($N=128$) models to observe the information bottleneck in real-time.
4.  **Decimation ($M$):** Reduce the number of sampled points (e.g., from 100 to 20) to see how robust Deep Sets are to data sparsity.
5.  **Permutation Test:** "Shuffle" the input order of points to verify that the model's prediction remains mathematically identical.

## 📂 Project Structure

* `model.py`: PyTorch implementation of the Deep Set architecture ($\phi$ and $\rho$ networks).
* `dataset.py`: Custom PyTorch Dataset that converts MNIST images to point clouds on the fly.
* `experiment.py`: Script to train multiple models with varying latent dimensions and plot the accuracy curve.
* `prepare_demo.py`: Script to pre-train and save specific models ($N=2, 16, 128$) for the web app.
* `app.py`: Flask backend that serves the model and processes drawing data.
* `templates/index.html`: The frontend interface.

## 🛠️ Installation & Usage

### 1. Clone and Install Dependencies
Ensure you have Python 3.8+ installed.

```bash
git clone [https://github.com/yourusername/deepsets-mnist-demo.git](https://github.com/yourusername/deepsets-mnist-demo.git)
cd deepsets-mnist-demo
pip install torch torchvision numpy matplotlib flask pillow tqdm

```

### 2. Generate Pre-trained Models

Before running the web server, you need to train the models that the demo will use. This script trains three variations () and saves them to `/saved_models`.

```bash
python prepare_demo.py

```

*Note: This utilizes the training logic defined in `train.py`.*

### 3. Run the Web App

Start the Flask server:

```bash
python app.py

```

### 4. Open in Browser

Navigate to `http://127.0.0.1:5000` in your web browser.

## 🧪 Experiments

The repository also contains the code to reproduce the "Accuracy vs. Latent Dimension" graph found in Wagstaff et al.

To run the full experiment:

```bash
python experiment.py

```

This will generate `wagstaff_experiment_result.png` showing the drop in accuracy when .

## 📚 References

1. **Deep Sets**: Zaheer, M., et al. (2017). *Deep Sets*. NIPS.
2. **Limitations**: Wagstaff, E., et al. (2019). *On the Limitations of Representing Functions on Sets*. ICML.
3. **MNIST**: LeCun, Y., et al. *Gradient-based learning applied to document recognition*.

---

*Created for educational purposes to demonstrate the capabilities of Set-based Neural Networks.*

```

```