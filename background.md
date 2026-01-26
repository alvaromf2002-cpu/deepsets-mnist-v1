Here is the document formatted as an academic paper in Markdown. It is designed to be technical, rigorous, and educational, incorporating the specific theories from the provided sources and the results of your experiment.

---

# Empirical Analysis of Latent Space Dimensionality in Deep Sets Architectures using MNIST Point Clouds

**Abstract**

This study investigates the structural limitations of permutation-invariant neural networks, specifically the **Deep Sets** architecture proposed by Zaheer et al. (2017). While Deep Sets provides a theoretical framework for universal function approximation on sets, subsequent work by Wagstaff et al. (2019) suggests that the dimensionality of the latent embedding space imposes a critical information bottleneck. By transforming the MNIST dataset into 2D point clouds of fixed cardinality , we perform a controlled experiment varying the latent dimension . Our results empirically demonstrate the "bottleneck" phenomenon at low dimensions () and reveal a saturation point at , suggesting that the intrinsic dimensionality of the classification task allows for convergence below the theoretical lower bound required for universal representation.

---

## 1. Introduction

Machine learning typically relies on structured data inputs, such as fixed-length vectors or ordered sequences. However, many engineering applications—ranging from LiDAR processing in autonomous vehicles to sensor network aggregation—produce data in the form of **sets**. Sets are characterized by their permutation invariance; the order of elements does not alter the information content of the set.

Traditional neural networks, such as Multilayer Perceptrons (MLPs) or Recurrent Neural Networks (RNNs), depend on input ordering. Feeding a set  into an MLP yields a different output than , violating the fundamental property of the data structure.

To address this, Zaheer et al. introduced **Deep Sets**, an architecture that enforces permutation invariance through symmetric operations. While successful, Wagstaff et al. later questioned the representational capacity of these models, specifically regarding the dimension of the latent space used for aggregation. They conjectured that to map a set of size  injectively, the latent dimension  must satisfy .

This paper replicates the theoretical conditions of these studies using a simplified geometric task: classifying MNIST digits represented as point clouds. We aim to empirically verify the relationship between latent space capacity and classification accuracy.

---

## 2. Theoretical Framework

### 2.1 Permutation Invariance and Deep Sets

A function  acting on a set  is permutation invariant if for any permutation :


.

Zaheer et al. proved that any such function defined on a countable set can be decomposed as:


where  maps elements to a latent space and  maps the aggregated sum to the output. The summation operation is commutative, thereby guaranteeing permutation invariance.

### 2.2 The Topological Bottleneck

Wagstaff et al. argue that the decomposition  relies heavily on the dimensionality of the latent space (the codomain of ). If we define the aggregation map as , then  must be **injective** (one-to-one) to distinguish between any two distinct sets  and .

Using arguments from topology, specifically regarding embeddings of , Wagstaff et al. demonstrate that a continuous, injective mapping of sets of cardinality  into a Euclidean space of dimension  is only guaranteed if . If , the model forces a "bottleneck" where distinct sets may map to the same latent vector, rendering them indistinguishable to the decoder .

---

## 3. Methodology

### 3.1 Dataset: Spatial MNIST

To simulate a set-based engineering problem, we transformed the MNIST dataset. Instead of processing  pixel grids with Convolutional Neural Networks (CNNs), we converted digits into **Point Clouds**:

1. **Extraction:** Coordinates  of active pixels (intensity ) are extracted.
2. **Normalization:** Coordinates are scaled to the range .
3. **Cardinality Enforcement:** Each input set  is forced to a fixed size . Images with fewer points are padded/repeated; images with more are subsampled.

This formulation forces the model to learn the *topology* of the digit from an unordered bag of 2D vectors.

### 3.2 Model Architecture

We employed a standard Deep Sets architecture consisting of two distinct MLPs:

* **Encoder ():** A 3-layer MLP with ReLU activations. This network lifts each 2D point into the -dimensional latent space.
* **Aggregator:** Global Sum Pooling ().
* **Decoder ():** A 2-layer MLP mapping the latent sum to the 10 class logits.

### 3.3 Experimental Design

We trained the model for 20 epochs using the Adam optimizer and Cross Entropy Loss. The independent variable was the **Latent Dimension ()**, which was varied across the set . The set size was fixed at .

---

## 4. Results

The experiment yielded the following relationship between Latent Dimension and Test Accuracy.

*Figure 1: Impact of Latent Dimension () on classification accuracy. The red dashed line indicates the theoretical bound  proposed by Wagstaff et al.*

### 4.1 Observations

1. **Structural Failure at Low :** At , the model achieves only  accuracy. This confirms the existence of the information bottleneck; a 2D vector cannot preserve the geometric variance of a 100-point set.
2. **Rapid Convergence:** Between  and , accuracy improves drastically as the bottleneck widens.
3. **Saturation:** Performance plateaus around  ( accuracy), showing diminishing returns for  and .

---

## 5. Discussion

### 5.1 Validation of the Bottleneck

The poor performance at  empirically validates the concerns raised by Wagstaff et al. When the latent space is too small, the summation operation destroys topological information, causing "collisions" where different digits map to similar aggregate vectors. This proves that  cannot be an arbitrary compression; it requires sufficient width to maintain separability.

### 5.2 Task Specificity vs. Universal Representation

Wagstaff et al. proved that  is necessary for **universal** function representation (i.e., representing *any* possible function on sets of size ). However, our results show convergence at , which is significantly less than .

This apparent contradiction is explained by the **intrinsic dimensionality** of the task. Classification is a "coarser" task than reconstruction or universal approximation. The manifold of valid MNIST digits occupies a small subspace of all possible 100-point arrangements. As noted by Wagstaff et al., simpler functions (like the mean) can be represented with . Similarly, distinguishing a '0' from a '1' does not require preserving the exact location of every point, allowing the model to succeed with .

---

## 6. Conclusion

This study confirms that the latent dimension is a critical hyperparameter in Deep Sets. We validated the theoretical "bottleneck" at low dimensions, showing that Deep Sets are not magically invariant to capacity constraints. However, we also demonstrated that for practical engineering classification tasks, the required latent dimension is often governed by the task's complexity rather than the strict  theoretical bound.

---

## References

1. **Zaheer, M., et al.** (2017). *Deep Sets*. Advances in Neural Information Processing Systems.
2. **Wagstaff, E., et al.** (2019). *On the Limitations of Representing Functions on Sets*. International Conference on Machine Learning.