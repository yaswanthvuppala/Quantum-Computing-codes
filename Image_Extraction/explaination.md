# Hybrid Quantum-Classical Computer Vision: FRQI Edge & Corner Detection

This document provides a comprehensive explanation of the project architecture, the mathematical formulas used, the step-by-step implementation, and instructions on how to run and verify the framework.

---

## 1. Project Overview
This project implements a hybrid quantum-classical pipeline to process grayscale images. The workflow consists of:
1. **Classical Pre-processing**: Loading real dataset images (e.g. from Fashion-MNIST) or generating high-contrast synthetic shapes, downsampling them to exactly $16 \times 16$ pixels, and normalizing pixel intensities.
2. **Quantum Image State Encoding**: Representing the image on a quantum register using the **Flexible Representation of Quantum Images (FRQI)** protocol.
3. **Quantum Simulation & Measurement**: Measuring the register using $100,000$ shots via the `AerSimulator` backend.
4. **Classical Image Reconstruction**: Decoding measurement probabilities into a classical grayscale image matrix.
5. **Classical Feature Extraction**: Computing horizontal and vertical spatial gradients ($I_x$, $I_y$) using OpenCV Sobel kernels, and computing the Harris Corner Response Matrix ($R$) to detect corners.
6. **Fidelity Analysis**: Evaluation of reconstruction quality using Mean Squared Error (MSE) and Structural Similarity Index (SSIM).

---

## 2. Mathematical Framework

### A. Flexible Representation of Quantum Images (FRQI)
An image is encoded into a quantum state that couples color (intensity) information with spatial (coordinate) information.

For a $2^{n_y} \times 2^{n_x}$ image, we define a normalized quantum state $|I\rangle$:
$$|I\rangle = \frac{1}{\sqrt{2^{n_y + n_x}}} \sum_{y=0}^{2^{n_y}-1} \sum_{x=0}^{2^{n_x}-1} |c_{y,x}\rangle \otimes |y\rangle \otimes |x\rangle$$

where:
- $|y\rangle \otimes |x\rangle$ are the spatial coordinate basis states.
- $|c_{y,x}\rangle$ is the color qubit state representing the grayscale intensity at coordinate $(y, x)$:
$$|c_{y,x}\rangle = \cos\theta_{y,x}|0\rangle + \sin\theta_{y,x}|1\rangle$$

For our $16 \times 16$ image:
- $n_x = 4$ qubits represent the X-coordinate ($x \in [0, 15]$).
- $n_y = 4$ qubits represent the Y-coordinate ($y \in [0, 15]$).
- $1$ color qubit represents the gray intensity.
- **Total Register Size**: $9$ qubits.

### B. Pixel Intensity Encoding
We scale each normalized pixel value $I(y, x) \in [0.0, 1.0]$ to a rotation angle $\theta_{y,x} \in [0, \pi/2]$:
$$\theta_{y,x} = I(y, x) \times \frac{\pi}{2}$$

Applying a Multi-Controlled $R_y(2\theta_{y,x})$ rotation on the color qubit, conditioned on the spatial qubits being in state $|y\rangle|x\rangle$, yields the desired state:
$$R_y(2\theta_{y,x})|0\rangle = \left(\cos\theta_{y,x}|0\rangle + \sin\theta_{y,x}|1\rangle\right)$$

---

## 3. Quantum-Classical Reconstruction Pipeline

### Step 1: Equal Superposition
Start with all qubits in $|0\rangle^{\otimes 9}$. Apply Hadamard gates to all 8 position qubits (qubits 1 to 8):
$$H^{\otimes 8} |0\rangle^{\otimes 8} = \frac{1}{16} \sum_{y=0}^{15} \sum_{x=0}^{15} |y\rangle |x\rangle$$

### Step 2: Multi-Controlled Rotation State Preparation
Apply the loop-based Multi-Controlled $R_y(2\theta_{y,x})$ sequence. For each pixel $(y, x)$, if $I(y,x) > 0$, we append a multi-controlled gate targeting qubit 0 (color) with controls being qubits 1..8 in the state matching the binary values of $y$ and $x$.
The rotation operator is:
$$R_y(2\theta_{y,x}) = \begin{pmatrix} \cos\theta_{y,x} & -\sin\theta_{y,x} \\ \sin\theta_{y,x} & \cos\theta_{y,x} \end{pmatrix}$$

### Step 3: Measurement
We measure all 9 qubits. For a given coordinate $(y, x)$, the probability of measuring the color qubit in state $|1\rangle$ is:
$$P(1 | y, x) = \sin^2\theta_{y,x}$$
The probability of measuring the color qubit in state $|0\rangle$ is:
$$P(0 | y, x) = \cos^2\theta_{y,x}$$

### Step 4: Reconstruction
Let $N_1(y, x)$ be the measured count of the state $|1\rangle \otimes |y\rangle \otimes |x\rangle$, and $N_0(y, x)$ be the count of the state $|0\rangle \otimes |y\rangle \otimes |x\rangle$. The empirical probability of measuring color 1 is:
$$P_{empirical}(c=1 | y, x) = \frac{N_1(y, x)}{N_0(y, x) + N_1(y, x)}$$

The reconstructed intensity $I_{recon}(y, x)$ is calculated by taking the inverse mapping:
$$\theta_{y,x} = \arcsin\left(\sqrt{P_{empirical}(c=1 | y, x)}\right)$$
$$I_{recon}(y, x) = \theta_{y,x} \times \frac{2}{\pi} = \frac{2}{\pi} \arcsin\left(\sqrt{\frac{N_1(y, x)}{N_0(y, x) + N_1(y, x)}}\right)$$

---

## 4. Edge & Corner Detection (Classical Post-Processing)

Using the quantum-reconstructed image matrix $I_{recon}$, we compute:

### A. Spatial Gradients ($I_x$, $I_y$)
Calculated using a $3 \times 3$ Sobel filter. The convolution masks are:
$$K_x = \begin{pmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{pmatrix}, \quad K_y = \begin{pmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{pmatrix}$$

$$I_x = I_{recon} * K_x, \quad I_y = I_{recon} * K_y$$

### B. Structure Tensor ($M$)
For each pixel, we compute the structure tensor $M$ representing local neighborhood gradients:
$$M = \begin{pmatrix} S_{xx} & S_{xy} \\ S_{xy} & S_{yy} \end{pmatrix}$$
where:
- $S_{xx} = \text{GaussianBlur}(I_x^2)$
- $S_{yy} = \text{GaussianBlur}(I_y^2)$
- $S_{xy} = \text{GaussianBlur}(I_x I_y)$

### C. Harris Corner Response ($R$)
The corner response score at each pixel is calculated via:
$$R = \det(M) - k \cdot (\text{trace}(M))^2$$
where:
- $\det(M) = S_{xx} S_{yy} - S_{xy}^2$
- $\text{trace}(M) = S_{xx} + S_{yy}$
- $k = 0.04$ (empirical constant)

A high positive response $R$ indicates a sharp corner, values near zero indicate flat regions, and negative values indicate straight edges.

---

## 5. Metrics Layer

- **Mean Squared Error (MSE)**:
$$\text{MSE} = \frac{1}{256} \sum_{y=0}^{15} \sum_{x=0}^{15} \left(I_{orig}(y, x) - I_{recon}(y, x)\right)^2$$

- **Structural Similarity Index (SSIM)**:
$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$
Evaluates luminance, contrast, and structure preservation between original and reconstructed matrices.

---

## 6. How to Run the Project & Check Metrics

### Prerequisites
Make sure the required libraries are installed:
```bash
pip install numpy pandas opencv-python scikit-image qiskit qiskit-aer
```

### Running the Pipeline
Execute the Python script from the root workspace directory:
```bash
python Image_Extraction/frqi_vision.py
```

### Checking Output Metrics
The script outputs logs detailing:
1. **Source Identification**: Reports whether the image was loaded from `fashion-mnist_test.csv` or generated synthetically.
2. **Reconstruction Performance**: Prints the **MSE** and **SSIM** scores directly in the console. High-fidelity simulations with 100,000 shots will result in:
   - $\text{MSE} < 0.0002$
   - $\text{SSIM} > 99.7\%$
3. **Feature Maps**: Prints the $16 \times 16$ spatial gradients ($I_x$ and $I_y$) and the Corner Response Matrix ($R$) in rounded matrix format.
4. **Detected Corners**: Enumerates the pixel coordinates $(Y, X)$ where the Harris Response exceeds $10\%$ of the maximum response value.
