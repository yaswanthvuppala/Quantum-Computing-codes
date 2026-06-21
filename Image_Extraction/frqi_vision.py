"""
Hybrid Quantum-Classical Computer Vision Framework for Edge and Corner Detection using FRQI.
Developed by: Quantum Software Engineer & Computer Vision Researcher

This script encodes a 16x16 image into a 9-qubit quantum state using the Flexible Representation of 
Quantum Images (FRQI) protocol. It simulates the state measurement using Qiskit Aer, reconstructs 
the image, and classically calculates spatial gradients and the Harris Corner Response Matrix.
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
from skimage.metrics import structural_similarity as ssim_func
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import RYGate

def load_and_preprocess(dataset_path=None, label_to_select=None):
    """
    Loads an image from the dataset or generates a synthetic shape if the dataset is unavailable.
    Downsamples the image to exactly 16x16 pixels and normalizes pixel intensities to [0, 1].

    Parameters:
    -----------
    dataset_path : str, optional
        Path to the CSV dataset (e.g. fashion-mnist_test.csv).
    label_to_select : int, optional
        Class label to select from the dataset.
        For Fashion-MNIST:
            0: T-shirt/top, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat,
            5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot.
        For MNIST (if using standard MNIST):
            Digits 0-9.

    Returns:
    --------
    image : numpy.ndarray (16x16, float64)
        Normalized 16x16 grayscale image.
    source_name : str
        A string describing where the image was loaded from.
    """
    # 1. Resolve possible paths for the dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = []
    if dataset_path:
        possible_paths.append(dataset_path)
    possible_paths.extend([
        os.path.join(script_dir, "fashion-mnist_test.csv"),
        os.path.join(os.path.dirname(script_dir), "fashion-mnist_test.csv"),
        "fashion-mnist_test.csv"
    ])

    loaded_pixels = None
    source_name = ""

    for path in possible_paths:
        if os.path.exists(path):
            try:
                print(f"Attempting to load dataset from: {path}")
                df = pd.read_csv(path)
                
                # Check if it has a label column and pixel columns
                if 'label' in df.columns:
                    # Default class label: 8 (Bag) or 1 (Trouser) has clear edges for Fashion-MNIST.
                    # If standard MNIST was passed, label 7 or 4 is selected.
                    target_label = label_to_select if label_to_select is not None else 8
                    
                    filtered_df = df[df['label'] == target_label]
                    if filtered_df.empty:
                        print(f"Label {target_label} not found in dataset. Loading the first available row.")
                        row = df.iloc[0]
                        target_label = row['label']
                    else:
                        row = filtered_df.iloc[0]
                    
                    # Read 784 pixels (28x28 image)
                    raw_pixels = row.drop('label').values.astype(np.uint8)
                    loaded_pixels = raw_pixels.reshape(28, 28)
                    source_name = f"Fashion-MNIST (Label {target_label})"
                    print(f"Successfully loaded image from {path} with label: {target_label}")
                    break
            except Exception as e:
                print(f"Error reading dataset at {path}: {e}")

    # 2. Fallback to generating a synthetic 16x16 geometric shape if dataset loading failed
    if loaded_pixels is None:
        print("\n[WARNING] Dataset could not be loaded. Generating a high-contrast synthetic 16x16 square.")
        # Create a 16x16 black canvas
        loaded_pixels = np.zeros((16, 16), dtype=np.uint8)
        # Draw a solid 8x8 white square in the center (from index 4 to 11 inclusive)
        # This provides exactly 4 sharp 90-degree corners and 4 clear straight edges.
        loaded_pixels[4:12, 4:12] = 255
        source_name = "Synthetic 16x16 Geometric Square"

    # 3. Downsample to exactly 16x16 pixels
    if loaded_pixels.shape != (16, 16):
        resized_image = cv2.resize(loaded_pixels, (16, 16), interpolation=cv2.INTER_AREA)
    else:
        resized_image = loaded_pixels.copy()

    # 4. Normalize intensities to the range [0.0, 1.0]
    normalized_image = resized_image.astype(np.float64) / 255.0

    return normalized_image, source_name

def build_frqi_circuit(image_matrix):
    """
    Constructs the 9-qubit Flexible Representation of Quantum Images (FRQI) circuit.
    
    Qubit Register Allocation & Mapping:
    -----------------------------------
    - Qubit 0: Color Qubit (stores intensity information as rotation state |c>).
    - Qubits 1..4: X Coordinate Qubits (|x_0, x_1, x_2, x_3>).
                   LSB is qubit 1, MSB is qubit 4.
    - Qubits 5..8: Y Coordinate Qubits (|y_0, y_1, y_2, y_3>).
                   LSB is qubit 5, MSB is qubit 8.

    State preparation is executed via a loop-based Multi-Controlled Ry pipeline.
    """
    # 9 qubits total: 1 color qubit + 8 position qubits (4 for X, 4 for Y)
    num_qubits = 9
    qc = QuantumCircuit(num_qubits)

    # 1. Apply Hadamard gates to all position qubits (1 to 8) to create an equal spatial superposition
    # State: 1/16 * Sum_{y=0}^{15} Sum_{x=0}^{15} |0> \otimes |y> |x>
    qc.h(range(1, num_qubits))

    # 2. Encode grayscale intensity using loop-based Multi-Controlled RY rotations on the color qubit
    for y in range(16):
        y_bin = f"{y:04b}"  # 4-bit binary representation of Y (MSB on left, LSB on right)
        for x in range(16):
            x_bin = f"{x:04b}"  # 4-bit binary representation of X (MSB on left, LSB on right)
            
            # Map pixel intensity from [0, 1] to angle [0, pi/2]
            intensity = image_matrix[y, x]
            theta = intensity * (np.pi / 2.0)

            # Optimization: Skip rotation if intensity is zero (no rotation needed)
            if theta > 1e-6:
                # Qiskit's ctrl_state maps the rightmost character to the first qubit in the control list,
                # and the leftmost character to the last qubit.
                # Control register order: [q_1, q_2, q_3, q_4, q_5, q_6, q_7, q_8]
                #   - q_1 to q_4 represent X coordinate LSB to MSB (x_bin elements read from right to left).
                #   - q_5 to q_8 represent Y coordinate LSB to MSB (y_bin elements read from right to left).
                # Therefore, ctrl_state string = f"{y_bin}{x_bin}" matches the list indices exactly.
                ctrl_state = f"{y_bin}{x_bin}"
                
                # Apply Multi-Controlled RY gate with angle 2*theta.
                # Controls are qubits 1..8, Target is qubit 0 (color qubit)
                mcry_gate = RYGate(2.0 * theta).control(num_ctrl_qubits=8, ctrl_state=ctrl_state)
                qc.append(mcry_gate, list(range(1, 9)) + [0])

    # 3. Add measurements to all qubits
    qc.measure_all()

    return qc

def reconstruct_image(counts, shots):
    """
    Decodes the measurements from the AerSimulator to reconstruct the 16x16 image matrix.

    Mathematical Reconstruction:
    ----------------------------
    P(c=1 | y, x) = Counts(1, y, x) / (Counts(0, y, x) + Counts(1, y, x))
    theta_(y, x) = arcsin(sqrt(P(c=1 | y, x)))
    Intensity(y, x) = theta_(y, x) * (2 / pi)
    """
    reconstructed_image = np.zeros((16, 16), dtype=np.float64)
    
    # 2D arrays to accumulate counts for color = 0 and color = 1 at each position (y, x)
    counts_0 = np.zeros((16, 16), dtype=np.float64)
    counts_1 = np.zeros((16, 16), dtype=np.float64)

    # Parse counts keys (plain binary strings of length 9)
    # Format of bitstring: b_8 b_7 b_6 b_5 b_4 b_3 b_2 b_1 b_0
    #   - b_0 (bitstring[-1]) is the color qubit state.
    #   - b_4 b_3 b_2 b_1 (bitstring[-5:-1]) represents X coordinate (qubits 1..4).
    #   - b_8 b_7 b_6 b_5 (bitstring[-9:-5]) represents Y coordinate (qubits 5..8).
    for bitstring, count in counts.items():
        color_val = int(bitstring[-1])
        x_val = int(bitstring[-5:-1], 2)
        y_val = int(bitstring[-9:-5], 2)

        if color_val == 0:
            counts_0[y_val, x_val] += count
        else:
            counts_1[y_val, x_val] += count

    # Compute intensity for each pixel location
    for y in range(16):
        for x in range(16):
            n0 = counts_0[y, x]
            n1 = counts_1[y, x]
            total_pixel_shots = n0 + n1

            if total_pixel_shots == 0:
                reconstructed_image[y, x] = 0.0
            else:
                prob_c1 = n1 / total_pixel_shots
                # Clip probability to [0.0, 1.0] to prevent floating-point representation anomalies
                prob_c1 = np.clip(prob_c1, 0.0, 1.0)
                
                # Retrieve the rotation angle and map back to intensity
                theta = np.arcsin(np.sqrt(prob_c1))
                reconstructed_image[y, x] = theta * (2.0 / np.pi)

    return reconstructed_image

def compute_vision_metrics(original, reconstructed, ksize=3, harris_k=0.04):
    """
    Computes structural metrics (MSE, SSIM) between the original and reconstructed images.
    Also computes spatial gradients (Ix, Iy) and the Harris Corner Response Matrix (R).
    """
    # 1. Structural Comparison Metrics
    mse = np.mean((original - reconstructed) ** 2)
    ssim_val = ssim_func(original, reconstructed, data_range=1.0)

    # 2. Compute Spatial Gradients (Ix, Iy) via OpenCV Sobel filter
    # OpenCV's cv2.Sobel accepts float64 images and performs gradient calculations
    Ix = cv2.Sobel(reconstructed, cv2.CV_64F, 1, 0, ksize=ksize)
    Iy = cv2.Sobel(reconstructed, cv2.CV_64F, 0, 1, ksize=ksize)

    # 3. Compute Harris Corner Response Matrix (R)
    # Compute local gradient products
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # Apply Gaussian smoothing to elements of the structure tensor M (3x3 kernel)
    Sxx = cv2.GaussianBlur(Ixx, (3, 3), 0)
    Syy = cv2.GaussianBlur(Iyy, (3, 3), 0)
    Sxy = cv2.GaussianBlur(Ixy, (3, 3), 0)

    # Calculate Det(M) and Trace(M) for each pixel
    det_M = Sxx * Syy - Sxy * Sxy
    trace_M = Sxx + Syy
    
    # R = Det(M) - k * Trace(M)^2
    R = det_M - harris_k * (trace_M ** 2)

    return {
        'mse': mse,
        'ssim': ssim_val,
        'Ix': Ix,
        'Iy': Iy,
        'R': R
    }

def main():
    print("=" * 70)
    print(" Hybrid Quantum-Classical Computer Vision Framework (FRQI) ".center(70, "="))
    print("=" * 70)

    # Step 1: Load and Preprocess Image
    # Choose class label 8 (Bag) for Fashion-MNIST as default (features clear rectangular shape)
    # For MNIST, standard choices are digits 7 or 4.
    label_to_select = 8
    image, source_name = load_and_preprocess(label_to_select=label_to_select)
    
    print(f"\n[1/5] Image loaded from: {source_name}")
    print("Original 16x16 Pixel Intensities:")
    print(np.round(image, 2))

    # Step 2: Build FRQI Quantum Circuit
    print("\n[2/5] Constructing FRQI Quantum Circuit (9 Qubits)...")
    qc = build_frqi_circuit(image)
    print(f"Quantum Circuit built with {qc.num_qubits} qubits.")
    print(f"Number of circuit operations: {len(qc.data)}")

    # Step 3: Run Quantum Simulation on Qiskit Aer
    shots = 100000
    print(f"\n[3/5] Transpiling and simulating circuit on AerSimulator ({shots:,} shots)...")
    
    backend = AerSimulator()
    transpiled_qc = transpile(qc, backend)
    
    job = backend.run(transpiled_qc, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    print("Simulation complete.")
    print(f"Number of distinct statevector measurements: {len(counts)}")

    # Step 4: Reconstruct the Grayscale Image
    print("\n[4/5] Reconstructing image from measurement probabilities...")
    reconstructed = reconstruct_image(counts, shots)
    print("Reconstructed 16x16 Pixel Intensities:")
    print(np.round(reconstructed, 2))

    # Step 5: Compute Vision Metrics and Corner Response Matrix
    print("\n[5/5] Computing Spatial Gradients & Harris Corner Response (R)...")
    metrics = compute_vision_metrics(image, reconstructed, ksize=3)
    
    print("-" * 50)
    print(f"Mean Squared Error (MSE) : {metrics['mse']:.6f}")
    print(f"Structural Similarity    : {metrics['ssim']:.6f}")
    print("-" * 50)

    # Print gradients and Harris response
    print("\nSpatial Gradient Ix (Horizontal):")
    print(np.round(metrics['Ix'], 2))
    
    print("\nSpatial Gradient Iy (Vertical):")
    print(np.round(metrics['Iy'], 2))

    print("\nHarris Corner Response Matrix R:")
    # Rounding and showing R values. High positive values denote corners.
    print(np.round(metrics['R'], 4))

    # Identify coordinates of detected corners (R > threshold)
    # We use a threshold of 10% of the maximum response value
    max_R = np.max(metrics['R'])
    threshold = 0.1 * max_R if max_R > 0 else 0.01
    corners = np.argwhere(metrics['R'] > threshold)
    
    print(f"\nDetected Corner Coordinates (R > {threshold:.4f}):")
    if len(corners) > 0:
        for cy, cx in corners:
            print(f"  - Corner at (Y={cy}, X={cx}) with R = {metrics['R'][cy, cx]:.5f}")
    else:
        print("  - No sharp corners detected above threshold.")

    print("\n" + "=" * 70)
    print(" Framework Process Completed Successfully ".center(70, "="))
    print("=" * 70)

if __name__ == "__main__":
    main()
