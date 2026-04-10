# 🚀 GPU vs CPU Deep Learning Benchmark Suite

## 📌 Overview
This project presents a comprehensive performance benchmarking study comparing **CPU** and **CUDA-enabled GPU** execution for deep learning workloads using **PyTorch**.

The goal is to analyze how hardware acceleration, batch size, model complexity, and precision modes (FP32 vs FP16) affect key performance indicators, including training efficiency and resource utilization.

## 🎯 Objectives
* **Evaluate** performance differences between CPU and GPU training.
* **Analyze** the impact of batch size scaling on hardware efficiency.
* **Compare** model complexity effects using a custom SimpleCNN versus an industry-standard ResNet18.
* **Investigate** the benefits of Mixed Precision training (FP16 vs FP32).
* **Measure** real-world deployment metrics such as inference latency.
* **Analyze** GPU memory utilization trends across different workloads.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Framework:** PyTorch
* **Hardware Acceleration:** CUDA (Tested on Google Colab GPU)
* **Dataset:** CIFAR-10
* **Libraries:** `torchvision`, `numpy`, `matplotlib`, `pandas`

---

## 🧠 Models Used

### 1. SimpleCNN
A custom, lightweight convolutional neural network designed for rapid testing:
* 3 Convolutional layers
* ReLU activations & Max pooling
* Fully connected classifier

### 2. ResNet18
A deeper, residual architecture adapted for CIFAR-10:
* Pretrained architecture (weights adjusted for benchmarking)
* Represents real-world production-level model complexity.

---

## ⚙️ Experimental Setup

### Variables
| Parameter | Values Tested |
| :--- | :--- |
| **Model** | SimpleCNN, ResNet18 |
| **Device** | CPU, GPU (CUDA) |
| **Batch Size** | 32, 64, 128 |
| **Precision Mode** | FP32, FP16 (CUDA AMP) |

**Total Experiments:** 24 configurations (Note: FP16 is applied to GPU only).

### 📊 Metrics Collected
For each configuration, the following data points were recorded:
* **Training Time:** Total time taken per epoch (seconds).
* **Accuracy:** Model classification performance (%).
* **Throughput:** Processing speed measured in **images/sec**.
* **Inference Latency:** Seconds per batch during evaluation.
* **GPU Memory Usage:** Dedicated VRAM consumption (MB).

---

## 📈 Results Summary

### 🔹 1. Training Performance
* GPU significantly outperformed CPU across all configurations.
* **Speedup:** ~3× for SimpleCNN, scaling up to **~9×+** for ResNet18.

### 🔹 2. Throughput Scaling
* Throughput increased proportionally with batch size on both devices.
* **Peak Throughput:** ~7,945 images/sec (ResNet18, batch size 128 on GPU).

### 🔹 3. Model Complexity Impact
* ResNet18 showed significantly higher GPU acceleration gains compared to SimpleCNN.
* **Insight:** Deeper models with more parameters benefit more from the massive parallelism of GPU cores.

### 🔹 4. Mixed Precision (FP16 vs FP32)
* FP16 successfully reduced GPU memory footprints.
* **Performance Note:** Speed gains were inconsistent for smaller workloads (CIFAR-10), indicating that the overhead of mixed precision may negate benefits on very small models.

### 🔹 5. Inference Latency
* GPU latency remained consistently lower than CPU.
* **Example:** CPU (~0.012–0.029s) vs. GPU (~0.0025s).

### 🔹 6. GPU Memory Usage
Memory consumption increased linearly with batch size and model depth:
* **SimpleCNN:** ~40–85 MB
* **ResNet18:** ~235–257 MB

---

## 🔍 Key Insights
1.  **Hardware Matters:** GPU acceleration provides substantial performance improvements across all deep learning configurations.
2.  **Utilization:** Larger batch sizes improve hardware utilization and maximize throughput.
3.  **Scalability:** Deeper architectures like ResNet18 are the primary beneficiaries of GPU parallelism.
4.  **Efficiency Tradeoffs:** Mixed precision is a powerful tool for memory efficiency but requires larger workloads to showcase significant speed improvements.
5.  **Optimization:** There is a distinct tradeoff between throughput optimization and model accuracy when scaling to very large batch sizes.

---
