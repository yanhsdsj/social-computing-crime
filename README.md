# Crime Prediction Models

This repository contains the implementation of crime prediction models based on **Graph Convolutional Networks (GCN)** and **Recurrent Neural Networks (RNN)**. It includes both the **original baseline models** and **our modified models**. The models use spatiotemporal features to predict crime trends across different cities.

## 🔧 Project Structure

### 1. **cf/** — Our Self-Developed Model
This folder contains our custom modifications and enhancements to the crime prediction model. These improvements include better feature extraction, temporal convolution blocks, and advanced regularization techniques.

- **`run.py`**: Entry point to train the model.
- **`supervisor.py`**: Manages model training, evaluation, and logging.
- **`save_predictions.py`**: Script for saving model predictions.
- **`model/`**: Model architecture files, including various network blocks and layers.

### 2. **origin/** — Baseline Reference Model (DCRNN)
This folder contains the original model used for comparison. We have implemented a **DCRNN** (Diffusion Convolutional Recurrent Neural Network) as the baseline model for spatiotemporal forecasting. 

- **`Code/`**: Core model code.
- **`Geographical-Chicago/`**: Model configuration and data for Chicago crime data.
- **`Geographical-LA/`**: Model configuration and data for Los Angeles crime data.
- **`README.md`**: Instructions for running the baseline model.
- **`1168_timesnet_temporal_2d_variation.pdf`**: Research paper for the baseline model.

---

