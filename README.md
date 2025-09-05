# 🚀 Anomaly Detection in Time-Series Data using Autoencoders  

This project implements an **anomaly detection system** on sensor temperature data using **deep learning autoencoders**. The goal is to identify unusual patterns (potential system failures) in time-series data for **predictive maintenance** and **fault detection**.  

---

## 📌 Features
- ✅ **Autoencoder-based anomaly detection** using TensorFlow/Keras  
- ✅ **Data preprocessing** with Pandas for clean and consistent input  
- ✅ **Unsupervised learning** approach (no labeled failures required)  
- ✅ **Evaluation metrics**: Precision, Recall, and F1-Score  
- ✅ **Visualization** of anomalies in time-series data using Matplotlib  

---

## 🛠️ Tech Stack
- **Language**: Python  
- **Libraries**: TensorFlow/Keras, Pandas, Matplotlib, scikit-learn  
- **Techniques**: Autoencoders, Deep Learning, Time-Series Analysis, Anomaly Detection  

---

## 📂 Dataset
The dataset used: **ambient_temperature_system_failure.csv**  
- Contains time-series sensor readings with a timestamp and temperature values.  
- Preprocessed to handle missing values and convert features into numeric format.  

---

## ⚙️ How It Works
1. **Data Preprocessing**  
   - Load CSV, drop missing values, normalize features.  

2. **Autoencoder Training**  
   - Encoder compresses data into latent space (dim=10).  
   - Decoder reconstructs input to learn normal patterns.  

3. **Anomaly Scoring**  
   - Reconstruction error (MSE) is used to assign anomaly scores.  
   - Thresholding applied (99th percentile) to classify anomalies.  

4. **Evaluation & Visualization**  
   - Precision, Recall, F1-Score calculated.  
   - Anomalies plotted in red over time-series chart.  

---

## 📊 Results
- High anomaly detection accuracy demonstrated with F1-Score evaluation.  
- Successfully identified abnormal points in sensor data.  
- Visual anomaly plots provide intuitive insights for system monitoring.  

---

## 🚀 How to Run
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/anomaly-detection-autoencoder.git
   cd anomaly-detection-autoencoder

## Install dependencies:

pip install -r requirements.txt


## Run the script:

python anomaly_detection.py
