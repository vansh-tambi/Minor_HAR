# Human Activity Recognition (HAR) Engine  
### Real-Time Activity Classification using Conv1D + LSTM

---

## Overview  

A real-time Human Activity Recognition system that classifies user movements using 6-axis sensor data (Accelerometer + Gyroscope) streamed directly from a mobile device.

The system processes live data through a Flask-based ML API and predicts activities using a hybrid deep learning model (1D CNN + LSTM).

This system is designed for real-world usability rather than basic motion classification demos.

---

## Key Highlights  

- Real-time sensor data streaming from mobile browser  
- Hybrid deep learning model (Conv1D + LSTM)  
- Live visualization dashboard using React and Chart.js  
- Low-latency predictions via Flask API  
- Optimized 8-class activity classification  

---

## Problem Statement  

Traditional HAR systems:
- Rely only on accelerometer data  
- Struggle with overlapping activities  
- Perform poorly in noisy real-world environments  

This system improves performance by:
- Using combined accelerometer and gyroscope data (6 channels)  
- Applying temporal modeling using LSTM  
- Consolidating noisy labels into meaningful activity groups  

---

## Dataset & Classes  

**Dataset:** WISDM (Wireless Sensor Data Mining)

The original dataset contains 18 complex and overlapping activities. These were consolidated into 8 practical categories to improve real-world performance.

### Final Classes:
- Walking  
- Jogging  
- Stairs  
- Still  
- Eating  
- Hand Activity  
- Active Hands  
- Sports  

This restructuring improves both model stability and prediction reliability.

---

## Model Architecture  

**Input Configuration:**
- 10-second sliding window  
- 50% overlap  
- Shape: `200 × 6` (time steps × sensor channels)

### Pipeline:

1. **Conv1D Layers**
   - Extract spatial features across sensor channels  
   - Includes Batch Normalization and MaxPooling  

2. **LSTM Layer**
   - Captures temporal dependencies  
   - Learns motion patterns over time  

3. **Dense Layer (Softmax)**
   - Outputs probabilities across 8 activity classes  

---

## System Design  

### Frontend:
- React (Vite)  
- Chart.js for real-time visualization  
- Mobile browser sensor integration  

### Backend:
- Flask API  
- TensorFlow/Keras model inference  

### Data Flow:
Mobile Sensors → React → Flask API → Model → Prediction → UI  

---

## Results  

- Accuracy: ~73% (8-class classification)  
- Improved performance compared to baseline CNN models  
- More stable predictions due to class consolidation  

---

## Setup Instructions  

```bash
# Backend
cd backend
pip install -r requirements.txt
python app.py

# Frontend
cd frontend
npm install
npm run dev
