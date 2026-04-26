# Human Activity Recognition (HAR) Engine
### Real-Time Motion Intelligence with Hybrid Deep Learning & AI Health Reporting

![Dashboard Preview](docs/images/dashboard_preview.png)

---

## Overview

This repository contains an end-to-end Human Activity Recognition (HAR) system that classifies eight distinct physical activities using real-time smartphone sensor data. The application captures 6-axis inertial measurement unit (IMU) data—comprising accelerometer and gyroscope readings—and processes it through a hybrid Conv1D and LSTM neural network architecture. 

In addition to real-time classification, the system features an automated health reporting module that aggregates daily activity logs and utilizes the Google Gemini API to generate structured, human-readable health summaries.

---

## Features

- **Real-Time Data Acquisition:** Captures and buffers sensor data at 20Hz directly from mobile web browsers using the W3C DeviceMotion API.
- **Deep Learning Architecture:** Utilizes a hybrid Conv1D + LSTM model designed to extract spatial features across sensor channels and model temporal dependencies over sliding time windows.
- **Signal Preprocessing Pipeline:** Implements real-time Butterworth low-pass filtering for noise reduction and gravity separation, alongside magnitude feature engineering for rotation invariance.
- **Automated Health Reporting:** Aggregates daily prediction logs (tracking active minutes, intensity, and estimated caloric expenditure) and generates descriptive health reports using LLM integration.
- **Secure Authentication & Logging:** Integrates Google OAuth 2.0 for user authentication, JWT for session management, and MongoDB for persistent activity logging.

---

## System Architecture

```mermaid
graph TD
    subgraph Client Application
        A["Mobile Browser (React)"] --> B["DeviceMotion API (20Hz)"]
        B --> C["Sliding Window Buffer (60 samples)"]
    end

    subgraph API Backend (Flask)
        C -->|"HTTP POST /predict"| D["Signal Filter (Butterworth)"]
        D --> E["Feature Scaler"]
        E --> F["Conv1D + LSTM Inference"]
        F --> G["Temporal Smoothing (Majority Vote)"]
    end

    subgraph Data & External Services
        G --> H["MongoDB (Activity Logs)"]
        H --> I["Report Generator (Gemini API)"]
        I --> J["PDF Export (FPDF/Matplotlib)"]
    end

    J --> K["User Dashboard"]
```

---

## Project Workflow

### 1. Data Collection & Preprocessing
The frontend buffers sensor readings into 3-second sliding windows (60 samples per window). Upon reaching the backend, the data undergoes causal Butterworth filtering to remove high-frequency noise and isolate body acceleration from gravitational forces.

### 2. Model Inference
The preprocessed data is normalized and passed to the Keras model. The model consists of 1D Convolutional layers that capture local spatial patterns, followed by LSTM layers that analyze the sequence over time. To stabilize the user interface, the system applies a confidence thresholding mechanism and a majority voting algorithm across the most recent predictions.

### 3. Reporting and Export
Verified predictions are logged to the database. Upon user request, the system computes aggregated statistics (e.g., total active time, caloric burn) based on Metabolic Equivalent of Task (MET) values. This data is passed to the Gemini API to construct a professional health summary, which can be viewed in the dashboard or exported as a formatted PDF.

---

## Technology Stack

| Component | Technologies |
| :--- | :--- |
| **Frontend** | React, Vite, Chart.js, Framer Motion |
| **Backend API** | Flask, PyJWT, PyMongo |
| **Machine Learning** | TensorFlow / Keras, SciPy, Scikit-learn |
| **Database** | MongoDB Atlas |
| **External APIs** | Google OAuth 2.0, Google Gemini 1.5 Flash |

---

## Model Performance

The classification model was trained on an aggregated dataset comprising samples from WISDM, Heterogeneity Activity Recognition, UCI HAR, and custom-recorded data. To improve robustness against real-world sensor noise, custom data was heavily augmented using techniques such as Gaussian jitter, magnitude scaling, and time warping.

- **Overall Test Accuracy:** 82.99%
- **Jogging Precision:** 99%
- **Eating Precision:** 96%
- **Still Precision:** 94%

---

## Setup Instructions

### Prerequisites
- Python 3.12+
- Node.js 18+
- MongoDB instance (local or Atlas)

### 1. Repository Setup
```bash
git clone https://github.com/your-username/Minor_HAR.git
cd Minor_HAR
```

### 2. Backend Configuration
```bash
cd backend
pip install -r requirements.txt
```
Create a `.env` file in the `backend` directory and configure the required environment variables:
```env
MONGODB_URI=your_mongodb_connection_string
VITE_GOOGLE_CLIENT_ID=your_google_oauth_client_id
JWT_SECRET=your_jwt_secret_key
GEMINI_API_KEY=your_gemini_api_key
```
Start the backend server:
```bash
python app.py
```

### 3. Frontend Configuration
```bash
cd frontend
npm install
```
Create a `.env` file in the `frontend` directory:
```env
VITE_GOOGLE_CLIENT_ID=your_google_oauth_client_id
```
Start the development server with network exposure:
```bash
npm run dev -- --host
```

### 4. Client Connection
To test the real-time sensor streaming, navigate to the network IP address provided by the Vite server output using a mobile device connected to the same local network.

---

## Contributors

- **Vansh Tambi** - [vanshtambi@gmail.com](mailto:vanshtambi@gmail.com)
- **Vivek Pasi** - [vivekpasi43@gmail.com](mailto:vivekpasi43@gmail.com)
