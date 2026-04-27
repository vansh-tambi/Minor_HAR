# Minor HAR - System Architecture

> Visual overview of system components, data flow, and training pipelines.

## System Architecture

The following diagram illustrates the real-time operational flow of the system. It showcases how sensor data is captured on the frontend, processed through the Flask backend, passed into the hybrid deep learning engine, and finally stored and utilized for AI-generated health reporting.

```mermaid
graph TB
    classDef frontend fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#0D47A1;
    classDef backend fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20;
    classDef ml fill:#FFF3E0,stroke:#EF6C00,stroke-width:2px,color:#E65100;
    classDef db fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px,color:#4A148C;
    classDef api fill:#FFEBEE,stroke:#C62828,stroke-width:2px,color:#B71C1C;

    subgraph "Frontend Architecture (React 19 & Vite)"
        UI["📱 User Interface<br/>(Dashboard & Reports)"]:::frontend
        SensorCapture["⏱️ Sensor Capture Engine<br/>(Acc & Gyro @ 20Hz)"]:::frontend
        Buffer["📦 Data Buffer<br/>(3-second / 60-sample windows)"]:::frontend
        AuthFront["🔐 Google OAuth Login"]:::frontend

        SensorCapture --> Buffer
        AuthFront -.-> UI
    end

    subgraph "Backend API (Flask & PyJWT)"
        Gateway["🌐 API Gateway<br/>(REST Endpoints)"]:::backend
        Preprocess["⚙️ Feature Engineering<br/>(Scale, Filter, Extract)"]:::backend
        Smoothing["📊 Prediction Smoothing<br/>(Majority Voting)"]:::backend
        GeminiService["🤖 AI Reporting Service<br/>(Generates Summaries)"]:::backend
        
        Gateway --> Preprocess
    end

    subgraph "Deep Learning Engine"
        ModelInference["🧠 Hybrid Model Inference<br/>(Conv1D + LSTM)"]:::ml
        ModelArtifacts["💾 Trained Artifacts<br/>(har_model.keras, scaler.pkl)"]:::ml
        
        ModelArtifacts -.-> ModelInference
        Preprocess --> ModelInference
        ModelInference --> Smoothing
    end

    subgraph "Storage & External Services"
        MongoDB[("🗄️ MongoDB Atlas<br/>(Activity Logs & Reports)")]:::db
        GeminiAPI{{"☁️ Google Gemini API<br/>(LLM Integration)"}}:::api
        OAuthAPI{{"☁️ Google Auth API"}}:::api
    end

    %% Connections
    Buffer -- "POST /predict" --> Gateway
    UI -- "JWT Auth Calls" --> Gateway
    AuthFront <--> OAuthAPI

    Smoothing --> MongoDB
    GeminiService <--> GeminiAPI
    GeminiService --> MongoDB
    
    Gateway --> GeminiService
```

## Data & Training Pipeline

The offline training pipeline is responsible for merging multiple academic datasets with custom mobile recordings, engineering magnitude features, applying heavy data augmentation, and training the optimal `Conv1D + LSTM` topology.

```mermaid
graph TD
    classDef data fill:#E1F5FE,stroke:#0277BD,stroke-width:2px,color:#01579B;
    classDef process fill:#FBE9E7,stroke:#D84315,stroke-width:2px,color:#BF360C;
    classDef output fill:#F1F8E9,stroke:#33691E,stroke-width:2px,color:#1B5E20;

    subgraph "Data Sources"
        Wisdm["WISDM Dataset"]:::data
        Uci["UCI HAR Dataset"]:::data
        Hetero["Heterogeneity Dataset"]:::data
        Custom["Custom Mobile CSVs"]:::data
    end

    subgraph "Data Preparation (prepare_data.py)"
        Filter["Butterworth Low-pass Filter"]:::process
        Sep["Gravity/Body Accel Separation"]:::process
        Mag["Magnitude Feature Engineering"]:::process
        Window["Sliding Window (Size 60, Step 30)"]:::process
        Augment["Data Augmentation (Jitter, Scale, Shift)"]:::process
        
        Wisdm & Uci & Hetero --> Filter
        Custom --> Filter
        Filter --> Sep --> Mag --> Window
        Custom -.-> Augment -.-> Window
    end

    subgraph "Model Training (train_model.py)"
        Scale["Global Feature Scaling"]:::process
        Train["Train Conv1D + LSTM Model"]:::process
        ClassWeights["Apply Class Weights"]:::process
        
        Window -->|X_all.npy, y_all.npy| Scale
        Scale --> Train
        ClassWeights -.-> Train
    end

    subgraph "Artifacts Generation"
        Keras["har_model.keras"]:::output
        Scaler["scaler.pkl"]:::output
        Encoder["label_encoder.pkl"]:::output
    end

    Train --> Keras & Scaler & Encoder
```

---

## Implementation

> Tools, technologies, and development details

### Tech Stack

| Category | Technologies Used | Details |
| :--- | :--- | :--- |
| **Frontend Language** | JavaScript, JSX, HTML5, CSS3 | Used for the web application UI and mobile browser sensor capture via Web APIs. |
| **Frontend Framework** | React 19, Vite | Fast, modern client-side framework with component-driven architecture. |
| **Backend Language** | Python 3.11+ | Primary language for API routing, data processing, and ML training pipelines. |
| **Backend Framework** | Flask, Flask-CORS | Lightweight Python microframework serving the REST API and prediction endpoints. |
| **Machine Learning** | TensorFlow / Keras, Scikit-learn | Used for defining, training, and running inference on the Conv1D + LSTM model. |
| **Data Processing** | NumPy, SciPy, Pandas | Used heavily in the data preparation and runtime feature engineering. |
| **Database** | MongoDB Atlas (PyMongo) | NoSQL document database used to store daily activity logs and health reports. |
| **Authentication** | Google OAuth, PyJWT | Secure user login workflow exchanging Google tokens for signed JWTs. |
| **External APIs** | Google Gemini API | Powers the automated AI-generated daily health summary reports. |
| **Visualizations** | Chart.js, Framer Motion | Interactive timeline graphs and smooth UI animations on the React client. |

### Core Project Flow

1. **Capture**: The React frontend (`useAccelerometer.js`) hooks into the device's accelerometer and gyroscope, sampling at exactly **20 Hz**.
2. **Buffer**: It buffers these readings until it reaches **60 samples (3 seconds)**.
3. **Transmit**: The 60-sample window is sent over HTTP to the Flask `POST /predict` API.
4. **Engineer**: Flask passes the data through `SciPy` Butterworth filters, computes magnitudes, and shapes it into an `(1, 60, 8)` tensor.
5. **Infer**: The pre-trained Keras model predicts the activity out of 7 possible classes (Walking, Jogging, Stairs, Still, Eating, Hand Activity, Sports).
6. **Smooth**: A sliding window majority-vote mechanism prevents erratic UI flickering.
7. **Store & Analyze**: The final predictions are saved to MongoDB. Once a day, the Gemini API is called to summarize the day's activity timeline into a human-readable health report.
