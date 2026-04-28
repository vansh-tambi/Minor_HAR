# Project Synopsis: Human Activity Recognition (HAR) System using Hybrid Deep Learning

**Project Title:** Minor_HAR - Real-time Activity Tracking & AI Health Reporting
**Team Members:** Vansh Tambi, Vivek Pasi

---

### 1. Introduction (Project ka Basic Idea)
Yeh project ek real-time **Human Activity Recognition (HAR)** system hai jo mobile sensors (accelerometer aur gyroscope) ka use karke user ki activities ko detect karta hai. Iska main goal user ki physical health ko monitor karna aur Gemini AI ke through daily health reports generate karna hai.

### 2. Problem Statement (Humein iski zaroorat kyun hai?)
Aaj kal ke lifestyle mein log apni physical activity track nahi kar paate. Existing apps ya toh expensive wearables mangte hain ya phir utne accurate nahi hote. Humein ek aisa solution chahiye tha jo normal mobile browser se hi data capture kare aur high accuracy ke saath activities classify kar sake.

### 3. Objectives (Main Goal kya hai?)
- Mobile sensor data ko real-time mein capture karna.
- **Hybrid Deep Learning Model** (Conv1D + BiLSTM + Attention) ka use karke 88.6% accuracy achieve karna.
- User ko daily AI-generated summaries dena taaki wo apni health progress dekh sake.
- Secure login (Google OAuth) aur PDF report export provide karna.

### 4. Methodology (Kaise Kaam Karta Hai?)
Is project ko humne 3 parts mein divide kiya hai:

1.  **Frontend (React 19 + Vite):** 
    - Mobile browser se accelerometer aur gyroscope ka data capture karta hai (20 Hz frequency par).
    - 60 samples (3 seconds) ka buffer banakar backend ko bhejta hai.
2.  **Backend (Flask):**
    - Received data ko preprocess karta hai (Butterworth Filter aur Magnitude scaling).
    - ML model se activity predict karwata hai.
3.  **Deep Learning Model:**
    - **Conv1D:** Spatial features extract karne ke liye.
    - **BiLSTM:** Time-series patterns ko samajhne ke liye.
    - **Temporal Attention:** Important time intervals par focus karne ke liye.

### 5. Technology Stack (Software aur Tools)
- **Frontend:** React.js, Vite, Chart.js, Framer Motion.
- **Backend:** Flask (Python), PyJWT (Authentication).
- **Machine Learning:** TensorFlow/Keras, NumPy, SciPy, Scikit-learn.
- **Database:** MongoDB Atlas (Activity logs store karne ke liye).
- **AI Integration:** Google Gemini API (Reports ke liye).

### 6. Key Features (Main Highlights)
- **Real-time Detection:** Walking, Jogging, Sitting, Stairs, etc. ko instantly detect karta hai.
- **AI Health Reports:** Gemini AI user ke pure din ka data analyze karke summary likhta hai.
- **PDF Export:** User apni reports ko download ya email par share kar sakta hai.
- **Cross-Platform:** Mobile browser par directly kaam karta hai, kisi app installation ki zaroorat nahi.

### 7. Future Scope (Aage kya ho sakta hai?)
- Isme heart rate aur sleep patterns ka data bhi add kiya ja sakta hai.
- More specific activities jaise Cycling ya Gym exercises ko recognize karna.
- Diet recommendation system integrate karna based on activity intensity.

---
