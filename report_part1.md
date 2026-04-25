# REAL-TIME HUMAN ACTIVITY RECOGNITION SYSTEM WITH AI-BASED DAILY HEALTH REPORTING

## Minor Project Report

Submitted in partial fulfillment of the requirements for the degree of
Bachelor of Technology in Computer Science and Engineering

**Submitted by:**
Vansh Tambi
Vivek Pasi

**Indian Institute of Information Technology, Bhopal**
2026

---

## ABSTRACT

Human Activity Recognition (HAR) has emerged as a critical area of research in ubiquitous computing, with applications spanning healthcare monitoring, fitness tracking, elderly care, and smart home automation. This project presents the design, development, and deployment of a real-time Human Activity Recognition system capable of classifying eight distinct physical activities using six-axis inertial measurement unit (IMU) data streamed directly from a mobile device's accelerometer and gyroscope sensors. The system employs a hybrid deep learning architecture combining one-dimensional convolutional neural networks (Conv1D) with Long Short-Term Memory (LSTM) networks, enabling both spatial feature extraction across sensor channels and temporal dependency modeling over sliding time windows.

The data pipeline aggregates three established public datasets — WISDM (Wireless Sensor Data Mining), Heterogeneity Activity Recognition, and UCI HAR — alongside custom-recorded mobile sensor data. A custom-priority training strategy is adopted wherein user-recorded data is augmented thirty-fold using six distinct augmentation techniques including Gaussian jitter, magnitude warping, time warping, channel permutation, temporal shifting, and signal inversion, while public dataset contributions are capped at five hundred samples per class. This ensures the model prioritizes real-world mobile sensor characteristics over laboratory-collected data.

The trained model achieves an overall test accuracy of 82.99 percent across eight activity classes: Walking, Jogging, Stairs, Still, Eating, Hand Activity, Active Hands, and Sports. Notably, the system achieves 99 percent precision for Jogging, 96 percent precision for Eating, and 94 percent precision for Still activities. The inference pipeline incorporates real-time Butterworth signal filtering, variance-based stationary detection, confidence thresholding, and majority voting to deliver stable, low-latency predictions.

The system architecture follows a client-server paradigm with a React-based frontend dashboard built on Vite, a Flask-based REST API backend for model inference and user management, MongoDB Atlas for persistent activity logging, Google OAuth 2.0 for secure authentication, and integration with Google's Gemini 1.5 Flash large language model for automated generation of natural-language daily health reports. The frontend captures sensor data at 20 Hz, buffers sixty samples into three-second sliding windows, and transmits them to the backend for real-time classification with live visualization using Chart.js.

**Keywords:** Human Activity Recognition, Conv1D, LSTM, Deep Learning, Inertial Sensors, Real-Time Classification, Health Monitoring, Sensor Data Augmentation

---

## LIST OF FIGURES

- Figure 1.1: Growth of HAR Research Publications (2015-2025)
- Figure 3.1: High-Level System Architecture
- Figure 3.2: Data Flow Diagram (Level 0)
- Figure 3.3: Data Flow Diagram (Level 1)
- Figure 5.1: Data Preprocessing Pipeline
- Figure 5.2: Conv1D + LSTM Model Architecture
- Figure 5.3: Prediction Pipeline Flowchart
- Figure 5.4: End-to-End System Sequence Diagram
- Figure 5.5: Authentication Flow Diagram
- Figure 6.1: Algorithm Pseudocode — Data Preparation
- Figure 6.2: Algorithm Pseudocode — Real-Time Prediction
- Figure 8.1: Confusion Matrix (8-Class Classification)
- Figure 8.2: Class-Wise Precision, Recall, and F1-Score Comparison
- Figure 8.3: Confidence Distribution Across Activities

## LIST OF TABLES

- Table 2.1: Comparison of HAR Approaches in Literature
- Table 4.1: Dataset Sources and Contributions
- Table 4.2: Activity Class Consolidation Mapping
- Table 4.3: Augmentation Techniques Applied to Custom Data
- Table 7.1: Tools and Technologies Used
- Table 7.2: Python Backend Dependencies
- Table 7.3: Frontend Dependencies
- Table 8.1: Classification Report (Per-Class Metrics)
- Table 8.2: Comparison with Traditional ML Approaches
- Table 8.3: Impact of Augmentation on Model Accuracy

---

## CHAPTER 1: INTRODUCTION

### 1.1 Background

The proliferation of smartphones equipped with high-precision inertial measurement units (IMUs) has created unprecedented opportunities for continuous, non-invasive monitoring of human physical activities. Modern smartphones universally incorporate tri-axial accelerometers and gyroscopes capable of sampling at rates between 20 and 200 Hz, generating rich streams of motion data that encode distinctive patterns corresponding to different physical activities. Human Activity Recognition (HAR) leverages this sensor data to automatically classify user movements, enabling applications in personalized healthcare, rehabilitation monitoring, sports analytics, context-aware computing, and ambient assisted living for elderly populations.

The fundamental challenge in HAR lies in accurately distinguishing between activities that produce similar sensor signatures. For instance, walking and climbing stairs generate comparable accelerometer patterns, while sedentary activities such as sitting and standing produce nearly identical readings during stationary phases. Furthermore, the same activity performed by different individuals exhibits significant inter-subject variability due to differences in gait, body composition, device placement, and movement intensity. These challenges necessitate sophisticated machine learning approaches capable of learning discriminative features from raw or minimally processed sensor data.

Traditional approaches to HAR relied on handcrafted feature engineering, wherein domain experts manually designed statistical and frequency-domain features — such as mean, variance, signal magnitude area, spectral entropy, and autoregressive coefficients — from fixed-size time windows of sensor data. These features were subsequently classified using conventional machine learning algorithms including Support Vector Machines (SVM), Random Forests, k-Nearest Neighbors (k-NN), and Decision Trees. While these methods achieved reasonable accuracy for simple activity sets (walking, sitting, standing), they exhibited limited scalability and poor generalization when confronted with complex, overlapping activities or heterogeneous sensor configurations.

The advent of deep learning has fundamentally transformed the HAR landscape. Convolutional Neural Networks (CNNs) eliminate the need for manual feature engineering by automatically learning hierarchical spatial features directly from raw sensor signals. Recurrent Neural Networks (RNNs), particularly Long Short-Term Memory (LSTM) networks, excel at modeling temporal dependencies inherent in sequential sensor data. Hybrid architectures combining CNNs and LSTMs have demonstrated state-of-the-art performance by jointly extracting spatial features and capturing temporal dynamics within a unified end-to-end framework.

### 1.2 Motivation

Despite significant academic progress in HAR, most existing systems remain confined to offline, batch-processing paradigms evaluated on benchmark datasets under controlled laboratory conditions. The translation of these research prototypes into deployable, real-time systems that operate on commodity hardware with live sensor streams presents numerous engineering challenges that are rarely addressed in academic literature. These challenges include continuous data acquisition at consistent sampling rates, real-time signal preprocessing with minimal latency, efficient model inference within strict time budgets, graceful handling of noisy and incomplete sensor readings, and meaningful presentation of classification results to end users.

Furthermore, the practical utility of activity classification extends beyond mere labeling. Healthcare professionals, fitness coaches, and individuals themselves benefit from aggregated, contextualized summaries of daily activity patterns rather than raw classification outputs. The integration of large language models (LLMs) for automated generation of human-readable health reports from activity logs represents a novel application of generative AI in the HAR domain.

This project is motivated by the objective of bridging the gap between research-grade HAR models and production-ready activity monitoring systems, delivering a complete end-to-end solution encompassing real-time data capture, deep learning inference, persistent data logging, secure authentication, and AI-powered health reporting.

### 1.3 Scope of the Project

This project encompasses the following technical contributions:

1. Design and implementation of a hybrid Conv1D + LSTM deep learning model trained on a custom-priority aggregation of three public HAR datasets and user-recorded mobile sensor data.
2. Development of a real-time signal preprocessing pipeline incorporating Butterworth filtering, gravity separation, and engineered magnitude features.
3. Construction of a responsive React-based dashboard for live sensor data visualization, activity prediction display, and user interaction.
4. Implementation of a Flask-based REST API for model inference, user authentication via Google OAuth 2.0, JWT session management, and MongoDB-backed activity logging.
5. Integration of Google Gemini 1.5 Flash for automated daily health report generation from accumulated activity logs.
6. Deployment infrastructure supporting local area network access for mobile device testing and Vercel-compatible frontend hosting.

### 1.4 Organization of the Report

The remainder of this report is organized as follows. Chapter 2 presents a comprehensive review of existing literature on HAR methodologies. Chapter 3 defines the problem statement and project objectives. Chapter 4 details the proposed methodology including data preparation, model architecture, and system design. Chapter 5 presents the algorithms in pseudocode form. Chapter 6 provides flowcharts and architectural diagrams. Chapter 7 describes the tools and technologies employed along with implementation details. Chapter 8 presents result analysis and discussion. Chapter 9 concludes the report with a summary of contributions and directions for future work.

---

## CHAPTER 2: BACKGROUND DETAILS AND LITERATURE REVIEW

### 2.1 Overview of Human Activity Recognition

Human Activity Recognition is a multidisciplinary research area intersecting signal processing, machine learning, and pervasive computing. The objective is to automatically identify physical activities performed by a user based on data captured from wearable or ambient sensors. HAR systems can be broadly categorized based on their sensing modality: vision-based systems employing cameras and depth sensors, ambient sensor systems utilizing environmental sensors such as pressure mats and door sensors, and wearable sensor systems leveraging accelerometers, gyroscopes, magnetometers, and barometers embedded in smartphones or dedicated wearable devices.

Wearable IMU-based HAR has received the most research attention due to the ubiquity of smartphones, the low cost of MEMS (Micro-Electro-Mechanical Systems) sensors, and the minimal privacy concerns compared to camera-based approaches. The standard pipeline for IMU-based HAR consists of four stages: data collection, signal preprocessing, feature extraction and selection, and classification. The evolution of this pipeline from manual feature engineering to end-to-end deep learning constitutes the primary narrative of HAR research over the past decade.

### 2.2 Traditional Machine Learning Approaches

Early HAR systems relied extensively on handcrafted features extracted from fixed-size sliding windows of sensor data. Anguita et al. [1] introduced the UCI HAR dataset and demonstrated that a multiclass SVM trained on 561 time-domain and frequency-domain features extracted from accelerometer and gyroscope data could achieve 96 percent accuracy on six basic activities (walking, walking upstairs, walking downstairs, sitting, standing, and laying). However, this approach required substantial domain expertise for feature engineering and exhibited limited transferability to novel activity sets or sensor configurations.

Kwapisz et al. [2] utilized the WISDM dataset comprising accelerometer data collected from smartphones carried in users' pockets during six activities. They evaluated multiple classifiers — Decision Trees, Logistic Regression, and Multilayer Perceptrons — on a feature set consisting of mean, standard deviation, absolute deviation, resultant acceleration, time between peaks, and binned distributions. The Multilayer Perceptron achieved the highest accuracy of 91.7 percent on this dataset. Their work highlighted the challenge of inter-subject variability, noting significant performance degradation when models were evaluated on data from unseen users.

Stisen et al. [3] investigated the impact of device heterogeneity on HAR performance using data from nine different smartphone models and two smartwatch models. Their Heterogeneity Activity Recognition dataset demonstrated that models trained on data from one device type often failed to generalize to other devices due to differences in sensor sampling rates, noise characteristics, and coordinate system orientations. This finding underscored the importance of training on heterogeneous data sources for robust real-world deployment.

Random Forests emerged as a popular choice for HAR due to their robustness to overfitting and ability to handle high-dimensional feature spaces. Reyes-Ortiz et al. [4] demonstrated that ensemble methods combining multiple decision tree classifiers achieved competitive accuracy while offering interpretable feature importance rankings that provided insights into which sensor channels and statistical features were most discriminative for activity classification.

### 2.3 Convolutional Neural Network-Based Approaches

The application of Convolutional Neural Networks to sensor-based HAR was pioneered by Yang et al. [5], who proposed applying 1D convolutions directly to raw accelerometer signals, treating the multi-axis sensor data as multi-channel input analogous to color channels in image processing. Their approach eliminated the need for manual feature engineering and achieved accuracy improvements of 3 to 5 percent over traditional methods on the Opportunity and Skoda datasets. The key insight was that convolutional filters learn to detect local patterns such as peaks, valleys, and oscillations in sensor signals that correspond to characteristic motion primitives.

Zeng et al. [6] extended this work by exploring deeper architectures with multiple convolutional layers interspersed with max-pooling operations, demonstrating that hierarchical feature learning — where lower layers capture fine-grained signal patterns and higher layers compose these into activity-level representations — improved classification performance on complex activity taxonomies. Their architecture processed each sensor axis independently through parallel convolutional streams before concatenating the learned features for classification.

Ha and Choi [7] conducted a systematic comparison of CNN architectures for HAR, evaluating single-channel, multi-channel, and cascade architectures on multiple benchmark datasets. Their results indicated that multi-channel architectures processing all sensor axes jointly outperformed single-channel approaches, as inter-axis correlations encode important information about the spatial orientation and dynamics of body movements.

### 2.4 LSTM and Recurrent Network Approaches

Long Short-Term Memory networks address a fundamental limitation of CNNs in the HAR context: the inability to model long-range temporal dependencies that span multiple time steps within a sensor data window. Hammerla et al. [8] conducted a comprehensive benchmark of deep learning architectures for HAR, comparing fully connected networks, CNNs, and vanilla LSTMs across multiple datasets. Their findings indicated that LSTMs consistently outperformed CNNs on datasets with complex temporal structures, particularly for activities involving extended sequences of sub-movements such as cooking or cleaning.

Guan and Plotz [9] proposed attention-based LSTM architectures for HAR, introducing a mechanism that allowed the network to selectively focus on the most informative time steps within a window rather than treating all time steps equally. This attention mechanism improved accuracy on the PAMAP2 and Opportunity datasets while providing interpretable visualizations of which temporal segments the model deemed most relevant for classification.

Bidirectional LSTMs, which process input sequences in both forward and reverse directions, were shown by Zhao et al. [10] to capture richer temporal context by leveraging both past and future information within a window. This architecture achieved improved recognition of transition activities — such as sit-to-stand or stand-to-walk — where the temporal context surrounding the transition point is critical for accurate classification.

### 2.5 Hybrid CNN-LSTM Architectures

The complementary strengths of CNNs (spatial feature extraction) and LSTMs (temporal dependency modeling) motivated the development of hybrid architectures that combine both approaches. Ordonez and Roggen [11] proposed DeepConvLSTM, a seminal architecture consisting of four convolutional layers followed by two LSTM layers. The convolutional layers extract local features from short temporal segments, while the LSTM layers model the temporal evolution of these features across the entire window. DeepConvLSTM achieved state-of-the-art results on the Opportunity and Skoda datasets, establishing the CNN-LSTM paradigm as the dominant approach in HAR research.

The theoretical motivation for the hybrid approach rests on the observation that human activities are inherently hierarchical: low-level motion primitives (e.g., individual foot strikes during walking) compose into higher-level temporal patterns (e.g., the rhythmic gait cycle) that collectively characterize an activity. CNNs are well-suited to detecting the low-level primitives through local receptive fields, while LSTMs excel at capturing the higher-level compositional structure through their recurrent memory mechanism.

Subsequent works refined the hybrid paradigm. Mutegeki and Han [12] introduced batch normalization between convolutional layers to stabilize training and improve generalization. Xia et al. [13] explored deeper convolutional encoders with residual connections to facilitate gradient flow in very deep networks. Chen et al. [14] combined the hybrid architecture with multi-task learning, simultaneously predicting the activity class and estimating physical parameters such as walking speed and energy expenditure.

### 2.6 Data Augmentation for Sensor Data

Data augmentation, a technique well-established in computer vision, has been adapted for sensor-based HAR to address the chronic problem of limited labeled training data. Um et al. [15] systematically evaluated augmentation techniques for IMU data, including rotation, permutation, time warping, magnitude warping, jittering, and scaling. Their findings indicated that combining multiple augmentation strategies yielded more robust models than any single technique, with time warping and jittering providing the most consistent improvements across datasets.

### 2.7 Summary and Research Gap

Table 2.1 summarizes the key approaches reviewed in this chapter.

**Table 2.1: Comparison of HAR Approaches in Literature**

| Approach | Representative Work | Strengths | Limitations |
|----------|-------------------|-----------|-------------|
| SVM + Handcrafted Features | Anguita et al. [1] | High accuracy on simple activities; interpretable features | Requires domain expertise; poor scalability |
| Random Forest | Reyes-Ortiz et al. [4] | Robust to overfitting; feature importance | Limited to engineered features |
| 1D CNN | Yang et al. [5] | Automatic feature learning; no manual engineering | Limited temporal modeling |
| LSTM | Hammerla et al. [8] | Strong temporal dependency modeling | Computationally expensive; limited spatial features |
| DeepConvLSTM | Ordonez and Roggen [11] | Joint spatial-temporal learning; state-of-the-art | High parameter count; offline evaluation only |
| Attention-LSTM | Guan and Plotz [9] | Interpretable; selective temporal focus | Complex architecture |

The literature reveals that while hybrid CNN-LSTM architectures represent the state of the art, most studies evaluate models exclusively on pre-segmented benchmark datasets in offline settings. Critical aspects of real-world deployment — including real-time sensor streaming, signal preprocessing for live data, confidence-aware prediction, user authentication, persistent activity logging, and automated health reporting — remain largely unaddressed. This project bridges this gap by delivering a complete, deployable HAR system that integrates a custom-trained Conv1D + LSTM model with a production-grade software stack.

---

## CHAPTER 3: PROBLEM DEFINITION AND OBJECTIVES

### 3.1 Problem Statement

Existing Human Activity Recognition systems suffer from several limitations that impede their practical deployment:

1. **Dependence on single-modality sensor data.** Many systems utilize only accelerometer data, neglecting the complementary information provided by gyroscope sensors regarding rotational dynamics and angular velocity, which are critical for disambiguating activities with similar translational motion profiles.

2. **Reliance on controlled laboratory datasets.** Models trained exclusively on benchmark datasets collected under controlled conditions exhibit significant performance degradation when deployed in unconstrained real-world environments characterized by diverse device models, variable sensor placement, and ambient noise.

3. **Absence of real-time inference infrastructure.** The majority of HAR research evaluates models in offline, batch-processing settings, neglecting the engineering challenges of continuous sensor data streaming, real-time preprocessing, and low-latency inference required for responsive user-facing applications.

4. **Lack of actionable health insights.** Raw activity classification labels provide limited value to end users without contextual aggregation, temporal analysis, and human-readable interpretation of daily activity patterns.

### 3.2 Objectives

The primary objectives of this project are:

1. To design and train a hybrid Conv1D + LSTM deep learning model for eight-class human activity recognition using six-axis IMU data (accelerometer and gyroscope), achieving a minimum test accuracy of 80 percent.

2. To develop a custom-priority data preparation pipeline that aggregates multiple public datasets (WISDM, Heterogeneity, UCI HAR) with user-recorded mobile sensor data, employing aggressive data augmentation to ensure model alignment with real-world sensor characteristics.

3. To implement a real-time prediction pipeline incorporating Butterworth signal filtering, variance-based stationary detection, confidence thresholding, and temporal smoothing via majority voting.

4. To build a responsive web-based dashboard enabling live sensor data capture from mobile browsers at 20 Hz, real-time activity visualization, and prediction display with confidence indicators.

5. To implement secure user authentication using Google OAuth 2.0 with JWT-based session management, and persistent activity logging using MongoDB Atlas.

6. To integrate Google Gemini 1.5 Flash for automated generation of daily health reports from accumulated activity logs, with support for report sharing between users.
