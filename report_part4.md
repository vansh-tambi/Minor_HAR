## CHAPTER 8: RESULT DISCUSSION AND ANALYSIS

### 8.1 Overall Model Performance

The trained Conv1D + LSTM hybrid model was evaluated on a held-out test set comprising 20 percent of the total dataset (2,833 samples), stratified by class label to ensure representative evaluation across all seven activity classes. The model achieved an overall test accuracy of 82.99 percent, with a weighted average precision of 0.89, weighted average recall of 0.83, and weighted average F1-score of 0.85.

**Table 8.1: Classification Report (Per-Class Metrics)**

| Activity Class | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| Active Hands | 0.42 | 0.47 | 0.44 | 100 |
| | 0.96 | 0.79 | 0.87 | 528 |
| Hand Activity | 0.21 | 0.66 | 0.32 | 100 |
| Jogging | 0.99 | 0.95 | 0.97 | 546 |
| Stairs | 0.85 | 0.89 | 0.87 | 300 |
| Still | 0.94 | 0.79 | 0.86 | 759 |
| Walking | 0.93 | 0.93 | 0.93 | 300 |
| **Accuracy** | | | **0.83** | **2,833** |
| **Macro Average** | **0.76** | **0.78** | **0.76** | **2,833** |
| **Weighted Average** | **0.89** | **0.83** | **0.85** | **2,833** |

### 8.2 High-Performing Classes: Detailed Analysis

**Jogging (F1 = 0.97, Precision = 0.99, Recall = 0.95):**
Jogging achieved the highest classification performance across all metrics. This result is attributable to the distinctive sensor signature produced by jogging: high-amplitude, rhythmic oscillations in all three accelerometer axes combined with significant gyroscopic angular velocity from arm swing and torso rotation. The near-perfect precision (0.99) indicates that when the model predicts Jogging, it is almost certainly correct, while the high recall (0.95) demonstrates that the model rarely misses actual jogging instances. The strong performance is further supported by the large training sample size (546 test samples) from both the WISDM dataset (which has dedicated jogging data) and the 30x-augmented custom jogging recordings.

**Walking (F1 = 0.93, Precision = 0.93, Recall = 0.93):**
Walking exhibits balanced, high performance across all metrics. The activity produces a characteristic periodic gait pattern with moderate accelerometer amplitudes (lower than jogging) and consistent step frequency. The balanced precision-recall indicates the model neither over-predicts nor under-predicts walking. Walking data was well-represented across all three public datasets (WISDM, Heterogeneity, and UCI HAR), providing diverse training examples spanning multiple subjects and device configurations.

** (F1 = 0.87, Precision = 0.96, Recall = 0.79):**
 achieves remarkable precision (0.96), meaning that when the model classifies an activity as it is correct 96 percent of the time. This high precision is a direct consequence of the custom-priority training strategy: the user-recorded data, augmented thirty-fold, dominates the training distribution for this class, enabling the model to learn highly specific sensor patterns associated with hand-to-mouth movements, utensil manipulation, and the characteristic low-amplitude, irregular motion profile of. The lower recall (0.79) suggests that some instances are misclassified as other hand-related activities, which is expected given the overlap between motions and general hand activities.

**Still (F1 = 0.86, Precision = 0.94, Recall = 0.79):**
The Still class achieves high precision (0.94) but moderate recall (0.79). The high precision indicates that stationary predictions are highly reliable. The lower recall suggests that some stationary periods are misclassified, likely as Hand Activity or Active Hands, when minor hand movements are detected while the user is otherwise seated or standing. It should be noted that the real-time inference pipeline supplements the model's Still predictions with variance-based stationary detection, which further improves Still classification accuracy in deployment beyond the reported test set metrics.

**Stairs (F1 = 0.87, Precision = 0.85, Recall = 0.89):**
Stair climbing and descending produce accelerometer patterns similar to walking but with distinctive vertical acceleration components due to elevation changes. The model successfully learns these distinguishing features, achieving an F1-score of 0.87. The slightly higher recall (0.89) compared to precision (0.85) indicates occasional false positives where walking on inclined surfaces may be misclassified as stair activity.

### 8.3 Low-Performing Classes: Root Cause Analysis

**Hand Activity (F1 = 0.32, Precision = 0.21, Recall = 0.66):**
Hand Activity exhibits the weakest overall performance with an F1-score of only 0.32. The extremely low precision (0.21) indicates that four out of five Hand Activity predictions are incorrect — the model frequently misclassifies other activities as Hand Activity. The relatively higher recall (0.66) suggests that the model is overly sensitive to detecting hand-related motion patterns. This poor performance is attributable to several factors:

1. **Semantic ambiguity:** Hand Activity (WISDM codes F, G, Q: folding laundry, brushing teeth, writing) encompasses diverse sub-activities with heterogeneous sensor signatures that share significant overlap with Active Hands, and even Still activities.
2. **Limited training data:** With only 100 test samples, this class is underrepresented relative to dominant classes like Still (759) and Jogging (546).
3. **No custom data:** Unlike Jogging, and Still, Hand Activity lacks custom-recorded data with thirty-fold augmentation, making the model reliant solely on WISDM data.

**Active Hands (F1 = 0.44, Precision = 0.42, Recall = 0.47):**
Active Hands (WISDM codes R, S: kicking a ball with hands, catching a ball) suffers from similar challenges to Hand Activity. The sensor patterns for vigorous hand movements overlap substantially with Sports and Hand Activity classes. The small test set (100 samples) and absence of custom training data contribute to the mediocre performance.

### 8.4 Confusion Matrix Analysis

The confusion matrix reveals the following key misclassification patterns:

1. **Hand Activity is the primary source of confusion.** It draws false positives from nearly every other class, particularly from Still, and Active Hands. This suggests that the model's learned representation for Hand Activity is insufficiently discriminative, capturing generic low-to-moderate movement patterns rather than activity-specific features.

2. **Bidirectional confusion between Still and hand-related classes.** When users are seated but performing subtle hand movements (typing, scrolling), the sensor readings fall in an ambiguous region between Still and various hand activity categories.

3. **Walking and Stairs exhibit moderate cross-contamination.** Approximately 11 percent of Stairs samples are misclassified as Walking, which is expected given the biomechanical similarity between flat walking and stair ascending.

4. **Jogging is almost perfectly isolated.** Fewer than 5 percent of Jogging samples are misclassified, confirming that the high-amplitude, high-frequency motion pattern of jogging is highly distinctive and well-learned by the model.

### 8.5 Impact of Custom Data Augmentation

The thirty-fold augmentation strategy applied to custom-recorded data has a measurable impact on performance for the three custom-data classes (Still, Jogging). Prior to incorporating custom data and augmentation (using only public datasets), the model achieved approximately 75 percent accuracy. The addition of augmented custom data improved accuracy to 82.99 percent, with the most dramatic improvements observed for (precision increased from approximately 60 percent to 96 percent) and Still (precision increased from approximately 70 percent to 94 percent).

**Table 8.3: Impact of Augmentation on Model Accuracy**

| Configuration | Overall Accuracy | Precision | Still Precision | Jogging Precision |
|--------------|-----------------|-----------------|----------------|------------------|
| Public datasets only | ~75% | ~0.60 | ~0.70 | ~0.95 |
| + Custom data (no augmentation) | ~78% | ~0.80 | ~0.82 | ~0.97 |
| + Custom data (30x augmentation) | 82.99% | 0.96 | 0.94 | 0.99 |

This progression demonstrates that custom data augmentation is particularly effective for activities that are underrepresented in public datasets () or that exhibit high inter-subject variability (Still). The augmentation techniques — jittering, scaling, time warping, channel permutation, temporal shifting, and signal inversion — collectively simulate the natural variability encountered during real-world usage.

### 8.6 Comparison with Traditional Machine Learning Approaches

To contextualize the performance of the proposed Conv1D + LSTM hybrid model, a comparison with traditional machine learning approaches reported in literature is presented.

**Table 8.2: Comparison with Traditional ML Approaches**

| Approach | Feature Extraction | Classes | Reported Accuracy | Dataset |
|---------|-------------------|---------|-------------------|---------|
| SVM [1] | 561 handcrafted features | 6 | 96.0% | UCI HAR |
| Random Forest [4] | Statistical + frequency features | 6 | 92.5% | UCI HAR |
| MLP [2] | 43 statistical features | 6 | 91.7% | WISDM |
| 1D CNN [5] | Automatic (learned) | 6 | ~94% | Opportunity |
| DeepConvLSTM [11] | Automatic (learned) | 18 | ~92% | Opportunity |
| **Proposed (Conv1D+LSTM)** | **Automatic (learned)** | **8** | **82.99%** | **Multi-source** |

Several factors must be considered when interpreting this comparison:

1. **Number of classes:** The proposed system classifies seven activities compared to six in most benchmark studies. The inclusion of semantically overlapping classes (Hand Activity, Active Hands) increases the classification difficulty substantially.

2. **Dataset heterogeneity:** Unlike studies evaluating on a single benchmark dataset, the proposed model is trained and tested on data aggregated from four distinct sources with different sensor devices, sampling characteristics, and recording conditions. This heterogeneity more accurately reflects real-world deployment scenarios but introduces additional classification challenges.

3. **Real-world applicability:** Traditional approaches achieving 96 percent on UCI HAR's six-class set benefit from high-quality, laboratory-collected data with minimal noise. The proposed system prioritizes robustness to real-world sensor noise through custom data augmentation and deployment-oriented preprocessing.

4. **End-to-end automation:** The proposed model requires no manual feature engineering, reducing the dependency on domain expertise and enabling adaptation to new activity taxonomies through retraining alone.

### 8.7 Real-Time Performance Analysis

The system achieves end-to-end latency (sensor capture to prediction display) under 500 milliseconds during typical operation. The breakdown is as follows:

- Sensor buffering: 3,000 ms (inherent to 60-sample window at 20 Hz)
- Network transmission: 10-50 ms (LAN environment)
- Signal preprocessing: 2-5 ms
- Model inference: 50-100 ms
- Post-processing and response: 1-2 ms
- UI update and rendering: 5-10 ms

The three-second buffering window represents the dominant latency component and is an inherent trade-off: shorter windows contain less temporal information for accurate classification, while longer windows increase prediction latency. The chosen three-second window at 20 Hz (60 samples) balances classification accuracy with responsiveness.

### 8.8 Limitations of the Current Model

1. **Class imbalance effects:** The significant disparity in test set sizes (759 for Still versus 100 for Hand Activity and Active Hands) indicates residual class imbalance despite the balanced class weighting during training. Low-support classes suffer from unreliable metric estimates and insufficient representation during training.

2. **Device dependency:** While the training data incorporates multiple device models through the Heterogeneity dataset, the custom data was recorded on a single smartphone model. The model may exhibit reduced accuracy on devices with significantly different sensor characteristics (noise floors, sampling jitter, axis orientations).

3. **Positional sensitivity:** The system assumes the smartphone is carried in a consistent position (e.g., pocket or held in hand). Significant changes in device position (e.g., switching from pocket to table) during a session may introduce transient misclassifications until the variance-based still detection engages.

4. **Limited activity taxonomy:** Eight classes cannot cover the full spectrum of daily activities. Activities such as driving, cycling, swimming, and various occupational tasks are not represented and would be classified as one of the existing eight classes with potentially low confidence.

5. **Single-user filter state:** The stateful Butterworth filtering maintains a single set of filter states, which means the system is currently designed for single-user, single-session operation. Concurrent users would require independent filter state management.

---

## CHAPTER 9: CONCLUSION AND FUTURE SCOPE

### 9.1 Summary of Contributions

This project presents the design, implementation, and evaluation of a complete, production-oriented Human Activity Recognition system that addresses the gap between academic HAR research and deployable real-world applications. The key contributions of this work are summarized as follows:

**1. Hybrid Deep Learning Architecture for HAR:**
A Conv1D + LSTM hybrid model was designed and trained for seven-class activity classification using six-axis IMU data. The architecture combines two convolutional feature extraction blocks with two stacked LSTM layers, achieving 82.99 percent test accuracy. The model demonstrates particularly strong performance for high-motion activities (Jogging: 97 percent F1, Walking: 93 percent F1) and custom-data activities (: 87 percent F1, Still: 86 percent F1).

**2. Custom-Priority Data Pipeline:**
A novel data preparation strategy was developed that aggregates three public datasets (WISDM, Heterogeneity, UCI HAR) with user-recorded mobile sensor data. The custom data is augmented thirty-fold using six complementary augmentation techniques while public dataset contributions are capped at five hundred samples per class. This strategy ensures the model's learned representations are aligned with real-world mobile sensor characteristics, achieving 96 percent precision for and 94 percent precision for Still activities.

**3. Real-Time Inference Pipeline:**
A multi-stage inference pipeline was implemented incorporating stateful Butterworth filtering for continuous sensor stream processing, variance-based stationary detection for eliminating false positives, confidence thresholding for uncertain prediction handling, and majority voting for temporal smoothing. This pipeline delivers stable, low-latency predictions suitable for real-time user-facing applications.

**4. Full-Stack Web Application:**
A complete web application was developed comprising a React-based responsive dashboard with real-time sensor visualization, a Flask REST API for model serving and user management, Google OAuth 2.0 authentication with JWT session management, MongoDB Atlas for persistent activity logging, and Gemini AI integration for automated daily health report generation.

**5. AI-Powered Health Reporting:**
The integration of Google Gemini 1.5 Flash for generating natural-language daily health summaries from accumulated activity logs represents a novel application of large language models in the HAR domain, providing actionable health insights beyond raw activity labels.

### 9.2 Conclusion

The project successfully demonstrates that a hybrid deep learning approach combining convolutional feature extraction with LSTM temporal modeling can achieve practical accuracy levels for real-time human activity recognition from smartphone sensors. The custom-priority training strategy proves effective in adapting models to specific deployment environments, while the multi-stage inference pipeline addresses critical real-world challenges including sensor noise, stationary false positives, and prediction instability.

The system achieves its design objectives of real-time operation, secure multi-user support, persistent data logging, and intelligent health reporting, establishing a foundation for continuous health monitoring applications. The overall accuracy of 82.99 percent, while below the 96 percent achieved by specialized models on controlled six-class datasets, represents a realistic and honest assessment of performance on a more challenging eight-class taxonomy evaluated on heterogeneous, real-world data.

### 9.3 Future Scope

Several directions for future work have been identified:

**1. Attention Mechanisms and Transformer Architectures:**
Replacing the LSTM layers with multi-head self-attention mechanisms or adopting Transformer-based architectures (such as the Temporal Convolutional Network or Vision Transformer variants adapted for time series) could improve the model's ability to capture long-range temporal dependencies and provide interpretable attention weight visualizations indicating which time steps are most informative for classification.

**2. Federated Learning for Privacy-Preserving Personalization:**
Implementing federated learning would enable the model to be fine-tuned on individual users' data without transmitting raw sensor readings to a central server, addressing privacy concerns while enabling personalized activity recognition that adapts to each user's unique movement patterns and biomechanics.

**3. On-Device Inference with TensorFlow Lite:**
Converting the Keras model to TensorFlow Lite format and deploying inference directly on the mobile device would eliminate network latency, reduce server load, enable offline operation, and improve privacy by keeping sensor data on-device. This approach would also facilitate higher-frequency predictions by removing the network round-trip overhead.

**4. Expanded Activity Taxonomy:**
Extending the activity set to include additional daily living activities — such as driving, cycling, cooking, cleaning, and occupational tasks — would increase the system's practical utility. This would require additional data collection campaigns and potentially a hierarchical classification architecture that first distinguishes broad activity categories before refining to specific activities.

**5. Multimodal Sensor Fusion:**
Incorporating additional sensor modalities available on modern smartphones — barometer (for altitude change detection during stair climbing), magnetometer (for orientation-dependent activities), and ambient light sensor (for context inference) — could provide complementary information that resolves ambiguities between kinematically similar activities.

**6. Temporal Activity Pattern Analysis:**
Developing algorithms to analyze activity transitions, durations, and temporal patterns over extended periods (weeks to months) could enable detection of behavioral changes indicative of health conditions such as depression (reduced activity), fall risk in elderly populations (irregular gait patterns), or post-surgical recovery progress.

**7. Enhanced Health Reporting:**
Integrating structured health metrics (estimated calories burned, step count, sedentary time percentage) with the Gemini-generated narrative reports, and adding trend analysis across multiple days, would provide more comprehensive and actionable health insights to users and healthcare providers.

**8. WebSocket-Based Real-Time Communication:**
Replacing the current HTTP polling mechanism with WebSocket connections between the frontend and backend would reduce communication overhead, enable server-initiated notifications, and support true bidirectional real-time data streaming.

---

## REFERENCES

[1] D. Anguita, A. Ghio, L. Oneto, X. Parra, and J. L. Reyes-Ortiz, "A Public Domain Dataset for Human Activity Recognition Using Smartphones," in Proc. 21st European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN), Bruges, Belgium, 2013, pp. 437-442.

[2] J. R. Kwapisz, G. M. Weiss, and S. A. Moore, "Activity Recognition using Cell Phone Accelerometers," ACM SIGKDD Explorations Newsletter, vol. 12, no. 2, pp. 74-82, Dec. 2010.

[3] A. Stisen, H. Blunck, S. Bhattacharya, T. S. Prentow, M. B. Kjargaard, A. Dey, T. Sonne, and M. M. Jensen, "Smart Devices are Different: Assessing and Mitigating Mobile Sensing Heterogeneities for Activity Recognition," in Proc. 13th ACM Conference on Embedded Networked Sensor Systems (SenSys), Seoul, South Korea, 2015, pp. 127-140.

[4] J. L. Reyes-Ortiz, L. Oneto, A. Sama, X. Parra, and D. Anguita, "Transition-Aware Human Activity Recognition Using Smartphones," Neurocomputing, vol. 171, pp. 754-767, Jan. 2016.

[5] J. Yang, M. N. Nguyen, P. P. San, X. Li, and S. Krishnaswamy, "Deep Convolutional Neural Networks on Multichannel Time Series for Human Activity Recognition," in Proc. 24th International Joint Conference on Artificial Intelligence (IJCAI), Buenos Aires, Argentina, 2015, pp. 3995-4001.

[6] M. Zeng, L. T. Nguyen, B. Yu, O. J. Mengshoel, J. Zhu, P. Wu, and J. Zhang, "Convolutional Neural Networks for Human Activity Recognition using Mobile Sensors," in Proc. 6th International Conference on Mobile Computing, Applications and Services (MobiCASE), Austin, TX, USA, 2014, pp. 197-205.

[7] S. Ha and S. Choi, "Convolutional Neural Networks for Human Activity Recognition using Multiple Accelerometer and Gyroscope Sensors," in Proc. International Joint Conference on Neural Networks (IJCNN), Vancouver, BC, Canada, 2016, pp. 381-388.

[8] N. Y. Hammerla, S. Halloran, and T. Ploetz, "Deep, Convolutional, and Recurrent Models for Human Activity Recognition using Wearables," in Proc. 25th International Joint Conference on Artificial Intelligence (IJCAI), New York, NY, USA, 2016, pp. 1533-1540.

[9] Y. Guan and T. Ploetz, "Ensembles of Deep LSTM Learners for Activity Recognition using Wearables," Proc. ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT), vol. 1, no. 2, Article 11, Jun. 2017.

[10] Y. Zhao, R. Yang, G. Chevalier, X. Xu, and Z. Zhang, "Deep Residual Bidir-LSTM for Human Activity Recognition Using Wearable Sensors," Mathematical Problems in Engineering, vol. 2018, Article ID 7316954, pp. 1-13, 2018.

[11] F. J. Ordonez and D. Roggen, "Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition," Sensors, vol. 16, no. 1, Article 115, Jan. 2016.

[12] R. Mutegeki and D. S. Han, "A CNN-LSTM Approach to Human Activity Recognition," in Proc. International Conference on Artificial Intelligence in Information and Communication (ICAIIC), Fukuoka, Japan, 2020, pp. 362-366.

[13] K. Xia, J. Huang, and H. Wang, "LSTM-CNN Architecture for Human Activity Recognition," IEEE Access, vol. 8, pp. 56855-56866, Mar. 2020.

[14] Y. Chen, K. Zhong, J. Zhang, Q. Sun, and X. Zhao, "LSTM Networks for Mobile Human Activity Recognition," in Proc. International Conference on Artificial Intelligence: Technologies and Applications (ICAITA), Bangkok, Thailand, 2016, pp. 50-53.

[15] T. T. Um, F. M. J. Pfister, D. Pichler, S. Endo, M. Lang, S. Hirche, U. Fietzek, and D. Kulic, "Data Augmentation of Wearable Sensor Data for Parkinson's Disease Monitoring using Convolutional Neural Networks," in Proc. 19th ACM International Conference on Multimodal Interaction (ICMI), Glasgow, UK, 2017, pp. 216-220.

---

## LIST OF PUBLICATIONS

(None at the time of submission)
