EEG Seizure Detection System (MLOps Pipeline)

This project is an end-to-end deep learning system for automated seizure detection from EEG signals, built using the Temple University Hospital (TUH) EEG Corpus. It covers the full machine learning lifecycle, from raw biomedical data processing to cloud deployment and backend integration.

📌 Overview

The system classifies EEG segments into seizure / non-seizure using a deep learning model trained on clinically annotated EEG recordings. It is designed as a production-oriented pipeline integrating ML, backend services, database storage, and cloud deployment.

📌 Pipeline Workflow
1. Data Acquisition (TUH EEG Dataset)
Raw EDF EEG recordings from TUH corpus
Clinically annotated seizure events
Multi-patient, real-world EEG variability

2. Data Preprocessing
EEG signal filtering and noise removal
Segmentation into fixed time windows
Normalization and channel selection
Label alignment (seizure vs non-seizure)
Class balancing for training stability

3. Deep Learning Model
Supervised learning on EEG time-window segments
Binary classification (seizure / non-seizure)
Outputs prediction + confidence score
Evaluated using accuracy, precision, recall

4. Backend (Flask API)
Serves trained ML model via REST API
Handles inference requests in real time
Returns prediction results and confidence
Prepares data for storage and UI display

6. Database (PostgreSQL)
Stores prediction results per patient
Fields: patient name, number, result, confidence, timestamp
Enables tracking and historical analysis

8. Cloud Deployment (AWS EC2)
Flask backend deployed on EC2 instance
Secure server configuration and persistent runtime
Remote accessibility for inference service

10. MLOps Extensions (Partial / Future Work)
AWS SNS for alert notifications (seizure detection alerts)
Grafana for monitoring and visualization
Logging pipeline for model performance tracking

📌 Tech Stack
ML / Data: Python, NumPy, Pandas, SciPy, Scikit-learn
Deep Learning: TensorFlow / PyTorch
Dataset: TUH EEG Seizure Corpus
Backend: Flask (REST API)
Database: PostgreSQL
Cloud: AWS EC2
Monitoring (planned): Grafana, AWS SNS

Raw EEG Data → Preprocessing → Deep Learning Model → Flask API → PostgreSQL → AWS EC2 Deployment → (Optional: SNS Alerts, Grafana Monitoring)

📌 Objective

To build a scalable, cloud-deployable seizure detection system that bridges biomedical signal processing and real-world MLOps deployment, demonstrating an end-to-end machine learning engineering workflow.
