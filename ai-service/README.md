# BioSense AI - AI Service

The BioSense AI Service is the Python-based AI layer of the BioSense AI research platform.

It is built using FastAPI and will eventually provide machine-learning inference for electronic-nose, VOC, and sensor-pattern data.

## Technology Stack

- Python 3.11+
- FastAPI
- Uvicorn
- Pydantic
- NumPy
- pandas
- scikit-learn
- XGBoost
- SHAP
- joblib

## Architecture

```text
React Frontend
       |
       v
Spring Boot Backend
       |
       v
FastAPI AI Service
       |
       v
Machine Learning Model