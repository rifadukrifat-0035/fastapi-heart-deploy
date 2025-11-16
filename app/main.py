import joblib
import pandas as pd
from fastapi import FastAPI
from .schemas import HeartDiseaseInput, HeartDiseaseOutput

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="A simple API to predict heart disease using a trained model.",
    version="0.1.0"
)

# Load the trained model
try:
    model = joblib.load("model/heart_model.joblib")
except FileNotFoundError:
    print("Error: Model file 'model/heart_model.joblib' not found.")
    print("Please run 'train_model.py' first to create the model file.")
    model = None # Set model to None to avoid further errors if file not found
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


# Define the features used for prediction
# This list MUST match the features used during training
FEATURE_LIST = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

@app.get("/health", summary="Check if API is running")
def get_health():
    """
    Health check endpoint to confirm the API is up and running.
    """
    return {"status": "OK"}

@app.get("/info", summary="Get model information")
def get_info():
    """
    Returns information about the model and features.
    """
    if model:
        return {
            "model_type": type(model).__name__,
            "features": FEATURE_LIST
        }
    return {"error": "Model not loaded"}

@app.post("/predict", summary="Predict heart disease", response_model=HeartDiseaseOutput)
def post_predict(data: HeartDiseaseInput):
    """
    Predicts the presence of heart disease based on input features.

    - **Input:** A JSON object with 13 required features.
    - **Output:** A JSON object with `heart_disease: true` (presence) or `false` (absence).
    """
    if not model:
        return {"error": "Model is not loaded. Cannot predict."}

    # Convert Pydantic input to a DataFrame (as expected by the model)
    # The model expects a 2D array or DataFrame
    input_data = pd.DataFrame([data.dict()])
    
    # Ensure columns are in the correct order
    input_data = input_data[FEATURE_LIST]

    # Make prediction
    prediction = model.predict(input_data) # Returns an array (e.g., [1])

    # Convert prediction (0 or 1) to boolean
    is_heart_disease = bool(prediction[0] == 1)

    return HeartDiseaseOutput(heart_disease=is_heart_disease)