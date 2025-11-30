import numpy as np
import librosa
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io
import os

# 1. Initialize API
app = FastAPI()

# 2. Allow CORS (So your website can talk to this API from a different domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Load Model (Global variable)
MODEL_PATH = "model/atm_crnn_enhanced.h5"
model = None

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# 4. Constants
SAMPLE_RATE = 22050
DURATION = 4
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
EXPECTED_FRAMES = int((SAMPLE_RATE * DURATION) / HOP_LENGTH) + 1

# 5. Helper Function
def process_audio(audio_bytes):
    # Load audio from bytes
    signal, sr = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, duration=DURATION)
    
    # Pad/Truncate
    target_len = SAMPLE_RATE * DURATION
    if len(signal) < target_len:
        signal = np.pad(signal, (0, target_len - len(signal)), mode='constant')
    else:
        signal = signal[:target_len]

    # MFCC
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    
    # Pad Frames
    if mfcc.shape[1] < EXPECTED_FRAMES:
        mfcc = np.pad(mfcc, ((0, 0), (0, EXPECTED_FRAMES - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :EXPECTED_FRAMES]
        
    return mfcc

# 6. The API Endpoint
@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read file
        contents = await file.read()
        
        # Process
        features = process_audio(contents)
        
        # Predict
        X_input = features[np.newaxis, ..., np.newaxis]
        prediction_prob = float(model.predict(X_input, verbose=0)[0][0])
        
        # Logic
        result_type = "ANOMALY" if prediction_prob > 0.5 else "NORMAL"
        confidence = prediction_prob if result_type == "ANOMALY" else (1 - prediction_prob)
        
        return {
            "status": "success",
            "prediction": result_type,
            "confidence": round(confidence * 100, 2),
            "raw_score": prediction_prob
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

# To run locally: uvicorn main:app --reload