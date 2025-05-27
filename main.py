from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io
import logging
import os
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== Load Models ==========
try:
    yield_model = joblib.load("model.pkl")
    crop_model = joblib.load("random_forest_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    model = tf.keras.models.load_model("leaf_disease_model.keras")
    logger.info("Models loaded successfully.")
except Exception as e:
    logger.error(f"Model loading error: {e}")
    raise RuntimeError("Model loading failed. Ensure all model files are present.")

# ========== Constants ==========
crop_categories = [ "Crop_Arecanut", "Crop_Arhar/Tur", "Crop_Bajra", "Crop_Banana", "Crop_Barley", "Crop_Black pepper",
    "Crop_Cardamom", "Crop_Cashewnut", "Crop_Castor seed", "Crop_Coconut", "Crop_Coriander", "Crop_Cotton(lint)",
    "Crop_Cowpea(Lobia)", "Crop_Dry chillies", "Crop_Garlic", "Crop_Ginger", "Crop_Gram", "Crop_Groundnut",
    "Crop_Guar seed", "Crop_Horse-gram", "Crop_Jowar", "Crop_Jute", "Crop_Khesari", "Crop_Linseed", "Crop_Maize",
    "Crop_Masoor", "Crop_Mesta", "Crop_Moong(Green Gram)", "Crop_Moth", "Crop_Niger seed", "Crop_Oilseeds total",
    "Crop_Onion", "Crop_Other  Rabi pulses", "Crop_Other Cereals", "Crop_Other Kharif pulses",
    "Crop_Other Summer Pulses", "Crop_Peas & beans (Pulses)", "Crop_Potato", "Crop_Ragi", "Crop_Rapeseed &Mustard",
    "Crop_Rice", "Crop_Safflower", "Crop_Sannhamp", "Crop_Sesamum", "Crop_Small millets", "Crop_Soyabean",
    "Crop_Sugarcane", "Crop_Sunflower", "Crop_Sweet potato", "Crop_Tapioca", "Crop_Tobacco", "Crop_Turmeric",
    "Crop_Urad", "Crop_Wheat", "Crop_other oilseeds"
]
season_categories = ["Season_Autumn", "Season_Kharif", "Season_Rabi", "Season_Summer", "Season_Whole Year", "Season_Winter"]
state_categories = [ "State_Andhra Pradesh", "State_Arunachal Pradesh", "State_Assam", "State_Bihar", "State_Chhattisgarh",
    "State_Delhi", "State_Goa", "State_Gujarat", "State_Haryana", "State_Himachal Pradesh", "State_Jammu and Kashmir",
    "State_Jharkhand", "State_Karnataka", "State_Kerala", "State_Madhya Pradesh", "State_Maharashtra", "State_Manipur",
    "State_Meghalaya", "State_Mizoram", "State_Nagaland", "State_Odisha", "State_Puducherry", "State_Punjab",
    "State_Sikkim", "State_Tamil Nadu", "State_Telangana", "State_Tripura", "State_Uttar Pradesh", "State_Uttarakhand",
    "State_West Bengal"
]
CLASS_NAMES = [ 'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
    'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy', 'Wheat_Brown rust',
    'Wheat_Healthy', 'Wheat_Loose Smut', 'Wheat_Septoria', 'Wheat_Yellow rust'
]

# ========== Helper Functions ==========
def encode_input(crop: str, season: str, state: str):
    try:
        crop_encoded = [int(f"Crop_{crop}" == c) for c in crop_categories]
        season_encoded = [int(f"Season_{season}" == s) for s in season_categories]
        state_encoded = [int(f"State_{state}" == st) for st in state_categories]
        return crop_encoded + season_encoded + state_encoded
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid crop, season, or state input")

def preprocess_image(img: Image.Image):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# ========== Pydantic Models ==========
class YieldPredictionRequest(BaseModel):
    crop: str
    season: str
    state: str
    rainfall: float
    area: float
    production: float
    fertilizer: float
    pesticide: float

class CropRecommendationRequest(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# ========== Routes ==========

@app.get("/")
def root():
    return {"message": "Welcome to AgriPredictAI — your smart agriculture assistant."}

@app.post("/predict")
def predict_yield(data: YieldPredictionRequest):
    categorical_features = encode_input(data.crop, data.season, data.state)
    numerical_features = [data.area, data.production, data.rainfall, data.fertilizer, data.pesticide]
    input_vector = np.array([numerical_features + categorical_features])

    if input_vector.shape[1] != 96:
        raise HTTPException(status_code=400, detail=f"Expected 96 features, got {input_vector.shape[1]}")

    prediction = yield_model.predict(input_vector)
    return {"predicted_yield": round(float(prediction[0]), 2)}

@app.post("/crop-prediction")
def recommend_crop(data: CropRecommendationRequest):
    try:
        input_array = np.array([[data.N, data.P, data.K, data.temperature, data.humidity, data.ph, data.rainfall]])
        predicted_index = crop_model.predict(input_array)[0]
        predicted_crop = label_encoder.inverse_transform([predicted_index])[0]
        return {"predicted_crop": predicted_crop}
    except Exception as e:
        logger.error(f"Crop prediction error: {e}")
        raise HTTPException(status_code=500, detail="Crop prediction failed.")

@app.post("/leaf-disease")
async def predict_leaf_disease(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
        
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image_array = preprocess_image(image)

        predictions = model.predict(image_array)
        predicted_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(np.max(predictions[0]))

        return JSONResponse({
            "predicted_class": predicted_class,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        logger.error(f"Image prediction error: {e}")
        raise HTTPException(status_code=500, detail="Image processing failed.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # fallback for local
    uvicorn.run(app, host="0.0.0.0", port=port)