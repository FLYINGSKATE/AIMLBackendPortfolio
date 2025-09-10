import os
import numpy as np
import cv2
import ssl
import urllib.request
import zipfile
import tempfile
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import joblib
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Disable SSL verification for model downloading
ssl._create_default_https_context = ssl._create_unverified_context

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize model and class names
model = None
class_names = []

# Model configuration
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_BATCH_SIZE = 32

def create_model(num_classes: int):
    """Create a new model with transfer learning from MobileNetV2.
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained MobileNetV2 without top layer
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Add custom classification head
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model

def load_model():
    global model, class_names
    model_path = "models/image_classification_model.h5"
    classes_path = "models/class_names.joblib"
    
    if os.path.exists(model_path) and os.path.exists(classes_path):
        try:
            model = tf.keras.models.load_model(model_path)
            class_names = joblib.load(classes_path)
            logger.info(f"Loaded model with {len(class_names)} classes")
            return
        except Exception as e:
            logger.warning(f"Error loading saved model: {e}")
    
    # If loading fails or no saved model exists, use pre-trained MobileNetV2
    logger.info("Loading pre-trained MobileNetV2")
    model = MobileNetV2(weights='imagenet')
    class_names = []  # Empty for pre-trained model
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

class ClassificationResult(BaseModel):
    filename: str
    predictions: List[dict]

class TrainingResponse(BaseModel):
    """Response model for training operations.
    
    Attributes:
        message: Status message about the training operation
        accuracy: Accuracy of the model on the validation data (0.0 to 1.0)
        classes: List of class names
        samples_trained: Number of training samples used
        model_summary: Summary of the model architecture
    """
    message: str = Field(..., example="Model retrained successfully")
    accuracy: Optional[float] = Field(None, example=0.95, ge=0.0, le=1.0)
    classes: Optional[List[str]] = Field(None, example=["cat", "dog"])
    samples_trained: Optional[int] = Field(None, example=100)
    model_summary: Optional[Dict[str, Any]] = None

class TrainingConfig(BaseModel):
    """Configuration for model training.
    
    Attributes:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for the optimizer
        image_size: Target size for input images (width, height)
        validation_split: Fraction of data to use for validation (0.0 to 1.0)
    """
    epochs: int = Field(5, ge=1, le=100)
    batch_size: int = Field(32, ge=1, le=128)
    learning_rate: float = Field(0.0001, ge=1e-6, le=0.1)
    image_size: tuple = (224, 224)
    validation_split: float = Field(0.2, ge=0.1, le=0.5)

def preprocess_image(img):
    # Resize image to 224x224 (required by MobileNetV2)
    img = cv2.resize(img, (224, 224))
    # Convert BGR to RGB (OpenCV loads as BGR by default)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert to array and preprocess for MobileNetV2
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

@router.post(
    "/predict",
    response_model=ClassificationResult,
    status_code=status.HTTP_200_OK,
    summary="Classify an image",
    response_description="Image classification result"
)
async def classify_image(file: UploadFile = File(...)):
    """
    Classify an image using a pre-trained model.
    """
    try:
        if model is None:
            load_model()
        
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not decode image"
            )
        
        # Preprocess the image
        processed_img = preprocess_image(img)
        
        # Make prediction
        predictions = model.predict(processed_img)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        
        # Format predictions
        result = [
            {"label": label, "probability": float(prob)}
            for (_, label, prob) in decoded_predictions
        ]
        
        return {
            "filename": file.filename,
            "predictions": result
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image: {str(e)}"
        )

@router.post(
    "/retrain",
    response_model=TrainingResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrain the image classification model",
    response_description="Training result with accuracy and model info"
)
async def retrain_model(
    file: UploadFile = File(..., description="ZIP file containing training images organized in subdirectories by class"),
    config: TrainingConfig = Body(
        default=TrainingConfig(),
        description="Training configuration"
    )
):
    """
    Retrain the image classification model with custom data.
    
    The ZIP file should contain subdirectories where each subdirectory represents a class
    and contains the training images for that class.
    
    Example directory structure:
    ```
    training_data.zip
    ├── cat/
    │   ├── cat1.jpg
    │   └── cat2.jpg
    └── dog/
        ├── dog1.jpg
        └── dog2.jpg
    ```
    
    **Example cURL**:
    ```bash
    curl -X 'POST' \
      'http://localhost:8001/api/cv/image-classification/retrain' \
      -H 'accept: application/json' \
      -H 'Content-Type: multipart/form-data' \
      -F 'file=@training_data.zip;type=application/zip' \
      -F 'config={"epochs": 5, "batch_size": 32}'
    ```
    """
    global model, class_names
    
    # Create a temporary directory to extract the zip file
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Save the uploaded zip file
            zip_path = os.path.join(temp_dir, "uploaded.zip")
            with open(zip_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Extract the zip file
            extract_dir = os.path.join(temp_dir, "extracted")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Find class directories
            class_dirs = [d for d in os.listdir(extract_dir) 
                         if os.path.isdir(os.path.join(extract_dir, d))]
            
            if not class_dirs:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No class directories found in the zip file"
                )
            
            # Set class names and create data generators
            class_names = sorted(class_dirs)
            num_classes = len(class_names)
            
            # Create data generators with data augmentation
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                validation_split=config.validation_split
            )
            
            # Load training data
            train_generator = train_datagen.flow_from_directory(
                extract_dir,
                target_size=config.image_size,
                batch_size=config.batch_size,
                class_mode='categorical',
                subset='training',
                shuffle=True
            )
            
            # Load validation data
            val_generator = train_datagen.flow_from_directory(
                extract_dir,
                target_size=config.image_size,
                batch_size=config.batch_size,
                class_mode='categorical',
                subset='validation',
                shuffle=False
            )
            
            # Create or load model
            if model is None or not hasattr(model, 'layers'):
                model = create_model(num_classes)
            
            # Compile the model
            model.compile(
                optimizer=Adam(learning_rate=config.learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train the model
            history = model.fit(
                train_generator,
                epochs=config.epochs,
                validation_data=val_generator,
                verbose=1
            )
            
            # Get the final validation accuracy
            val_accuracy = history.history['val_accuracy'][-1]
            
            # Save the model and class names
            os.makedirs("models", exist_ok=True)
            model.save("models/image_classification_model.h5")
            joblib.dump(class_names, "models/class_names.joblib")
            
            # Prepare response
            return {
                "message": f"Model retrained successfully on {train_generator.samples} samples",
                "accuracy": float(val_accuracy),
                "classes": class_names,
                "samples_trained": train_generator.samples,
                "model_summary": {
                    "layers": len(model.layers),
                    "trainable_parameters": int(np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])),
                    "non_trainable_parameters": int(np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]))
                }
            }
            
        except zipfile.BadZipFile:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid zip file"
            )
        except Exception as e:
            logger.error(f"Error during training: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error during training: {str(e)}"
            )

@router.get(
    "/available_models",
    response_model=List[str],
    status_code=status.HTTP_200_OK,
    summary="List available models",
    response_description="List of available image classification models"
)
async def list_models():
    """
    List all available image classification models and their status.
    
    Returns:
        List of available models with their status
    """
    models = ["MobileNetV2 (pre-trained on ImageNet)"]
    
    if os.path.exists("models/image_classification_model.h5"):
        models.append("Custom Model (fine-tuned)")
    
    return models

def create_model(num_classes):
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    
    return model

def load_model():
    global model, class_names
    
    if os.path.exists("models/image_classification_model.h5"):
        model = tf.keras.models.load_model("models/image_classification_model.h5")
        class_names = joblib.load("models/class_names.joblib")
    else:
        model = create_model(1000)  # Default to 1000 classes for ImageNet

def preprocess_image(img):
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    
    return img

def decode_predictions(predictions, top=3):
    results = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=top)
    
    return results
