# 🤖 AI Portfolio API

A comprehensive collection of machine learning models served as RESTful APIs using FastAPI. This project showcases various ML techniques including supervised learning, unsupervised learning, NLP, computer vision, and reinforcement learning.

## 🌟 Features

### Supervised Learning
- 🏠 **House Price Prediction**: Predict house prices using a Random Forest Regressor
- 😊 **Sentiment Analysis**: Classify text sentiment (positive/negative/neutral)

### Unsupervised Learning
- 🎯 **Customer Segmentation**: K-means clustering for customer data

### Natural Language Processing
- 📝 **Text Classification**: Categorize text into different classes
- 🔍 **Named Entity Recognition**: Extract entities like persons, organizations, locations

### Computer Vision
- 🖼️ **Image Classification**: Classify images using ResNet-18
- 📦 **Object Detection**: Detect and localize objects in images

### Time Series Forecasting
- 📈 **LSTM Forecasting**: Long Short-Term Memory networks for time series prediction
- 🔮 **ARIMA Modeling**: AutoRegressive Integrated Moving Average for statistical time series forecasting

### Reinforcement Learning
- 🎮 **CartPole-v1**: Deep Q-Network (DQN) implementation for the classic control problem

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/FlyingSkates/ai-portfolio-api.git
   cd ai-portfolio-api
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download additional models:
   ```bash
   python -m spacy download en_core_web_sm
   ```

### Running the API

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`

## 📚 API Documentation

Once the server is running, you can access:

- **Interactive API Docs**: `http://127.0.0.1:8000/docs`
- **Alternative API Docs**: `http://127.0.0.1:8000/redoc`

## 🧩 Example API Endpoints

### House Price Prediction
```bash
POST /supervised/house-price/predict
{
  "MedInc": 8.3,
  "HouseAge": 21.0,
  "AveRooms": 6.2,
  "AveBedrms": 1.1,
  "Population": 240.0,
  "AveOccup": 3.0,
  "Latitude": 37.8,
  "Longitude": -122.2
}
```

### Sentiment Analysis
```bash
POST /supervised/sentiment/predict
{
  "text": "I really enjoyed this product!"
}
```

### Time Series Forecasting

#### Train a New Model
```bash
POST /time-series/forecast/train
{
  "model_type": "lstm",
  "model_id": "sales_forecast",
  "lstm_units": 32,
  "epochs": 50,
  "batch_size": 32,
  "look_back": 7,
  "data": {
    "timestamps": ["2023-01-01", "2023-01-02", ...],
    "values": [100, 105, 98, ...],
    "freq": "D"
  }
}
```

#### Generate Forecasts
```bash
POST /time-series/forecast/forecast
{
  "model_id": "sales_forecast",
  "steps": 14,
  "confidence_level": 0.95
}
```

#### List Available Models
```bash
GET /time-series/forecast/models
```

### Image Classification
```bash
POST /cv/classify/predict
Content-Type: multipart/form-data
file: [upload an image file]
```

## 🛠️ Project Structure

```
app/
├── __init__.py
├── main.py                 # Main FastAPI application
└── api/
    ├── __init__.py
    ├── supervised/         # Supervised learning models
    │   ├── __init__.py
    │   ├── house_price.py
    │   └── sentiment_analysis.py
    ├── unsupervised/       # Unsupervised learning models
    │   ├── __init__.py
    │   └── clustering.py
    ├── nlp/                # NLP models
    │   ├── __init__.py
    │   ├── text_classification.py
    │   └── ner.py
    ├── cv/                 # Computer Vision models
    │   ├── __init__.py
    │   ├── image_classification.py
    │   └── object_detection.py
    └── rl/                 # Reinforcement Learning
        ├── __init__.py
        └── cartpole.py
```

## 📦 Dependencies

- FastAPI - Modern, fast web framework
- PyTorch - Deep learning framework
- scikit-learn - Machine learning library
- spaCy - NLP library
- OpenCV - Computer vision library
- Gym - Reinforcement learning environments

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
  - HouseAge: Median house age in block group (years)
  - AveRooms: Average number of rooms per household
  - AveBedrms: Average number of bedrooms per household
  - Population: Block group population
  - AveOccup: Average number of household members
  - Latitude: Block group latitude
  - Longitude: Block group longitude

## Setup

1. Clone the repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the server:
   ```bash
   uvicorn app.main:app --reload
   ```
5. Access the API documentation at: http://127.0.0.1:8000/docs

## API Endpoints

- `GET /`: Welcome message
- `POST /predict`: Make house price predictions
- `POST /retrain`: Retrain the model with current data

## Free Deployment Options

### 1. Render (Recommended)
1. Create a free account on [Render](https://render.com/)
2. Connect your GitHub repository
3. Select "New" → "Web Service"
4. Choose your repository and configure:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Deploy!

### 2. Railway.app
1. Sign up at [Railway](https://railway.app/)
2. Click "New Project" → "Deploy from GitHub"
3. Select your repository
4. Add environment variable: `PORT=8000`
5. Deploy!

### 3. PythonAnywhere
1. Create a free account on [PythonAnywhere](https://www.pythonanywhere.com/)
2. Upload your code
3. Configure a web app with Flask
4. Update the WSGI file to point to your FastAPI app
5. Reload the web app

## Example Request

```bash
curl -X 'POST' \
  'http://your-app-url/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "MedInc": 8.3,
  "HouseAge": 21.0,
  "AveRooms": 6.2,
  "AveBedrms": 1.1,
  "Population": 240.0,
  "AveOccup": 3.0,
  "Latitude": 37.8,
  "Longitude": -122.2
}'
```

## License

MIT

make this Model and serve using fastAPI and tell me how to deploy it for free and For Long time?