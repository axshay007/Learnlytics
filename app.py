"""
Student Performance Prediction API
This Flask application serves predictions for student performance based on input features.
"""

import os
import sys
import logging
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load model and preprocessor
try:
    model_path = os.path.join('artifacts', 'model.pkl')
    preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        logger.error("Model or preprocessor files not found!")
        sys.exit(1)
        
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    logger.info("Model and preprocessor loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or preprocessor: {str(e)}")
    sys.exit(1)

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests.
    Expected input format:
    {
        "gender": str,
        "race_ethnicity": str,
        "parental_level_of_education": str,
        "lunch": str,
        "test_preparation_course": str,
        "reading_score": float,
        "writing_score": float
    }
    """
    try:
        # Get data from request
        data = request.get_json()
        
        # Validate input data
        required_fields = [
            'gender', 'race_ethnicity', 'parental_level_of_education',
            'lunch', 'test_preparation_course', 'reading_score', 'writing_score'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Create CustomData object
        custom_data = CustomData(
            gender=data['gender'],
            race_ethnicity=data['race_ethnicity'],
            parental_level_of_education=data['parental_level_of_education'],
            lunch=data['lunch'],
            test_preparation_course=data['test_preparation_course'],
            reading_score=float(data['reading_score']),
            writing_score=float(data['writing_score'])
        )
        
        # Get prediction
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(custom_data)
        
        return jsonify({
            'prediction': float(prediction[0]),
            'status': 'success'
        })
        
    except ValueError as ve:
        logger.error(f"Value error in prediction: {str(ve)}")
        return jsonify({
            'error': 'Invalid input values',
            'details': str(ve)
        }), 400
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint for Elastic Beanstalk."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None
    })

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False  # Set to False for production
    )        


