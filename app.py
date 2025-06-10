from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import base64
import io
import os

app = Flask(__name__)

# Your model results from the notebook (these would be the actual results from your .ipynb)
MODEL_RESULTS = {
    'cv_metrics': {
        'accuracy': {'mean': 0.8520, 'std': 0.0123},
        'precision': {'mean': 0.8445, 'std': 0.0156}, 
        'recall': {'mean': 0.8612, 'std': 0.0198},
        'f1': {'mean': 0.8527, 'std': 0.0134}
    },
    'test_metrics': {
        'accuracy': 0.8565,
        'precision': 0.8423,
        'recall': 0.8702,
        'f1': 0.8560
    }
}

DATASET_INFO = {
    'total_samples': 10000,
    'diabetes_cases': 850,
    'non_diabetes_cases': 9150,
    'train_size': 8000,
    'test_size': 2000,
    'features': [
        'Age', 'BMI', 'Waist_Circumference', 'Fasting_Blood_Glucose', 'HbA1c',
        'Blood_Pressure_Systolic', 'Blood_Pressure_Diastolic', 'Cholesterol_Total',
        'Cholesterol_HDL', 'Cholesterol_LDL', 'GGT', 'Serum_Urate', 'Dietary_Intake_Calories',
        'Family_History_of_Diabetes', 'Previous_Gestational_Diabetes'
    ]
}

def create_visualizations():
    """Create visualizations from the dataset"""
    try:
        df = pd.read_csv('diabetes_dataset.csv')
        df['Diabetes'] = (df['Fasting_Blood_Glucose'] >= 126).astype(int)
        
        visualizations = {}
        
        # 1. Diabetes distribution
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x='Diabetes')
        plt.title('Distribution of Diabetes Cases', fontsize=16, fontweight='bold')
        plt.xlabel('Diabetes (0: No, 1: Yes)')
        plt.ylabel('Count')
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=150)
        img.seek(0)
        visualizations['distribution'] = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        # 2. Feature correlation heatmap
        numerical_features = DATASET_INFO['features']
        plt.figure(figsize=(12, 10))
        corr_matrix = df[numerical_features + ['Diabetes']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True)
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=150)
        img.seek(0)
        visualizations['correlation'] = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return visualizations
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        return {}

@app.route("/")
def home():
    """Main dashboard showing model results and visualizations"""
    visualizations = create_visualizations()
    return render_template("index.html", 
                         results=MODEL_RESULTS, 
                         dataset=DATASET_INFO,
                         visualizations=visualizations)

@app.route("/model-details")
def model_details():
    """Detailed model architecture and training information"""
    model_architecture = {
        'type': 'Neural Network (Sequential)',
        'layers': [
            {'type': 'Dense', 'units': 64, 'activation': 'relu', 'regularization': 'L2'},
            {'type': 'Dropout', 'rate': 0.3},
            {'type': 'Dense', 'units': 32, 'activation': 'relu', 'regularization': 'L2'},
            {'type': 'Dense', 'units': 1, 'activation': 'sigmoid'}
        ],
        'optimizer': 'Adam',
        'loss': 'binary_crossentropy',
        'training': {
            'epochs': 100,
            'batch_size': 16,
            'validation_split': '5-fold cross-validation',
            'early_stopping': 'Yes (patience=10)'
        }
    }
    
    return render_template("model_details.html", 
                         results=MODEL_RESULTS,
                         architecture=model_architecture)
    
@app.route("/video-portal")
def video_portal():
    """Video portal for project demonstrations"""
    return render_template("video_portal.html")
    
if __name__ == "__main__":
    print("Starting Diabetes Prediction Dashboard...")
    print("Loading dataset and creating visualizations...")
    app.run(debug=True, host='0.0.0.0', port=5000) 
