from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            no_of_dependents=int(request.form.get('no_of_dependents')),
            education=request.form.get('education'),
            self_employed=request.form.get('self_employed'),
            income_annum=float(request.form.get('income_annum')),
            loan_amount=float(request.form.get('loan_amount')),
            loan_term=int(request.form.get('loan_term')),
            cibil_score=int(request.form.get('cibil_score')),
            residential_assets_value=float(request.form.get('residential_assets_value')),
            commercial_assets_value=float(request.form.get('commercial_assets_value')),
            luxury_assets_value=float(request.form.get('luxury_assets_value')),
            bank_asset_value=float(request.form.get('bank_asset_value'))
        )
        pred_df = data.get_data_as_data_frame()
        print("Input Data:")
        print(pred_df)
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        # Get prediction probability for more detailed results
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = PredictPipeline().load_object(file_path=model_path)
            preprocessor = PredictPipeline().load_object(file_path=preprocessor_path)
            
            data_scaled = preprocessor.transform(pred_df)
            prediction_proba = model.predict_proba(data_scaled)
            
            approval_probability = prediction_proba[0][1] * 100  # Probability of approval
            rejection_probability = prediction_proba[0][0] * 100  # Probability of rejection
            
            return render_template('home.html', 
                                 results=results[0],
                                 approval_probability=f"{approval_probability:.2f}%",
                                 rejection_probability=f"{rejection_probability:.2f}%")
            
        except Exception as e:
            # Fallback if probability prediction fails
            return render_template('home.html', results=results[0])

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)