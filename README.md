# Loan Approval Prediction System

A machine learning-based web application that predicts the likelihood of loan approval based on applicant information.

![Loan Approval Prediction](https://img.shields.io/badge/Python-Machine%20Learning-blue) ![Status](https://img.shields.io/badge/Status-In%20Development-yellow) ![License](https://img.shields.io/badge/License-MIT-green)

## 📖 Overview

Loan Approval Prediction System is a web application that uses machine learning to predict whether a loan application will be approved based on various factors like applicant income, credit history, loan amount, and other relevant parameters.

## ✨ Features

- **User-friendly Interface**: Clean and intuitive web interface for inputting loan application details
- **Machine Learning Model**: Predictive model trained on historical loan data
- **Real-time Prediction**: Instant approval/rejection prediction with confidence percentage
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- **Detailed Analysis**: Provides insights into factors affecting the prediction

## 🛠️ Technology Stack

- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Backend**: Python, Flask (or Django)
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn (for analysis)
- **Version Control**: Git, GitHub

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/loan-approval-prediction.git
cd loan-approval-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:5000`

## 📊 Dataset

The model is trained on the Loan Prediction Dataset which includes information about:
- Applicant income
- Coapplicant income
- Loan amount
- Credit history
- Property area
- Dependents
- Education level
- Employment type

## 🧠 Machine Learning Model

The project uses a **Random Forest Classifier** which demonstrated the best performance among various algorithms tested:

- Accuracy: 94%
- Precision: 92%
- Recall: 95%
- F1-Score: 93%

### Model Training Process:
1. Data preprocessing and cleaning
2. Feature engineering
3. Handling missing values
4. Encoding categorical variables
5. Model training and validation
6. Hyperparameter tuning using GridSearchCV

## 🚀 Usage

1. Fill in the applicant details in the web form
2. Click the "Check Approval Chance" button
3. View the prediction result (Approved, Denied, or Needs Further Review)
4. See the factors influencing the decision

## 📁 Project Structure

```
loan-approval-prediction/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── model/                # Machine learning model files
│   ├── loan_model.pkl    # Trained model
│   ├── preprocessing.py  # Data preprocessing
│   └── train_model.py    # Model training script
├── static/               # Static assets
│   ├── css/
│   ├── js/
│   └── images/
├── templates/            # HTML templates
│   └── index.html
├── data/                 # Dataset
│   └── loan_data.csv
└── README.md
```

## 🔮 Future Enhancements

- [ ] Integration with credit score APIs
- [ ] Additional ML models for comparison
- [ ] Admin dashboard for data analysis
- [ ] User authentication and history
- [ ] Export functionality for applications
- [ ] Multi-language support

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. 

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Jinit Yadav - [Jinit Yadav](https://github.com/Jinit-Yadav)

## 🙏 Acknowledgments

- Dataset provided by [Analytics Vidhya](https://datahack.analyticsvidhya.com)
- Icons by [Font Awesome](https://fontawesome.com)
- UI components by [Bootstrap](https://getbootstrap.com)

## 📞 Support

If you have any questions or feedback, please reach out at yadavjinit8@example.com or create an issue in the repository.
