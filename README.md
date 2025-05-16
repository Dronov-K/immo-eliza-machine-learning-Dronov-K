# 🏠Price Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue) ![License](https://img.shields.io/badge/License-MIT-green)

## 📝 Description
The **Price Prediction** project is designed to help predict property prices across Belgium using machine learning. The project focuses on building and training a machine learning model that can accurately predict property prices based on various features from the dataset.

The project is structured into the following main steps:

1. **Data Cleaning & Preprocessing**:
Raw data is cleaned using the DataCleaner class, which removes duplicates, fills in or drops missing values, and standardizes columns based on predefined parameters. Categorical and numerical features are then processed using a unified ColumnTransformer pipeline that handles imputation, scaling, and encoding.

2. **Model Training & Tuning**:
A machine learning pipeline is built and trained using the cleaned data. Hyperparameter tuning is performed using GridSearchCV to select the best model configuration. Supported models include Random Forest, XGBoost, and others.

3. **Model Evaluation**:
The trained model is evaluated on a test set using metrics such as Mean Absolute Error (MAE) and R² score to assess prediction accuracy.

4. **Web Application (Streamlit)**:
The final trained model is integrated into a Streamlit web application. Users can input property features and instantly receive price predictions via an intuitive user interface.

## 🌳 Project Structure

```
Price Prediction/
│
├── data/
|   ├── Kangaroo.csv
|   └── updated_Kangaroo.csv
├── models
|   └── xgb_model.pkl
├── src/
│   ├── __init__.py
|   ├── app_styles.py
|   ├── cleaner.py
|   ├── cleaning_config.py
|   ├── data_io.py
|   ├── input_form.py
|   ├── location_utils.py
|   └── model.py
├── __init__.py
├── .gitignore
├── app.py
├── main.py
├── README.md  
└── requirements.txt   
```

## 🚀 Installation and Execution

🔧 Local Setup

1. **Clone the repository:**
```bash
git clone https://github.com/becodeorg/immo-eliza-machine-learning-Dronov-K.git
cd immo-eliza-machine-learning-Dronov-K
```

2. Create and activate a virtual environment (optional):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Streamlit app locally:
```bash
streamlit run main.py
```

☁️ Cloud Deployment
The application is also deployed using Streamlit Community Cloud, allowing you to use the app directly in your browser without installing anything locally.

🔗 Live App Link: Coming soon / Insert link here

## ⚖️ License

This project is licensed under the MIT License.