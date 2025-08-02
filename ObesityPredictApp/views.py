import pickle
import joblib
import pandas as pd
import numpy as np
from django.shortcuts import render
from sklearn.preprocessing import LabelEncoder

try:
    log_model_package = joblib.load('ObesityPredictApp/model_files/logistic_regression_model.pkl')
    rf_model_package = joblib.load('ObesityPredictApp/model_files/random_forest_model.pkl')
    
    log_model = log_model_package['model']
    rf_model = rf_model_package['model']
    scaler = log_model_package['scaler']  
    encoders = log_model_package['encoders']
    feature_names = log_model_package['feature_names']
    
    print("Models loaded from Colab notebook format")
    
except FileNotFoundError:
    try:
        with open('ObesityPredictApp/model_files/log_reg_model.pkl', 'rb') as f:
            log_model = pickle.load(f)
        
        with open('ObesityPredictApp/model_files/rf_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        
        scaler = None
        encoders = None
        feature_names = log_model.feature_names_in_
        
        print("Models loaded from old pickle format")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        log_model = None
        rf_model = None

target_le = LabelEncoder()
target_le.classes_ = np.array([
    'Insufficient_Weight',
    'Normal_Weight', 
    'Overweight_Level_I',
    'Overweight_Level_II',
    'Obesity_Type_I',
    'Obesity_Type_II',
    'Obesity_Type_III'
])

def encode_categorical_data(data, encoders=None):
    data_encoded = data.copy()
    
    if encoders:
        categorical_columns = ['Gender', 'family_history_with_overweight', 'FAVC', 
                             'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
        
        for col in categorical_columns:
            if col in data_encoded.columns and col in encoders:
                try:

                    data_encoded[col] = encoders[col].transform(data_encoded[col])
                except ValueError as e:
                    print(f"Warning: Unknown category in {col}: {e}")
                    data_encoded[col] = 0
    else:
        data_encoded['Gender'] = data_encoded['Gender'].map({'Male': 1, 'Female': 0})
        
        binary_cols = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
        for col in binary_cols:
            data_encoded[col] = data_encoded[col].map({'yes': 1, 'no': 0})
        
        caec_map = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
        data_encoded['CAEC'] = data_encoded['CAEC'].map(caec_map)
        
        calc_map = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
        data_encoded['CALC'] = data_encoded['CALC'].map(calc_map)
        
        mtrans_map = {
            'Automobile': 0,
            'Motorbike': 1, 
            'Bike': 2,
            'Public_Transportation': 3,
            'Walking': 4
        }
        data_encoded['MTRANS'] = data_encoded['MTRANS'].map(mtrans_map)
    
    return data_encoded

def prepare_data_for_prediction(data, feature_names, scaler=None):
    df = pd.DataFrame([data])
    
    df_encoded = encode_categorical_data(df, encoders)
    
    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    df_encoded = df_encoded[feature_names]
    
    if scaler is not None:
        df_scaled = scaler.transform(df_encoded)
        return df_encoded, df_scaled
    
    return df_encoded, df_encoded

def index(request):
    if request.method == 'POST':
        try:
            data = {
                'Gender': request.POST.get('Gender'),
                'Age': float(request.POST.get('Age')),
                'Height': float(request.POST.get('Height')),
                'Weight': float(request.POST.get('Weight')),
                'family_history_with_overweight': request.POST.get('family_history_with_overweight'),
                'FAVC': request.POST.get('FAVC'),
                'FCVC': float(request.POST.get('FCVC')),
                'NCP': float(request.POST.get('NCP')),
                'CAEC': request.POST.get('CAEC'),
                'SMOKE': request.POST.get('SMOKE'),
                'CH2O': request.POST.get('CH2O'),
                'SCC': request.POST.get('SCC'),
                'FAF': float(request.POST.get('FAF')),
                'TUE': float(request.POST.get('TUE')),
                'CALC': request.POST.get('CALC'),
                'MTRANS': request.POST.get('MTRANS'),
            }

            print("Input data:", data)

            df_encoded, df_for_lr = prepare_data_for_prediction(data, feature_names, scaler)

            pred_log = log_model.predict(df_for_lr)[0]
            proba_log_all = log_model.predict_proba(df_for_lr)[0]
            proba_log = np.max(proba_log_all)

            pred_rf = rf_model.predict(df_encoded)[0]
            proba_rf_all = rf_model.predict_proba(df_encoded)[0]
            proba_rf = np.max(proba_rf_all)

            print(f"Predictions - LR: {pred_log}, RF: {pred_rf}")

            result_log = target_le.inverse_transform([pred_log])[0]
            result_rf = target_le.inverse_transform([pred_rf])[0]

            proba_table = list(zip(
                target_le.classes_, 
                [round(p*100, 2) for p in proba_log_all], 
                [round(p*100, 2) for p in proba_rf_all]
            ))

            return render(request, 'index/index.html', {
                'log_result': result_log,
                'rf_result': result_rf,
                'log_proba': round(proba_log * 100, 2),
                'rf_proba': round(proba_rf * 100, 2),
                'proba_table': proba_table,
                'labels': target_le.classes_,
                'input_data': data
            })

        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            
            return render(request, 'index/index.html', {
                'error': f"Prediction error: {str(e)}"
            })

    return render(request, 'index/index.html')