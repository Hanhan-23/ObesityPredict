import pickle
import pandas as pd
import numpy as np
from django.shortcuts import render
from sklearn.preprocessing import LabelEncoder

with open('ObesityPredictApp/model_files/log_reg_model.pkl', 'rb') as f:
    log_model = pickle.load(f)

with open('ObesityPredictApp/model_files/rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

le = LabelEncoder()
le.classes_ = np.array([
    'Insufficient_Weight',
    'Normal_Weight',
    'Overweight_Level_I',
    'Overweight_Level_II',
    'Obesity_Type_I',
    'Obesity_Type_II',
    'Obesity_Type_III'
])

def index(request):
    if request.method == 'POST':
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
            'CH2O': float(request.POST.get('CH2O')),
            'SCC': request.POST.get('SCC'),
            'FAF': float(request.POST.get('FAF')),
            'TUE': float(request.POST.get('TUE')),
            'CALC': request.POST.get('CALC'),
            'MTRANS': request.POST.get('MTRANS'),
        }

        df = pd.DataFrame([data])
        df_encoded = pd.get_dummies(df, drop_first=True)

        expected_cols = log_model.feature_names_in_
        for col in expected_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[expected_cols]

        pred_log = log_model.predict(df_encoded)[0]
        pred_rf = rf_model.predict(df_encoded)[0]

        proba_log = np.max(log_model.predict_proba(df_encoded))
        proba_rf = np.max(rf_model.predict_proba(df_encoded))

        result_log = le.inverse_transform([pred_log])[0]
        result_rf = le.inverse_transform([pred_rf])[0]

        return render(request, 'index/index.html', {
            'log_result': result_log,
            'rf_result': result_rf,
            'log_proba': round(proba_log * 100, 2),
            'rf_proba': round(proba_rf * 100, 2),
            'input_data': data
        })

    return render(request, 'index/index.html')
