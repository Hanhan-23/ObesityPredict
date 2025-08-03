# Obesity Levels Predict

A machine learning project for multiclass classification to predict obesity levels based on physical condition and eating habits.  
Developed by **Farhan Ramadhan**.

---

## Features
- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Data Visualization
- Model Building: Logistic Regression & Random Forest
- Model Evaluation & Comparison
- Model Serialization (Pickle)
- Simple Web Interface to Input Features and Predict Obesity Level

---

## Tech Stack
- Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- Django (for Web App)
- Google Colab (Development Environment)
- Pickle (Model Saving)
- HTML/CSS (Simple Interface)

---

## Demo

▶️ **Installation & App Demo Video:**  
[https://youtu.be/ZRAhlPZw_RA?si=tYuC1iRsO3VaNcPU](https://youtu.be/ZRAhlPZw_RA?si=tYuC1iRsO3VaNcPU)

---


## How to Run Locally

```bash
# Clone the repository
git clone https://github.com/Hanhan-23/ObesityPredict.git

# Download Model files and .env
https://drive.google.com/drive/folders/1oSyiBnwnWKwoCsMPNy5kzh5mdlxwyYxR?usp=sharing

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python manage.py runserver
