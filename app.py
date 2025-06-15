import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# --- Sample Data ---
data = {
    'marks': [90, 45, 33, 77, 25, 60, 85, 40, 70, 20],
    'appear': ['Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No']
}
df = pd.DataFrame(data)

# --- Data Preprocessing ---
df['appear'] = df['appear'].map({'Yes': 1, 'No': 0})

X = df[['marks']]
y = df['appear']

# --- Model Training ---
model = LogisticRegression()
model.fit(X, y)

# --- Streamlit UI ---
st.set_page_config(page_title="Student Appearance Predictor", layout="centered")

st.title("🎓 Student Exam Appearance Predictor")
st.markdown("Enter student's marks to predict whether they will **Appear** or **Not Appear** in the exam.")

# --- Input ---
marks_input = st.number_input("📌 Enter Marks (0 - 100):", min_value=0, max_value=100, step=1)

if st.button("🔍 Predict"):
    prediction = model.predict(np.array([[marks_input]]))[0]
    result = "✅ Will Appear in Exam" if prediction == 1 else "❌ Will NOT Appear in Exam"
    st.success(result)
