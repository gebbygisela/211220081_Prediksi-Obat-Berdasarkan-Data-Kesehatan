import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestClassifier  
import streamlit as st  
  
# Load dataset  
data = pd.read_csv('classification.csv')  
  
# Preprocessing  
data['Drug'] = data['Drug'].str.upper()  # Normalize drug names  
X = data.drop('Drug', axis=1)  
y = data['Drug']  
  
# Convert categorical variables to numerical  
X = pd.get_dummies(X, drop_first=True)  
  
# Split the dataset  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
  
# Train the model  
model = RandomForestClassifier(random_state=42)  
model.fit(X_train, y_train)  
  
# Calculate accuracy  
accuracy = model.score(X_test, y_test)  
  
# Streamlit app  
st.set_page_config(page_title="Prediksi Obat", layout="wide")  
  
# Custom CSS for background color  
st.markdown("""  
    <style>  
    .stApp {  
        background-color:rgb(224, 177, 224);  /* Change this to your desired color */  
        color: #333;  /* Change text color if needed */  
    }  
    </style>  
""", unsafe_allow_html=True)  
  
st.title("Prediksi Obat Berdasarkan Data Kesehatan")  
st.write("Masukkan data kesehatan Anda:")  
  
# Input fields in a sidebar  
with st.sidebar:  
    st.header("Input Data Kesehatan")  
    age = st.number_input("Usia", min_value=0, max_value=120, value=30)  
    sex = st.selectbox("Jenis Kelamin", options=["L", "P"])  
    bp = st.selectbox("Tekanan Darah", options=["LOW", "NORMAL", "HIGH"])  
    cholesterol = st.selectbox("Kolesterol", options=["NORMAL", "HIGH"])  
    na_to_k = st.number_input("Rasio Natrium terhadap Kalium", value=15.0)  
  
# Prepare input data for prediction  
input_data = pd.DataFrame({  
    'Age': [age],  
    'Sex_M': [1 if sex == 'L' else 0],  # Adjusted to match input options  
    'Sex_F': [1 if sex == 'P' else 0],  
    'BP_LOW': [1 if bp == 'LOW' else 0],  
    'BP_NORMAL': [1 if bp == 'NORMAL' else 0],  
    'BP_HIGH': [1 if bp == 'HIGH' else 0],  
    'Cholesterol_NORMAL': [1 if cholesterol == 'NORMAL' else 0],  
    'Cholesterol_HIGH': [1 if cholesterol == 'HIGH' else 0],  
    'Na_to_K': [na_to_k]  
})  
  
# Reindex input data to match the training data  
input_data = input_data.reindex(columns=X.columns, fill_value=0)  
  
# Prediction  
if st.button("Prediksi Obat"):  
    prediction = model.predict(input_data)  
    st.success(f"Obat yang direkomendasikan: {prediction[0]}")  
    st.write(f"Akurasi model: {accuracy * 100:.2f}%")  
  
# Dashboard Section  
st.header("Visualisasi")  
  
# Visualisasi Distribusi Usia  
st.subheader("Distribusi Usia")  
plt.figure(figsize=(10, 5))  
sns.histplot(data['Age'], bins=10, kde=True, color='skyblue')  
plt.title("Distribusi Usia", fontsize=16)  
plt.xlabel("Usia", fontsize=12)  
plt.ylabel("Frekuensi", fontsize=12)  
st.pyplot(plt)  
  
# Visualisasi Jenis Kelamin  
st.subheader("Distribusi Jenis Kelamin")  
plt.figure(figsize=(10, 5))  
sns.countplot(x='Sex', data=data, palette='pastel')  
plt.title("Distribusi Jenis Kelamin", fontsize=16)  
plt.xlabel("Jenis Kelamin", fontsize=12)  
plt.ylabel("Jumlah", fontsize=12)  
st.pyplot(plt)  
  
# Visualisasi Tekanan Darah  
st.subheader("Distribusi Tekanan Darah")  
plt.figure(figsize=(10, 5))  
sns.countplot(x='BP', data=data, palette='Set2')  
plt.title("Distribusi Tekanan Darah", fontsize=16)  
plt.xlabel("Tekanan Darah", fontsize=12)  
plt.ylabel("Jumlah", fontsize=12)  
st.pyplot(plt)  
  
# Visualisasi Kolesterol  
st.subheader("Distribusi Kolesterol")  
plt.figure(figsize=(10, 5))  
sns.countplot(x='Cholesterol', data=data, palette='coolwarm')  # Ensure 'Cholesterol' is the correct column name  
plt.title("Distribusi Kolesterol", fontsize=16)  
plt.xlabel("Kolesterol", fontsize=12)  
plt.ylabel("Jumlah", fontsize=12)  
st.pyplot(plt)