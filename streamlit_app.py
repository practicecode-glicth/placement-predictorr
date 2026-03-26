import streamlit as st
import requests

st.title("🎓 Placement Predictor")

cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)
aptitude = st.slider("Aptitude Score", 0, 100, 70)
communication = st.slider("Communication", 1, 10, 5)
projects = st.slider("Projects", 0, 5, 2)

if st.button("Predict"):
    url = "https://placement-predictorr.onrender.com/predict"
    
    data = {
        "cgpa": cgpa,
        "aptitude": aptitude,
        "communication": communication,
        "projects": projects
    }
    
    response = requests.post(url, json=data)
    result = response.json()
    
    if result['prediction'] == 1:
        st.success("🎉 You will get placed!")
    else:
        st.error("❌ Not likely to be placed")
