"""Streamlit client for the existing Practical Task 6 FastAPI backend."""
import os
import requests
import streamlit as st

API_URL = os.getenv('API_URL', 'http://127.0.0.1:8000').rstrip('/')
st.set_page_config(page_title='Iris Classifier | SIS-3', page_icon='🌱', layout='centered')
st.title('Iris Flower Classifier')
st.caption('SIS-3 · Random Forest · FastAPI + Streamlit + MLflow')
st.write('Enter four flower measurements in centimeters to predict the Iris species.')
with st.form('iris_measurements'):
    left, right = st.columns(2)
    with left:
        sepal_length = st.number_input('Sepal length (cm)', min_value=0.1, value=5.1, step=0.1)
        sepal_width = st.number_input('Sepal width (cm)', min_value=0.1, value=3.5, step=0.1)
    with right:
        petal_length = st.number_input('Petal length (cm)', min_value=0.1, value=1.4, step=0.1)
        petal_width = st.number_input('Petal width (cm)', min_value=0.1, value=0.2, step=0.1)
    submitted = st.form_submit_button('Predict species', type='primary')
if submitted:
    payload = dict(sepal_length=sepal_length, sepal_width=sepal_width,
                   petal_length=petal_length, petal_width=petal_width)
    try:
        response = requests.post(f'{API_URL}/predict', json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        st.success(f'Predicted species: {result["prediction"]}')
        st.caption(f'Class ID: {result["class_id"]}')
        with st.expander('API response'):
            st.json(result)
    except (requests.RequestException, ValueError, KeyError) as exc:
        st.error(f'Prediction failed. Check that FastAPI is running at {API_URL}.')
        st.caption(str(exc))
st.divider()
st.caption('Model training and version history are available in the local MLflow dashboard.')
st.markdown('[FastAPI docs](http://127.0.0.1:8000/docs) · [MLflow dashboard](http://127.0.0.1:5000)')
