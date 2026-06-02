import streamlit as st
import pandas as pd
import numpy as np
import sqlalchemy as sa
import joblib
import os

st.set_page_config(page_title="Churn Analytics Dashboard", page_icon="📊", layout="wide")

st.title("Customer Churn Analytics & Prediction Platform 📊")
st.write("Ecosistema portátil de Data Science impulsado por Docker, PostgreSQL y XGBoost.")

# --- CADENA DE CONEXIÓN Y MODELO ---
DATABASE_URL = "postgresql://ds_user:chiara01!@db:5432/customer_churn"
MODEL_PATH = "01.models/pipeline_xgboost_telco.pkl"

@st.cache_resource
def get_db_engine():
    return sa.create_engine(DATABASE_URL)

@st.cache_resource
def load_ml_pipeline():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

engine = get_db_engine()
pipeline = load_ml_pipeline()

st.header("1. Monitoreo de Clientes en Riesgo Crítico (PostgreSQL)")

try:
    query = "SELECT * FROM view_high_risk_revenue_customers LIMIT 10;"
    df_high_risk = pd.read_sql(query, con=engine)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Clientes Críticos Listados", value=len(df_high_risk))
    with col2:
        st.metric(label="Cargo Mensual Máximo Detectado", value=f"${df_high_risk['monthly_charges'].max():.2f}")
    with col3:
        st.metric(label="Permanencia Promedio (Meses)", value=f"{df_high_risk['tenure'].mean():.1f} meses")
        
    st.subheader("Top 10 Clientes con Mayor Impacto de Fuga")
    st.dataframe(df_high_risk, use_container_width=True)
except Exception as e:
    st.error(f"No se pudo conectar a la base de datos PostgreSQL: {e}")

st.markdown("---")
st.header("2. Simulador de Riesgo de Abandono (Inferencia en Vivo)")

if pipeline is None:
    st.warning("⚠️ El pipeline predictivo 'pipeline_xgboost_telco.pkl' aún no está exportado en la carpeta 01.models. Por favor, ejecuta tu Notebook 4 dentro del contenedor para generarlo.")
else:
    st.write("Modifica las variables operativas del cliente para calcular instantáneamente su probabilidad de Churn:")
    
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    
    with col_f1:
        gender = st.selectbox("Género", ["Male", "Female"])
        senior_citizen = st.selectbox("Adulto Mayor", [0, 1])
        partner = st.selectbox("Tiene Pareja", ["Yes", "No"])
    with col_f2:
        dependents = st.selectbox("Tiene Dependientes", ["Yes", "No"])
        tenure = st.slider("Meses de Permanencia (Tenure)", min_value=1, max_value=72, value=12)
        phone_service = st.selectbox("Servicio Telefónico", ["Yes", "No"])
    with col_f3:
        multiple_lines = st.selectbox("Líneas Múltiples", ["No phone service", "No", "Yes"])
        internet_service = st.selectbox("Servicio de Internet", ["DSL", "Fiber optic", "No"])
        contract = st.selectbox("Tipo de Contrato", ["Month-to-month", "One year", "Two year"])
    with col_f4:
        paperless_billing = st.selectbox("Facturación Electrónica", ["Yes", "No"])
        monthly_charges = st.number_input("Cargo Mensual ($)", min_value=10.0, max_value=150.0, value=75.0)
        total_charges = st.number_input("Cargos Totales Acumulados ($)", min_value=10.0, max_value=8000.0, value=900.0)

    input_data = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": senior_citizen, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone_service, "MultipleLines": multiple_lines,
        "InternetService": internet_service, "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": contract, "PaperlessBilling": paperless_billing, "PaymentMethod": "Electronic check",
        "MonthlyCharges": monthly_charges, "TotalCharges": total_charges
    }])

    if st.button("🔮 Calcular Riesgo de Churn"):
        # --- CORRECCIÓN CRÍTICA DE TIPOS DE DATOS ---
        # Extraemos la probabilidad de Churn (columna 1) y la convertimos a float nativo de Python
        prob_array = pipeline.predict_proba(input_data)
        probabilidad_porcentaje = float(prob_array[0, 1]) * 100
        probabilidad_progreso = float(prob_array[0, 1])
        
        st.markdown("### Resultado del Análisis de Riesgo:")
        if probabilidad_porcentaje > 50.0:
            st.error(f"🚨 **ALTO RIESGO DE FUGA:** Este cliente tiene un **{probabilidad_porcentaje:.1f}%** de probabilidad de abandonar la empresa.")
            st.progress(probabilidad_progreso)
        else:
            st.success(f"✅ **CLIENTE ESTABLE:** La probabilidad de abandono es baja (**{probabilidad_porcentaje:.1f}%**).")
            st.progress(probabilidad_progreso)
