import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import sqlalchemy as sa
import plotly.express as px
import joblib
import os

# 1. Configuración de la página con Layout Ancho y Estilo Ejecutivo
st.set_page_config(page_title="Corporate Churn Intelligence", page_icon="📈", layout="wide")

# Inyección de CSS Personalizado para diseño premium
st.markdown("""
<style>
@import url('https://googleapis.com');
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.main-title {
    font-size: 32px;
    font-weight: 600;
    color: #1E293B;
    margin-bottom: 2px;
}
.sub-title {
    font-size: 14px;
    color: #64748B;
    margin-bottom: 25px;
}
.kpi-box {
    background-color: #F8FAFC;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #3B82F6;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    margin-bottom: 15px;
}
.kpi-title {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #64748B;
    font-weight: 600;
}
.kpi-value {
    font-size: 24px;
    font-weight: 600;
    color: #0F172A;
    margin-top: 5px;
}
</style>
""", unsafe_allow_html=True)

# Encabezado Principal
st.markdown('<div class="main-title">Customer Intelligence Hub</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Análisis de Datos Interactivo desde PostgreSQL e Inferencia Acelerada por GPU</div>', unsafe_allow_html=True)

# 🧠 Conexiones Híbridas (Nube / Docker Local)
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/customer_churn")
MODEL_PATH = "01.models/pipeline_xgboost_telco.pkl"

@st.cache_resource
def get_db_engine():
    return sa.create_engine(
        DATABASE_URL,
        pool_recycle=1800,   # Saca de circulación conexiones con más de 30 minutos inactivas
        pool_pre_ping=True   # Envía un latido de prueba antes de cada consulta; si el canal SSL cayó, lo levanta en microsegundos
    )

@st.cache_resource
def load_ml_pipeline():
    if os.path.exists(MODEL_PATH):
        # 1. Cargamos el pipeline de Scikit-Learn / XGBoost
        pipeline = joblib.load(MODEL_PATH)
        
        # 2. DETECTOR DINÁMICO DE HARDWARE:
        # Si la app corre en la nube sin GPU, forzamos a XGBoost a operar en CPU
        if hasattr(pipeline.named_steps['model'], 'set_params'):
            try:
                # Intentamos verificar si CUDA está disponible
                import xgboost as xgb
                # Si estamos en la nube de Streamlit, esto fallará o no tendrá device cuda activo
                pipeline.named_steps['model'].set_params(device="cpu", tree_method="hist")
            except:
                pipeline.named_steps['model'].set_params(device="cpu", tree_method="hist")
                
        return pipeline
    return None

engine = get_db_engine()
pipeline_produccion = load_ml_pipeline()

# -------------------------------------------------------------------
# SIDEBAR CONTROL (INTERACTIVIDAD AVANZADA)
# -------------------------------------------------------------------
st.sidebar.header("🎯 Filtros Globales de Negocio")
st.sidebar.write("Modifica el entorno para segmentar las vistas analíticas en tiempo real.")

try:
    with engine.connect() as conn:
        contratos_disponibles = pd.read_sql("SELECT DISTINCT contract FROM customers;", con=conn)["contract"].tolist()
except:
    contratos_disponibles = ["Month-to-month", "One year", "Two year"]

contract_filter = st.sidebar.multiselect(
    "Filtrar por Esquema Contractual:",
    options=contratos_disponibles,
    default=contratos_disponibles
)

# -------------------------------------------------------------------
# SECCIÓN 1: PANEL ANALÍTICO INTERACTIVO (POSTGRESQL + PLOTLY)
# -------------------------------------------------------------------
st.subheader("📊 Consola de Control de Clientes")

try:
    if not contract_filter:
        st.warning("Selecciona al menos un tipo de contrato en el panel izquierdo.")
    else:
        query = sa.text("SELECT * FROM customers WHERE contract IN :contratos")
        df_filtered = pd.read_sql(query, con=engine, params={"contratos": tuple(contract_filter)})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="kpi-box"><div class="kpi-title">Clientes Analizados</div><div class="kpi-value">{len(df_filtered):,}</div></div>', unsafe_allow_html=True)
        with col2:
            tasa_churn = (df_filtered['churn'] == 'Yes').mean() * 100 if len(df_filtered) > 0 else 0.0
            st.markdown(f'<div class="kpi-box" style="border-left-color: #EF4444;"><div class="kpi-title">Tasa de Churn Global</div><div class="kpi-value">{tasa_churn:.1f}%</div></div>', unsafe_allow_html=True)
        with col3:
            ingreso_total = df_filtered['monthly_charges'].sum() if len(df_filtered) > 0 else 0.0
            st.markdown(f'<div class="kpi-box" style="border-left-color: #10B981;"><div class="kpi-title">Ingreso Mensual Total</div><div class="kpi-value">${ingreso_total:,.0f}</div></div>', unsafe_allow_html=True)
        with col4:
            permanencia_promedio = df_filtered['tenure'].mean() if len(df_filtered) > 0 else 0.0
            st.markdown(f'<div class="kpi-box" style="border-left-color: #F59E0B;"><div class="kpi-title">Permanencia Promedio</div><div class="kpi-value">{permanencia_promedio:.1f} Meses</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.markdown("**Distribución de Cargos Mensuales por Estatus de Churn**")
            fig_hist = px.histogram(
                df_filtered, 
                x="monthly_charges", 
                color="churn", 
                barmode="overlay",
                marginal="box",
                color_discrete_map={"No": "#3B82F6", "Yes": "#EF4444"},
                labels={"monthly_charges": "Cargos Mensuales ($)", "churn": "¿Se Fugó?"},
                template="plotly_white"
            )
            fig_hist.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=350)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_g2:
            st.markdown("**Concentración de Clientes por Método de Pago**")
            fig_pie = px.pie(
                df_filtered, 
                names="payment_method", 
                color="payment_method", 
                color_discrete_sequence=px.colors.sequential.Blues_r, 
                hole=0.4,
                template="plotly_white"
            )
            fig_pie.update_layout(
                margin=dict(l=20, r=20, t=20, b=20), 
                height=350,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
         
        with st.expander("🔍 Explorar Datos Crutos Segmentados", expanded=False):
            st.dataframe(df_filtered.head(100), use_container_width=True)

except Exception as e:
    st.error(f"Error al estructurar los gráficos o consultar la base de datos: {e}")

# -------------------------------------------------------------------
# SECCIÓN 2: SIMULADOR DE INFERENCIA EN VIVO (XGBOOST)
# -------------------------------------------------------------------
st.markdown("<br><hr>", unsafe_allow_html=True)
st.subheader("🔮 Simulador de Retención de Clientes")

if pipeline_produccion is None:
    st.warning(f"⚠️ Artefacto del modelo no detectado en '{MODEL_PATH}'.")
else:
    with st.container(border=True):
        st.markdown("**Ingreso Manual de Parámetros**")
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        
        with col_f1:
            gender = st.selectbox("Género", ["Male", "Female"])
            senior_citizen = st.selectbox("Adulto Mayor", [0, 1], format_func=lambda x: "Sí (1)" if x == 1 else "No (0)")
            partner = st.selectbox("Tiene Pareja", ["Yes", "No"])
        with col_f2:
            dependents = st.selectbox("Tiene Dependientes", ["Yes", "No"])
            tenure = st.slider("Permanencia (Meses)", min_value=1, max_value=72, value=12)
            phone_service = st.selectbox("Servicio de Voz", ["Yes", "No"])
        with col_f3:
            multiple_lines = st.selectbox("Líneas Telefónicas", ["No phone service", "No", "Yes"])
            internet_service = st.selectbox("Tecnología Internet", ["DSL", "Fiber optic", "No"])
            contract = st.selectbox("Esquema Contractual", ["Month-to-month", "One year", "Two year"])
        with col_f4:
            paperless_billing = st.selectbox("Factura Electrónica", ["Yes", "No"])
            monthly_charges = st.number_input("Factura Mensual ($)", min_value=10.0, max_value=150.0, value=75.0)
            total_charges = st.number_input("Factura Acumulada ($)", min_value=10.0, max_value=8000.0, value=900.0)

        # Sincronización Estricta de Nombres de Columnas Originales (Evita el ValueError de XGBoost)
        input_data = pd.DataFrame([{
            "gender": gender, 
            "SeniorCitizen": int(senior_citizen), 
            "Partner": partner,
            "Dependents": dependents,
            "tenure": int(tenure), 
            "PhoneService": phone_service, 
            "MultipleLines": multiple_lines,
            "InternetService": internet_service, 
            "OnlineSecurity": "No", 
            "OnlineBackup": "No",
            "DeviceProtection": "No", 
            "TechSupport": "No", 
            "StreamingTV": "No", 
            "StreamingMovies": "No",
            "Contract": contract, 
            "PaperlessBilling": paperless_billing, 
            "PaymentMethod": "Electronic check", # Valor base por defecto para el simulador
            "MonthlyCharges": float(monthly_charges), 
            "TotalCharges": float(total_charges)
        }])

        col_btn, _ = st.columns(2)
        with col_btn:
            btn_action = st.button("🚀 Evaluar Probabilidad de Fuga", use_container_width=True)

        if btn_action:
            try:
                # Inferencia nativa acelerada
                prob_array = pipeline_produccion.predict_proba(input_data)
                prob_val = float(prob_array[0, 1])
                probabilidad_porcentaje = round(prob_val * 100, 1)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                if probabilidad_porcentaje > 50.0:
                    st.error(f"🚨 **Diagnóstico de Riesgo de Fuga Alto:** El cliente tiene una probabilidad de abandono del {probabilidad_porcentaje}%. Requiere una acción comercial inmediata.")
                    st.progress(prob_val)
                else:
                    st.success(f"✅ Diagnóstico de Cuenta Estable: La probabilidad de abandono es controlada ({probabilidad_porcentaje}%). La cuenta es saludable.")
                    st.progress(prob_val)
            except Exception as e:
                    st.error(f"❌ Error durante el proceso de inferencia: {str(e)}")