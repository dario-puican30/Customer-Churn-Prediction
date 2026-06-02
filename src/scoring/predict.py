import os
import pandas as pd
import joblib

def load_model(model_path: str):
    """Carga de forma segura el pipeline entrenado desde el archivo pkl."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró ningún modelo en la ruta: {model_path}")
    return joblib.load(model_path)

def predict(df: pd.DataFrame, model) -> pd.DataFrame:
    """Genera predicciones y probabilidades de abandono (Churn)."""
    df_scoring = df.copy()
    
    # Preprocesamiento idéntico al entrenamiento
    df_scoring["TotalCharges"] = pd.to_numeric(df_scoring["TotalCharges"], errors="coerce")
    
    if "customerID" in df_scoring.columns:
        df_scoring = df_scoring.drop(columns=["customerID"])
    if "Churn" in df_scoring.columns:
        df_scoring = df_scoring.drop(columns=["Churn"])
        
    print("Generando probabilidades en lote...")
    predicciones = model.predict(df_scoring)
    probabilidades = model.predict_proba(df_scoring)[:, 1]
    
    df_result = df.copy()
    df_result["Prediccion_Churn"] = predicciones
    df_result["Probabilidad_Churn"] = probabilidades
    
    return df_result

def save_results(df: pd.DataFrame, output_path: str):
    """Guarda el reporte final de scoring en la carpeta correspondiente."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Archivo guardado con éxito en: {output_path}")

def run_scoring(input_csv: str, model_path: str, output_path: str):
    """Flujo principal que orquesta el proceso automatizado de scoring."""
    print("📥 Cargando datos nuevos de clientes...")
    df_nuevos = pd.read_csv(input_csv)
    
    print("🧠 Cargando Pipeline de Machine Learning...")
    model = load_model(model_path)
    
    print("🔮 Ejecutando inferencia automatizada...")
    df_final = predict(df_nuevos, model)
    
    print("💾 Almacenando base de datos enriquecida...")
    save_results(df_final, output_path)
