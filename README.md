# Corporate Churn Intelligence Hub 📈

Ecosistema portátil de producción para la predicción, análisis y prevención de la fuga de clientes (_Customer Churn_) en la industria de telecomunicaciones. Este proyecto implementa una arquitectura desacoplada y contenerizada, asegurando portabilidad absoluta (**cero instalación local**) y aceleración por hardware.

## 🏗️ Arquitectura del Sistema

El proyecto está orquestado mediante **Docker** y consta de tres servicios independientes interconectados a través de una red interna aislada (`data_net` interna y `web_net` externa):

1. **`customer_churn_app` (Entorno DS):** Contenedor basado en el entorno de desarrollo seguro de producción. Cuenta con acceso nativo a la GPU local mediante drivers de NVIDIA y CUDA, permitiendo entrenamientos acelerados de modelos complejos sin bloqueos de `ipykernel` usando la extensión Dev Containers de VS Code (`remoteUser: "root"`).
2. **`customer_churn_db` (Almacenamiento):** Motor relacional `postgres:16-alpine` (Puerto `5432`). Mantiene persistencia de datos mediante volúmenes de Docker (`pgdata`) e implementa restricciones de integridad y vistas analíticas de negocio en caliente. Credenciales externalizadas mediante `.env`.
3. **`customer_churn_streamlit` (Dashboard Ejecutivo):** Panel corporativo interactivo en gama de azules desarrollado con Streamlit (Puerto `8501`) y Plotly. Permite la exploración visual de métricas y simulaciones de inferencia unitaria en tiempo real mediante una lógica híbrida (GPU local / CPU en la nube con Neon Postgres).

---

## 📂 Estructura del Proyecto

```text
01.Customer-churn/
├── .devcontainer/            # Configuración del entorno de desarrollo remoto para VS Code
├── 01.data/                  # Datos del ciclo de vida del proyecto (ignorados en Git)
│   ├── processed/            # Datasets limpios (telco_churn_clean.csv / telco_churn_model_ready.csv)
│   ├── raw/                  # Dataset crudo original (telco_churn.csv)
│   └── scoring/              # Clientes nuevos para inferencia por lotes y sus resultados
├── 01.models/                # Binarios de modelos serializados (.pkl de XGBoost y baselines)
├── 02.sql/                   # Scripts de estructura e inicialización de la Base de Datos PostgreSQL
├── 03.notebooks/             # Jupyter Notebooks ordenados para el ciclo de vida experimental
├── 04.streamlit/             # Código fuente de la interfaz web interactiva en producción
├── src/                      # Código de producción modularizado (Source Layer)
│   ├── data/                 # Inicializadores de ingesta de datos
│   ├── features/             # Funciones reutilizables de feature engineering
│   ├── models/               # Lógica secundaria de modelado
│   ├── scoring/              # Script predict.py optimizado para inferencia en lotes de producción
│   └── utils/                # Funciones de soporte técnico
├── .gitattributes            # Configuración de atributos de Git y asignación de punteros LFS
├── .gitignore                # Reglas estrictas de exclusión por extensión para evitar fugas binarias
├── Dockerfile                # Receta de construcción del entorno DS basado en CUDA de NVIDIA
├── README.md                 # Documentación principal del sistema
├── docker-compose.yml        # Orquestador del ecosistema multi-contenedor (App, DB, UI)
├── requirements-dev.txt      # Dependencias estrictas fijadas para el entorno Docker y uso local
└── requirements.txt          # Dependencias ligeras requeridas específicamente para Streamlit Cloud
```

---

## 🛠️ Tecnologías y Hardware Utilizados

- **Sistema Base Local:** Ubuntu 26.04 LTS (Resolute Raccoon).
- **Procesamiento Acelerado Local:** NVIDIA RTX 4060 GPU (8GB VRAM) + NVIDIA Container Toolkit.
- **Modelado Core:** Python (Scikit-Learn, XGBoost, Pandas 3, NumPy, SciPy).
- **Base de Datos:** PostgreSQL 16 Alpine + `psycopg2` compilado nativo en C.
- **Infraestructura y Portabilidad:** Docker Engine + Docker Compose.

---

## 📊 Fuente de Datos y Rendimiento del Modelo

### Origen de los Datos

El pipeline se alimenta del dataset público [Kaggle Telco Customer Churn](https://kaggle.com). Contiene información de 7,043 clientes de una compañía de telecomunicaciones en California, detallando perfiles demográficos, servicios contratados y estados de cuenta financieros.

### Efectividad y Resultados de Machine Learning

El modelo final seleccionado es un **XGBClassifier** optimizado por hardware mediante histogramas acelerados por CUDA (`tree_method="hist"`). Para mitigar la multicolinealidad categórica en variables como los servicios adicionales, se implementó el cálculo real de **Cramer's V** en matrices de asociación nominal en lugar del coeficiente clásico de Pearson.

Las métricas clave del clasificador consolidado en el reporte de rendimiento son:

- **F1-Score (Clase Churn):** **~0.62 - 0.65** (Diseñado específicamente para equilibrar la precisión y la exhaustividad en la detección del cliente que realmente va a abandonar la compañía).
- **Exactitud General (Accuracy):** **~80.5%** en el conjunto de prueba independiente (`X_test`).
- **Manejo de Desbalanceo:** Configurado mediante optimización de hiperparámetros (`max_depth=4`, `subsample=0.8`, `colsample_bytree=0.6`) para prevenir el sobreajuste (_overfitting_) en la clase minoritaria.

---

## 🧠 Características Avanzadas de Ingeniería de Datos

- **Robustez en Inferencia (Scikit-Learn Pipelines):** Toda la lógica de ingeniería de características, imputación de valores nulos mediante medianas estadísticas y el escalado numérico financiero se encuentran empaquetados dentro de un objeto `Pipeline` unificado. Esto elimina el riesgo de fuga de datos (_Data Leakage_).
- **Gestión de Entornos Dual (`requirements-dev.txt` vs `requirements.txt`):** El proyecto mantiene un aislamiento estricto de paquetes. El archivo `requirements-dev.txt` provee el stack completo de desarrollo con aceleración CUDA e `ipykernel` optimizado para la ejecución local en Ubuntu. El archivo `requirements.txt` se reserva exclusivamente como un manifiesto ligero de producción que la plataforma de Streamlit Cloud interpreta de manera nativa para desplegar la UI sin conflictos de drivers de hardware o compiladores.
- **Inferencia Desacoplada (CPU ➡️ GPU):** Para evitar caídas de rendimiento y alertas de discrepancia de hardware (`mismatched devices`), el backend de producción en `src/scoring/predict.py` extrae explícitamente el preprocesamiento secuencial en CPU (`ColumnTransformer`) antes de enviar el arreglo numérico plano de forma directa al motor predictivo de la GPU de la RTX 4060. Esto reduce la latencia de transferencia de memoria en producción a milisegundos.
- **Persistencia Segura:** El `.gitignore` actúa como un escudo corporativo, evitando la subida accidental de datos crudos pesados de clientes o credenciales en texto plano.

---

## 🚀 Guía de Despliegue Rápido (Portabilidad Total)

Gracias a la centralización en Docker, puedes levantar todo este ecosistema en cualquier computadora con solo clonar el repositorio bajo el usuario `dario-puican30` y ejecutar los siguientes comandos:

### 1. Requisitos Previos

Asegúrate de tener instalado Docker, Docker Compose y el driver oficial de NVIDIA junto al NVIDIA Container Toolkit en el sistema anfitrión.

### 2. Levantar la Infraestructura Local

En la raíz del proyecto, ejecuta el orquestador en segundo plano (el cual se construye utilizando el entorno robusto de desarrollo de `requirements-dev.txt`):

```bash
docker compose up -d --build
```

### 3. Ejecutar la Ingesta de Datos y el Entrenamiento

1. Abre VS Code en esta carpeta y selecciona **"Reopen in Container"** mediante la extensión _Dev Containers_.
2. Ejecuta secuencialmente los notebooks de la carpeta `03.notebooks/` (`01_load_raw_data.ipynb` hasta `04_modeling.ipynb`) para poblar la base de datos PostgreSQL de forma relacional y exportar el Pipeline serializado con soporte híbrido hacia la ruta `01.models/pipeline_xgboost_telco.pkl`.

### 4. Automatización por Lotes (Scoring Batch)

Para procesar clientes nuevos sin abrir la interfaz gráfica, el contenedor de producción puede ejecutar el script modularizado de inferencia:

```bash
python src/scoring/predict.py
```

El resultado enriquecido con las columnas de `Prediccion_Churn` y `Probabilidad_Churn` se consolidará automáticamente en `01.data/scoring/scoring_resultado.csv`.

### 5. Acceder al Dashboard Ejecutivo (Local)

Abre tu navegador e ingresa a:
📌 **http://localhost:8501**
