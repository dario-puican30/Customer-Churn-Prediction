# Corporate Churn Intelligence Hub 📈

Ecosistema portátil de producción para la predicción, análisis y prevención de la fuga de clientes (*Customer Churn*) en la industria de telecomunicaciones. Este proyecto implementa una arquitectura desacoplada y contenerizada, asegurando portabilidad absoluta (**cero instalación local**) y aceleración por hardware.

## 🏗️ Arquitectura del Sistema

El proyecto está orquestado mediante **Docker** y consta de tres servicios independientes interconectados a través de una red interna aislada:

1. **`customer_churn_app` (Entorno DS):** Contenedor basado en Miniconda optimizado para Machine Learning. Cuenta con acceso nativo a la GPU local mediante el driver NVIDIA y CUDA, permitiendo entrenamientos acelerados de modelos complejos.
2. **`customer_churn_db` (Almacenamiento):** Motor relacional PostgreSQL 16 sobre una imagen alpina ligera. Mantiene persistencia de datos mediante volúmenes de Docker e implementa restricciones de integridad y vistas analíticas de negocio en caliente.
3. **`customer_churn_streamlit` (Dashboard):** Panel corporativo interactivo en tonos monocromáticos azules desarrollado con Streamlit y Plotly. Permite la exploración visual de métricas desde la base de datos y simulaciones de inferencia unitaria en tiempo real.

---

## 📂 Estructura del Proyecto

```text
01.Customer-churn/
├── .devcontainer/         # Configuración para desarrollo aislado dentro de Docker en VS Code
├── 01.data/               # Datos del ciclo de vida del proyecto (ignorados en Git)
│   ├── raw/               # Dataset crudo original (telco_churn.csv)
│   ├── processed/         # Datasets limpios y preprocesados
│   └── scoring/           # Clientes nuevos para inferencia por lotes
├── 02.sql/                # Scripts de estructura e inicialización de la Base de Datos
├── 03.notebooks/          # Jupyter Notebooks de experimentación del pipeline
├── 04.streamlit/          # Código fuente de la aplicación web interactiva (app.py)
├── 05.docs/               # Documentación adicional y reportes de negocio
├── src/                   # Código de producción modularizado (Source Layer)
│   └── scoring/           # Scripts automatizados para inferencia por lotes (predict.py)
├── Dockerfile             # Receta de construcción del entorno DS basado en Conda
├── docker-compose.yml     # Orquestador del ecosistema multi-contenedor (App, DB, UI)
├── requirements.txt       # Dependencias estrictas de Python fijadas para producción
└── README.md              # Documentación principal del sistema
```

---

## 🛠️ Tecnologías y Hardware Utilizados

- **Sistema Base:** Ubuntu OS (Optimizado para contenedores y drivers NVIDIA)
- **Procesamiento Acelerado:** NVIDIA RTX 4060 GPU (8GB VRAM) + CUDA Execution Provider
- **Modelado Core:** Python (Scikit-Learn, XGBoost, Pandas, SQLAlchemy)
- **Base de Datos:** PostgreSQL 16 + `psycopg2-binary`
- **Infraestructura y Portabilidad:** Docker Engine + Docker Compose + NVIDIA Container Toolkit

---

## 🚀 Guía de Despliegue Rápido (Portabilidad Total)

Gracias a la centralización en Docker, puedes levantar todo este ecosistema en cualquier computadora con solo clonar el repositorio y ejecutar los siguientes comandos:

### 1. Requisitos Previos
Asegúrate de tener instalado Docker, Docker Compose y el driver oficial de NVIDIA en el sistema anfitrión.

### 2. Levantar la Infraestructura
En la raíz del proyecto, ejecuta el orquestador en segundo plano:
```bash
docker compose up -d --build
```

### 3. Ejecutar la Ingesta de Datos y el Entrenamiento
1. Abre VS Code en esta carpeta y selecciona **"Reopen in Container"** mediante la extensión *Dev Containers*.
2. Ejecuta secuencialmente los notebooks de la carpeta `03.notebooks/` para poblar la base de datos PostgreSQL de forma relacional y exportar el Pipeline serializado (`pipeline_xgboost_telco.pkl`) con soporte para GPU hacia la carpeta `01.models/`.

### 4. Acceder al Dashboard Ejecutivo
Abre tu navegador e ingresa a:
📌 **http://localhost:8501**

Desde aquí podrás segmentar los datos de PostgreSQL interactivamente con filtros globales y simular el riesgo de abandono de un cliente usando el modelo predictivo de XGBoost.

---

## 🧠 Características Avanzadas de Ingeniería de Datos

- **Robustez en Inferencia (Scikit-Learn Pipelines):** Toda la lógica de ingeniería de características, imputación de valores nulos mediante medianas estadísticas y el escalado numérico financiero se encuentran empaquetados dentro de un objeto `Pipeline` unificado. Esto elimina el riesgo de fuga de datos (*Data Leakage*) y garantiza consistencia total entre entrenamiento y producción.
- **Persistencia Segura:** El `.gitignore` del proyecto actúa como un escudo corporativo, evitando la subida accidental de datos pesados de clientes, credenciales en texto plano o archivos binarios de modelos a repositorios públicos.
