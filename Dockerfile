# Usamos una base oficial de Miniconda
FROM continuumio/miniconda3:latest

# Instalar herramientas básicas del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /workspace

# Copiar el archivo de requerimientos e instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto por defecto de Streamlit (por si acaso)
EXPOSE 8501
