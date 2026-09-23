# ==============================================================================
# ETAPA 1: BUILDER (Compilación y Preparación de Dependencias)
# ==============================================================================
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Instalar herramientas necesarias para descargar y compilar
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Instalar Miniconda de forma silenciosa
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    /bin/bash /miniconda.sh -b -p $CONDA_DIR && \
    rm /miniconda.sh

# Copiar archivos de requerimientos e instalar dependencias (Compila psycopg2 nativo en C)
COPY requirements-dev.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-dev.txt


# ==============================================================================
# ETAPA 2: RUNTIME (Entorno Final Seguro y Ligero para Ejecución)
# ==============================================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
# Inyectamos tanto Conda como las librerías precompiladas de la etapa anterior
ENV PATH=/home/ds_user/.local/bin:$CONDA_DIR/bin:$PATH
ENV PYTHONPATH=/workspace

# Instalar SOLO las librerías compartidas de tiempo de ejecución (Runtime) para Postgres y Git
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copiar la instalación limpia de Miniconda desde la etapa de compilación
COPY --from=builder $CONDA_DIR $CONDA_DIR
# Copiar las librerías de Python ya compiladas sin arrastrar herramientas de desarrollo
COPY --from=builder /install /opt/conda

# 🔐 SEGURIDAD: Crear un usuario sin privilegios root
RUN useradd -m -s /bin/bash ds_user && \
    mkdir -p /workspace && \
    chown -R ds_user:ds_user /workspace

# Establecer el espacio de trabajo y cambiar al usuario seguro
WORKDIR /workspace
USER ds_user

# Exponer el puerto por defecto de tu simulador
EXPOSE 8501
