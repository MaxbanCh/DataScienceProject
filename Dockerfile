FROM python:3.11-slim

WORKDIR /app

# Copier les requirements
COPY visualizer/requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code du visualiseur et les données
COPY visualizer /app/visualizer
COPY ina-api/data /app/data

WORKDIR /app/visualizer

# Exposer le port Streamlit
EXPOSE 8501

# Lancer Streamlit
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]