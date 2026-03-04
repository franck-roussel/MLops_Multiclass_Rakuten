FROM python:3.8-slim-bullseye

# Évite la génération de fichiers .pyc et force l'affichage des logs en temps réel
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Installation des libs système nécessaires pour OpenCV et les extensions C
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Mise à jour de pip et installation des dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY . .

# Port exposé par FastAPI
EXPOSE 8000

# Commande de lancement (adapter 'main:app' au nom de votre fichier d'entrée)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]