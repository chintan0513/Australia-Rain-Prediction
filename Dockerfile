# Use official Python slim image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Set working directory inside container
WORKDIR /app

# Install system dependencies (needed for numpy, pandas, sklearn, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first and install
COPY requirements.txt /app/

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model artifacts
COPY app.py /app/
COPY model.h5 /app/
COPY scaler.pkl /app/
COPY feature_columns.pkl /app/
COPY label_encoders.pkl /app/
COPY encoders.pkl /app/

# Expose API port
EXPOSE 8080

# Run FastAPI app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
