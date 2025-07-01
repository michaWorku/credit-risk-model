# Use a lightweight Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements file first to leverage Docker cache
# If requirements.txt doesn't change, this layer is cached
COPY requirements.txt .

# Install dependencies
# Use --no-cache-dir to reduce image size
# Use --upgrade pip to ensure pip is up-to-date pip install --upgrade pip && \
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code AFTER dependencies are installed
# This ensures that changes to your source code don't invalidate the dependency layer
COPY . /app

# Set environment variables for MLflow (optional, but good practice for clarity)
# ENV MLFLOW_TRACKING_URI=file:///app/mlruns
# ENV REGISTERED_MODEL_NAME=CreditRiskClassifier
# ENV MODEL_VERSION=latest
# # Ensure the path to raw data for processor fit is correct inside the container
# ENV RAW_DATA_PATH_FOR_PROCESSOR_FIT=/app/data/raw/data.csv

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn
# --host 0.0.0.0 makes the server accessible from outside the container
# --port 8000 specifies the port
# --workers 1 (or more, depending on CPU cores) for production
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

