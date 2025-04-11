FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages
RUN pip install --no-cache-dir \
    flask==2.3.2 \
    pyyaml==6.0 \
    opencv-python-headless==4.7.0.72 \
    numpy \
    scikit-learn==1.2.2

# Create directory and set permissions
RUN mkdir -p /app/configs && \
    chmod -R a+rwx /app

# Copy files
COPY services/data_loader/app.py .
COPY configs/data_config.yml /app/configs/

# Verify files were copied
RUN ls -l /app/configs/

# CMD ["python", "app.py"]

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
