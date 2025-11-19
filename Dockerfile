# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
#COPY train.py .
#COPY test.py .
COPY app.py .
COPY templates/ templates/

# Copy pre-trained model (if it exists in the build context)
COPY savedmodel.pth .
COPY test_data.pkl .

# Expose port 5000 for Flask
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:5000/health')" || exit 1

# Run the Flask application
CMD ["python", "app.py"]
