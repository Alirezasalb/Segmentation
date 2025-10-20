# sam-universal-segmenter/Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
# Note: segment-anything requires git during install
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .

# Create directory for model
RUN mkdir -p /app/model

# Expose Streamlit port
EXPOSE 8501

# Instructions for user
ENV MODEL_PATH=/app/model/sam_vit_b_01ec64.pth

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health

# Run app
CMD ["sh", "-c", "streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]