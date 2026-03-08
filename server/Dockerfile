FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt

# Copy application code
COPY models.py /app/
COPY server/ /app/server/

# Expose port for HF Spaces
EXPOSE 7860

# Run server on port 7860 (HF Spaces default)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
