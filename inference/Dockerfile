# inference/Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code + downloaded model
COPY . .

EXPOSE 8000

ENV MODEL_PATH=/app/model/model.keras

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
