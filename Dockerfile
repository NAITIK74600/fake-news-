FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONPATH=/app/src

EXPOSE 8000
CMD ["python", "scripts/run_api.py"]
