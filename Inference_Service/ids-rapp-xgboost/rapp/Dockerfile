FROM python:3.10-slim
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY app /app
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8005
CMD ["python3", "-u", "main.py"]
