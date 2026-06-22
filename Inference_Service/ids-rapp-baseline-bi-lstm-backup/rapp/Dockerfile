# GPU 이미지 (CUDA 사용) — TensorFlow 포함
FROM tensorflow/tensorflow:2.17.0-gpu
ENV PYTHONUNBUFFERED=1
# build context = baseline 루트 (model/ 이 rapp/ 의 형제이므로 함께 베이크하려면 필요)
COPY rapp/app /app
# /bench/retraining 가 트리거하는 재학습 스크립트
COPY train/train.py /app/train.py
# 학습 산출물 베이크 인 (model.keras + scaler.pkl)
COPY model /model
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8005
CMD ["python3", "-u", "main.py"]
