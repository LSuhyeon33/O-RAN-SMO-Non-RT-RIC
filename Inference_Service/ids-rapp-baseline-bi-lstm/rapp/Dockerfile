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
# 베이스 TF 이미지에 OS(distutils)로 깔린 blinker 1.4 를 pip 이 제거 못 해 실패하므로
# --ignore-installed blinker 로 기존 것을 건드리지 않고 새 버전을 설치한다 (Flask 3.0 의존성).
RUN pip install --no-cache-dir --ignore-installed blinker -r requirements.txt
EXPOSE 8005
CMD ["python3", "-u", "main.py"]
