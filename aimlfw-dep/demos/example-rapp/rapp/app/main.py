# ==================================================================================
#
#       Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ==================================================================================
from flask import Flask, request, jsonify
import simplejson as json
import os
import requests
import numpy as np

app = Flask(__name__)

# 환경 변수 설정 (KServe 연결 정보)
# 배포 시 환경 변수가 설정되지 않았을 경우를 대비해 기본값을 넣거나 예외처리를 합니다.
KSERVE_HOST = os.environ.get('KSERVE_HOST')
MODEL_NAME = os.environ.get('MODEL_NAME')
PREDICTION_URL = f"{KSERVE_HOST}/v1/models/{MODEL_NAME}:predict"

def predict_single_at_time(model_input: list):
    """
    KServe Inference Service에 추론을 요청하는 함수
    """
    headers = {"Content-Type": "application/json"}

    try:
        # [핵심 수정] 2차원 입력을 CNN용 3차원(Batch, 1, 20)으로 변환
        # xApp에서 넘겨준 데이터가 [[f1, f2...], [f1, f2...]] 형태일 때
        # KServe CNN 모델은 [[[f1, f2...]], [[f1, f2...]]] 형태를 원함
        
        instances = np.array(model_input)
        
        # 데이터가 2차원인 경우 (Batch, Features) -> (Batch, 1, Features)로 변경
        if instances.ndim == 2:
            instances = instances[:, np.newaxis, :]
        
        # 다시 리스트로 변환하여 JSON 직렬화 가능하게 만듦
        payload = {
            "signature_name": "serving_default",
            "instances": instances.tolist()
        }

        response = requests.post(PREDICTION_URL, headers=headers, json=payload, timeout=10)
        if response.status_code != 200:
            print(f"Error| Status-code: {response.status_code}| {response.text}")
            return None
        
        predictions = json.loads(response.text)
        return predictions['predictions'][0]
    except Exception as e:
        print(f"Prediction Request Failed: {e}")
        return None

@app.route('/callback', methods=['POST'])
def test_endpoint():
    # 1. 콜백 데이터 수신
    data = request.json
    print("Received Data for Prediction --> ", data)
    
    # 2. 데이터 추출 (xApp 예시처럼 'data' 필드에서 리스트를 가져온다고 가정)
    # 테스트 시 전송하는 JSON 구조에 따라 data['input'] 등으로 수정 가능합니다.
    model_input = data.get('input', [[2.56, 2.56]] * 10) 
    
    # 3. Inference Service 호출
    predicted_result = predict_single_at_time(model_input)
    
    if predicted_result is not None:
        # 4. 성공 시 결과 반환
        print(f"Prediction Success: {predicted_result}")
        return jsonify({
            "status": "success",
            "input_data": model_input,
            "prediction": predicted_result
        }), 200
    else:
        # 5. 실패 시 에러 반환
        return jsonify({
            "status": "error",
            "message": "Inference service request failed"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8005)