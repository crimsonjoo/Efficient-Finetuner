#!/bin/bash

echo "환경 설정을 시작합니다..."
conda create --name unsloth_env python=3.10 -y
conda activate unsloth_env

# 사용자로부터 CUDA 버전 입력 받기
read -p "CUDA 버전을 선택하세요 (예: 11.8, 12.1): " cuda_version

conda install pytorch-cuda=$cuda_version pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers -y

# PyTorch 버전 확인
pytorch_version=$(python -c "import torch; print(torch.__version__)")

# PyTorch 버전에 따른 Unsloth 설치
if [[ $pytorch_version == "2.1.0" ]]; then
  pip install "unsloth[cu${cuda_version//./}] @ git+https://github.com/unslothai/unsloth.git"
elif [[ $pytorch_version == "2.1.1" ]]; then
  # 예시: PyTorch 2.1.1용 Unsloth 설치 명령
  # 추가 조건 분기 및 설치 명령 작성
else
  echo "지원되지 않는 PyTorch 버전입니다."
fi

pip install --no-deps trl peft accelerate bitsandbytes
echo "환경 설정이 완료되었습니다."
