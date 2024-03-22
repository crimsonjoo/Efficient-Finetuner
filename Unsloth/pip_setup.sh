#!/bin/bash

# CUDA 버전 확인
CUDA_VERSION=$(python -c 'import torch; print(torch.version.cuda)')

# PyTorch 버전 확인
PYTORCH_VERSION=$(python -c 'import torch; print(torch.__version__)')

# Ampere 아키텍처 여부 확인
IS_AMPERE=false
# 여기서는 사용자가 직접 Ampere 아키텍처 여부를 입력하게 하는 예시입니다.
# 자동으로 확인하는 방법도 구현할 수 있으나, 복잡성이 증가합니다.
read -p "Are you using an Ampere architecture GPU? (yes/no): " AMPERE_ANSWER
if [ "$AMPERE_ANSWER" == "yes" ]; then
    IS_AMPERE=true
fi

# 필요한 패키지 설치 로직 구현
# 예: CUDA 11.8, PyTorch 2.1.0, Ampere 아키텍처인 경우
if [ "$CUDA_VERSION" == "11.8" ] && [ "$PYTORCH_VERSION" == "2.1.0" ] && $IS_AMPERE ; then
    pip install --upgrade --force-reinstall --no-cache-dir torch==2.1.0 triton --index-url https://download.pytorch.org/whl/cu118
    pip install "unsloth[cu118-ampere] @ git+https://github.com/unslothai/unsloth.git"
fi

# 다른 조건에 따른 설치 명령어 추가...
