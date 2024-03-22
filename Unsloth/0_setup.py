
###### ※스크립트 사용 전 주의사항※ #############################################################################################################
# - 이 스크립트는 Pip을 사용한 설치를 가정하고 있습니다. (Conda 환경에서는 사용하지 않는 것이 좋습니다.)
# - 사용자의 환경에 따라 PyTorch 버전이나 기타 라이브러리의 호환성을 확인하고 필요에 따라 버전을 조정할 필요가 있습니다.
# - CUDA 버전은 사용자의 GPU와 호환되는 버전을 선택해야 합니다.(CUDA 버전을 정확히 알고 있지 않은 경우, NVIDIA 공식 문서 또는 GPU의 사양을 참조하세요.)
################################################################################################################################################


''' 방법1'''
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    # Pip과 Python이 최신 버전인지 확인
    install("--upgrade pip")
    
    # 사용자의 CUDA 버전 확인
    cuda_version = input("CUDA 버전을 입력하세요 (예: 11.8, 12.1): ")
    pytorch_version = "2.1.0" # 예시로 사용된 PyTorch 버전, 필요에 따라 변경 가능
    
    # PyTorch와 관련 종속성 설치
    if cuda_version == "11.8":
        install(f"--upgrade --force-reinstall --no-cache-dir torch=={pytorch_version}+cu118 --index-url https://download.pytorch.org/whl/cu118")
    elif cuda_version == "12.1":
        install(f"--upgrade --force-reinstall --no-cache-dir torch=={pytorch_version}+cu121 --index-url https://download.pytorch.org/whl/cu121")
    else:
        print("지원되지 않는 CUDA 버전입니다.")
        sys.exit(1)
    
    # Unsloth 설치
    unsloth_version = "cu121" if cuda_version == "12.1" else "cu118"
    install(f"unsloth[{unsloth_version}] @ git+https://github.com/unslothai/unsloth.git")
    
    # 기타 필요한 라이브러리 설치
    install("--no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes")
    print("설치가 완료되었습니다.")






'''
# 방법2

###### ※스크립트 사용 전 주의사항※ #############################################################################################################
# 이 스크립트는 nvcc 명령어를 사용하여 CUDA 버전을 감지합니다. CUDA Toolkit이 시스템에 설치되어 있고, nvcc가 시스템의 PATH에 추가되어 있어야 합니다.
# 설치 중 오류가 발생할 경우, 스크립트는 해당 오류 메시지를 출력하고 종료됩니다.
# 이 스크립트는 파이썬 3.6 이상에서 테스트되었습니다.
################################################################################################################################################


import subprocess
import sys

def run_command(command):
    """주어진 명령어를 실행하는 함수"""
    result = subprocess.run(command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Error executing command: {command}\n{result.stderr}", file=sys.stderr)
        sys.exit(result.returncode)
    return result.stdout.strip()

def get_cuda_version():
    """CUDA 버전을 확인하는 함수"""
    cuda_version = run_command("nvcc --version | grep 'release' | awk '{print $6}' | cut -c2-")
    return cuda_version

def install_requirements(cuda_version):
    """필요한 패키지들을 설치하는 함수"""
    # Pip 업그레이드
    run_command("pip install --upgrade pip")
    # PyTorch 및 Triton 설치 (예: CUDA 11.8을 사용하는 경우)
    if cuda_version == "11.8":
        run_command("pip install --upgrade --force-reinstall --no-cache-dir torch==2.1.0 triton --index-url https://download.pytorch.org/whl/cu118")
        run_command("pip install 'unsloth[cu118] @ git+https://github.com/unslothai/unsloth.git'")
    # 추가 CUDA 버전에 대한 조건부 설치 로직을 여기에 추가...
    else:
        print("Unsupported CUDA version for automatic installation.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    cuda_version = get_cuda_version()
    print(f"Detected CUDA version: {cuda_version}")
    install_requirements(cuda_version)
'''