'''
명령줄 인자:
--action: 수행할 작업을 지정합니다. 가능한 값은 train, save_local, load, upload_hf, save_gguf 입니다.
--model_name: 사용할 모델의 이름을 지정합니다. 기본값은 "saltlux/luxia-21.4b-alignment-v1.1" 입니다.
--token: Hugging Face API 토큰입니다. Hugging Face에 모델을 업로드할 때 필요합니다.
--save_format: 저장할 모델의 형식을 지정합니다. 가능한 값은 lora, merged_16bit, merged_4bit, q4_k_m 입니다.
--save_path: 모델을 저장할 경로입니다. 기본값은 "lora_model" 입니다.
--token 옵션은 upload_hf 액션을 사용할 때 필요합니다.
--model_name, --save_format, --save_path는 필요에 따라 조정할 수 있습니다.


[예시 1: 모델 파인튜닝]
python 2_Instruction_tuned.py --action train --model_name saltlux/luxia-21.4b-alignment-v1.1

[예시 2: 로컬에 모델 저장]
python 2_Instruction_tuned.py --action save_local --save_format lora --save_path ./my_model

[예시 3: Hugging Face에 모델 업로드]
python 2_Instruction_tuned.py --action upload_hf --token your_huggingface_token --save_path ./my_model
'''



import argparse
from unsloth import FastLanguageModel, get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer
import torch


# 명령줄 인자 처리를 위한 함수
def parse_args():
    parser = argparse.ArgumentParser(description="모델 파인튜닝, 저장, 로딩 및 Hugging Face에 업로드하는 스크립트")
    parser.add_argument("--action", choices=['train', 'save_local', 'load', 'upload_hf', 'save_gguf'], help="수행할 작업을 지정합니다.")
    parser.add_argument("--model_name", type=str, default="saltlux/luxia-21.4b-alignment-v1.1", help="사용할 모델의 이름입니다.")
    parser.add_argument("--token", type=str, default="", help="Hugging Face 토큰입니다.")
    parser.add_argument("--save_format", choices=['lora', 'merged_16bit', 'merged_4bit', 'q4_k_m'], default='merged_16bit', help="저장할 모델의 형식을 지정합니다.")
    parser.add_argument("--save_path", type=str, default="lora_model", help="모델을 저장할 경로입니다.")
    return parser.parse_args()


# GPU 사용 가능 여부를 확인하고 출력하는 함수
def check_gpu():
    if torch.cuda.is_available():
        print("GPU를 사용할 수 있습니다.")
    else:
        print("GPU가 없습니다. CPU를 사용합니다.")


# 모델 학습 함수
def train_model(args):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    dataset = load_dataset("philschmid/guanaco-sharegpt-style", split="train")
    # 여기에 데이터 포매팅 및 트레이너 설정을 추가하세요.


# 모델 저장 함수
def save_model(args, model, tokenizer):
    if args.save_format == 'lora':
        model.save_pretrained(args.save_path)
    elif args.save_format in ['merged_16bit', 'merged_4bit']:
        model.save_pretrained_merged(args.save_path, tokenizer, save_method=args.save_format)
    elif args.save_format == 'q4_k_m':
        model.save_pretrained_gguf(args.save_path, tokenizer, quantization_method=args.save_format)
    else:
        raise ValueError("지원하지 않는 저장 형식입니다.")


# Hugging Face에 모델 업로드 함수
def upload_to_huggingface(args, model, tokenizer):
    if args.token == "":
        raise ValueError("Hugging Face 토큰이 필요합니다.")
    # 업로드 로직 구현


# 메인 함수
def main():
    args = parse_args()
    
    if args.action == 'train':
        train_model(args)
    elif args.action in ['save_local', 'upload_hf', 'save_gguf']:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        if args.action == 'save_local':
            save_model(args, model, tokenizer)
        elif args.action == 'upload_hf':
            upload_to_huggingface(args, model, tokenizer)
        elif args.action == 'save_gguf':
            save_model(args, model, tokenizer)
    else:
        print("정의되지 않은 작업입니다.")


if __name__ == "__main__":
    main()
