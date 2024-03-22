"""
명령줄 인자 설명:
  --task: 수행할 작업을 지정합니다. 'finetune', 'save', 'upload' 등을 선택할 수 있습니다.
  --model_name: 사용할 모델의 이름을 지정합니다. 이는 파인튜닝 작업에 사용됩니다.
  --dataset_name: 파인튜닝에 사용할 데이터셋의 이름을 지정합니다.
  --model_path: 모델이 저장된 경로를 지정합니다. 이는 모델을 저장하거나 업로드할 때 사용됩니다.
  --hf_username: Hugging Face에 업로드할 때 사용할 사용자 이름입니다.
  --hf_token: Hugging Face에 업로드할 때 사용할 토큰입니다.



[예시 1: 모델 파인튜닝]
  python 3_RL_tuned.py --task finetune --model_name your_model_name_here --dataset_name your_dataset_name_here

[예시 2: 로컬에 모델 저장]
  python 3_RL_tuned.py --task save --model_path your_model_path_here

[예시 3: Hugging Face에 모델 업로드]
  python 3_RL_tuned.py --task upload --model_path your_model_path_here --hf_username your_hf_username_here --hf_token your_hf_token_here

"""


import argparse
import torch
from unsloth import FastLanguageModel, PatchDPOTrainer
from transformers import TrainingArguments, TextStreamer
from trl import DPOTrainer
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="모델 파인튜닝 및 관리 스크립트")
    parser.add_argument("--action", type=str, choices=["finetune", "save", "upload"], required=True,
                        help="실행할 작업: finetune, save, upload 중 선택")
    parser.add_argument("--model_name", type=str, default="unsloth/zephyr-sft-bnb-4bit",
                        help="사용할 모델의 이름 또는 경로")
    parser.add_argument("--max_seq_length", type=int, default=4096,
                        help="모델에 사용할 최대 시퀀스 길이")
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16", "None"], default="None",
                        help="데이터 타입 설정. Tesla T4, V100의 경우 'float16', Ampere+의 경우 'bfloat16'")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="4비트 양자화를 사용하여 메모리 사용량 줄이기. 기본값은 False.")
    parser.add_argument("--finetune_epochs", type=int, default=3,
                        help="파인튜닝할 때의 에포크 수")
    parser.add_argument("--save_path", type=str, default="./model",
                        help="모델을 저장할 경로")
    parser.add_argument("--hub_name", type=str,
                        help="Hugging Face Hub에 업로드할 때 사용할 모델 이름")
    parser.add_argument("--hub_token", type=str,
                        help="Hugging Face Hub에 업로드할 때 필요한 토큰")
    
    args = parser.parse_args()

    # 데이터 타입 설정
    dtype = None if args.dtype == "None" else args.dtype
    
    # 모델 및 토크나이저 불러오기
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=args.load_in_4bit,
    )

    if args.action == "finetune":
        finetune(model, tokenizer, args)
    elif args.action == "save":
        save_model(model, args)
    elif args.action == "upload":
        upload_model(model, tokenizer, args)
    else:
        raise ValueError("지원하지 않는 작업입니다.")

def finetune(model, tokenizer, args):
    """
    모델을 파인튜닝하는 함수입니다. 파라미터에 따라 DPOTrainer를 사용하여 모델을 학습시킵니다.
    """
    # 필요한 패치 적용
    PatchDPOTrainer()
    
    # 트레이닝 인수 설정
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        num_train_epochs=args.finetune_epochs,
        learning_rate=5e-6,
        fp16=dtype == "float16",
        bf16=dtype == "bfloat16",
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.0,
        lr_scheduler_type="linear",
        seed=42,
        output_dir=args.save_path,
    )
    
    # DPOTrainer 인스턴스 생성 및 훈련 시작
    dpo_trainer = DPOTrainer(
        model=model,
        args=training_args,
        beta=0.1,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        # 추가적인 설정 필요시 여기에 추가
    )
    
    # 학습 시작
    dpo_trainer.train()

def save_model(model, args):
    """
    모델을 지정된 경로에 저장합니다.
    """
    model.save_pretrained(args.save_path)
    print(f"모델이 {args.save_path}에 저장되었습니다.")

def upload_model(model, tokenizer, args):
    """
    모델을 Hugging Face Hub에 업로드합니다.
    """
    if args.hub_name is None or args.hub_token is None:
        print("Hugging Face Hub에 업로드하기 위해서는 모델 이름과 토큰이 필요합니다.")
        sys.exit(1)
    
    model.push_to_hub(args.hub_name, use_auth_token=args.hub_token)
    tokenizer.push_to_hub(args.hub_name, use_auth_token=args.hub_token)
    print(f"모델이 {args.hub_name}으로 Hugging Face Hub에 업로드되었습니다.")

if __name__ == "__main__":
    main()