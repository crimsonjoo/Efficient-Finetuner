"""
명령줄 인자 설명:
    --finetune: 모델을 파인튜닝합니다.
    --save_model: 모델을 저장합니다.
    --upload_huggingface: 모델을 Hugging Face에 업로드합니다.
    --model_name: 사용할 모델의 이름입니다.
    --dataset: 파인튜닝에 사용할 데이터셋입니다.
    --output_dir: 모델을 저장할 디렉토리 경로입니다.
    --model_dir: 업로드할 모델이 위치한 디렉토리입니다.
    --hf_username: Hugging Face의 사용자 이름입니다.
    --hf_token: Hugging Face의 API 토큰입니다.


[예시 1: 모델 파인튜닝]
    python 1_Pretrained.py --finetune --model_name "unsloth/mistral-7b-bnb-4bit" --dataset "roneneldan/TinyStories"

[예시 2: 로컬에 모델 저장]
    python 1_Pretrained.py --save_model --output_dir "./saved_models"

[예시 3: Hugging Face에 모델 업로드]
    python 1_Pretrained.py --upload_huggingface --model_dir "./saved_models" --hf_username "your_username" --hf_token "your_token"

"""

# 필요한 라이브러리를 불러옵니다.
import argparse
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

def main():
    # 명령줄 인자를 파싱하기 위한 ArgumentParser 객체를 생성합니다.
    parser = argparse.ArgumentParser(description="모델 파인튜닝, 저장 및 Hugging Face에 업로드하는 스크립트")
    
    # 사용자로부터 입력받을 명령줄 인자들을 정의합니다.
    parser.add_argument('--finetune', action='store_true', help='모델을 파인튜닝합니다.')
    parser.add_argument('--save_model', action='store_true', help='모델을 저장합니다.')
    parser.add_argument('--upload_huggingface', action='store_true', help='모델을 Hugging Face에 업로드합니다.')
    parser.add_argument('--model_name', type=str, help='사용할 모델의 이름입니다.')
    parser.add_argument('--dataset', type=str, help='파인튜닝에 사용할 데이터셋입니다.')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='모델을 저장할 디렉토리 경로입니다.')
    parser.add_argument('--model_dir', type=str, help='업로드할 모델이 위치한 디렉토리입니다.')
    parser.add_argument('--hf_username', type=str, help='Hugging Face의 사용자 이름입니다.')
    parser.add_argument('--hf_token', type=str, help='Hugging Face의 API 토큰입니다.')

    # 파싱한 인자들을 args 변수에 저장합니다.
    args = parser.parse_args()

    # 모델을 파인튜닝하는 경우
    if args.finetune:
        print("모델 파인튜닝을 시작합니다.")
        
        # 주어진 데이터셋을 불러옵니다.
        dataset = load_dataset(args.dataset, split='train[:5000]')

        # 모델과 토크나이저를 불러옵니다.
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True
        )

        # 파인튜닝을 위한 Trainer를 설정합니다.
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            dataset_text_field="text",
            tokenizer=tokenizer,
            max_seq_length=2048,
            packing=True,
            formatting_func=lambda example: example["text"] + tokenizer.eos_token,
            args=TrainingArguments(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_ratio=0.05,
                max_grad_norm=1.0,
                num_train_epochs=1,
                learning_rate=2e-5,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.1,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir=args.output_dir,
            )
        )

        # 모델을 파인튜닝합니다.
        trainer.train()

        print("모델 파인튜닝이 완료되었습니다.")

    # 모델을 저장하는 경우
    if args.save_model:
        print(f"모델을 {args.output_dir}에 저장합니다.")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    # 모델을 Hugging Face에 업로드하는 경우
    if args.upload_huggingface:
        print("모델을 Hugging Face에 업로드합니다.")
        # Hugging Face에 업로드하는 로직을 추가합니다.
        # 예시: model.push_to_hub(args.model_dir, use_auth_token=args.hf_token)

if __name__ == "__main__":
    main()