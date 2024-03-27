# Efficient-Fientuner

## 소개
Efficient-Finetuner는 LLM의 파인튜닝에 대한 다양한 고급 기능을 쉽고 효율적으로 활용할 수 있습니다.
* 파인튜닝을 완료한 이후, LLM 모델을 사용자가 원하는 형식으로 저장할 수 있습니다.
* (참고): Easy-Finetuner는 초심자, Efficient-Finetuner는 실무자를 위한 레포지토리입니다.

---


# Requirement
## Mandatory

|                 | Minimum | Recommend |
|-----------------|---------|-----------|
| **python**      | 3.8     | 3.10      |
| **torch**       | 1.13.1  | 2.2.0     |
| **transformers**| 4.37.2  | 4.39.1    |
| **datasets**    | 2.14.3  | 2.17.1    |
| **accelerate**  | 0.27.2  | 0.28.0    |
| **peft**        | 0.9.0   | 0.10.0    |
| **trl**         | 0.8.1   | 0.8.1     |

## Optional

|                 | Minimum | Recommend |
|-----------------|---------|-----------|
| **CUDA**        | 11.6    | 12.2      |
| **deepspeed**   | 0.10.0  | 0.14.0    |
| **bitsandbytes**| 0.39.0  | 0.43.0    |
| **flash-attn**  | 2.3.0   | 2.5.6     |

## Hardware Requirement

*estimated*

| Method  | Bits | 7B  | 13B | 30B | 70B | 8x7B |
|---------|------|-----|-----|-----|-----|------|
| Full    | AMP  |120GB|240GB|600GB|1200GB|900GB|
| Full    | 16   |60GB |120GB|300GB|600GB |400GB|
| GaLore  | 16   |16GB |32GB |64GB |160GB |120GB|
| Freeze  | 16   |20GB |40GB |80GB |200GB |160GB|
| LoRA    | 16   |16GB |32GB |64GB |160GB |120GB|
| QLoRA   | 8    |10GB |20GB |40GB |80GB  |60GB |
| QLoRA   | 4    |6GB  |12GB |24GB |48GB  |30GB |
| QLoRA   | 2    |4GB  |8GB  |16GB |24GB  |18GB |


### Unsloth(추천)
`Unsloth` 폴더 CPU 상에서 효율적으로 rerank를 수행할 수 있는 방법론, "Flashrank"에 대해 다룹니다. GPU에 의존하지 않고도 빠르고 효율적인 reranking을 가능하게 하는 다양한 기술과 최적화 방법이 포함되어 있으며, 특히 대규모 데이터셋에서의 성능 향상에 초점을 맞추고 있습니다.


