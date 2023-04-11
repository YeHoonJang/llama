# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    # =========================================
    local_rank: int,    # for parallel process
    world_size: int,    # for parallel process
    # =========================================
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()

    # =========================================
    # path에서 checkpoint를 가져오기 위한 process
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]     # 병렬 처리
    # =========================================

    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # params.json -> 모델 사이즈 별로 하이퍼파라미터 지정 ex) dim, n_heads, n_layers...
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    # =========================================
    # tokenizer part
    tokenizer = Tokenizer(model_path=tokenizer_path)
    # -1로 초기화 되어 있는 vocab_size에 tokenizer를 통해 나온 n_words를 할당
    model_args.vocab_size = tokenizer.n_words
    # =========================================

    torch.set_default_tensor_type(torch.cuda.HalfTensor)

    # =========================================
    # model setting part
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    # =========================================

    # =========================================
    # generate part
    generator = LLaMA(model, tokenizer)
    # =========================================

    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
):
    # =========================================
    # 병렬 처리 세팅
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")
    # =========================================

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        "Building a website can be done in 10 simple steps:\n",
        # Few shot prompts: https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api
        """Tweet: "I hate it when my phone battery dies."
Sentiment: Negative
###
Tweet: "My day has been 👍"
Sentiment: Positive
###
Tweet: "This is the link to the article"
Sentiment: Neutral
###
Tweet: "This new music video was incredibile"
Sentiment:""",
        """Translate English to French:

sea otter => loutre de mer

peppermint => menthe poivrée

plush girafe => girafe peluche

cheese =>""",
    ]
    # =========================================
    # generator에 prompt를 입력한 결과값
    # temperature -> 무작위성을 조절하는 하이퍼파라미터
    # 낮은 온도는 보수적으로 예측 가능한 출력을 생성 하고 모델이 확률이 높은 단어를 선택 하는 경향
    # 높은 온도는 더 다양 하고 예측 하지 못하는 출력을 생성 하고 대안 옵션을 더 탐색
    #
    results = generator.generate(
        prompts, max_gen_len=256, temperature=temperature, top_p=top_p
    )
    #
    # =========================================

    for result in results:
        print(result)
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
