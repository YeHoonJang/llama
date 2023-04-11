# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

import torch

from llama.tokenizer import Tokenizer
from llama.model import Transformer


class LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:

        # prompt가 input 이니까 prompt의 size 받아서 bsz에 저장
        bsz = len(prompts)

        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)   # 조건, 메시지

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        # 일단 (bsz, total_len) 사이즈의 tensor를 pad token으로 초기화
        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()

        # token 채우기
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()

        # token이 있는 (pad token)이 아닌 인덱스는 True, pad token이면 False
        input_text_mask = tokens != self.tokenizer.pad_id

        start_pos = min_prompt_size
        prev_pos = 0

        # 문장 생성 part
        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)

            # temperature 적용할지 말지 -> temperature softmax 식 같이 봐
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            # where: condition, x, y -> condition이 true(nonzero)이면 x, 그렇지 않으면 y
            # input_text_mask가 0이면 next_token을 채우고 그렇지 않으면 그 전의 값 그대로 두기
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token

            # token 인덱스 업데이트
            prev_pos = cur_pos

        # token generation 끝
        # =========================================
        # token을 문자로 decoding 시작
        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)    # 누적합
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
