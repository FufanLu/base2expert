"""verl's MultiTurnSFTDataset with system prompts that Qwen3.5's chat template accepts.

verl renders every message of a conversation on its own and concatenates the
tokens (verl/utils/dataset/multiturn_sft_dataset.py, _process_single_message).
Qwen3.5's template refuses a message list without a user message ("No user
query found in messages."), so verl's apply_chat_template retries with an empty
user message in front (verl/utils/tokenizer/chat_template.py). For a system
message that puts it second, and the template raises "System message must be
at the beginning." (SFT smoke 11378546, first batch of the law mix: the
smoltalk system-chat rows carry a system prompt). Here a system message is
rendered with an empty user message after it instead, and that message's
tokens are cut off the end. Every other message goes through verl unchanged,
and verl's sanity_check still compares the concatenated turns with the whole
conversation rendered at once, so a wrong cut fails the row instead of
training on it.
"""

from __future__ import annotations

import torch
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset


class SystemPromptSFTDataset(MultiTurnSFTDataset):
    def _process_single_message(self, index, message, full_message, tools=None, enable_thinking=None):
        if message["role"] != "system":
            return super()._process_single_message(index, message, full_message, tools, enable_thinking)
        processor = self.processor if self.processor is not None else self.tokenizer
        kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        # same content form as the message (_build_messages turns text into a list when there is a processor)
        user = {"role": "user", "content": [{"type": "text", "text": ""}] if isinstance(message["content"], list) else ""}

        def render(msgs, tools):
            return dict(processor.apply_chat_template(msgs, tools=tools, add_generation_prompt=False, tokenize=True,
                                                      return_dict=True, return_tensors="pt", **kwargs))

        both, tail = render([message, user], tools), render([user], None)
        n = tail["input_ids"].shape[1]
        if not torch.equal(both["input_ids"][0, -n:], tail["input_ids"][0]):
            raise ValueError("the empty user message is not a clean suffix of the system message's rendering")
        input_ids = both.pop("input_ids")[0][:-n]
        attention_mask = both.pop("attention_mask")[0][:-n]
        if "mm_token_type_ids" in both:
            both["mm_token_type_ids"] = both["mm_token_type_ids"][:, :-n]
        return input_ids, torch.zeros_like(attention_mask), attention_mask, both
