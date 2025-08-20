# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import List, Optional

import torch
import transformer_lens
import transformers
from fancy_einsum import einsum
from jaxtyping import Float, Int
from typeguard import typechecked
import streamlit as st

from llm_transparency_tool.models.transparent_llm import ModelInfo, TransparentLlm
from transformer_lens.loading_from_pretrained import MODEL_ALIASES, get_official_model_name


@dataclass
class _RunInfo:
    tokens: Int[torch.Tensor, "batch pos"]
    logits: Float[torch.Tensor, "batch pos d_vocab"]
    cache: transformer_lens.ActivationCache


@st.cache_resource(
    max_entries=1,
    show_spinner=True,
    hash_funcs={
        transformers.PreTrainedModel: id,
        transformers.PreTrainedTokenizer: id
    }
)
def load_hooked_transformer(
    model_name: str,
    hf_model: Optional[transformers.PreTrainedModel] = None,
    tlens_device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    supported_model_name: Optional[str] = None,
):
    if supported_model_name is None:
        supported_model_name = model_name
    supported_model_name = get_official_model_name(supported_model_name)
    if model_name not in MODEL_ALIASES:
        MODEL_ALIASES[supported_model_name] = []
    if model_name not in MODEL_ALIASES[supported_model_name]:
        MODEL_ALIASES[supported_model_name].append(model_name)
    tlens_model = transformer_lens.HookedTransformer.from_pretrained(
        model_name,
        hf_model=hf_model,
        fold_ln=False,  # Keep layer norm where it is.
        center_writing_weights=False,
        center_unembed=False,
        device=tlens_device,
        dtype=dtype,
    )
    tlens_model.eval()
    return tlens_model


# TODO(igortufanov): If we want to scale the app to multiple users, we need more careful
# thread-safe implementation. The simplest option could be to wrap the existing methods
# in mutexes.
class TransformerLensTransparentLlm(TransparentLlm):
    """
    Implementation of Transparent LLM based on transformer lens.

    Args:
    - model_name: The official name of the model from HuggingFace. Even if the model was
        patched or loaded locally, the name should still be official because that's how
        transformer_lens treats the model.
    - hf_model: The language model as a HuggingFace class.
    - tokenizer,
    - device: "gpu" or "cpu"
    """

    def __init__(
        self,
        model_name: str,
        hf_model: Optional[transformers.PreTrainedModel] = None,
        tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
        device: str = "gpu",
        dtype: torch.dtype = torch.float32,
        supported_model_name: str = None,
        first: bool = True
    ):
        if device == "gpu":
            self.device = "cuda"
            if not torch.cuda.is_available():
                RuntimeError("Asked to run on gpu, but torch couldn't find cuda")
        elif device == "cpu":
            self.device = "cpu"
        else:
            raise RuntimeError(f"Specified device {device} is not a valid option")

        self.dtype = dtype
        self.hf_tokenizer = tokenizer
        self.hf_model = hf_model

        # self._model = tlens_model
        self._model_name = model_name
        self._supported_model_name = supported_model_name
        self._prepend_bos = False
        self._last_run = None
        self._run_exception = RuntimeError(
            "Tried to use the model output before calling the `run` method"
        )
        
        self._use_hf_chat_template: bool = True
        self._manual_inst_brackets: bool = False
        self._add_generation_prompt: bool = True
        self._system_prompt: Optional[str] = None
        self._strip_post_instruction: bool = False
        self._last_post_instruction_suffix_ids: List[int] = []
            
        


    def copy(self):
        import copy
        return copy.copy(self)

    @property
    def _model(self):
        tlens_model = load_hooked_transformer(
            self._model_name,
            hf_model=self.hf_model,
            tlens_device=self.device,
            dtype=self.dtype,
            supported_model_name=self._supported_model_name,
        )

        if self.hf_tokenizer is not None:
            tlens_model.set_tokenizer(self.hf_tokenizer, default_padding_side="left")

        tlens_model.set_use_attn_result(True)
        tlens_model.set_use_attn_in(False)
        tlens_model.set_use_split_qkv_input(False)

        return tlens_model

    def model_info(self) -> ModelInfo:
        cfg = self._model.cfg
        return ModelInfo(
            name=self._model_name,
            n_params_estimate=cfg.n_params,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            d_model=cfg.d_model,
            d_vocab=cfg.d_vocab,
        )

    @torch.no_grad()
    def run(self, sentences: List[str]) -> None:
        # Optionally wrap raw user strings with a chat / instruction template
        prepared = self._prepare_inputs(sentences)
        tokens = self._model.to_tokens(prepared, prepend_bos=self._prepend_bos)
        logits, cache = self._model.run_with_cache(tokens)

        self._last_run = _RunInfo(
            tokens=tokens,
            logits=logits,
            cache=cache,
        )

    def batch_size(self) -> int:
        if not self._last_run:
            raise self._run_exception
        return self._last_run.logits.shape[0]

    @typechecked
    def tokens(self) -> Int[torch.Tensor, "batch pos"]:
        if not self._last_run:
            raise self._run_exception
        return self._last_run.tokens

    @typechecked
    def tokens_to_strings(self, tokens: Int[torch.Tensor, "pos"]) -> List[str]:
        return self._model.to_str_tokens(tokens)

    @typechecked
    def logits(self) -> Float[torch.Tensor, "batch pos d_vocab"]:
        if not self._last_run:
            raise self._run_exception
        return self._last_run.logits

    @torch.no_grad()
    @typechecked
    def unembed(
        self,
        t: Float[torch.Tensor, "d_model"],
        normalize: bool,
    ) -> Float[torch.Tensor, "vocab"]:
        # t: [d_model] -> [batch, pos, d_model]
        tdim = t.unsqueeze(0).unsqueeze(0)
        if normalize:
            normalized = self._model.ln_final(tdim)
            result = self._model.unembed(normalized)
        else:
            result = self._model.unembed(tdim.to(self.dtype))
        return result[0][0]

    def _get_block(self, layer: int, block_name: str) -> torch.Tensor:
        if not self._last_run:
            raise self._run_exception
        return self._last_run.cache[f"blocks.{layer}.{block_name}"]

    # ================= Methods related to the residual stream =================

    @typechecked
    def residual_in(self, layer: int) -> Float[torch.Tensor, "batch pos d_model"]:
        if not self._last_run:
            raise self._run_exception
        return self._get_block(layer, "hook_resid_pre")

    @typechecked
    def residual_after_attn(
        self, layer: int
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        if not self._last_run:
            raise self._run_exception
        return self._get_block(layer, "hook_resid_mid")

    @typechecked
    def residual_out(self, layer: int) -> Float[torch.Tensor, "batch pos d_model"]:
        if not self._last_run:
            raise self._run_exception
        return self._get_block(layer, "hook_resid_post")

    # ================ Methods related to the feed-forward layer ===============

    @typechecked
    def ffn_out(self, layer: int) -> Float[torch.Tensor, "batch pos d_model"]:
        if not self._last_run:
            raise self._run_exception
        return self._get_block(layer, "hook_mlp_out")

    @torch.no_grad()
    @typechecked
    def decomposed_ffn_out(
        self,
        batch_i: int,
        layer: int,
        pos: int,
    ) -> Float[torch.Tensor, "hidden d_model"]:
        # Take activations right before they're multiplied by W_out, i.e. non-linearity
        # and layer norm are already applied.
        processed_activations = self._get_block(layer, "mlp.hook_post")[batch_i][pos]
        return torch.mul(processed_activations.unsqueeze(-1), self._model.blocks[layer].mlp.W_out)

    @typechecked
    def neuron_activations(
        self,
        batch_i: int,
        layer: int,
        pos: int,
    ) -> Float[torch.Tensor, "hidden"]:
        return self._get_block(layer, "mlp.hook_pre")[batch_i][pos]

    @typechecked
    def neuron_output(
        self,
        layer: int,
        neuron: int,
    ) -> Float[torch.Tensor, "d_model"]:
        return self._model.blocks[layer].mlp.W_out[neuron]

    # ==================== Methods related to the attention ====================

    @typechecked
    def attention_matrix(
        self, batch_i: int, layer: int, head: int
    ) -> Float[torch.Tensor, "query_pos key_pos"]:
        return self._get_block(layer, "attn.hook_pattern")[batch_i][head]

    @typechecked
    def attention_output_per_head(
        self,
        batch_i: int,
        layer: int,
        pos: int,
        head: int,
    ) -> Float[torch.Tensor, "d_model"]:
        return self._get_block(layer, "attn.hook_result")[batch_i][pos][head]

    @typechecked
    def attention_output(
        self,
        batch_i: int,
        layer: int,
        pos: int,
    ) -> Float[torch.Tensor, "d_model"]:
        return self._get_block(layer, "hook_attn_out")[batch_i][pos]

    @torch.no_grad()
    @typechecked
    def decomposed_attn(
        self, batch_i: int, layer: int
    ) -> Float[torch.Tensor, "pos key_pos head d_model"]:
        if not self._last_run:
            raise self._run_exception
        hook_v = self._get_block(layer, "attn.hook_v")[batch_i]
        b_v = self._model.blocks[layer].attn.b_V

        # support for gqa
        num_head_groups = b_v.shape[-2] // hook_v.shape[-2]
        hook_v = hook_v.repeat_interleave(num_head_groups, dim=-2)

        v = hook_v + b_v
        pattern = self._get_block(layer, "attn.hook_pattern")[batch_i].to(v.dtype)
        z = einsum(
            "key_pos head d_head, "
            "head query_pos key_pos -> "
            "query_pos key_pos head d_head",
            v,
            pattern,
        )
        decomposed_attn = einsum(
            "pos key_pos head d_head, "
            "head d_head d_model -> "
            "pos key_pos head d_model",
            z,
            self._model.blocks[layer].attn.W_O,
        )
        return decomposed_attn

    # ==================== Chat template helpers ====================
    def configure_chat(
        self,
        system_prompt: Optional[str] = None,
        use_hf_chat_template: bool = True,
        add_generation_prompt: bool = True,
        manual_inst_brackets: bool = False,
    ) -> None:
        """Configure automatic prompt templating for instruction / chat models.

        If ``use_hf_chat_template`` is True and the tokenizer exposes
        ``apply_chat_template`` (recent HF tokenizers for Llama / Mistral etc.),
        inputs passed to ``run`` will be converted from plain user strings to
        a chat-formatted prompt composed of (optional) system + user messages.

        Fallbacks:
          * If HF chat template unavailable or disabled, and ``manual_inst_brackets``
            is True, each sentence becomes ``[INST] <system?> <user> [/INST]``.
          * Otherwise sentences are used as-is (default original behaviour).
        """
        self._system_prompt = system_prompt
        self._use_hf_chat_template = use_hf_chat_template
        self._add_generation_prompt = add_generation_prompt
        self._manual_inst_brackets = manual_inst_brackets

    # ---------------------- Post-instruction control ----------------------
    def set_strip_post_instruction(self, value: bool) -> None:
        """Enable/disable removal of post-instruction tokens inside the chat prompt.

        When enabled and using an HF chat template with generation prompt, we build
        two variants of the prompt (with & without generation prompt), compute the
        suffix token ids (assistant header etc.), store them for inspection, and
        return only the *without* version to the model so that the last token is the
        instruction terminator (<eot_id> in Llama3). This lets downstream code treat
        that as t_inst without manual sequence slicing.
        """
        self._strip_post_instruction = value

    def post_instruction_suffix_ids(self) -> List[int]:
        """Return the token ids that constitute the last extracted post-instruction
        suffix for the most recent prepared prompt (empty if none / not applicable)."""
        return self._last_post_instruction_suffix_ids

    def _prepare_inputs(self, sentences: List[str]) -> List[str]:
        """Return possibly templated prompt strings.

        The original behaviour (no modification) is preserved unless
        ``configure_chat`` has been called to enable a mode.
        """
        # Fast path: nothing configured
        if not (self._use_hf_chat_template or self._manual_inst_brackets):
            return sentences

        prepared: List[str] = []
        can_use_hf = (
            self._use_hf_chat_template
            and self.hf_tokenizer is not None
            and hasattr(self.hf_tokenizer, "apply_chat_template")
        )
        for s in sentences:
            if can_use_hf:
                messages = []
                if self._system_prompt:
                    messages.append({"role": "system", "content": self._system_prompt})
                messages.append({"role": "user", "content": s})
                # HF tokenizer returns a string when tokenize=False. We optionally
                # construct both variants to identify & optionally remove the
                # post-instruction suffix (assistant header) tokens.
                prompt_with_gen = self.hf_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                ) if self._add_generation_prompt else None

                prompt_without_gen = self.hf_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )

                if self._add_generation_prompt and prompt_with_gen is not None:
                    # Compute token id difference (suffix) between with_gen and without_gen.
                    ids_full = self.hf_tokenizer.encode(
                        prompt_with_gen, add_special_tokens=False
                    )
                    ids_no = self.hf_tokenizer.encode(
                        prompt_without_gen, add_special_tokens=False
                    )
                    # Guard: ensure without_gen is prefix of with_gen
                    if len(ids_no) <= len(ids_full) and ids_full[: len(ids_no)] == ids_no:
                        suffix = ids_full[len(ids_no) :]
                    else:
                        suffix = []  # fallback (unexpected template change)
                    # Optionally also strip the terminal <eot_id> token which belongs
                    # to post-instruction region per your definition (blue segment).
                    if self._strip_post_instruction:
                        eot_id = None
                        try:
                            eot_id = self.hf_tokenizer.convert_tokens_to_ids("<|eot_id|>")
                        except Exception:
                            eot_id = None
                        if eot_id is not None and len(ids_no) > 0 and ids_no[-1] == eot_id:
                            # Prepend eot to suffix list; remove it from base sequence.
                            suffix = [eot_id] + suffix
                            ids_no = ids_no[:-1]
                            # Reconstruct prompt_without_gen without the eot token.
                            prompt_without_gen = self.hf_tokenizer.decode(
                                ids_no, skip_special_tokens=False
                            )
                    self._last_post_instruction_suffix_ids = suffix
                    if self._strip_post_instruction:
                        # Use the version without generation prompt.
                        prompt = prompt_without_gen
                    else:
                        prompt = prompt_with_gen
                else:
                    # Generation prompt disabled globally; no suffix.
                    ids_no = self.hf_tokenizer.encode(
                        prompt_without_gen, add_special_tokens=False
                    )
                    suffix = []
                    if self._strip_post_instruction:
                        # Strip <eot_id> even when no generation prompt.
                        eot_id = None
                        try:
                            eot_id = self.hf_tokenizer.convert_tokens_to_ids("<|eot_id|>")
                        except Exception:
                            eot_id = None
                        if eot_id is not None and len(ids_no) > 0 and ids_no[-1] == eot_id:
                            suffix = [eot_id]
                            ids_no = ids_no[:-1]
                            prompt_without_gen = self.hf_tokenizer.decode(
                                ids_no, skip_special_tokens=False
                            )
                    self._last_post_instruction_suffix_ids = suffix
                    prompt = prompt_without_gen

                prepared.append(prompt)
            elif self._manual_inst_brackets:
                sys_part = (self._system_prompt + " ") if self._system_prompt else ""
                prepared.append(f"[INST] {sys_part}{s} [/INST]")
            else:
                prepared.append(s)
        return prepared
