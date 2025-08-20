# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from llm_transparency_tool.models.tlens_model import TransformerLensTransparentLlm
from llm_transparency_tool.models.transparent_llm import ModelInfo


class TransparentLlmTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Picking the smallest model possible so that the test runs faster. It's ok to
        # change this model, but you'll need to update tokenization specifics in some
        # tests.
        cls._llm = TransformerLensTransparentLlm(
            model_name="EleutherAI/pythia-14m",
            device="cpu",
        )

    def setUp(self):
        self._llm.run(["test", "test 1"])
        self._eps = 1e-5

    def test_model_info(self):
        info = self._llm.model_info()
        self.assertEqual(
            info,
            ModelInfo(
                name="EleutherAI/pythia-14m",
                n_params_estimate=1179648,
                n_layers=6,
                n_heads=4,
                d_model=128,
                d_vocab=50304,
            ),
        )

    def test_tokens(self):
        tokens = self._llm.tokens()
    # Pythia-14M (GPTNeoX) tokenizes inputs into two tokens each here.
    # "test" -> [2566, 0]; "test 1" -> [2566, 337]
    self.assertEqual(tokens.tolist(), [[2566, 0], [2566, 337]])

    def test_tokens_to_strings(self):
    seq = torch.tensor([2566, 0], dtype=torch.int)
    s = self._llm.tokens_to_strings(seq)
    self.assertEqual(len(s), 2)
    self.assertIn("test", s[0])

    def test_manage_state(self):
        # One llm.run was called at the setup. Call one more and make sure the object
        # returns values for the new state.
        self._llm.run(["one", "two", "three", "four"])
        self.assertEqual(self._llm.tokens().shape[0], 4)

    def test_residual_in_and_out(self):
        """
        Test that residual_in is a residual_out for the previous layer.
        """
    for layer in range(1, 6):
            prev_residual_out = self._llm.residual_out(layer - 1)
            residual_in = self._llm.residual_in(layer)
            diff = torch.max(torch.abs(residual_in - prev_residual_out)).item()
            self.assertLess(diff, self._eps, f"layer {layer}")

    def test_residual_plus_block(self):
        """
        Make sure that new residual = old residual + block output. Here, block is an ffn
        or attention. It's not that obvious because it could be that layer norm is
        applied after the block output, but before saving the result to residual.
        Luckily, this is not the case in TransformerLens, and we're relying on that.
        """
        layer = 3
        batch = 0
        pos = 0

        residual_in = self._llm.residual_in(layer)[batch][pos]
        residual_mid = self._llm.residual_after_attn(layer)[batch][pos]
        residual_out = self._llm.residual_out(layer)[batch][pos]
        ffn_out = self._llm.ffn_out(layer)[batch][pos]
        attn_out = self._llm.attention_output(batch, layer, pos)

        a = residual_mid
        b = residual_in + attn_out
        diff = torch.max(torch.abs(a - b)).item()
        self.assertLess(diff, self._eps, "attn")

        a = residual_out
        b = residual_mid + ffn_out
        diff = torch.max(torch.abs(a - b)).item()
        self.assertLess(diff, self._eps, "ffn")

    def test_tensor_shapes(self):
        # Not much we can do about the tensors, but at least check their shapes and
        # that they don't contain NaNs.
    vocab_size = 50304
    n_batch = 2
    n_tokens = 2
    d_model = 128
    d_hidden = 512  # intermediate_size from config
    n_heads = 4
    layer = 3

        device = self._llm.residual_in(0).device

        for name, tensor, expected_shape in [
            ("r_in", self._llm.residual_in(layer), [n_batch, n_tokens, d_model]),
            (
                "r_mid",
                self._llm.residual_after_attn(layer),
                [n_batch, n_tokens, d_model],
            ),
            ("r_out", self._llm.residual_out(layer), [n_batch, n_tokens, d_model]),
            ("logits", self._llm.logits(), [n_batch, n_tokens, vocab_size]),
            ("ffn_out", self._llm.ffn_out(layer), [n_batch, n_tokens, d_model]),
            (
                "decomposed_ffn_out",
                self._llm.decomposed_ffn_out(0, 0, 0),
                [d_hidden, d_model],
            ),
            ("neuron_activations", self._llm.neuron_activations(0, 0, 0), [d_hidden]),
            ("neuron_output", self._llm.neuron_output(0, 0), [d_model]),
            (
                "attention_matrix",
                self._llm.attention_matrix(0, 0, 0),
                [n_tokens, n_tokens],
            ),
            (
                "attention_output_per_head",
                self._llm.attention_output_per_head(0, 0, 0, 0),
                [d_model],
            ),
            (
                "attention_output",
                self._llm.attention_output(0, 0, 0),
                [d_model],
            ),
            (
                "decomposed_attn",
                self._llm.decomposed_attn(0, layer),
                [n_tokens, n_tokens, n_heads, d_model],
            ),
            (
                "unembed",
                self._llm.unembed(torch.zeros([d_model]).to(device), normalize=True),
                [vocab_size],
            ),
        ]:
            self.assertEqual(list(tensor.shape), expected_shape, name)
            self.assertFalse(torch.any(tensor.isnan()), name)


if __name__ == "__main__":
    unittest.main()
