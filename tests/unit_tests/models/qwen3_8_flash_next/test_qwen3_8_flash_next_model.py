# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU execution tests for the Qwen3.8-Flash-Next HC decoder."""

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_8_flash_next.config import (
    Qwen3_8_FlashNextConfig,
    Qwen3_8_FlashNextTextConfig,
)
from nemo_automodel.components.models.qwen3_8_flash_next.engram import Qwen3_8_FlashNextEngramTableConfig
from nemo_automodel.components.models.qwen3_8_flash_next.model import (
    ModelClass,
    Qwen3_8_FlashNextForConditionalGeneration,
)
from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.components.speculative.dspark.config import build_draft_config
from nemo_automodel.components.speculative.dspark.draft_qwen3 import Qwen3DSparkModel
from nemo_automodel.components.speculative.dspark.registry import resolve_dspark_draft_spec
from nemo_automodel.components.speculative.dspark.target import HFDSparkTargetModel


def _tiny_config() -> Qwen3_8_FlashNextConfig:
    text = Qwen3_8_FlashNextTextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=["full_attention"],
        full_attention_interval=1,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=2,
        num_experts_per_tok=1,
        hc_count=2,
        hc_lowrank=4,
        ple_layer_ids=[],
        indexer_budget=8,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        # The main rotary width is 8 here, so the index head must be at least
        # that wide just like the released 64-rope/128-index configuration.
        indexer_head_dim=8,
        max_position_embeddings=16,
        rope_parameters={
            "rope_theta": 10000.0,
            "rope_type": "default",
            "partial_rotary_factor": 1.0,
        },
        partial_rotary_factor=1.0,
        dtype="float32",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=1,
        tie_word_embeddings=False,
    )
    return Qwen3_8_FlashNextConfig(text_config=text, language_model_only=True, tie_word_embeddings=False)


def _tiny_moe_config(config: Qwen3_8_FlashNextTextConfig) -> MoEConfig:
    return MoEConfig(
        dim=config.hidden_size,
        inter_dim=config.hidden_size,
        moe_inter_dim=config.moe_intermediate_size,
        n_routed_experts=config.num_experts,
        n_shared_experts=1,
        n_activated_experts=config.num_experts_per_tok,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=config.router_aux_loss_coef,
        norm_topk_prob=True,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        softmax_before_topk=True,
        shared_expert_gate=True,
        shared_expert_inter_dim=config.shared_expert_intermediate_size,
        dtype=torch.float32,
    )


def test_model_class_alias_selects_qwen3_8_flash_next_conditional_generation() -> None:
    assert ModelClass is Qwen3_8_FlashNextForConditionalGeneration


def test_multimodal_configuration_fails_closed() -> None:
    config = _tiny_config()
    config.language_model_only = False

    with pytest.raises(NotImplementedError, match="currently language-only"):
        Qwen3_8_FlashNextForConditionalGeneration.from_config(
            config,
            moe_config=_tiny_moe_config(config.text_config),
            backend=BackendConfig(enable_hf_state_dict_adapter=False),
        )


def test_tiny_qwen3_8_flash_next_forward_backward_and_state_layout() -> None:
    config = _tiny_config()
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
    )
    model = Qwen3_8_FlashNextForConditionalGeneration.from_config(
        config,
        moe_config=_tiny_moe_config(config.text_config),
        backend=backend,
    )
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    model.train()

    input_ids = torch.randint(2, config.text_config.vocab_size, (2, 6))
    output = model(input_ids=input_ids, output_hidden_states=True)
    loss = output.logits.square().mean()
    loss.backward()

    assert output.logits.shape == (2, 6, config.text_config.vocab_size)
    assert output.hidden_states[0].shape == (2, 6, config.text_config.hidden_size)
    assert output.hidden_states[1].shape == (
        2,
        6,
        config.text_config.hc_count * config.text_config.hidden_size,
    )
    assert output.hidden_states[-1].shape == (2, 6, config.text_config.hidden_size)
    assert torch.isfinite(output.logits).all()
    assert model.get_input_embeddings().weight.grad is not None

    keys = model.state_dict()
    prefix = "model.language_model.layers.0"
    assert f"{prefix}.attn_hyper_connection.hc_norm.weight" in keys
    assert f"{prefix}.attn_hyper_connection.block_inject_weight.weight" in keys
    assert f"{prefix}.self_attn.indexer.index_qk_proj.weight" in keys
    assert f"{prefix}.input_layernorm.weight" not in keys
    assert "model.language_model.hyper_connection_mixer.block_inject_weight.weight" not in keys


def test_output_hidden_states_defaults_to_model_config() -> None:
    config = _tiny_config()
    config.output_hidden_states = True
    model = Qwen3_8_FlashNextForConditionalGeneration.from_config(
        config,
        moe_config=_tiny_moe_config(config.text_config),
        backend=BackendConfig(
            linear="torch",
            attn="sdpa",
            rms_norm="torch",
            experts="torch",
            dispatcher="torch",
            enable_hf_state_dict_adapter=False,
        ),
    )
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)

    output = model(input_ids=torch.randint(2, config.text_config.vocab_size, (1, 6)))

    assert isinstance(output.hidden_states, torch.Tensor)
    assert output.hidden_states.shape == (1, 6, config.text_config.hidden_size)

    explicit_output = model(
        input_ids=torch.randint(2, config.text_config.vocab_size, (1, 6)),
        output_hidden_states=True,
    )
    assert isinstance(explicit_output.hidden_states, tuple)
    assert len(explicit_output.hidden_states) == config.text_config.num_hidden_layers + 2
    assert explicit_output.hidden_states[-1].shape == (1, 6, config.text_config.hidden_size)


def test_dspark_target_capture_collapses_hyper_connection_streams() -> None:
    config = _tiny_config()
    config.text_config.num_hidden_layers = 2
    config.text_config.layer_types = ["full_attention", "full_attention"]
    model = Qwen3_8_FlashNextForConditionalGeneration.from_config(
        config,
        moe_config=_tiny_moe_config(config.text_config),
        backend=BackendConfig(
            linear="torch",
            attn="sdpa",
            rms_norm="torch",
            experts="torch",
            dispatcher="torch",
            enable_hf_state_dict_adapter=False,
        ),
    )
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    wrapper = HFDSparkTargetModel(model, target_layer_ids=[0, 1])
    input_ids = torch.randint(2, config.text_config.vocab_size, (2, 6))

    batch = wrapper.generate_batch(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        loss_mask=torch.ones_like(input_ids, dtype=torch.uint8),
    )

    assert batch.target_hidden_states.shape == (2, 6, 2 * config.text_config.hidden_size)
    assert batch.target_last_hidden_states.shape == (2, 6, config.text_config.hidden_size)
    assert torch.isfinite(batch.target_hidden_states).all()


def test_dspark_target_config_build_is_independent_and_forwards_backend(monkeypatch) -> None:
    import nemo_automodel._transformers as transformers_bridge

    outer_config = Qwen3_8_FlashNextConfig(
        text_config=Qwen3_8_FlashNextTextConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
            hc_count=2,
            ple_layer_ids=[2],
            vocab_size=128,
        )
    )
    captured = {}

    class FakeAutoModel:
        @staticmethod
        def from_config(config=None, **kwargs):
            captured["config"] = config
            captured.update(kwargs)
            return "target-model"

    monkeypatch.setattr(transformers_bridge, "NeMoAutoModelForCausalLM", FakeAutoModel)
    text_config, target_model = outer_config.build_dspark_target(
        target_path="qwen3.8-flash-next",
        distributed_setup="distributed-setup",
        device=torch.device("cuda"),
        compute_dtype=torch.bfloat16,
        target_num_hidden_layers=2,
        target_attn_backend="flex",
        target_dispatcher="hybridep",
        target_experts="torch_mm",
        target_enable_fsdp_optimizations=True,
        trust_remote_code=False,
    )

    assert target_model == "target-model"
    assert text_config.num_hidden_layers == 2
    assert text_config.layer_types == ["linear_attention", "full_attention"]
    assert captured["config"] is not outer_config
    assert captured["config"].language_model_only is True
    assert captured["load_base_model"] is True
    assert captured["distributed_setup"] == "distributed-setup"
    assert captured["torch_dtype"] == torch.bfloat16
    assert captured["backend"].attn == "flex"
    assert captured["backend"].experts == "torch_mm"
    assert captured["backend"].dispatcher == "hybridep"
    assert captured["backend"].gate_precision == torch.float32
    assert captured["backend"].enable_hf_state_dict_adapter is True
    assert outer_config.language_model_only is False
    assert outer_config.text_config.num_hidden_layers == 4


def test_dspark_target_config_builds_dense_qwen_draft_contract() -> None:
    class Args(dict):
        def __getattr__(self, key):
            return self[key]

    target_config = Qwen3_8_FlashNextTextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
        hc_count=2,
        ple_layer_ids=[2],
    )
    draft_config = build_draft_config(
        target_config,
        Args(
            num_draft_layers=2,
            target_layer_ids=[0, 2],
            block_size=7,
            num_anchors=4,
            mask_token_id=127,
            confidence_head_alpha=1.0,
            confidence_head_with_markov=True,
            markov_rank=8,
            markov_head_type="vanilla",
        ),
    )

    outer_config = Qwen3_8_FlashNextConfig(text_config=target_config)
    spec = resolve_dspark_draft_spec(list(outer_config.dspark_draft_architectures))
    assert spec.draft_cls is Qwen3DSparkModel
    assert outer_config.dspark_draft_config_kind == "dense"
    assert draft_config.num_hidden_layers == 2
    assert draft_config.layer_types == ["full_attention", "full_attention"]
    assert draft_config.target_layer_ids == [0, 2]
    assert draft_config.architectures == ["Qwen3DSparkModel"]
    assert draft_config._attn_implementation == "flex_attention"
    assert target_config.num_hidden_layers == 4
    assert target_config.layer_types[0] == "linear_attention"


def test_dspark_target_config_rejects_reduction_before_ple() -> None:
    outer_config = Qwen3_8_FlashNextConfig(
        text_config=Qwen3_8_FlashNextTextConfig(num_hidden_layers=4, hc_count=2, ple_layer_ids=[2])
    )

    with pytest.raises(ValueError, match="removes required PLE layers"):
        outer_config.prepare_dspark_target_config(
            target_path="qwen3.8-flash-next",
            target_num_hidden_layers=1,
        )


def test_dspark_target_config_requires_cuda() -> None:
    outer_config = Qwen3_8_FlashNextConfig()

    with pytest.raises(RuntimeError, match="requires CUDA"):
        outer_config.build_dspark_target(
            target_path="qwen3.8-flash-next",
            distributed_setup="distributed-setup",
            device=torch.device("cpu"),
            compute_dtype=torch.float32,
            target_num_hidden_layers=None,
            target_attn_backend="flex",
            target_dispatcher="hybridep",
            target_experts="torch_mm",
            target_enable_fsdp_optimizations=True,
            trust_remote_code=False,
        )


def test_qsa_sparse_training_runs_above_budget() -> None:
    config = _tiny_config()
    config.text_config.indexer_budget = 4
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
    )
    model = Qwen3_8_FlashNextForConditionalGeneration.from_config(
        config,
        moe_config=_tiny_moe_config(config.text_config),
        backend=backend,
    )
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)

    input_ids = torch.randint(2, config.text_config.vocab_size, (1, 9))
    output = model(input_ids=input_ids)
    output.logits.square().mean().backward()

    assert output.logits.shape == (1, 9, config.text_config.vocab_size)
    assert torch.isfinite(output.logits).all()
    attention = model.model.language_model.layers["0"].self_attn
    assert attention.q_proj.weight.grad is not None
    assert all(not parameter.requires_grad for parameter in attention.indexer.parameters())
    assert all(parameter.grad is None for parameter in attention.indexer.parameters())


def test_released_ple_table_requires_distributed_owner_group() -> None:
    config = _tiny_config()
    config.text_config.ple_layer_ids = [1]
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
    )

    with pytest.raises(RuntimeError, match="must be constructed after torch.distributed initialization"):
        Qwen3_8_FlashNextForConditionalGeneration.from_config(
            config,
            moe_config=_tiny_moe_config(config.text_config),
            backend=backend,
        )


def test_engram_owner_markers_are_restored_after_meta_materialization() -> None:
    from nemo_automodel.components.checkpoint.checkpointing import to_empty_parameters_only

    config = _tiny_config()
    config.text_config.ple_layer_ids = [1]
    config.text_config.ngram_size = 3
    config.text_config.heads_per_ngram = 8
    config.text_config.ple_embed_dim = 32
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
    )
    table_config = Qwen3_8_FlashNextEngramTableConfig(num_embeddings=36, embedding_dim=2, initializer_range=0.05)
    with torch.device("meta"):
        model = Qwen3_8_FlashNextForConditionalGeneration.from_config(
            config,
            moe_config=_tiny_moe_config(config.text_config),
            backend=backend,
            engram_table_config=table_config,
        )

    table = model.model.language_model.layers["0"].ple.ple_embedding.ngram_embedding
    assert model.model.language_model.layers["0"]._nemo_disable_activation_checkpointing is True
    original_weight = table.weight
    to_empty_parameters_only(model, device=torch.device("cpu"))
    assert table.weight is not original_weight
    assert not hasattr(table.weight, "_nemo_owner_sharded_spec")
    with torch.no_grad():
        table.weight.fill_(torch.nan)

    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    assert torch.isfinite(table.weight).all()
    assert torch.count_nonzero(table.weight) > 0
    assert not hasattr(table.weight, "_nemo_model_owned_dtensor_spec")
