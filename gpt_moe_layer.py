import math
import warnings

import torch
from torch import nn
import peft
from peft.tuners.tuners_utils import check_adapters_to_merge
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts


def _grouped_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    offs: torch.Tensor,
    bias: torch.Tensor | None = None,
    is_transposed: bool = False,
) -> torch.Tensor:
    """Grouped linear layer supporting optional bias and transposed weights.

    Args:
        input (`torch.Tensor`):
            Input tensor of shape (S, input_dim).
        weight (`torch.Tensor`):
            Weight tensor of shape (num_experts, input_dim, output_dim) if `is_transposed`,
            else of shape (num_experts, output_dim, input_dim).
        offs (`torch.Tensor`):
            Offsets tensor indicating the boundaries of each group in the input tensor.
        bias (`torch.Tensor`, *optional*):
            Bias tensor of shape (S, output_dim). Default is `None`.
        is_transposed (`bool`, *optional*, defaults to `False`):
            Whether the weight tensor is transposed.
    Returns:
        `torch.Tensor`: Output tensor of shape (S, output_dim).
    """
    if is_transposed:
        # (S, input_dim) @ grouped (num_experts, input_dim, output_dim) -> (S, output_dim)
        out = nn.functional.grouped_mm(input, weight, offs=offs)
    else:
        # (S, input_dim) @ grouped (num_experts, output_dim, input_dim).T -> (S, output_dim)
        out = nn.functional.grouped_mm(input, weight.transpose(-2, -1), offs=offs)

    if bias is not None:
        # We should be able to pass bias to the grouped_mm call, but it's not yet supported.
        out = out + bias

    return out


def _compute_routing(
    expert_ids: torch.Tensor,
    num_experts: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort tokens by expert and compute grouped_mm offsets.

    Returns:
        perm: argsort permutation over expert_ids
        inv_perm: inverse of perm (restores original order)
        offsets: cumulative token counts per expert (shape: num_experts,)
    """
    perm = torch.argsort(expert_ids)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=device)

    expert_ids_sorted = expert_ids[perm]
    # using histc instead of bincount to avoid cuda graph issues
    # With deterministic algorithms, CPU only supports float input, CUDA only supports int input.
    histc_input = expert_ids_sorted.float() if device.type == "cpu" else expert_ids_sorted.int()
    tokens_per_expert = torch.histc(histc_input, bins=num_experts, min=0, max=num_experts - 1)
    offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

    return perm, inv_perm, offsets


def grouped_mm_experts_forward(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    num_experts: int,
    has_bias: bool,
    is_transposed: bool,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    apply_gate,
    gate_up_proj_bias: torch.Tensor | None = None,
    down_proj_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)

    # S is the number of selected token-expert pairs (S = num_tokens * num_top_k)
    token_idx = (
        torch.arange(num_tokens, device=device)
        .unsqueeze(1)
        .expand(-1, num_top_k)
        .reshape(-1)
    )  # (S,)
    sample_weights = top_k_weights.reshape(-1)  # (S,)
    expert_ids = top_k_index.reshape(-1)  # (S,)
    selected_hidden_states = hidden_states[token_idx]

    perm, inv_perm, offsets = _compute_routing(expert_ids, num_experts, device)

    expert_ids_g = expert_ids[perm]
    sample_weights_g = sample_weights[perm]
    selected_hidden_states_g = selected_hidden_states[perm]

    # Select expert weights and biases
    # NOTE: We keep all experts here and rely on offsets to target the active ones.
    # I have already implemented a version that only passes the active experts, but
    # to do so I had to use torch.unique which breaks the graph capture (data-dependent).
    # Also there were no speedup gains from it in my experiments, even in eager mode.

    # --- Up projection per expert (grouped) ---
    proj_out = _grouped_linear(
        selected_hidden_states_g,
        gate_up_proj,
        offsets,
        bias=gate_up_proj_bias[expert_ids_g] if has_bias else None,
        is_transposed=is_transposed,
    )  # (S, 2 * intermediate_dim)

    # Apply gating mechanism
    proj_out = apply_gate(proj_out)  # (S, intermediate_dim)

    # --- Down projection per expert (grouped) ---
    proj_out = _grouped_linear(
        proj_out,
        down_proj,
        offsets,
        bias=down_proj_bias[expert_ids_g] if has_bias else None,
        is_transposed=is_transposed,
    )  # (S, hidden_dim)

    # Apply routing weights
    weighted_out = proj_out * sample_weights_g.unsqueeze(-1)  # (S, hidden_dim)

    # Restore original order
    weighted_out = weighted_out[inv_perm]  # (S, hidden_dim)

    # Accumulate results using deterministic reshape+sum instead of index_add_
    # index_add_ with duplicate indices is non-deterministic on CUDA due to atomicAdd
    # index_add_ accumulates in-place using the dtype of the output tensor (fp16/bf16)
    # reshape+sum accumulates in fp32 which is more stable for low precision training/inference.
    final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)

    return final_hidden_states.to(hidden_states.dtype)


class GptOssExpertsLora(nn.Module, peft.tuners.lora.layer.LoraLayer):
    adapter_layer_names: tuple[str, ...] = ("lora_A", "lora_B")
    other_param_names: tuple[str, ...] = ("r", "lora_alpha", "scaling")

    def __init__(self, base_layer: GptOssExperts, adapter_name: str, **kwargs):
        nn.Module.__init__(self)
        peft.tuners.lora.layer.LoraLayer.__init__(self, base_layer)
        print("initializing lora injected layer")

        r = kwargs["r"]
        lora_alpha = kwargs["lora_alpha"]
        self.adapter_name = adapter_name
        # LoraLayer.__init__ already created these as empty dicts; populate rather than overwrite
        # so multiple adapters can coexist and PEFT utilities (merge/unmerge, etc.) work correctly.
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.scaling[adapter_name] = lora_alpha / r

        ne = base_layer.num_experts
        h = base_layer.hidden_size
        i = base_layer.intermediate_size

        self.lora_A[adapter_name] = nn.ParameterDict(
            {
                "gate_up_proj": nn.Parameter(torch.empty(ne, h, r), requires_grad=True),
                "down_proj": nn.Parameter(torch.empty(ne, i, r), requires_grad=True),
            }
        )
        self.lora_B[adapter_name] = nn.ParameterDict(
            {
                "gate_up_proj": nn.Parameter(torch.empty(ne, r, 2 * i), requires_grad=True),
                "down_proj": nn.Parameter(torch.empty(ne, r, h), requires_grad=True),
            }
        )

        self.reset_lora_parameters()

    def _lora_deltas(self, adapter_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute LoRA weight deltas for gate_up and down projections.

        Returns:
            gate_up_delta: (num_experts, hidden_size, 2 * intermediate_size)
            down_delta:    (num_experts, intermediate_size, hidden_size)
        """
        scaling = self.scaling[adapter_name]
        lora_A = self.lora_A[adapter_name]
        lora_B = self.lora_B[adapter_name]
        gate_up_delta = (
            torch.bmm(lora_A["gate_up_proj"], lora_B["gate_up_proj"]) * scaling
        )
        down_delta = (
            torch.bmm(lora_A["down_proj"], lora_B["down_proj"]) * scaling
        )
        return gate_up_delta, down_delta

    def merge(self, safe_merge: bool = False, adapter_names=None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter != self.adapter_name:
                continue

            gate_up_delta, down_delta = self._lora_deltas(active_adapter)

            if safe_merge:
                merged_gate_up = self.base_layer.gate_up_proj.data + gate_up_delta
                merged_down = self.base_layer.down_proj.data + down_delta
                if not (torch.isfinite(merged_gate_up).all() and torch.isfinite(merged_down).all()):
                    raise ValueError(
                        f"NaNs/Infs detected in merged weights for adapter '{active_adapter}'. "
                        "Use safe_merge=False to skip this check."
                    )
                self.base_layer.gate_up_proj.data = merged_gate_up
                self.base_layer.down_proj.data = merged_down
            else:
                self.base_layer.gate_up_proj.data += gate_up_delta
                self.base_layer.down_proj.data += down_delta

            self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged, nothing to unmerge.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter != self.adapter_name:
                continue
            gate_up_delta, down_delta = self._lora_deltas(active_adapter)
            self.base_layer.gate_up_proj.data -= gate_up_delta
            self.base_layer.down_proj.data -= down_delta

    def reset_lora_parameters(self):
        # Initialize A with kaiming_uniform_ (same as nn.Linear default) and B to zero.
        for adapter_dict in self.lora_A.values():
            for param in adapter_dict.values():
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        for adapter_dict in self.lora_B.values():
            for param in adapter_dict.values():
                nn.init.zeros_(param)

    # def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
    #     gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    #     gate = gate.clamp(min=None, max=self.base_layer.limit)
    #     up = up.clamp(min=-self.base_layer.limit, max=self.base_layer.limit)
    #     glu = gate * torch.sigmoid(gate * self.base_layer.alpha)
    #     return (up + 1) * glu

    # def forward(
    #     self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None
    # ) -> torch.Tensor:
    #     """
    #     Args:
    #         hidden_states (torch.Tensor): (batch_size, seq_len, hidden_size)
    #         router_indices (torch.Tensor): (batch_size * token_num, top_k)
    #         routing_weights (torch.Tensor): (batch_size * token_num, top_k)
    #     Returns:
    #         torch.Tensor
    #     """
    #     bl = self.base_layer
    #     hidden_states_2d = hidden_states.view(-1, bl.hidden_size)
    #     has_bias = bl.gate_up_proj_bias is not None

    #     # When adapters are disabled (e.g. DPO reference pass) or merged into
    #     # base weights, fall back to the base layer so the LoRA delta is not
    #     # applied twice / applied when it shouldn't be.
    #     if self.disable_adapters or self.merged:
    #         return grouped_mm_experts_forward(
    #             hidden_states=hidden_states_2d,
    #             top_k_index=router_indices,
    #             top_k_weights=routing_weights,
    #             num_experts=bl.num_experts,
    #             has_bias=has_bias,
    #             is_transposed=True,
    #             gate_up_proj=bl.gate_up_proj,
    #             down_proj=bl.down_proj,
    #             apply_gate=self._apply_gate,
    #             gate_up_proj_bias=bl.gate_up_proj_bias,
    #             down_proj_bias=bl.down_proj_bias
    #         )

    #     scaling = self.scaling[self.adapter_name]

    #     device = hidden_states_2d.device
    #     num_top_k = router_indices.size(-1)
    #     num_tokens = hidden_states_2d.size(0)
    #     hidden_dim = hidden_states_2d.size(-1)

    #     # S is the number of selected token-expert pairs (S = num_tokens * num_top_k)
    #     token_idx = (
    #         torch.arange(num_tokens, device=device)
    #         .unsqueeze(1)
    #         .expand(-1, num_top_k)
    #         .reshape(-1)
    #     )
    #     sample_weights = routing_weights.reshape(-1)
    #     expert_ids = router_indices.reshape(-1)
    #     selected_hidden_states = hidden_states_2d[token_idx]

    #     perm, inv_perm, offsets = _compute_routing(expert_ids, bl.num_experts, device)

    #     expert_ids_g = expert_ids[perm]
    #     sample_weights_g = sample_weights[perm]
    #     selected_hidden_states_g = selected_hidden_states[perm]

    #     # --- Gate/up projection + LoRA ---
    #     proj_out = _grouped_linear(
    #         selected_hidden_states_g,
    #         bl.gate_up_proj,
    #         offsets,
    #         bias=bl.gate_up_proj_bias[expert_ids_g] if has_bias else None,
    #         is_transposed=True,
    #     )  # (S, 2 * intermediate_dim)

    #     # gate_up LoRA: hidden -> r -> 2*intermediate
    #     lora_A = self.lora_A[self.adapter_name]
    #     lora_B = self.lora_B[self.adapter_name]
    #     lora_gate_up = _grouped_linear(
    #         selected_hidden_states_g, lora_A["gate_up_proj"], offsets, is_transposed=True
    #     )
    #     lora_gate_up = _grouped_linear(
    #         lora_gate_up, lora_B["gate_up_proj"], offsets, is_transposed=True
    #     )
    #     proj_out = proj_out + lora_gate_up * scaling

    #     # Apply gating
    #     proj_out = self._apply_gate(proj_out)  # (S, intermediate_dim)

    #     # --- Down projection + LoRA ---
    #     down_out = _grouped_linear(
    #         proj_out,
    #         bl.down_proj,
    #         offsets,
    #         bias=bl.down_proj_bias[expert_ids_g] if has_bias else None,
    #         is_transposed=True,
    #     )  # (S, hidden_dim)

    #     # down LoRA: intermediate -> r -> hidden
    #     lora_down = _grouped_linear(
    #         proj_out, lora_A["down_proj"], offsets, is_transposed=True
    #     )
    #     lora_down = _grouped_linear(
    #         lora_down, lora_B["down_proj"], offsets, is_transposed=True
    #     )
    #     proj_out = down_out + lora_down * scaling

    #     # Apply routing weights, restore original order, and accumulate
    #     proj_out = (proj_out * sample_weights_g.unsqueeze(-1))[inv_perm]
    #     final_hidden_states = proj_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)

    #     return final_hidden_states.to(hidden_states.dtype)

    def forward(
        self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None
    ) -> torch.Tensor:
        """
        Forward pass using the per-expert for-loop (training path) with LoRA.

        Args:
            hidden_states (torch.Tensor): (batch_size, seq_len, hidden_size)
            router_indices (torch.Tensor): (batch_size * token_num, top_k)
            routing_weights (torch.Tensor): (batch_size * token_num, top_k)
        Returns:
            torch.Tensor
        """
        bl = self.base_layer
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, bl.hidden_size)  # (num_tokens, hidden_size)
        num_experts = routing_weights.shape[1]

        next_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                router_indices, num_classes=num_experts + 1
            )  # masking is also a class
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        # When adapters are disabled (e.g. DPO reference pass) or merged into
        # base weights, fall back to base-layer-only so the LoRA delta is not
        # applied twice / applied when it shouldn't be.
        if self.disable_adapters or self.merged:
            for expert_idx in expert_hit[:]:
                expert_idx = expert_idx[0]
                if expert_idx == num_experts:
                    continue
                with torch.no_grad():
                    _, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                gate_up = current_state @ bl.gate_up_proj[expert_idx] + bl.gate_up_proj_bias[expert_idx]
                gate, up = gate_up[..., ::2], gate_up[..., 1::2]
                gate = gate.clamp(min=None, max=bl.limit)
                up = up.clamp(min=-bl.limit, max=bl.limit)
                glu = gate * torch.sigmoid(gate * bl.alpha)
                gated_output = (up + 1) * glu
                out = gated_output @ bl.down_proj[expert_idx] + bl.down_proj_bias[expert_idx]
                weighted_output = out * routing_weights[token_idx, expert_idx, None]
                next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
            return next_states.view(batch_size, -1, bl.hidden_size)

        # LoRA active path
        scaling = self.scaling[self.adapter_name]
        lora_A = self.lora_A[self.adapter_name]
        lora_B = self.lora_B[self.adapter_name]

        for expert_idx in expert_hit[:]:
            expert_idx = expert_idx[0]
            if expert_idx == num_experts:
                continue
            with torch.no_grad():
                _, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            # --- Gate/up projection + LoRA ---
            gate_up = current_state @ bl.gate_up_proj[expert_idx] + bl.gate_up_proj_bias[expert_idx]
            lora_gate_up = (current_state @ lora_A["gate_up_proj"][expert_idx]) @ lora_B["gate_up_proj"][expert_idx]
            gate_up = gate_up + lora_gate_up * scaling

            # Apply gating
            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            gate = gate.clamp(min=None, max=bl.limit)
            up = up.clamp(min=-bl.limit, max=bl.limit)
            glu = gate * torch.sigmoid(gate * bl.alpha)
            gated_output = (up + 1) * glu

            # --- Down projection + LoRA ---
            out = gated_output @ bl.down_proj[expert_idx] + bl.down_proj_bias[expert_idx]
            lora_down = (gated_output @ lora_A["down_proj"][expert_idx]) @ lora_B["down_proj"][expert_idx]
            out = out + lora_down * scaling

            weighted_output = out * routing_weights[token_idx, expert_idx, None]
            next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))

        return next_states.view(batch_size, -1, bl.hidden_size)
