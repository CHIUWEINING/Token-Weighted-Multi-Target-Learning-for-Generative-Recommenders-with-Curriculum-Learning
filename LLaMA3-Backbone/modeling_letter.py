import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import LlamaForCausalLM


class LETTER(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(
        self,
        config,
        args=None,
        item_weights: Optional[torch.Tensor] = None,
        token_weights: Optional[torch.Tensor] = None,
        gain: Optional[torch.Tensor] = None,
        igd: Optional[dict] = None,
    ):
        super().__init__(config)

        self.temperature = 1.0

        self.item_weights = item_weights
        self.token_weights = token_weights
        self.warm_up_steps = 0
        self.gain = gain
        self.args = args
        self.igd = igd

        self.igd_beta = float(getattr(self.args, "igdbeta", 0.1))
        self.cft_lambda = float(getattr(self.args, "cft_lambda", 0.1))
        self.cft_beta = float(getattr(self.args, "cft_beta", 0.5))
        self.cft_weight_normal = bool(getattr(self.args, "cft_weight_normal", True))
        self.c = float(getattr(self.args, "c", 0.00003))

        self.log_vars = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        self._curriculum_eps = 1e-6
        self._logvar_bound = 20.0
        self.log_vars.register_hook(self._log_vars_grad_guard)
        self.log_vars.requires_grad_(True)

    def set_hyper(self, temperature):
        self.temperature = temperature

    def _log_vars_grad_guard(self, grad):
        if grad is None:
            return grad
        if not torch.isfinite(grad).all():
            raise FloatingPointError(
                f"[LETTER][ours] non-finite log_vars grad detected: {grad.detach().cpu().tolist()}"
            )
        # Prevent abrupt optimizer updates from driving log_vars into unstable regions.
        grad32 = grad.float().clamp(min=-10.0, max=10.0)
        return grad32.to(dtype=grad.dtype)

    def _method(self):
        return str(getattr(self.args, "method", "origin"))

    def _get_eos_token_ids(self):
        eos_token_id = getattr(self.config, "eos_token_id", None)
        if eos_token_id is None:
            return []
        if isinstance(eos_token_id, (list, tuple)):
            return [int(x) for x in eos_token_id]
        return [int(eos_token_id)]

    def ranking_loss(self, shift_logits, shift_labels):
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1).to(shift_logits.device)
        return loss_fct(shift_logits / self.temperature, shift_labels)

    def token_weight_loss(self, shift_logits, shift_labels):
        if shift_labels is None or self.token_weights is None:
            return None

        logits = shift_logits / self.temperature
        device = logits.device
        labels = shift_labels.to(device)

        bsz, seq_len, vocab = logits.shape
        ce = F.cross_entropy(
            logits.reshape(-1, vocab),
            labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).reshape(bsz, seq_len)

        weights_vec = self.token_weights.to(device)
        max_idx = weights_vec.numel() - 1
        safe_labels = labels.clamp(min=0, max=max_idx)
        token_w = weights_vec.gather(0, safe_labels.reshape(-1)).reshape(bsz, seq_len)

        out_of_vocab_mask = labels > max_idx
        if out_of_vocab_mask.any():
            token_w = torch.where(out_of_vocab_mask, torch.ones_like(token_w), token_w)

        valid_mask = labels != -100
        token_w = token_w * valid_mask.float()

        eos_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        for eos_id in self._get_eos_token_ids():
            eos_mask |= (labels == eos_id)
        eos_mask &= valid_mask

        non_eos_mask = valid_mask & (~eos_mask)

        eps = 1e-12
        non_eos_w = token_w * non_eos_mask.float()
        non_eos_sum = non_eos_w.sum(dim=1, keepdim=True).clamp(min=eps)
        valid_count = valid_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        has_eos = eos_mask.any(dim=1, keepdim=True)
        target_non_eos_sum = torch.where(has_eos, torch.full_like(valid_count, 4.0), valid_count)

        scaled_non_eos = non_eos_w * (target_non_eos_sum / non_eos_sum)
        token_w = scaled_non_eos + eos_mask.float()

        return (ce * token_w).sum() / (token_w.sum() + eps)

    def _build_token_pos_weights(self, labels):
        device = labels.device
        bsz, seq_len = labels.shape
        beta = float(self.cft_beta)

        w = torch.zeros_like(labels, dtype=torch.float32, device=device)
        for b in range(bsz):
            valid_idx = (labels[b] != -100).nonzero(as_tuple=False).view(-1)
            n = valid_idx.numel()
            if n == 0:
                continue
            if n == 1:
                w[b, valid_idx[0]] = 1.0
                continue

            pos = torch.arange(n, device=device, dtype=torch.float32)
            weights_seq = 1.0 - (1.0 - beta) * pos / (n - 1)
            w[b, valid_idx] = weights_seq

        w_flat = w.view(-1)
        omega = w_flat.sum().clamp(min=1e-12)
        return w_flat, omega

    def front_greater_loss(self, shift_logits, shift_labels):
        if shift_labels is None or self.gain is None:
            return None

        device = shift_logits.device
        logits = shift_logits
        labels = shift_labels.to(device)
        bsz, seq_len, vocab = logits.shape

        ce = F.cross_entropy(
            logits.view(-1, vocab),
            labels.view(-1),
            reduction="none",
            ignore_index=-100,
        )

        eps = 1e-12
        gain = self.gain.to(device).float()
        gain_len = gain.shape[0]
        if gain_len == 0:
            return self.ranking_loss(shift_logits, shift_labels)

        w_raw = torch.log1p(torch.clamp(gain, min=0.0))
        if torch.all(w_raw <= eps):
            base_w = torch.full((gain_len,), 4.0 / max(gain_len, 1), device=device)
        else:
            base_w = 4.0 * (w_raw / torch.clamp(w_raw.sum(), min=eps))

        pos_w = torch.zeros((bsz, seq_len), device=device, dtype=torch.float32)
        valid2d = labels != -100
        eos_ids = self._get_eos_token_ids()

        for b in range(bsz):
            valid_idx = valid2d[b].nonzero(as_tuple=False).view(-1)
            n_valid = valid_idx.numel()
            if n_valid == 0:
                continue

            if eos_ids:
                local_labels = labels[b, valid_idx]
                local_eos_mask = torch.zeros(n_valid, dtype=torch.bool, device=device)
                for eos_id in eos_ids:
                    local_eos_mask |= (local_labels == eos_id)
                eos_idx = valid_idx[local_eos_mask]
                semantic_idx = valid_idx[~local_eos_mask]
            else:
                eos_idx = valid_idx[:0]
                semantic_idx = valid_idx

            n_semantic = semantic_idx.numel()
            if n_semantic > 0:
                if n_semantic == gain_len:
                    w_local = base_w
                elif n_semantic == 1:
                    w_local = torch.tensor([4.0], device=device)
                else:
                    src = torch.linspace(0, gain_len - 1, steps=n_semantic, device=device)
                    left = torch.floor(src).long()
                    right = torch.clamp(left + 1, max=gain_len - 1)
                    alpha = src - left.float()
                    w_local = base_w[left] * (1.0 - alpha) + base_w[right] * alpha
                    w_local = 4.0 * (w_local / torch.clamp(w_local.sum(), min=eps))
                pos_w[b, semantic_idx] = w_local

            if eos_idx.numel() > 0:
                pos_w[b, eos_idx] = 1.0
            elif n_semantic == 0:
                pos_w[b, valid_idx] = 1.0

        token_w = pos_w.view(-1)
        return (ce * token_w).sum() / (token_w.sum() + eps)

    def igd_weight_loss(self, shift_logits, shift_labels):
        if self.igd is None:
            return None

        bsz, seq_len, vocab = shift_logits.shape
        ce = F.cross_entropy(
            shift_logits.view(-1, vocab),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        )

        wt_list = []
        for b in range(bsz):
            prefix = []
            for t in range(seq_len):
                y_t = int(shift_labels[b, t])
                if y_t == -100:
                    wt_list.append(1.0)
                    continue

                key = (tuple(prefix), y_t)
                ig_val = self.igd.get(key, 0.0)
                wt = 1.0 if ig_val > 0 else self.igd_beta
                wt_list.append(wt)
                prefix.append(y_t)

        wt_tensor = torch.tensor(wt_list, device=ce.device)
        return (ce * wt_tensor).sum() / wt_tensor.sum().clamp(min=1e-12)

    def _weighted_ce_loss(self, logits, labels, use_pos_weights=True, normalize_by_omega=True):
        device = logits.device
        _, _, vocab = logits.shape

        labels = labels.to(device)
        ce = F.cross_entropy(
            logits.view(-1, vocab),
            labels.view(-1),
            reduction="none",
            ignore_index=-100,
        )

        valid_mask = (labels.view(-1) != -100).float()

        if use_pos_weights:
            w_flat, omega = self._build_token_pos_weights(labels)
            w_flat = w_flat.to(device) * valid_mask
            denom = omega if normalize_by_omega else w_flat.sum().clamp(min=1e-12)
        else:
            w_flat = valid_mask
            denom = w_flat.sum().clamp(min=1e-12)

        return (ce * w_flat).sum() / denom

    def _curriculum_multi_target_weights(self):
        if not torch.isfinite(self.log_vars).all():
            # Safety fallback: recover from rare optimizer-state corruption without aborting training.
            with torch.no_grad():
                self.log_vars.nan_to_num_(nan=0.0, posinf=self._logvar_bound, neginf=-self._logvar_bound)
                self.log_vars.clamp_(min=-self._logvar_bound, max=self._logvar_bound)

        # Keep curriculum math in fp32 and smoothly bound log_vars to avoid unstable updates.
        log_vars_fp32 = self.log_vars.float()
        log_vars_eff = self._logvar_bound * torch.tanh(log_vars_fp32 / self._logvar_bound)
        step_scale = torch.tensor(-self.c * self.warm_up_steps, device=log_vars_eff.device, dtype=torch.float32)
        pace_front_origin = torch.exp(step_scale)
        pace_token = 1.0 - pace_front_origin
        pace = torch.stack([pace_front_origin, pace_front_origin, pace_token]).to(log_vars_eff.device)
        pace = torch.clamp(pace, min=self._curriculum_eps)

        # Numerically stable weights: softmax(log precision + log pace).
        logits = (-log_vars_eff) + torch.log(pace)
        w = torch.softmax(logits, dim=0)
        if not torch.isfinite(w).all():
            raise FloatingPointError(
                "[LETTER][ours] non-finite curriculum weights: "
                f"log_vars={self.log_vars.detach().cpu().tolist()}, "
                f"log_vars_eff={log_vars_eff.detach().cpu().tolist()}, "
                f"pace={pace.detach().cpu().tolist()}, "
                f"logits={logits.detach().cpu().tolist()}, "
                f"warm_up_steps={int(self.warm_up_steps)}"
            )

        reg = log_vars_eff.pow(2).mean()
        return w, reg

    def total_loss(self, shift_logits, shift_labels, shift_logits_cf=None):
        method = self._method()

        loss_token = self.token_weight_loss(shift_logits, shift_labels)
        loss_origin = self.ranking_loss(shift_logits, shift_labels)
        loss_front = self.front_greater_loss(shift_logits, shift_labels)

        if loss_token is None:
            loss_token = loss_origin
        if loss_front is None:
            loss_front = loss_origin

        # Curriculum pace should advance only on training updates, not validation forwards.
        if self.training:
            self.warm_up_steps += 1
        lbda_token = min(self.warm_up_steps * 0.000004, 0.4)
        lbda_front = min(self.warm_up_steps * 0.000010, 1.0)

        if method == "combine":
            return ((1 - lbda_front - lbda_token) * loss_front) + (lbda_front * loss_origin) + (lbda_token * loss_token)

        if method in {"front_gain", "front_reverse"}:
            return ((1 - lbda_front) * loss_front) + (lbda_front * loss_origin)

        if "origin" in method or "rank" in method:
            return loss_origin

        if "pos" in method:
            return self._weighted_ce_loss(
                shift_logits,
                shift_labels,
                use_pos_weights=True,
                normalize_by_omega=True,
            )

        if method == "_token_freq":
            return (lbda_token * loss_token) + ((1 - lbda_token) * loss_origin)

        if "igd" in method:
            loss_igd = self.igd_weight_loss(shift_logits, shift_labels)
            return loss_igd if loss_igd is not None else loss_origin

        if "cft" in method:
            if self.cft_weight_normal:
                loss_n = self._weighted_ce_loss(
                    shift_logits,
                    shift_labels,
                    use_pos_weights=True,
                    normalize_by_omega=True,
                )
            else:
                loss_n = loss_origin

            loss_c = 0.0
            if (shift_logits_cf is not None) and (self.cft_lambda > 0.0):
                logits_diff = shift_logits - shift_logits_cf
                loss_c = self._weighted_ce_loss(
                    logits_diff,
                    shift_labels,
                    use_pos_weights=True,
                    normalize_by_omega=True,
                )

            total = loss_n + self.cft_lambda * loss_c
            self.last_loss_components = {
                "L_n": float(loss_n.detach()),
                "L_c": float(loss_c if isinstance(loss_c, float) else loss_c.detach()),
                "lambda": float(self.cft_lambda),
            }
            return total

        if "ours" in method:
            ls = torch.stack([loss_origin, loss_front, loss_token])
            if not torch.isfinite(ls).all():
                raise FloatingPointError(
                    "[LETTER][ours] non-finite loss component detected: "
                    f"L_origin={float(loss_origin.detach().cpu())}, "
                    f"L_front={float(loss_front.detach().cpu())}, "
                    f"L_token={float(loss_token.detach().cpu())}, "
                    f"warm_up_steps={int(self.warm_up_steps)}"
                )

            w, reg = self._curriculum_multi_target_weights()
            loss_mix = torch.dot(w, ls)
            if not torch.isfinite(loss_mix):
                raise FloatingPointError(
                    "[LETTER][ours] non-finite mixed loss detected: "
                    f"L_origin={float(loss_origin.detach().cpu())}, "
                    f"L_front={float(loss_front.detach().cpu())}, "
                    f"L_token={float(loss_token.detach().cpu())}, "
                    f"w_origin={float(w[0].detach().cpu())}, "
                    f"w_front={float(w[1].detach().cpu())}, "
                    f"w_token={float(w[2].detach().cpu())}, "
                    f"log_vars={self.log_vars.detach().cpu().tolist()}, "
                    f"warm_up_steps={int(self.warm_up_steps)}"
                )

            self.last_loss_components = {
                "L_origin": float(loss_origin.detach()),
                "L_front": float(loss_front.detach()),
                "L_token": float(loss_token.detach()),
                "w_origin": float(w[0].detach()),
                "w_front": float(w[1].detach()),
                "w_token": float(w[2].detach()),
                "reg": float(reg.detach()),
            }
            return loss_mix

        return loss_origin

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        input_ids_cf: Optional[torch.LongTensor] = None,
        attention_mask_cf: Optional[torch.Tensor] = None,
        item_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        shift_logits_cf = None
        if ("cft" in self._method()) and (input_ids_cf is not None):
            outputs_cf = self.model(
                input_ids=input_ids_cf,
                attention_mask=attention_mask_cf,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states_cf = outputs_cf[0]
            if self.config.pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                logits_cf = [F.linear(hidden_states_cf, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                logits_cf = torch.cat(logits_cf, dim=-1)
            else:
                logits_cf = self.lm_head(hidden_states_cf)
            logits_cf = logits_cf.float()
            shift_logits_cf = logits_cf[..., :-1, :].contiguous()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.total_loss(shift_logits, shift_labels, shift_logits_cf=shift_logits_cf)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
