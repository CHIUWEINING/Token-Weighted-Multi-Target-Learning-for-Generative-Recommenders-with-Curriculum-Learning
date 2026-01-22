from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import (
    T5Stack, T5Block, T5LayerNorm, T5LayerSelfAttention, T5LayerFF, T5LayerCrossAttention,
    T5PreTrainedModel, T5ForConditionalGeneration
)
import torch
from torch import nn
import copy
import torch
import torch.nn as nn
import json
import torch.nn.functional as F
import numpy as np
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import ModelOutput, BaseModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, Seq2SeqModelOutput
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import logging
from transformers import BeamScorer, BeamSearchScorer
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


class LETTER(T5ForConditionalGeneration):

    def __init__(self, config: T5Config, args, item_weights:torch.Tensor = None, token_weights:dict = None,\
                gain:torch.Tensor=None, igd:dict = None):

        super().__init__(config)

        # You can add parameters out here.
        self.temperature = 1.0

        self.item_weights = item_weights
        self.token_weights = token_weights
        self.warm_up_steps = 0
        self.gain = gain
        self.args = args
        self.igd_beta = self.args.igdbeta
        
        self.cft_lambda = self.args.cft_lambda
        # beta for token-level linear decay weights (last token weight)
        self.cft_beta = self.args.cft_beta
        # whether to also use token-level weights on normal loss
        self.cft_weight_normal = self.args.cft_weight_normal

        self.c = self.args.c #adjust as you want
        self.igd = igd

        # order: [origin, front, token]  (match how you build Ls below)
        self.log_vars = nn.Parameter(torch.zeros(3)) # initialized so w_k = 1/3 each, you can adjust the initialization weight as you want
        # init_w = torch.tensor([0.0, 0.0, 0.0])             # [origin, front, token]
        # log_vars_init = -torch.log(init_w)                 # s_k = -log w_k
        # log_vars_init = log_vars_init - log_vars_init.mean()  # optional: zero-center to keep scale stable
        # self.log_vars = nn.Parameter(log_vars_init)  

    def set_hyper(self,temperature):
        self.temperature = temperature


    def ranking_loss(self, lm_logits, labels):
        if labels is not None:
            t_logits = lm_logits/self.temperature
            # weights = self.weights.to(t_logits.device)
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            # print(labels.shape, t_logits.shape)
            loss = loss_fct(t_logits.view(-1, t_logits.size(-1)), labels.view(-1))
        return loss

    def token_weight_loss(self, lm_logits, labels):
        if labels is None:
            return None

        logits = lm_logits / self.temperature  # (B, L, V)
        device = logits.device
        labels = labels.to(device)

        B, L, V = logits.shape

        # token-level CE (no reduction), shaped back to (B, L)
        ce = F.cross_entropy(
            logits.reshape(-1, V),
            labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).reshape(B, L)  # (B, L)

        # Build token weights from labels (B, L)
        weights_vec = self.token_weights.to(device)  # (num_tokens,)
        safe_labels = labels.clamp_min(0)            # avoid gather(-100)
        
        token_w = weights_vec.gather(0, safe_labels.reshape(-1)).reshape(B, L)
        # Mask out ignore_index/padding
        valid_mask = (labels != -100).float()        # (B, L)
        token_w = token_w * valid_mask               # invalid => 0

        # Find "last valid token" position per sample (treat as EOS position)
        # last_idx: (B,)
        valid_len = valid_mask.sum(dim=1).long()                 # number of valid tokens
        last_idx = (valid_len - 1).clamp_min(0)                  # in case of empty (shouldn't happen)
        eos_mask = torch.zeros((B, L), device=device, dtype=torch.float)
        eos_mask.scatter_(1, last_idx.unsqueeze(1), 1.0)         # (B, L) one-hot at last valid pos

        # Force EOS weight to 1.0 for each sample (and only if that position is valid)
        eos_mask = eos_mask * valid_mask
        token_w = token_w * (1.0 - eos_mask) + eos_mask * 1.0
        # Now normalize NON-EOS valid tokens to sum to 4.0 per sample
        non_eos_mask = valid_mask * (1.0 - eos_mask)
        non_eos_sum = (token_w * non_eos_mask).sum(dim=1, keepdim=True)  # (B, 1)

        eps = 1e-12
        scale = 4.0 / (non_eos_sum + eps)  # (B, 1)

        # scale only non-eos weights; keep eos at 1.0
        token_w = (token_w * non_eos_mask) * scale + eos_mask * 1.0

        # Final weighted mean loss over the whole batch/time
        loss = (ce * token_w).sum() / (token_w.sum() + eps)
        return loss

    def _build_token_pos_weights(self, labels):
        """
        Build token-position weights ω_t for each valid target token in labels of CFT paper:
            ω_t = 1 - (1 - β) * (t - 1) / (|y| - 1)
        where positions are counted only over valid (label != -100) tokens.

        Args:
            labels: (B, L) with -100 as ignore_index.

        Returns:
            w_flat: (B*L,) weights for each position.
            omega: scalar sum of all weights over valid positions.
        """
        device = labels.device
        B, L = labels.shape
        beta = float(self.cft_beta)

        # weights per position, same shape as labels
        w = torch.zeros_like(labels, dtype=torch.float32, device=device)

        for b in range(B):
            valid_idx = (labels[b] != -100).nonzero(as_tuple=False).view(-1)
            n = valid_idx.numel()
            if n == 0:
                continue
            if n == 1:
                # only one token → weight = 1.0
                w[b, valid_idx[0]] = 1.0
                continue
            # positions 0..n-1 along the item tokens
            pos = torch.arange(n, device=device, dtype=torch.float32)
            # linear decay from 1 → beta
            weights_seq = 1.0 - (1.0 - beta) * pos / (n - 1)
            w[b, valid_idx] = weights_seq

        w_flat = w.view(-1)
        omega = w_flat.sum()
        return w_flat, omega.clamp(min=1e-12)

    def front_greater_loss(self, lm_logits, labels):
        """
        Token-weighted CE using gain-derived weights:
        - For positions [0..L-2], weight_t = 4 * log(1 + gain_t) / sum_t log(1 + gain_t)
        - EOS positions (label == eos_token_id) keep weight = 1.0
        - Positions with ignore_index=-100 are masked out
        Shapes:
        - lm_logits: (B, L, V)
        - labels:    (B, L)
        - gain:      (L-1,)
        """
        if labels is None:
            return None

        device = lm_logits.device
        logits = lm_logits.to(device) #/ self.temperature
        B, L, V = logits.shape
        labels = labels.to(device)

        # ----- per-token CE (no reduction) -----
        ce = F.cross_entropy(
            logits.view(-1, V),            # (B*L, V)
            labels.view(-1),               # (B*L,)
            reduction="none",
            ignore_index=-100,
        )                                  # (B*L,)

        # ----- build per-position weights from 'gain' for first L-1 steps -----
        # w_raw = log(1 + gain); normalize to sum=1; then scale by 4.
        eps = 1e-12
        gain = self.gain.to(device).float()                   # (L-1,)
        assert gain.shape[0] == L - 1, "gain must be length L-1"

        w_raw = torch.log1p(torch.clamp(gain, min=0.0))  # (L-1,)
        denom = torch.clamp(w_raw.sum(), min=eps)
        w_pos = 4.0 * (w_raw / denom)                    # (L-1,), sums to 4.0

        # If gain is all zeros -> use uniform over first L-1 (still sums to 4)
        if torch.all(w_raw <= eps):
            w_pos = torch.full((L - 1,), 4.0 / max(L - 1, 1), device=device)

        # Create a (B, L) matrix of weights, fill first L-1 with w_pos, last step initially 0
        pos_w = torch.zeros((B, L), device=device, dtype=torch.float32)
        if L > 1:
            pos_w[:, :L - 1] = w_pos.unsqueeze(0).expand(B, L - 1)

        # ----- mask out ignore positions -----
        valid2d = (labels != -100)                        # (B, L)
        token_w = (pos_w * valid2d.float()).view(-1)      # (B*L,)

        # ----- force EOS tokens to weight = 1.0 -----
        eos_id = getattr(self.config, "eos_token_id", None)
        if eos_id is not None:
            labels_flat = labels.view(-1)
            eos_mask = (labels_flat == eos_id).float()    # 1 at EOS tokens
            # set weight=1 at EOS, keep others as-is
            token_w = token_w * (1.0 - eos_mask) + eos_mask * 1.0


        loss = (ce * token_w).sum() / (token_w.sum() + eps) # / den
        return loss

    def igd_weight_loss(self, lm_logits, labels, decoder_input_ids):
        # decoder_input_ids provide prefix
        if self.igd is None:
            return None  # fallback

        B, L, V = lm_logits.shape
        logits = lm_logits #/ self.temperature
        ce = F.cross_entropy(
            logits.view(-1, V),
            labels.view(-1),
            reduction="none",
            ignore_index=-100,
        )

        wt_list = []
        for b in range(B):
            prefix = []
            for t in range(L):
                y_t = int(labels[b, t])
                if y_t == -100:
                    wt_list.append(1.0)
                    continue

                key = (tuple(prefix), y_t)
                ig_val = self.igd.get(key, 0.0)
                # print(tuple(prefix), ig_val)
                wt = 1.0 if ig_val > 0 else self.igd_beta
                wt_list.append(wt)
                prefix.append(y_t)

        wt_tensor = torch.tensor(wt_list, device=ce.device)
        loss = (ce * wt_tensor).sum() / wt_tensor.sum()
        return loss

    def _weighted_ce_loss(self, logits, labels, use_pos_weights=True, normalize_by_omega=True):
        """
        logits: (B, L, V)
        labels: (B, L)
        use_pos_weights: whether to apply token-position weights ω_t
        normalize_by_omega: if True, divide by Ω=sum(ω_t), else by sum over valid positions

        Returns:
            scalar loss
        """
        device = logits.device
        B, L, V = logits.shape

        labels = labels.to(device)
        ce = F.cross_entropy(
            logits.view(-1, V),
            labels.view(-1),
            reduction="none",
            ignore_index=-100,
        )  # (B*L,)

        valid_mask = (labels.view(-1) != -100).float()  # (B*L,)

        if use_pos_weights:
            w_flat, omega = self._build_token_pos_weights(labels)
            w_flat = w_flat.to(device) * valid_mask
            denom = omega if normalize_by_omega else w_flat.sum().clamp(min=1e-12)
        else:
            w_flat = valid_mask
            denom = w_flat.sum().clamp(min=1e-12)

        loss = (ce * w_flat).sum() / denom
        return loss

    def _curriculum_multi_target_weights(self):
        
        precisions = torch.exp(-self.log_vars)

        pace_front_origin = torch.exp(torch.tensor(-self.c * self.warm_up_steps, device=precisions.device))#
        pace_token = 1.0 - torch.exp(torch.tensor(-self.c * self.warm_up_steps, device=precisions.device))
        pace = torch.tensor([pace_front_origin, pace_front_origin, pace_token], device=precisions.device)
        precisions = precisions #* pace
        w = precisions / (precisions.sum() + 1e-12)       # normalize to sum=1
        reg = self.log_vars.sum() * 0.5                 # Kendall regularizer (not used)
        return w, reg
    
    def total_loss(self, lm_logits, labels, decoder_input_ids,  item_ids=None, lm_logits_cf=None):
        
        loss_token = self.token_weight_loss(lm_logits, labels)
        loss_origin = self.ranking_loss(lm_logits, labels) #original loss
        loss_front = self.front_greater_loss(lm_logits, labels)
        
        self.warm_up_steps += 1
        lbda_token = self.warm_up_steps * 0.000004
        lbda_token = min(lbda_token, 0.4)

        lbda_front = self.warm_up_steps * 0.000010
        lbda_front = min(lbda_front, 1.0)
        if self.args.method == "combine":
            return ((1-lbda_front-lbda_token) * loss_front) + ((lbda_front) * loss_origin) + \
                (lbda_token * loss_token)

        elif self.args.method == "front_gain" or self.args.method == "front_reverse":
            return ((1-lbda_front) * loss_front) + (lbda_front * loss_origin)

        elif "origin" in self.args.method or "rank" in self.args.method:
            return loss_origin###reverse

        elif "pos" in self.args.method:
            loss_pos = self._weighted_ce_loss(
                    lm_logits, labels,
                    use_pos_weights=True,
                    normalize_by_omega=True,
                )
            return loss_pos
        elif self.args.method == "_token_freq":
            return (lbda_token * loss_token) + ((1-lbda_token) * loss_origin)
        elif "igd" in self.args.method:
            loss_igd = self.igd_weight_loss(lm_logits, labels, decoder_input_ids)
            return loss_igd
        elif "cft" in self.args.method:
            """
            CFT baseline implementation:
              L_n = normal loss (with or without position weights)
              L_c = causal loss using logits_diff = logits_with_hist - logits_without_hist
              Total L = L_n + λ * L_c
            """
            # 1) Normal loss: f_theta(x_h, y_<t)
            if self.cft_weight_normal:
                loss_n = self._weighted_ce_loss(
                    lm_logits, labels,
                    use_pos_weights=True,
                    normalize_by_omega=True,
                )
            else:
                # fallback: plain CE (your original ranking loss)
                loss_n = self.ranking_loss(lm_logits, labels)

            # 2) Causal loss: use difference logits
            loss_c = 0.0
            if (lm_logits_cf is not None) and (self.cft_lambda > 0.0):
                logits_diff = lm_logits - lm_logits_cf  # (B, L, V)
                
                loss_c = self._weighted_ce_loss(
                    logits_diff, labels,
                    use_pos_weights=True,
                    normalize_by_omega=True,
                )

            total = loss_n + self.cft_lambda * loss_c
            # logging
            self.last_loss_components = {
                'L_n': float(loss_n.detach()),
                'L_c': float(loss_c if isinstance(loss_c, float) else loss_c.detach()),
                'lambda': float(self.cft_lambda),
            }
            return total
        elif "ours" in self.args.method:

            # Pack in the SAME order as self.log_vars
            Ls = torch.stack([loss_origin, loss_front, loss_token])     # [3], dtype=float, device=...
            # Learnable weights (sum-to-1)
            w, reg = self._curriculum_multi_target_weights()                        # w.sum() == 1

            # Mixed loss stays on CE-like scale due to sum-to-1
            loss_mix = torch.dot(w, Ls)

            # Final loss (add tiny, scale-free regularizer)
            total = loss_mix #+ reg

            # (Optional) If you want to freeze dynamic weights during warmup:
            # if step is not None and step < self.warm_up_steps:
            #     total = Ls[0]  # e.g., use origin only during warmup
            #     # or: total = torch.mean(Ls)  # equal mixing without learning

            # For logging/inspection (no grads through logs)
            self.last_loss_components = {
                'L_origin': float(loss_origin.detach()),
                'L_front': float(loss_front.detach()),
                'L_token': float(loss_token.detach()),
                'w_origin': float(w[0].detach()),
                'w_front': float(w[1].detach()),
                'w_token': float(w[2].detach()),
                'reg': float(reg.detach()),
            }
            return total

    def forward(
        self,
        input_ids=None,
        whole_word_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        cross_attn_head_mask = None,
        past_key_values=None,
        use_cache=None,
        labels=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
        decoder_head_mask = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        item_ids=None,#for item freq weight
        input_ids_cf=None,
        attention_mask_cf=None,
        encoder_outputs_cf=None,
        reduce_loss=False,

        return_hidden_state=False,

        **kwargs,
    ):
        r"""

        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,

            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)
        

        # ------------------------------------------
        # (NEW) Counterfactual branch for CFT
        # ------------------------------------------
        lm_logits_cf = None
        if ("cft" in self.args.method) and (input_ids_cf is not None):
            # Encode counterfactual input: x_0 (history removed or set to "None")
            if encoder_outputs_cf is None:
                encoder_outputs_cf = self.encoder(
                    input_ids=input_ids_cf,
                    attention_mask=attention_mask_cf,
                    inputs_embeds=None,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            elif return_dict and not isinstance(encoder_outputs_cf, BaseModelOutput):
                encoder_outputs_cf = BaseModelOutput(
                    last_hidden_state=encoder_outputs_cf[0],
                    hidden_states=encoder_outputs_cf[1] if len(encoder_outputs_cf) > 1 else None,
                    attentions=encoder_outputs_cf[2] if len(encoder_outputs_cf) > 2 else None,
                )

            hidden_states_cf = encoder_outputs_cf[0]

            # If you are not using model_parallel,可以忽略這段
            if self.model_parallel:
                torch.cuda.set_device(self.decoder.first_device)
                hidden_states_cf = hidden_states_cf.to(self.decoder.first_device)

            # decoder uses the same decoder_input_ids / labels (y_<t)
            decoder_outputs_cf = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=hidden_states_cf,
                encoder_attention_mask=attention_mask_cf,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            sequence_output_cf = decoder_outputs_cf[0]

            if self.model_parallel:
                torch.cuda.set_device(self.encoder.first_device)
                self.lm_head = self.lm_head.to(self.encoder.first_device)
                sequence_output_cf = sequence_output_cf.to(self.lm_head.weight.device)

            if self.config.tie_word_embeddings:
                sequence_output_cf = sequence_output_cf * (self.model_dim ** -0.5)

            lm_logits_cf = self.lm_head(sequence_output_cf)
        # ------------------------------------------
        # Loss Computing!
        loss = None
        loss = self.total_loss(lm_logits, labels, decoder_input_ids, item_ids, lm_logits_cf=lm_logits_cf)

        # ------------------------------------------

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
