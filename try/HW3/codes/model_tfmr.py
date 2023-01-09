from turtle import position
import jittor as jt
import jittor.nn as jnn
import time
import random


def jt_masked_select(invar:jt.Var,mask:jt.Var)->jt.Var:
    assert invar.shape == mask.shape
    invar2 = invar.reshape(-1)
    mask2 = mask.reshape(-1)
    res = []
    for i in range(invar2.shape[0]):
        if (mask2[i]):
            res.append(invar2[i])
    return jt.concat(res)

def jt_multinomial(prob:jt.Var,samples:int)->jt.Var:
    assert len(prob.shape)==2
    prob2 = jt.cumsum(prob,dim=-1) 
    pmax = prob2[:,-1].view(1,-1).transpose().expand(-1,prob.shape[1])
    prob2 = prob2/pmax
    sample_rand = jt.random((prob.shape[0],samples))
    out_var = jt.searchsorted(prob2,sample_rand)
    return out_var
    
ACT2FN = {
    "relu": jt.nn.relu,
    "tanh": jt.tanh,
    "linear": lambda x: x,
    "sigmoid": jt.sigmoid,
    "gelu": jt.nn.gelu,
}

class TransposeLinear(jnn.Module):
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = jt.init.gauss((nx,nf),std=0.02)
        self.weight = w
        self.bias = jt.zeros(nf)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = jt.matmul(x.view(-1, x.size(-1)),self.weight) + self.bias
        # x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x

class TfmrAttention(jnn.Module):
    def __init__(self, config):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.bias=jt.tril(jt.ones((max_positions, max_positions))).view((1, 1, max_positions, max_positions))
        
        self.masked_bias=jt.float64(-1e4)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.c_attn = TransposeLinear(3 * self.embed_dim, self.embed_dim)
        self.c_proj = TransposeLinear(self.embed_dim, self.embed_dim)

        self.attn_dropout = jnn.Dropout(config.attn_pdrop)
        self.resid_dropout = jnn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()

    def _attn(self, query:jt.Var, key:jt.Var, value):
        # TODO START
        # implement the multi-head mask self-attnetion mechanism
        attn_weights = jt.matmul(query, key.transpose(-1, -2))
        # query_length, key_length = query.shape(-2), key.size(-2)
        # causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
        attn_weights = jnn.Softmax(dim=-1)(attn_weights)
        attn_output = jt.matmul(attn_weights, value)

        return attn_output, attn_weights
        # TODO END

    def _split_heads(self, tensor:jt.Var, num_heads, attn_head_size):
        # TODO START
        new_shape = tensor.shape[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        # TODO END

    def _merge_heads(self, tensor:jt.Var, num_heads, attn_head_size):
        # TODO START
        tensor = tensor.permute(0, 2, 1, 3)#.contiguous()
        new_shape = tensor.shape[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)
        # TODO END

    def forward(
        self,
        hidden_states,
        layer_past=None,
        use_cache=False,
    ):
        query, key, value = self.c_attn.forward(hidden_states).split(self.split_size, dim=2)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = jt.concat((past_key, key), dim=-2)
            value = jt.concat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj.forward(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class TfmrMLP(jnn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = TransposeLinear(intermediate_size, embed_dim)
        self.c_proj = TransposeLinear(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = jnn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.c_fc.forward(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj.forward(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class TfmrBlock(jnn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.prefix_word = jt.init.gauss((config.prefix_word_num,config.hidden_size),std=config.initializer_range) if config.prefix_word_num>0 else None

        self.ln_1 = jnn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = TfmrAttention(config)
        self.ln_2 = jnn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = TfmrMLP(inner_dim, config)

    # TODO add prefix before hidden_states and take it as trained paras.
    def forward(
        self,
        hidden_states,
        layer_past=None,
        use_cache=False,
    ):

        word_len = hidden_states.shape[-2]
        # prefix
        if self.prefix_word is not None:
            expand_shape = hidden_states.shape[:-2] + self.prefix_word.shape
            hidden_states = jt.concat(
                (hidden_states , self.prefix_word.expand(expand_shape)),
                dim=-2
            )
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn.forward(
            hidden_states,
            layer_past=layer_past,
            use_cache=use_cache,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]


        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp.forward(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        if self.prefix_word is not None:
            hidden_states =  hidden_states[...,0:word_len,:]

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs


class TfmrModel(jnn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size

        self.wte = jnn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = jnn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = jnn.Dropout(config.embd_pdrop)
        self.h = jnn.ModuleList([TfmrBlock(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = jnn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def get_input_embeddings(self):
        return self.wte

    def forward(
        self,
        input_ids,
        past_key_values=None,
        use_cache=None,
    ):
        input_shape = input_ids.shape
        input_ids = input_ids.view(-1, input_shape[-1])
        inputs_embeds = self.wte(input_ids)
        # TODO START
        # Implement the positional embeddings. Note that the length of cache hidden states used during inference
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        position_ids = jt.arange(past_length, input_shape[-1] + past_length)# , dtype="int64")
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        position_embeds = self.wpe(position_ids)
        # TODO END
        hidden_states = inputs_embeds + position_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = ()
        all_cross_attentions = ()
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = block.forward(
                hidden_states,
                layer_past=layer_past,
                use_cache=use_cache,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        all_hidden_states = all_hidden_states + (hidden_states,)

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": presents,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attentions,
            "cross_attentions": all_cross_attentions,
        }


class TfmrLMHeadModel(jnn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = TfmrModel(config)
        self.lm_head = jnn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids,
        past_key_values=None,
        labels=None,
        use_cache=None,
        PAD_ID=None,
    ):
        transformer_outputs = self.transformer.forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = transformer_outputs["last_hidden_state"]
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :]#.contiguous()
            shift_labels = labels[..., 1:]#.contiguous()

            pad_pos = jt.equal(shift_labels, PAD_ID).float32()
            pad_pos = jt.concat([jt.zeros([shift_labels.size()[0], 1]), pad_pos[:, :-1]], 1)
            loss_mask = 1. - pad_pos
            loss = jt.nn.cross_entropy_loss(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1),reduction="none")
            loss = jt.mean(jt.sum(loss.view(shift_labels.size()[0], -1) * loss_mask, 1) / (jt.sum(loss_mask, 1) + 1e-20))

        return {
            "loss": loss,
            "logits": lm_logits,
            "past_key_values": transformer_outputs["past_key_values"],
            "hidden_states": transformer_outputs["hidden_states"],
            "attentions": transformer_outputs["attentions"],
            "cross_attentions": transformer_outputs["cross_attentions"],
         }
        

    def inference(self, PAD_ID, batch_size, maxlen, decode_strategy, temperature, top_p=1.0, top_k=50267):
        allgen = []
        self.eval()
        with jt.no_grad():
            for i in range(0, int(5000/batch_size)+1):
                input_ids = jt.float64([[PAD_ID] for _ in range(batch_size)])
                past_key_values = None
                output_ids = input_ids
                for _ in range(maxlen):
                    outputs = self.forward(input_ids, past_key_values=past_key_values, use_cache=True)
                    logits = outputs["logits"]
                    past_key_values = outputs["past_key_values"]
                    logits = logits[:, -1, :] / temperature
                    if decode_strategy == "top-p" :
                        #sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        sorted_indices,sorted_logits= jt.argsort(logits, descending=True)
                        cumulative_probs = jt.cumsum(jt.nn.softmax(sorted_logits, dim=-1), dim=-1)

                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        # Shift the indices to the right to keep also the first token above the threshold
                        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1]
                        sorted_indices_to_remove[:, 0] = 0
                        sorted_indices = (sorted_indices + jt.arange(sorted_indices.shape[0]).int64().unsqueeze(dim=-1) * sorted_indices.shape[1])
                        # indices_to_remove = torch.masked_select(sorted_indices, sorted_indices_to_remove)
                        indices_to_remove = jt_masked_select(sorted_indices, sorted_indices_to_remove)
                        logits = logits.reshape(-1)
                        # logits = torch.index_fill(logits, 0, indices_to_remove, -float("inf"))
                        logits = jt.index_fill_(logits, 0, indices_to_remove, -1e20)
                        logits = logits.reshape(sorted_indices.shape[0], sorted_indices.shape[1])
                    elif decode_strategy == "top-k":
                        top_k = min(top_k, logits.size(-1))
                        indices_to_remove = logits < jt.topk(logits, top_k,dim=-1)[0][..., -1, None]
                        logits = logits.masked_fill(indices_to_remove, -1e20)
                    prob = logits.softmax(dim=-1) # shape: (batch_size, num_vocabs)
                    # now_token = torch.multinomial(prob, 1)[:, :1] # shape: (batch_size)
                    jt.set_global_seed(int(1000*time.time()) % 2147483647)
                    now_token = jt.multinomial(prob, 1)[:, :1] # shape: (batch_size)
                    output_ids = jt.concat([output_ids, now_token], 1)
                    input_ids = now_token
                allgen += output_ids.numpy().tolist()
        pro_allgen = []
        for gen in allgen[:5000]:
            pro_allgen.append([])
            for idx in gen[1:]:
                if idx == PAD_ID:
                    break
                pro_allgen[-1].append(idx)
        return pro_allgen
                