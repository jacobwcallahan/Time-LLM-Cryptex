from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer, AutoConfig, AutoModel, AutoTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    """
    Flattens the input tensor and applies a linear layer followed by dropout.
    Used as the final projection head for the model's output.
    Args:
        n_vars: Number of variables (features)
        nf: Flattened feature dimension
        target_window: Output window size (prediction length)
        head_dropout: Dropout rate for the head
    """
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        # x: [batch, n_vars, patch_nums, d_ff]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    TimeLLM Model: Integrates a pretrained LLM (e.g., Llama, GPT2, BERT, etc.)
    with time series patch embedding and reprogramming layers for forecasting.
    - Loads a frozen LLM backbone and tokenizer
    - Uses patch embedding and reprogramming to adapt time series to LLM input
    - Generates prompts with time series statistics for LLM context
    - Output head projects LLM output to prediction window
    """
    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        # Store key hyperparameters
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        # Make top_k robust: never greater than seq_len
        self.top_k = min(5, self.seq_len - 1)  # Number of lags to report in prompt
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        # Define supported LLM model configurations
        self.model_configs = {
            'LLAMA': {
                'config_class': LlamaConfig,
                'model_class': LlamaModel,
                'tokenizer_class': LlamaTokenizer,
                'model_name': 'huggyllama/llama-7b'
            },
            'GPT2': {
                'config_class': GPT2Config,
                'model_class': GPT2Model,
                'tokenizer_class': GPT2Tokenizer,
                'model_name': 'openai-community/gpt2'
            },
            'BERT': {
                'config_class': BertConfig,
                'model_class': BertModel,
                'tokenizer_class': BertTokenizer,
                'model_name': 'google-bert/bert-base-uncased'
            },
            'DEEPSEEK': {
                'config_class': AutoConfig,
                'model_class': AutoModel,
                'tokenizer_class': AutoTokenizer,
                'model_name': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
            },
            'QWEN': {
                'config_class': AutoConfig,
                'model_class': AutoModel,
                'tokenizer_class': AutoTokenizer,
                'model_name': 'Qwen/Qwen3-8B'
            },
            'MISTRAL': {
                'config_class': AutoConfig,
                'model_class': AutoModel,
                'tokenizer_class': AutoTokenizer,
                'model_name': 'mistralai/Mistral-7B-v0.1'
            },
            'GEMMA': {
                'config_class': AutoConfig,
                'model_class': AutoModel,
                'tokenizer_class': AutoTokenizer,
                'model_name': 'google/gemma-3-1b-it'
            },
            'LLAMA3.1': {
                'config_class': AutoConfig,
                'model_class': AutoModel,
                'tokenizer_class': AutoTokenizer,
                'model_name': 'meta-llama/Llama-3.1-8B'
            }
            # Add new models here
        }

        # Check if requested LLM is supported
        if configs.llm_model not in self.model_configs:
            raise Exception(f'LLM model {configs.llm_model} is not defined. Available models: {list(self.model_configs.keys())}')

        # Get model configuration for the selected LLM
        model_config = self.model_configs[configs.llm_model]
        
        # Initialize LLM config and set custom parameters
        self.llm_config = model_config['config_class'].from_pretrained(model_config['model_name'], trust_remote_code=True)
        self.llm_config.num_hidden_layers = configs.llm_layers
        self.llm_config.output_attentions = True
        self.llm_config.output_hidden_states = True

        # Load the LLM model (frozen)
        try:
            self.llm_model = model_config['model_class'].from_pretrained(
                model_config['model_name'],
                trust_remote_code=True,
                local_files_only=True,
                config=self.llm_config
            )
        except Exception as e:
            print(f"Error loading model {configs.llm_model}: {str(e)}")
            print("Attempting to download...")
            self.llm_model = model_config['model_class'].from_pretrained(
                model_config['model_name'],
                trust_remote_code=True,
                local_files_only=False,
                config=self.llm_config
            )

        # Set d_llm from the LLM's embedding dimension for robustness
        self.d_llm = self.llm_model.get_input_embeddings().embedding_dim

        # Load the tokenizer for the LLM
        try:
            self.tokenizer = model_config['tokenizer_class'].from_pretrained(
                model_config['model_name'],
                trust_remote_code=True,
                local_files_only=True,
                config=self.llm_config
            )
        except Exception as e:
            print(f"Error loading tokenizer for {configs.llm_model}: {str(e)}")
            print("Attempting to download...")
            self.tokenizer = model_config['tokenizer_class'].from_pretrained(
                model_config['model_name'],
                trust_remote_code=True,
                local_files_only=False,
                config=self.llm_config
            )
        # Ensure pad token is set
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        # Freeze LLM parameters (no fine-tuning)
        for param in self.llm_model.parameters():
            param.requires_grad = False

        # Require a prompt to be passed for context
        if not getattr(configs, 'content', None):
            raise ValueError('You must provide a prompt via configs.content.')
        self.description = configs.content

        self.dropout = nn.Dropout(configs.dropout)

        # Patch embedding for time series input
        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        # LLM word embeddings and mapping layer
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = configs.num_tokens
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        # Reprogramming layer to adapt time series to LLM
        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        # Calculate number of patches and head feature size
        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        # Output projection head (maps LLM output to prediction)
        self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len, head_dropout=configs.dropout)

        # Normalization layers for input/output
        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, input_data, mask=None):
        """
        Forward pass for TimeLLM.
        Args:
            input_data: [batch, seq_len, num_features]
            mask: (optional) attention mask
        Returns:
            output: [batch, pred_len, num_features]
        """
        # Normalize input
        input_data = self.normalize_layers(input_data, 'norm')
        B, T, N = input_data.size()
        # Reshape to [B*N, T, 1] for per-feature processing
        input_data = input_data.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        # Compute statistics for prompt
        min_values = torch.min(input_data, dim=1)[0]
        max_values = torch.max(input_data, dim=1)[0]
        medians = torch.median(input_data, dim=1).values
        lags = self.calcute_lags(input_data)
        trends = input_data.diff(dim=1).sum(dim=1)
        prompt = []
        for b in range(input_data.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top {self.top_k} lags are : {lags_values_str}<|<end_prompt>|>"
            )
            prompt.append(prompt_)
        # Reshape back to [B, N, T] for patch embedding
        input_data = input_data.reshape(B, N, T).permute(0, 2, 1).contiguous()
        # Tokenize prompt and get embeddings
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(input_data.device))
        # Map LLM word embeddings to num_tokens
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        # Patch embedding for time series
        input_data = input_data.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(input_data.to(torch.bfloat16))
        # Reprogramming layer
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        # Concatenate prompt and encoded input
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        # LLM forward pass
        output = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        output = output[:, :, :self.d_ff]
        # Reshape and project output
        output = torch.reshape(
            output, (-1, n_vars, output.shape[-2], output.shape[-1]))
        output = output.permute(0, 1, 3, 2).contiguous()
        output = self.output_projection(output[:, :, :, -self.patch_nums:])
        output = output.permute(0, 2, 1).contiguous()
        # Denormalize output
        output = self.normalize_layers(output, 'denorm')
        return output[:, -self.pred_len:, :]

    def calcute_lags(self, x_enc):
        """
        Calculate top-k lags using FFT-based autocorrelation.
        Args:
            x_enc: [batch, seq_len, 1]
        Returns:
            lags: [batch, top_k]
        """
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding

# --- Saving Note ---
# This model contains a frozen LLM loaded via HuggingFace Transformers.
# torch.save(model) will NOT save the LLM weights (only references to them).
# To save and reload the model safely:
# 1. Save the model config and state_dict (torch.save(model.state_dict(), ...))
# 2. Save the LLM separately using HuggingFace's save_pretrained() for both model and tokenizer.
# 3. On reload, reconstruct the TimeLLM model, load the LLM and tokenizer with from_pretrained(),
#    then load the state_dict for the rest of the model.
