from typing import Literal, Callable, Optional
import flax.linen as nn
import jax.numpy as jnp
import torch

from utils.quantum_layer import QuantumLayer

# See:
# - https://nlp.seas.harvard.edu/annotated-transformer/
# - https://github.com/rdisipio/qtransformer/blob/main/qtransformer.py
# - https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py


class MultiHeadSelfAttention(nn.Module):
    hidden_size: int
    num_heads: int
    dropout: float = 0.0

    quantum_w_shape: tuple = (1,)
    quantum_circuit: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, deterministic, q=None, k=None, v=None, mask=None):
        batch_size, seq_len, hidden_size = x.shape
        # x.shape = (batch_size, seq_len, hidden_size)
        assert (
            hidden_size == self.hidden_size
        ), f"Input hidden size ({hidden_size}) does not match layer hidden size ({self.hidden_size})"
        assert (
            hidden_size % self.num_heads == 0
        ), f"Hidden size ({hidden_size}) must be divisible by the number of heads ({self.num_heads})"
        head_dim = hidden_size // self.num_heads

        if q is None and k is None and v is None and self.quantum_circuit is None:
            q, k, v = [
                proj(x)
                .reshape(batch_size, seq_len, self.num_heads, head_dim)
                .swapaxes(1, 2)
                for proj, x in zip(
                    [
                        nn.Dense(features=hidden_size),
                        nn.Dense(features=hidden_size),
                        nn.Dense(features=hidden_size),
                    ],
                    [x, x, x],
                )
            ]
        elif q is None and k is None and v is None:
            q, k, v = [
                proj(x)
                .reshape(batch_size, seq_len, self.num_heads, head_dim)
                .swapaxes(1, 2)
                for proj, x in zip(
                    [
                        QuantumLayer(
                            num_qubits=hidden_size,
                            w_shape=self.quantum_w_shape,
                            circuit=self.quantum_circuit,
                        ),
                        QuantumLayer(
                            num_qubits=hidden_size,
                            w_shape=self.quantum_w_shape,
                            circuit=self.quantum_circuit,
                        ),
                        QuantumLayer(
                            num_qubits=hidden_size,
                            w_shape=self.quantum_w_shape,
                            circuit=self.quantum_circuit,
                        ),
                    ],
                    [x, x, x],
                )
            ]

        # Compute scaled dot-product attention
        attn_logits = (q @ k.swapaxes(-2, -1)) / jnp.sqrt(head_dim)
        # attn_logits.shape = (batch_size, num_heads, seq_len, seq_len)
        if mask is not None:
            mask = jnp.broadcast_to(mask, attn_logits.shape)
            attn_logits = jnp.where(mask == 0, -1e9, attn_logits)

        attn = nn.softmax(attn_logits, axis=-1)
        # attn.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = nn.Dropout(rate=self.dropout)(attn, deterministic=deterministic)

        # Compute output
        values = attn @ v

        # values.shape = (batch_size, num_heads, seq_len, head_dim)
        values = values.swapaxes(1, 2).reshape(batch_size, seq_len, hidden_size)
        # values.shape = (batch_size, seq_len, hidden_size)
        if self.quantum_circuit is None:
            x = nn.Dense(features=hidden_size)(values)
        else:
            x = QuantumLayer(
                num_qubits=hidden_size,
                w_shape=self.quantum_w_shape,
                circuit=self.quantum_circuit,
            )(values)
        # x.shape = (batch_size, seq_len, hidden_size)

        return x


class FeedForward(nn.Module):
    hidden_size: int
    mlp_hidden_size: int
    dropout: float = 0.0

    quantum_w_shape: tuple = (1,)
    quantum_circuit: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, deterministic):
        x = nn.Dense(features=self.mlp_hidden_size)(x)
        if self.quantum_circuit is not None:
            x = QuantumLayer(
                num_qubits=self.mlp_hidden_size,
                w_shape=self.quantum_w_shape,
                circuit=self.quantum_circuit,
            )(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        x = nn.gelu(x)
        x = nn.Dense(features=self.hidden_size)(x)
        return x


class TransformerDecoder(nn.Module):
    hidden_size: int
    num_heads: int
    mlp_hidden_size: int
    dropout: float = 0.0

    quantum_w_shape: tuple = (1,)
    quantum_attn_circuit: Optional[Callable] = None
    quantum_mlp_circuit: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, src_mask, trg_mask, e_outputs, deterministic, dim=None):
        if dim == None:
            attn1_output = nn.LayerNorm()(x)
        else:
            attn1_output = nn.LayerNorm(dim)(x)
        attn1_output = MultiHeadSelfAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
            quantum_circuit=self.quantum_attn_circuit,
        )(
            attn1_output,
            q=None,
            k=None,
            v=None,
            deterministic=deterministic,
            mask=trg_mask,
        )
        attn1_output = nn.Dropout(rate=self.dropout)(
            attn1_output, deterministic=deterministic
        )

        x = x + attn1_output

        if dim == None:
            attn2_output = nn.LayerNorm()(x)
        else:
            attn2_output = nn.LayerNorm(dim)(x)

        attn2_output = MultiHeadSelfAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
            quantum_circuit=self.quantum_attn_circuit,
        )(
            attn2_output,
            q=attn2_output,
            k=e_outputs,
            v=e_outputs,
            deterministic=deterministic,
            mask=src_mask,
        )
        attn2_output = nn.Dropout(rate=self.dropout)(
            attn2_output, deterministic=deterministic
        )

        x = x + attn2_output

        if dim == None:
            y = nn.LayerNorm()(x)
        else:
            y = nn.LayerNorm(dim)(x)

        y = FeedForward(
            hidden_size=self.hidden_size,
            mlp_hidden_size=self.mlp_hidden_size,
            quantum_circuit=self.quantum_mlp_circuit,
        )(y, deterministic=deterministic)
        y = nn.Dropout(rate=self.dropout)(y, deterministic=deterministic)

        return x + y


class TransformerRK1(nn.Module):
    hidden_size: int
    num_heads: int
    mlp_hidden_size: int
    dropout: float = 0.0

    quantum_w_shape: tuple = (1,)
    quantum_attn_circuit: Optional[Callable] = None
    quantum_mlp_circuit: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, mask, deterministic, dim=None):
        if dim == None:
            x_norm = nn.LayerNorm()(x)
        else:
            x_norm = nn.LayerNorm(dim)(x)

        attn_output = MultiHeadSelfAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
            quantum_circuit=self.quantum_attn_circuit,
        )(x_norm, q=None, k=None, v=None, deterministic=deterministic, mask=mask)
        attn_output = nn.Dropout(rate=self.dropout)(
            attn_output, deterministic=deterministic
        )

        y = FeedForward(
            hidden_size=self.hidden_size,
            mlp_hidden_size=self.mlp_hidden_size,
            quantum_circuit=self.quantum_mlp_circuit,
        )(x_norm, deterministic=deterministic)
        y = nn.Dropout(rate=self.dropout)(y, deterministic=deterministic)

        return attn_output + y + x_norm


class TransformerRK2(nn.Module):
    hidden_size: int
    num_heads: int
    mlp_hidden_size: int
    dropout: float = 0.0

    quantum_w_shape: tuple = (1,)
    quantum_attn_circuit: Optional[Callable] = None
    quantum_mlp_circuit: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, mask, deterministic, dim=None):
        if dim == None:
            x_norm = nn.LayerNorm()(x)
        else:
            x_norm = nn.LayerNorm(dim)(x)

        attn_output1 = MultiHeadSelfAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
            quantum_circuit=self.quantum_attn_circuit,
        )(x_norm, q=None, k=None, v=None, deterministic=deterministic, mask=mask)
        attn_output1 = nn.Dropout(rate=self.dropout)(
            attn_output1, deterministic=deterministic
        )

        y1 = FeedForward(
            hidden_size=self.hidden_size,
            mlp_hidden_size=self.mlp_hidden_size,
            quantum_circuit=self.quantum_mlp_circuit,
        )(x_norm, deterministic=deterministic)
        y1 = nn.Dropout(rate=self.dropout)(y1, deterministic=deterministic)

        attn_output2 = MultiHeadSelfAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
            quantum_circuit=self.quantum_attn_circuit,
        )(
            x_norm + attn_output1 + y1,
            q=None,
            k=None,
            v=None,
            deterministic=deterministic,
            mask=mask,
        )
        attn_output2 = nn.Dropout(rate=self.dropout)(
            attn_output2, deterministic=deterministic
        )

        y2 = FeedForward(
            hidden_size=self.hidden_size,
            mlp_hidden_size=self.mlp_hidden_size,
            quantum_circuit=self.quantum_mlp_circuit,
        )(x_norm + attn_output1 + y1, deterministic=deterministic)
        y2 = nn.Dropout(rate=self.dropout)(y2, deterministic=deterministic)

        return x_norm + 1 / 2 * (attn_output1 + y1) + 1 / 2 * (attn_output2 + y2)


class TransformerRK3(nn.Module):
    hidden_size: int
    num_heads: int
    mlp_hidden_size: int
    dropout: float = 0.0

    quantum_w_shape: tuple = (1,)
    quantum_attn_circuit: Optional[Callable] = None
    quantum_mlp_circuit: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, mask, deterministic, dim=None):
        if dim == None:
            x_norm = nn.LayerNorm()(x)
        else:
            x_norm = nn.LayerNorm(dim)(x)

        attn_output1 = MultiHeadSelfAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
            quantum_circuit=self.quantum_attn_circuit,
        )(x_norm, q=None, k=None, v=None, deterministic=deterministic, mask=mask)
        attn_output1 = nn.Dropout(rate=self.dropout)(
            attn_output1, deterministic=deterministic
        )

        y1 = FeedForward(
            hidden_size=self.hidden_size,
            mlp_hidden_size=self.mlp_hidden_size,
            quantum_circuit=self.quantum_mlp_circuit,
        )(x_norm, deterministic=deterministic)
        y1 = nn.Dropout(rate=self.dropout)(y1, deterministic=deterministic)

        attn_output2 = MultiHeadSelfAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
            quantum_circuit=self.quantum_attn_circuit,
        )(
            x_norm + 1 / 2 * (attn_output1 + y1),
            q=None,
            k=None,
            v=None,
            deterministic=deterministic,
            mask=mask,
        )
        attn_output2 = nn.Dropout(rate=self.dropout)(
            attn_output2, deterministic=deterministic
        )

        y2 = FeedForward(
            hidden_size=self.hidden_size,
            mlp_hidden_size=self.mlp_hidden_size,
            quantum_circuit=self.quantum_mlp_circuit,
        )(x_norm + 1 / 2 * (attn_output1 + y1), deterministic=deterministic)
        y2 = nn.Dropout(rate=self.dropout)(y2, deterministic=deterministic)

        attn_output3 = MultiHeadSelfAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
            quantum_circuit=self.quantum_attn_circuit,
        )(
            x_norm + 3 / 4 * (attn_output2 + y2),
            q=None,
            k=None,
            v=None,
            deterministic=deterministic,
            mask=mask,
        )
        attn_output3 = nn.Dropout(rate=self.dropout)(
            attn_output3, deterministic=deterministic
        )

        y3 = FeedForward(
            hidden_size=self.hidden_size,
            mlp_hidden_size=self.mlp_hidden_size,
            quantum_circuit=self.quantum_mlp_circuit,
        )(
            x_norm + 2 / 3 * (attn_output2 + y2),
            deterministic=deterministic,
        )
        y3 = nn.Dropout(rate=self.dropout)(y3, deterministic=deterministic)

        return x_norm + 1 / 9 * (
            2 * (attn_output1 + y1) + 3 * (attn_output2 + y2) + 4 * (attn_output3 + y3)
        )


class TransformerRK4(nn.Module):
    hidden_size: int
    num_heads: int
    mlp_hidden_size: int
    dropout: float = 0.0

    quantum_w_shape: tuple = (1,)
    quantum_attn_circuit: Optional[Callable] = None
    quantum_mlp_circuit: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, mask, deterministic, dim=None):
        if dim == None:
            x_norm = nn.LayerNorm()(x)
        else:
            x_norm = nn.LayerNorm(dim)(x)

        attn_output1 = MultiHeadSelfAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
            quantum_circuit=self.quantum_attn_circuit,
        )(x_norm, q=None, k=None, v=None, deterministic=deterministic, mask=mask)
        attn_output1 = nn.Dropout(rate=self.dropout)(
            attn_output1, deterministic=deterministic
        )

        y1 = FeedForward(
            hidden_size=self.hidden_size,
            mlp_hidden_size=self.mlp_hidden_size,
            quantum_circuit=self.quantum_mlp_circuit,
        )(x_norm, deterministic=deterministic)
        y1 = nn.Dropout(rate=self.dropout)(y1, deterministic=deterministic)

        attn_output2 = MultiHeadSelfAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
            quantum_circuit=self.quantum_attn_circuit,
        )(
            x_norm + 1 / 2 * (attn_output1 + y1),
            q=None,
            k=None,
            v=None,
            deterministic=deterministic,
            mask=mask,
        )
        attn_output2 = nn.Dropout(rate=self.dropout)(
            attn_output2, deterministic=deterministic
        )

        y2 = FeedForward(
            hidden_size=self.hidden_size,
            mlp_hidden_size=self.mlp_hidden_size,
            quantum_circuit=self.quantum_mlp_circuit,
        )(x_norm + 1 / 2 * (attn_output1 + y1), deterministic=deterministic)
        y2 = nn.Dropout(rate=self.dropout)(y2, deterministic=deterministic)

        attn_output3 = MultiHeadSelfAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
            quantum_circuit=self.quantum_attn_circuit,
        )(
            x_norm + 1 / 2 * (attn_output2 + y2),
            q=None,
            k=None,
            v=None,
            deterministic=deterministic,
            mask=mask,
        )
        attn_output3 = nn.Dropout(rate=self.dropout)(
            attn_output3, deterministic=deterministic
        )

        y3 = FeedForward(
            hidden_size=self.hidden_size,
            mlp_hidden_size=self.mlp_hidden_size,
            quantum_circuit=self.quantum_mlp_circuit,
        )(x_norm + 1 / 2 * (attn_output2 + y2), deterministic=deterministic)
        y3 = nn.Dropout(rate=self.dropout)(y3, deterministic=deterministic)

        attn_output4 = MultiHeadSelfAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
            quantum_circuit=self.quantum_attn_circuit,
        )(
            x_norm + attn_output3 + y3,
            q=None,
            k=None,
            v=None,
            deterministic=deterministic,
            mask=mask,
        )
        attn_output4 = nn.Dropout(rate=self.dropout)(
            attn_output4, deterministic=deterministic
        )

        y4 = FeedForward(
            hidden_size=self.hidden_size,
            mlp_hidden_size=self.mlp_hidden_size,
            quantum_circuit=self.quantum_mlp_circuit,
        )(x_norm + attn_output3 + y3, deterministic=deterministic)
        y4 = nn.Dropout(rate=self.dropout)(y4, deterministic=deterministic)

        return x_norm + 1 / 6 * (
            attn_output1
            + y1
            + 2 * (attn_output2 + y2)
            + 2 * (attn_output3 + y3)
            + attn_output4
            + y4
        )


class TransformerRK4Enhanced(nn.Module):
    hidden_size: int
    num_heads: int
    mlp_hidden_size: int
    dropout: float = 0.0

    quantum_w_shape: tuple = (1,)
    quantum_attn_circuit: Optional[Callable] = None
    quantum_mlp_circuit: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, mask, deterministic, dim=None):
        if dim == None:
            x_norm = nn.LayerNorm()(x)
        else:
            x_norm = nn.LayerNorm(dim)(x)

        attn_output1 = MultiHeadSelfAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
            quantum_circuit=self.quantum_attn_circuit,
        )(x_norm, q=None, k=None, v=None, deterministic=deterministic, mask=mask)
        attn_output1 = nn.Dropout(rate=self.dropout)(
            attn_output1, deterministic=deterministic
        )

        y1 = FeedForward(
            hidden_size=self.hidden_size,
            mlp_hidden_size=self.mlp_hidden_size,
            quantum_circuit=self.quantum_mlp_circuit,
        )(x_norm, deterministic=deterministic)
        y1 = nn.Dropout(rate=self.dropout)(y1, deterministic=deterministic)

        attn_output2 = MultiHeadSelfAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
            quantum_circuit=self.quantum_attn_circuit,
        )(
            x_norm + (attn_output1 + y1),
            q=None,
            k=None,
            v=None,
            deterministic=deterministic,
            mask=mask,
        )
        attn_output2 = nn.Dropout(rate=self.dropout)(
            attn_output2, deterministic=deterministic
        )

        y2 = FeedForward(
            hidden_size=self.hidden_size,
            mlp_hidden_size=self.mlp_hidden_size,
            quantum_circuit=self.quantum_mlp_circuit,
        )(x_norm + (attn_output1 + y1), deterministic=deterministic)
        y2 = nn.Dropout(rate=self.dropout)(y2, deterministic=deterministic)

        attn_output3 = MultiHeadSelfAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
            quantum_circuit=self.quantum_attn_circuit,
        )(
            x_norm +  (attn_output2 + y2),
            q=None,
            k=None,
            v=None,
            deterministic=deterministic,
            mask=mask,
        )
        attn_output3 = nn.Dropout(rate=self.dropout)(
            attn_output3, deterministic=deterministic
        )

        y3 = FeedForward(
            hidden_size=self.hidden_size,
            mlp_hidden_size=self.mlp_hidden_size,
            quantum_circuit=self.quantum_mlp_circuit,
        )(x_norm + (attn_output2 + y2), deterministic=deterministic)
        y3 = nn.Dropout(rate=self.dropout)(y3, deterministic=deterministic)

        attn_output4 = MultiHeadSelfAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
            quantum_circuit=self.quantum_attn_circuit,
        )(
            x_norm + attn_output3 + y3,
            q=None,
            k=None,
            v=None,
            deterministic=deterministic,
            mask=mask,
        )
        attn_output4 = nn.Dropout(rate=self.dropout)(
            attn_output4, deterministic=deterministic
        )

        y4 = FeedForward(
            hidden_size=self.hidden_size,
            mlp_hidden_size=self.mlp_hidden_size,
            quantum_circuit=self.quantum_mlp_circuit,
        )(x_norm + attn_output3 + y3, deterministic=deterministic)
        y4 = nn.Dropout(rate=self.dropout)(y4, deterministic=deterministic)

        return x_norm + (
            attn_output1
            + y1
            + attn_output4
            + y4
        )


class TransformerEncoder(nn.Module):
    hidden_size: int
    num_heads: int
    mlp_hidden_size: int
    dropout: float = 0.0

    quantum_w_shape: tuple = (1,)
    quantum_attn_circuit: Optional[Callable] = None
    quantum_mlp_circuit: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, mask, deterministic, dim=None):
        if dim == None:
            attn_output = nn.LayerNorm()(x)
        else:
            attn_output = nn.LayerNorm(dim)(x)
        attn_output = MultiHeadSelfAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
            quantum_circuit=self.quantum_attn_circuit,
        )(attn_output, q=None, k=None, v=None, deterministic=deterministic, mask=mask)
        attn_output = nn.Dropout(rate=self.dropout)(
            attn_output, deterministic=deterministic
        )
        x = x + attn_output

        if dim == None:
            y = nn.LayerNorm()(x)
        else:
            y = nn.LayerNorm(dim)(x)

        y = FeedForward(
            hidden_size=self.hidden_size,
            mlp_hidden_size=self.mlp_hidden_size,
            quantum_circuit=self.quantum_mlp_circuit,
        )(y, deterministic=deterministic)
        y = nn.Dropout(rate=self.dropout)(y, deterministic=deterministic)

        return x + y


def posemb_sincos_2d(
    sqrt_num_steps, hidden_size, temperature=10_000.0, dtype=jnp.float32
):
    """2D sin-cos position embedding. Follows the MoCo v3 logic."""
    # Code adapted from https://github.com/google-research/big_vision/blob/184d1201eb34abe7da84fc69f84fd89a06ad43c4/big_vision/models/vit.py#L33.
    y, x = jnp.mgrid[:sqrt_num_steps, :sqrt_num_steps]

    assert (
        hidden_size % 4 == 0
    ), f"Hidden size ({hidden_size}) must be divisible by 4 for 2D sin-cos position embedding"
    omega = jnp.arange(hidden_size // 4) / (hidden_size // 4 - 1)
    omega = 1.0 / (temperature**omega)
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
    return jnp.asarray(pe, dtype)[None, :, :]


class Transformer(nn.Module):
    num_tokens: int
    max_seq_len: int
    num_classes: int
    patch_size: int
    hidden_size: int
    num_heads: int
    num_transformer_encoder_blocks: int
    num_transformer_rk1_blocks: int
    num_transformer_rk2_blocks: int
    num_transformer_rk3_blocks: int
    num_transformer_rk4_blocks: int
    num_transformer_rk4_enhanced_blocks: int
    num_transformer_decoder_blocks: int
    mlp_hidden_size: int
    dropout: float = 0.1

    quantum_w_shape: tuple = (1,)
    quantum_attn_circuit: Optional[Callable] = None
    quantum_mlp_circuit: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, src, trg, src_mask, trg_mask, train):
        # Token embedding
        x_src = nn.Embed(num_embeddings=self.num_tokens, features=src)(x)
        x_trg = nn.Embed(num_embeddings=self.num_tokens, features=trg)(x)
        # x.shape = (batch_size, seq_len, hidden_size)

        # Positional embedding
        x_src += nn.Embed(num_embeddings=self.max_seq_len, features=src)(
            jnp.arange(x_src.shape[1])
        )
        x_trg += nn.Embed(num_embeddings=self.max_seq_len, features=trg)(
            jnp.arange(x_trg.shape[1])
        )

        # Dropout
        x_src = nn.Dropout(rate=self.dropout)(x_src, deterministic=not train)
        x_trg = nn.Dropout(rate=self.dropout)(x_trg, deterministic=not train)

        for _ in range(self.num_transformer_encoder_blocks):
            x_src = TransformerEncoder(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_hidden_size=self.mlp_hidden_size,
                dropout=self.dropout,
                quantum_attn_circuit=self.quantum_attn_circuit,
                quantum_mlp_circuit=self.quantum_mlp_circuit,
            )(x_src, deterministic=not train, mask=src_mask)

        for _ in range(self.num_transformer_rk1_blocks):
            x_src = TransformerRK1(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_hidden_size=self.mlp_hidden_size,
                dropout=self.dropout,
                quantum_attn_circuit=self.quantum_attn_circuit,
                quantum_mlp_circuit=self.quantum_mlp_circuit,
            )(x_src, deterministic=not train, mask=src_mask)

        for _ in range(self.num_transformer_rk2_blocks):
            x_src = TransformerRK2(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_hidden_size=self.mlp_hidden_size,
                dropout=self.dropout,
                quantum_attn_circuit=self.quantum_attn_circuit,
                quantum_mlp_circuit=self.quantum_mlp_circuit,
            )(x_src, deterministic=not train, mask=src_mask)

        for _ in range(self.num_transformer_rk3_blocks):
            x_src = TransformerRK3(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_hidden_size=self.mlp_hidden_size,
                dropout=self.dropout,
                quantum_attn_circuit=self.quantum_attn_circuit,
                quantum_mlp_circuit=self.quantum_mlp_circuit,
            )(x_src, deterministic=not train, mask=src_mask)

        for _ in range(self.num_transformer_rk4_blocks):
            x_src = TransformerRK4(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_hidden_size=self.mlp_hidden_size,
                dropout=self.dropout,
                quantum_attn_circuit=self.quantum_attn_circuit,
                quantum_mlp_circuit=self.quantum_mlp_circuit,
            )(x_src, deterministic=not train, mask=src_mask)

        for _ in range(self.num_transformer_rk4_enhanced_blocks):
            x_src = TransformerRK4Enhanced(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_hidden_size=self.mlp_hidden_size,
                dropout=self.dropout,
                quantum_attn_circuit=self.quantum_attn_circuit,
                quantum_mlp_circuit=self.quantum_mlp_circuit,
            )(x_src, deterministic=not train, mask=src_mask)

        # Layer normalization
        x_src = nn.LayerNorm()(x_src)

        for _ in range(self.num_transformer_decoder_blocks):
            x_trg = TransformerDecoder(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_hidden_size=self.mlp_hidden_size,
                dropout=self.dropout,
                quantum_attn_circuit=self.quantum_attn_circuit,
                quantum_mlp_circuit=self.quantum_mlp_circuit,
            )(
                x_src,
                src_mask=src_mask,
                trg_mask=trg_mask,
                e_outputs=x_src,
                deterministic=not train,
            )

        # Global average pooling
        x_trg = jnp.mean(x_trg, axis=1)
        # x.shape = (batch_size, hidden_size)

        # Classification logits
        x_trg = nn.Dense(self.num_classes)(x_trg)
        # x.shape = (batch_size, num_classes)

        return x_trg


class VisionTransformer(nn.Module):
    num_classes: int
    patch_size: int
    hidden_size: int
    num_heads: int
    num_transformer_encoder_blocks: int
    num_transformer_rk1_blocks: int
    num_transformer_rk2_blocks: int
    num_transformer_rk3_blocks: int
    num_transformer_rk4_blocks: int
    num_transformer_rk4_enhanced_blocks: int
    num_transformer_decoder_blocks: int
    mlp_hidden_size: int
    dropout: float = 0.1
    pos_embedding: Literal["none", "learn", "sincos"] = "learn"
    classifier: Literal["token", "gap"] = "gap"
    channels_last: bool = True

    quantum_w_shape: tuple = (1,)
    quantum_attn_circuit: Optional[Callable] = None
    quantum_mlp_circuit: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, src, trg, src_mask, trg_mask, train):
        assert x.ndim == 4, f"Input must be 4D, got {x.ndim}D ({x.shape})"

        if not self.channels_last:
            x = x.transpose((0, 3, 1, 2))
        # x.shape = (batch_size, height, width, num_channels)
        # Note that JAX's Conv expects the input to be in the format (batch_size, height, width, num_channels)

        batch_size, height, width, _ = x.shape
        assert height == width, f"Input must be square, got {height}x{width}"
        img_size = height
        num_steps = (img_size // self.patch_size) ** 2

        # Splitting an image into patches and linearly projecting these flattened patches can be
        # simplified as a single convolution operation, where both the kernel size and the stride size
        # are set to the patch size.
        x_src = nn.Conv(
            features=src,
            kernel_size=(self.patch_size, self.patch_size),
            strides=self.patch_size,
            padding="VALID",
        )(x)

        x_trg = nn.Conv(
            features=trg,
            kernel_size=(self.patch_size, self.patch_size),
            strides=self.patch_size,
            padding="VALID",
        )(x)

        # x.shape = (batch_size, sqrt(num_steps), sqrt(num_steps), hidden_size)
        sqrt_num_steps_src = x_src.shape[1]
        sqrt_num_steps_trg = x_trg.shape[1]

        x_src = jnp.reshape(x_src, (batch_size, num_steps, src))
        x_trg = jnp.reshape(x_trg, (batch_size, num_steps, trg))
        # x.shape = (batch_size, num_steps, hidden_size)

        # Positional embedding
        if self.pos_embedding == "learn":
            x_src += self.param(
                "pos_embedding_src",
                nn.initializers.normal(stddev=1 / jnp.sqrt(src)),
                (1, num_steps, src),
                x_src.dtype,
            )
            x_trg += self.param(
                "pos_embedding_trg",
                nn.initializers.normal(stddev=1 / jnp.sqrt(trg)),
                (1, num_steps, trg),
                x_trg.dtype,
            )
        elif self.pos_embedding == "sincos":
            x_src += posemb_sincos_2d(sqrt_num_steps_src, src, dtype=x_src.dtype)
            x_trg += posemb_sincos_2d(sqrt_num_steps_trg, trg, dtype=x_trg.dtype)
        elif self.pos_embedding == "none":
            pass
        else:
            raise ValueError(f"Unknown positional embedding type: {self.pos_embedding}")

        if self.classifier == "token":
            # CLS token
            cls_token_src = self.param("cls", nn.initializers.zeros, (1, 1, src))
            cls_token_src = jnp.tile(cls_token_src, (batch_size, 1, 1))

            cls_token_trg = self.param("cls", nn.initializers.zeros, (1, 1, trg))
            cls_token_trg = jnp.tile(cls_token_trg, (batch_size, 1, 1))

            x_src = jnp.concatenate([cls_token_src, x_src], axis=1)
            x_trg = jnp.concatenate([cls_token_trg, x_trg], axis=1)
            num_steps += 1
            # x.shape = (batch_size, num_steps, hidden_size)

        # Dropout
        x_src = nn.Dropout(rate=self.dropout)(x_src, deterministic=not train)
        x_trg = nn.Dropout(rate=self.dropout)(x_trg, deterministic=not train)

        # Transformer blocks
        for _ in range(self.num_transformer_encoder_blocks):
            x_src = TransformerEncoder(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_hidden_size=self.mlp_hidden_size,
                dropout=self.dropout,
                quantum_attn_circuit=self.quantum_attn_circuit,
                quantum_mlp_circuit=self.quantum_mlp_circuit,
            )(x_src, deterministic=not train, mask=src_mask)

        for _ in range(self.num_transformer_rk1_blocks):
            x_src = TransformerRK1(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_hidden_size=self.mlp_hidden_size,
                dropout=self.dropout,
                quantum_attn_circuit=self.quantum_attn_circuit,
                quantum_mlp_circuit=self.quantum_mlp_circuit,
            )(x_src, deterministic=not train, mask=src_mask)

        for _ in range(self.num_transformer_rk2_blocks):
            x_src = TransformerRK2(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_hidden_size=self.mlp_hidden_size,
                dropout=self.dropout,
                quantum_attn_circuit=self.quantum_attn_circuit,
                quantum_mlp_circuit=self.quantum_mlp_circuit,
            )(x_src, deterministic=not train, mask=src_mask)

        for _ in range(self.num_transformer_rk3_blocks):
            x_src = TransformerRK3(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_hidden_size=self.mlp_hidden_size,
                dropout=self.dropout,
                quantum_attn_circuit=self.quantum_attn_circuit,
                quantum_mlp_circuit=self.quantum_mlp_circuit,
            )(x_src, deterministic=not train, mask=src_mask)

        for _ in range(self.num_transformer_rk4_blocks):
            x_src = TransformerRK4(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_hidden_size=self.mlp_hidden_size,
                dropout=self.dropout,
                quantum_attn_circuit=self.quantum_attn_circuit,
                quantum_mlp_circuit=self.quantum_mlp_circuit,
            )(x_src, deterministic=not train, mask=src_mask)

        for _ in range(self.num_transformer_rk4_enhanced_blocks):
            x_src = TransformerRK4Enhanced(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_hidden_size=self.mlp_hidden_size,
                dropout=self.dropout,
                quantum_attn_circuit=self.quantum_attn_circuit,
                quantum_mlp_circuit=self.quantum_mlp_circuit,
            )(x_src, deterministic=not train, mask=src_mask)

        # Layer normalization
        x_src = nn.LayerNorm()(x_src)

        for _ in range(self.num_transformer_decoder_blocks):
            x_trg = TransformerDecoder(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_hidden_size=self.mlp_hidden_size,
                dropout=self.dropout,
                quantum_attn_circuit=self.quantum_attn_circuit,
                quantum_mlp_circuit=self.quantum_mlp_circuit,
            )(
                x_trg,
                src_mask=src_mask,
                trg_mask=trg_mask,
                e_outputs=x_src,
                deterministic=not train,
            )

        # Layer normalization
        x_trg = nn.LayerNorm()(x_trg)
        # x.shape = (batch_size, num_steps, hidden_size)

        if self.classifier == "token":
            # Get the classifcation token
            x_trg = x_trg[:, 0]
        elif self.classifier == "gap":
            # Global average pooling
            x_trg = jnp.mean(x_trg, axis=1)
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier}")
        # x.shape = (batch_size, hidden_size)

        # Classification logits
        x_trg = nn.Dense(features=self.num_classes)(x_trg)
        # x.shape = (batch_size, num_classes)

        return x_trg
