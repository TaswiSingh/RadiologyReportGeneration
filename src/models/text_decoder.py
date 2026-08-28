import torch
import torch.nn as nn


class TextDecoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model=512,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        max_length=128
    ):
        super().__init__()

        self.d_model = d_model
        self.max_length = max_length

        # Token IDs -> embeddings
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=0
        )

        # Position embeddings
        self.position_embedding = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=d_model
        )

        # One Transformer decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Stack decoder layers
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        # Decoder output -> vocabulary
        self.output_projection = nn.Linear(
            d_model,
            vocab_size
        )

    def forward(
        self,
        input_ids,
        memory,
        attention_mask=None
    ):
        """
        input_ids:
            [B, T]

        memory:
            [B, 49, 512]

        attention_mask:
            [B, T]
            True = real token
            False = padding

        returns:
            logits [B, T, vocab_size]
        """

        batch_size, seq_length = input_ids.shape

        # Safety check
        if seq_length > self.max_length:
            raise ValueError(
                f"Sequence length {seq_length} "
                f"exceeds max_length {self.max_length}"
            )

        # Position IDs: 0, 1, 2, ..., T-1
        positions = torch.arange(
            seq_length,
            device=input_ids.device
        )

        positions = positions.unsqueeze(0).expand(
            batch_size,
            seq_length
        )

        # Token embeddings
        token_embeddings = self.token_embedding(
            input_ids
        )

        # Position embeddings
        position_embeddings = self.position_embedding(
            positions
        )

        # Combine token + position information
        x = token_embeddings + position_embeddings

        # Scale embeddings
        x = x * (self.d_model ** 0.5)

        # Causal mask
        # Prevents the decoder from seeing future tokens
        causal_mask = torch.triu(
            torch.ones(
                 seq_length,
                 seq_length,
                 dtype=torch.bool,
                 device=input_ids.device
            ),
            diagonal=1
        )

        # Padding mask
        # Transformer expects:
        # True  = ignore
        # False = use
        if attention_mask is not None:
            padding_mask = ~attention_mask
        else:
            padding_mask = None

        # Transformer decoder
        x = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask
        )

        # Predict vocabulary scores
        logits = self.output_projection(x)

        return logits