"""Tier 1 — The Encoder (Frozen Prior Layer).

Trained during pretraining.  A few thousand Hebbian steps.  Never touched again.
Takes raw token IDs and produces structured representations for higher layers.

Encodes:
  - Token co-occurrence primitives
  - Basic syntactic structure
  - Fundamental attention primitives
  - Sequence directionality
  - Basic cause/effect linking

Does NOT encode facts, concepts, or semantic meaning.
"""

from __future__ import annotations

import torch

from .genome import Genome
from .tensor import DTYPE, get_device
from .types_ import StructuredRepresentation


class Encoder:
    """Frozen prior layer that converts raw tokens into structured representations."""

    def __init__(self, genome: Genome, rng: object | None = None) -> None:
        self._genome = genome
        self._device = get_device()
        topo = genome.topology

        # Learnable (during pretrain only) weight matrices
        self.token_embeddings = torch.randn(
            (topo.vocab_size, topo.embed_dim), dtype=DTYPE, device=self._device,
        ) * 0.02
        self.token_embeddings.requires_grad_(True)

        self.cooccurrence_weights = torch.randn(
            (topo.embed_dim, topo.encoder_hidden), dtype=DTYPE, device=self._device,
        ) * 0.02
        self.cooccurrence_weights.requires_grad_(True)

        self.syntax_weights = torch.randn(
            (topo.encoder_hidden, topo.encoder_hidden), dtype=DTYPE, device=self._device,
        ) * 0.02
        self.syntax_weights.requires_grad_(True)

        self.attention_query = torch.randn(
            (topo.encoder_hidden, topo.embed_dim), dtype=DTYPE, device=self._device,
        ) * 0.02
        self.attention_query.requires_grad_(True)

        self.attention_key = torch.randn(
            (topo.encoder_hidden, topo.embed_dim), dtype=DTYPE, device=self._device,
        ) * 0.02
        self.attention_key.requires_grad_(True)

        self.output_projection = torch.randn(
            (topo.encoder_hidden, topo.embed_dim), dtype=DTYPE, device=self._device,
        ) * 0.02
        self.output_projection.requires_grad_(True)

        self._frozen = False

    # ------------------------------------------------------------------
    # Pretraining
    # ------------------------------------------------------------------

    def pretrain(self, token_sequences: list[torch.Tensor]) -> None:
        """Run Hebbian pretraining on a corpus of token sequences.

        Each sequence is a 1-D array of token IDs.  We do genome.generation.
        encoder_pretrain_steps Hebbian updates across the sequences.
        """
        params = self._genome.hebbian
        steps = self._genome.generation.encoder_pretrain_steps
        n_seqs = len(token_sequences)
        if n_seqs == 0:
            self._frozen = True
            return

        try:
            from tqdm import trange
        except Exception:
            trange = range

        for step in trange(steps, desc="Encoder pretrain", leave=False):
            seq = token_sequences[step % n_seqs]
            if len(seq) < 2:
                continue

            seq = seq.to(device=self._device, dtype=torch.long)
            embeddings = self.token_embeddings[seq]  # (seq_len, embed_dim)
            hidden = self._relu(embeddings @ self.cooccurrence_weights)  # (seq_len, encoder_hidden)
            syntactic = self._relu(hidden @ self.syntax_weights)  # (seq_len, encoder_hidden)

            # Hebbian update: Δw = η · pre^T · post  (averaged over sequence)
            lr = params.learning_rate
            seq_len = len(seq)

            # Perform updates without tracking gradients
            with torch.no_grad():
                # Co-occurrence weights
                delta_co = (embeddings.T @ hidden) / seq_len
                self.cooccurrence_weights += lr * delta_co
                self.cooccurrence_weights.clamp_(params.min_weight, params.max_weight)

                # Syntax weights
                delta_syn = (hidden.T @ syntactic) / seq_len
                self.syntax_weights += lr * delta_syn
                self.syntax_weights.clamp_(params.min_weight, params.max_weight)

                # Embedding update via reconstruction error signal
                reconstructed = syntactic @ self.output_projection  # (seq_len, embed_dim)
                error = embeddings - reconstructed
                delta_emb = (syntactic.T @ error) / seq_len
                self.output_projection += lr * 0.5 * delta_emb
                self.output_projection.clamp_(params.min_weight, params.max_weight)

                # Attention weight update via co-activation of Q and K
                Q = syntactic @ self.attention_query  # (seq_len, embed_dim)
                K = syntactic @ self.attention_key
                delta_q = (syntactic.T @ Q) / seq_len
                delta_k = (syntactic.T @ K) / seq_len
                self.attention_query += lr * 0.3 * delta_q
                self.attention_key += lr * 0.3 * delta_k
                self.attention_query.clamp_(params.min_weight, params.max_weight)
                self.attention_key.clamp_(params.min_weight, params.max_weight)

            # --- Token embedding update via next-token prediction ---
            # For each position t, predict embedding of token t+1 from context at t.
            # The output_projection maps encoder_hidden → embed_dim, so we use
            # syntactic[t] @ output_projection as the predicted next embedding, and
            # compare against the actual next embedding.  This gives the embeddings
            # a sequential / predictive structure rather than staying random.
            if seq_len > 1:
                with torch.no_grad():
                    predicted_next = syntactic[:-1] @ self.output_projection  # (seq_len-1, embed_dim)
                    actual_next = self.token_embeddings[seq[1:].long()]        # (seq_len-1, embed_dim)
                    emb_error = actual_next - predicted_next                   # (seq_len-1, embed_dim)
                    # Update token_embeddings for the *next* tokens toward the prediction
                    for pos in range(seq_len - 1):
                        tid = int(seq[pos + 1].item())
                        self.token_embeddings[tid] += lr * 0.2 * emb_error[pos]
                    self.token_embeddings.clamp_(params.min_weight, params.max_weight)

        self._frozen = True

    def freeze(self) -> None:
        self._frozen = True

    def unfreeze(self) -> None:
        """Allow online Hebbian updates during conversation."""
        self._frozen = False

    # ------------------------------------------------------------------
    # Gradient helpers
    # ------------------------------------------------------------------

    def parameters(self) -> list[torch.Tensor]:
        """Return a list of tensors that should receive gradient updates.

        These are the same weights that Hebbian learning adjusts; marking them
        `requires_grad=True` and collecting them allows an optimizer to work
        alongside the existing plasticity rules.
        """
        return [
            self.token_embeddings,
            self.cooccurrence_weights,
            self.syntax_weights,
            self.attention_query,
            self.attention_key,
            self.output_projection,
        ]

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def online_update(
        self,
        token_ids: torch.Tensor,
        lr: float = 0.0001,
        ewc_scalars: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Light Hebbian update after each conversation turn.

        Keeps the encoder plastic during conversation at a much lower rate
        than pretraining, so it can adapt to the specific conversational
        vocabulary without catastrophically forgetting pretrain structure.
        Only runs when the encoder is unfrozen.
        """
        if self._frozen:
            return
        params = self._genome.hebbian
        seq = token_ids.to(device=self._device, dtype=torch.long)
        seq_len = int(seq.shape[0])
        if seq_len < 2:
            return

        embeddings = self.token_embeddings[seq]       # (seq_len, embed_dim)
        hidden     = self._relu(embeddings @ self.cooccurrence_weights)  # (seq_len, bs_h)
        syntactic  = self._relu(hidden @ self.syntax_weights)            # (seq_len, bs_h)

        lr = lr
        # perform updates without gradient tracking
        with torch.no_grad():
            # Co-occurrence (small update)
            delta_co = lr * (embeddings.T @ hidden) / seq_len
            co_protection = None if ewc_scalars is None else ewc_scalars.get("cooccurrence_weights")
            if co_protection is not None and co_protection.ndim == 1 and co_protection.numel() == delta_co.shape[1]:
                delta_co = delta_co * (1.0 - co_protection).unsqueeze(0)
            self.cooccurrence_weights += delta_co
            self.cooccurrence_weights.clamp_(params.min_weight, params.max_weight)

            # Next-token embedding prediction
            if seq_len > 1:
                predicted_next = syntactic[:-1] @ self.output_projection  # (seq_len-1, embed_dim)
                actual_next    = self.token_embeddings[seq[1:].long()]     # (seq_len-1, embed_dim)
                emb_error      = actual_next - predicted_next
                token_protection = None if ewc_scalars is None else ewc_scalars.get("token_embeddings")
                for pos in range(seq_len - 1):
                    tid = int(seq[pos + 1].item())
                    token_lr = lr
                    if token_protection is not None and tid < int(token_protection.numel()):
                        token_lr *= float(max(0.0, 1.0 - token_protection[tid].item()))
                    self.token_embeddings[tid] += token_lr * emb_error[pos]
                self.token_embeddings.clamp_(params.min_weight, params.max_weight)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def process(self, token_ids: torch.Tensor) -> StructuredRepresentation:
        """Convert raw token IDs into a structured representation.

        Args:
            token_ids: 1-D int array of token IDs.

        Returns:
            StructuredRepresentation ready for the reinforcement gate.
        """
        token_ids = token_ids.to(device=self._device, dtype=torch.long)
        seq_len = int(token_ids.shape[0])
        topo = self._genome.topology

        # Embed
        embeddings = self.token_embeddings[token_ids]  # (seq_len, embed_dim)

        # Positional encoding — sinusoidal
        pos_enc = self._positional_encoding(seq_len, topo.embed_dim)

        # Hidden representations via co-occurrence and syntax layers
        hidden = self._relu(embeddings @ self.cooccurrence_weights)
        syntactic = self._relu(hidden @ self.syntax_weights)

        # Self-attention: Q, K from syntactic; V = syntactic projected
        Q = syntactic @ self.attention_query   # (seq_len, embed_dim)
        K = syntactic @ self.attention_key     # (seq_len, embed_dim)
        scale = float(topo.embed_dim) ** 0.5
        attn_logits = (Q @ K.T) / scale        # (seq_len, seq_len)

        # Causal mask — sequence directionality (language moves forward)
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=DTYPE, device=self._device),
            diagonal=1,
        ) * -1e9
        attn_logits += causal_mask

        attn_weights = self._softmax(attn_logits, axis=-1)

        # Final structured output projected back to embed_dim
        context = attn_weights @ syntactic     # (seq_len, encoder_hidden)
        projected = context @ self.output_projection  # (seq_len, embed_dim)

        return StructuredRepresentation(
            token_ids=token_ids,
            embeddings=projected + pos_enc,
            positional_encoding=pos_enc,
            attention_weights=attn_weights,
        )

    # ------------------------------------------------------------------
    # Weight I/O
    # ------------------------------------------------------------------

    def get_weights(self) -> dict[str, torch.Tensor]:
        return {
            "token_embeddings": self.token_embeddings.detach().clone(),
            "cooccurrence_weights": self.cooccurrence_weights.detach().clone(),
            "syntax_weights": self.syntax_weights.detach().clone(),
            "attention_query": self.attention_query.detach().clone(),
            "attention_key": self.attention_key.detach().clone(),
            "output_projection": self.output_projection.detach().clone(),
        }

    def load_weights(self, weights: dict[str, torch.Tensor]) -> None:
        for name, arr in weights.items():
            current = getattr(self, name, None)
            if current is not None and current.shape == arr.shape:
                setattr(self, name, arr.to(device=self._device, dtype=DTYPE).clone())
        # Keep encoder unfrozen after load so conversation can continue training it.

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _relu(x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)

    @staticmethod
    def _softmax(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
        shifted = x - x.max(dim=axis, keepdim=True).values
        exp = torch.exp(shifted)
        return exp / (exp.sum(dim=axis, keepdim=True) + 1e-12)

    @staticmethod
    def _positional_encoding(seq_len: int, dim: int) -> torch.Tensor:
        pos = torch.arange(seq_len, dtype=DTYPE, device=get_device()).unsqueeze(1)
        i = torch.arange(dim, dtype=DTYPE, device=get_device()).unsqueeze(0)
        angle_rates = 1.0 / torch.pow(torch.tensor(10000.0, dtype=DTYPE, device=get_device()), (2 * torch.floor(i / 2)) / dim)
        angles = pos * angle_rates
        # sin on even indices, cos on odd
        angles[:, 0::2] = torch.sin(angles[:, 0::2])
        angles[:, 1::2] = torch.cos(angles[:, 1::2])
        return angles
