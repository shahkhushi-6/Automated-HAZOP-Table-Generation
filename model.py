"""
model.py — HAZOPModel
======================
Architecture:
    GNN (2 layers of SAGEConv)  →  node embeddings
    Guideword embedding table   →  guideword vector
    Projection MLP              →  T5 encoder hidden states
    T5 decoder                  →  text for each output field

The model is called with a batch dict (produced by hazop_collate_fn) and
returns the combined cross-entropy loss over all output text fields.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm
from transformers import T5ForConditionalGeneration, T5Config


# ──────────────────────────────────────────────────────────────────────────────
# Graph Neural Network
# ──────────────────────────────────────────────────────────────────────────────

class GraphEncoder(nn.Module):
    """
    Multi-layer GraphSAGE encoder with edge feature fusion.

    SAGEConv aggregates neighbour features by mean-pooling, then
    concatenates with the centre node's own features and applies a
    linear transform. Edge features are projected and fused into
    destination node representations.

    Input:
        x          — (N_total, node_feat_dim)  node feature matrix
        edge_index — (2, E_total)              directed edges [src, dst]
        edge_attr  — (E_total, edge_feat_dim)  edge features (optional)

    Output:
        node_emb   — (N_total, gnn_out)        context-aware embeddings
    """

    def __init__(self, node_feat_dim: int, hidden_dim: int,
                 out_dim: int, n_layers: int = 2,
                 edge_feat_dim: int = 3, dropout: float = 0.1):
        super().__init__()

        self.dropout = dropout
        self.out_dim = out_dim

        # Project edge features to node_feat_dim before message passing
        self.edge_proj = nn.Linear(edge_feat_dim, node_feat_dim)

        # Stack of SAGEConv layers with batch normalization
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = node_feat_dim
        for i in range(n_layers):
            out = out_dim if i == n_layers - 1 else hidden_dim
            self.convs.append(SAGEConv(in_dim, out, normalize=True))
            self.norms.append(BatchNorm(out))
            in_dim = out

    def _fuse_edge_features(self, x, edge_index, edge_attr):
        """
        Add projected edge features to destination nodes.
        
        This lets the model condition node representations on the
        properties of incoming edges.
        
        Args:
            x: (N, node_feat_dim) node features
            edge_index: (2, E) edge indices [src, dst]
            edge_attr: (E, edge_feat_dim) or None

        Returns:
            x + edge contributions: (N, node_feat_dim)
        """
        if edge_attr is None:
            return x

        edge_contrib = self.edge_proj(edge_attr)  # (E, node_feat_dim)
        dst = edge_index[1]  # destination node indices

        # Scatter sum edge contributions into destination nodes
        agg = torch.zeros_like(x)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_contrib),
                        edge_contrib)
        return x + agg

    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: (N, node_feat_dim) node features
            edge_index: (2, E) edge indices
            edge_attr: (E, edge_feat_dim) edge features, optional

        Returns:
            (N, gnn_out) node embeddings
        """
        x = self._fuse_edge_features(x, edge_index, edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x  # (N, gnn_out)


# ──────────────────────────────────────────────────────────────────────────────
# Combined HAZOP model
# ──────────────────────────────────────────────────────────────────────────────

class HAZOPModel(nn.Module):
    """
    GNN + T5 model for HAZOP text generation.

    forward() accepts a batch dict from hazop_collate_fn, runs the full
    pipeline, computes cross-entropy loss for each output field, and
    returns the summed loss (for training) or per-field loss dict
    (for evaluation).

    Workflow:
        1. Encode the batched graph using GraphEncoder (GNN)
        2. Extract target node embeddings and fuse with guideword embeddings
        3. Project combined context to T5 hidden size
        4. For each output field:
           - Append field-specific prefix token
           - Compute T5 loss in teacher-forcing mode
        5. Return sum of all field losses
    """

    def __init__(
        self,
        node_feat_dim: int,
        gnn_hidden: int,
        gnn_out: int,
        gnn_layers: int,
        num_guidewords: int,
        gw_embed_dim: int,
        t5_model_name: str,
        output_keys: list,
        edge_feat_dim: int = 3,
        gnn_dropout: float = 0.1,
    ):
        super().__init__()

        self.output_keys = output_keys

        # ── GNN ──────────────────────────────────────────────────────────
        self.gnn = GraphEncoder(
            node_feat_dim=node_feat_dim,
            hidden_dim=gnn_hidden,
            out_dim=gnn_out,
            n_layers=gnn_layers,
            edge_feat_dim=edge_feat_dim,
            dropout=gnn_dropout,
        )

        # ── Guideword embedding table ────────────────────────────────────
        self.gw_embed = nn.Embedding(num_guidewords, gw_embed_dim)

        # ── Context MLP: (gnn_out + gw_embed_dim) → t5_hidden_dim ───────
        t5_config = T5Config.from_pretrained(t5_model_name)
        t5_d_model = t5_config.d_model  # typically 512 for t5-small

        context_dim = gnn_out + gw_embed_dim
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, t5_d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(t5_d_model * 2, t5_d_model),
        )

        # ── T5 — one shared model, called per output field ──────────────
        # We replace the T5 encoder with our GNN context by injecting
        # a fake encoder output.
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        self.t5_d_model = t5_d_model
        
        # Get bos_token_id safely
        self.bos_token_id = self.t5.config.decoder_start_token_id
        if self.bos_token_id is None:
            self.bos_token_id = self.t5.config.bos_token_id
        if self.bos_token_id is None:
            self.bos_token_id = 0  # fallback

        # One learned prefix token per output field
        # This token is concatenated to context so T5 knows which field
        # to generate.
        n_fields = len(output_keys)
        self.field_prefix = nn.Embedding(n_fields, t5_d_model)

    # ── Freeze / unfreeze T5 ─────────────────────────────────────────────

    def freeze_t5(self, freeze: bool):
        """Freeze or unfreeze all T5 weights."""
        for param in self.t5.parameters():
            param.requires_grad = not freeze

    # ── Context builder ──────────────────────────────────────────────────

    def _build_context(self, batched_graph, node_idxs_global, guideword_idxs):
        """
        Build encoder hidden states for T5 from graph and guideword context.

        Args:
            batched_graph: PyGBatch object
            node_idxs_global: (B,) global node indices in batched graph
            guideword_idxs: (B,) guideword indices

        Returns:
            context: (B, t5_d_model) context vector per instance
        """
        # Run GNN on entire batched graph
        node_emb = self.gnn(
            batched_graph.x,
            batched_graph.edge_index,
            getattr(batched_graph, "edge_attr", None),
        )  # (N_total, gnn_out)

        # Extract target node embeddings for each instance
        target_emb = node_emb[node_idxs_global]  # (B, gnn_out)

        # Get guideword embeddings
        gw_emb = self.gw_embed(guideword_idxs)  # (B, gw_embed_dim)

        # Combine and project to T5 hidden size
        combined = torch.cat([target_emb, gw_emb], dim=-1)  # (B, context_dim)
        context = self.context_proj(combined)  # (B, t5_d_model)

        return context

    # ── Text generation (loss computation) per field ─────────────────────

    def _field_loss(self, context, field_idx, target_ids, attention_mask):
        """
        Compute cross-entropy loss for one output field in teacher-forcing mode.

        Args:
            context: (B, t5_d_model) context vector
            field_idx: int (which output field)
            target_ids: (B, L) tokenized target text
            attention_mask: (B, L) attention mask for target

        Returns:
            scalar loss
        """
        B = context.size(0)

        # Field prefix token — informs T5 which field to generate
        field_tok = self.field_prefix(
            torch.tensor(field_idx, device=context.device, dtype=torch.long)
                .expand(B)
        )  # (B, t5_d_model)

        # Encoder hidden states: two pseudo-tokens — context + field prefix
        encoder_hidden = torch.stack([context, field_tok], dim=1)  # (B, 2, d)

        # Both pseudo-tokens are real, so attention mask is all ones
        encoder_attn_mask = torch.ones(
            B, 2, device=context.device, dtype=torch.long)

        # Prepare labels: set padding tokens to -100 (ignored in loss)
        labels = target_ids.clone()
        labels[labels == 0] = -100  # 0 is T5's pad token id

        # T5 forward in teacher-forcing mode
        outputs = self.t5(
            encoder_outputs=(encoder_hidden,),
            attention_mask=encoder_attn_mask,
            labels=labels,
            decoder_start_token_id=self.bos_token_id,
        )
        return outputs.loss

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, batch, tokenizer, max_output_len, output_keys,
                return_field_losses=False):
        """
        Compute training loss (sum of all field losses).

        Args:
            batch: dict from hazop_collate_fn
                - batched_graph: PyGBatch
                - node_idxs_global: (B,) global node indices
                - guideword_idxs: (B,) guideword indices
                - target_texts: dict with lists of strings per field
            tokenizer: T5Tokenizer
            max_output_len: int (max tokens per field)
            output_keys: list of field names
            return_field_losses: if True, return (loss, field_loss_dict);
                                 if False, return scalar loss

        Returns:
            loss (scalar) or (loss, {field: loss})
        """
        device = next(self.parameters()).device

        # Build shared context
        context = self._build_context(
            batch["batched_graph"].to(device),
            batch["node_idxs_global"].to(device),
            batch["guideword_idxs"].to(device),
        )  # (B, t5_d_model)

        total_loss = 0.0
        field_losses = {}

        for field_idx, key in enumerate(output_keys):
            # Get target texts for this field
            texts = batch["target_texts"][key]  # list of B strings

            # Tokenize
            encoded = tokenizer(
                texts,
                max_length=max_output_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            target_ids = encoded.input_ids.to(device)
            attention_mask = encoded.attention_mask.to(device)

            # Compute field loss
            field_loss = self._field_loss(
                context, field_idx, target_ids, attention_mask)

            total_loss += field_loss
            field_losses[key] = field_loss.item()

        if return_field_losses:
            return total_loss, field_losses
        return total_loss

    # ── Inference: generate text for a new graph ──────────────────────────

    @torch.no_grad()
    def generate_hazop_row(self, graph, node_idx: int, guideword_idx: int,
                          tokenizer, max_output_len: int = 128,
                          num_beams: int = 4) -> dict:
        """
        Generate HAZOP text for a single (graph, node, guideword) triple.

        Args:
            graph: PyGData object (single system)
            node_idx: int index of target node within the graph
            guideword_idx: int guideword index
            tokenizer: T5Tokenizer
            max_output_len: max tokens per field
            num_beams: beam search width

        Returns:
            dict {field_name: generated_text, ...}
        """
        from torch_geometric.data import Batch as PyGBatch

        self.eval()
        device = next(self.parameters()).device

        # Wrap single graph as batch-of-one
        batched = PyGBatch.from_data_list([graph]).to(device)

        # Build context
        context = self._build_context(
            batched,
            torch.tensor([node_idx], device=device, dtype=torch.long),
            torch.tensor([guideword_idx], device=device, dtype=torch.long),
        )  # (1, t5_d_model)

        results = {}

        for field_idx, key in enumerate(self.output_keys):
            # Field prefix token
            field_tok = self.field_prefix(
                torch.tensor(field_idx, device=device, dtype=torch.long)
                    .unsqueeze(0)
            )  # (1, t5_d_model)

            # Encoder hidden states
            from transformers.modeling_outputs import BaseModelOutput

            # Encoder hidden states
            encoder_hidden = torch.stack([context, field_tok], dim=1)  # (1, 2, d)
            encoder_attn_mask = torch.ones(1, 2, device=device, dtype=torch.long)

            # Wrap encoder output properly for newer transformers versions
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)

            # ── Resolve decoder_start_token_id safely ──
            dstart = getattr(self, "bos_token_id", None)
            if dstart is None:
                dstart = self.t5.config.decoder_start_token_id
            if dstart is None:
                dstart = tokenizer.pad_token_id
            if dstart is None:
                dstart = 0

            generated = self.t5.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=encoder_attn_mask,
                max_new_tokens=max_output_len,
                num_beams=num_beams,
                early_stopping="never",          # avoids deprecation warning
                no_repeat_ngram_size=3,
                decoder_start_token_id=dstart,
                bos_token_id=dstart,
                pad_token_id=tokenizer.pad_token_id or 0,
            )
            
            text = tokenizer.decode(generated[0], skip_special_tokens=True)
            results[key] = text

        return results
