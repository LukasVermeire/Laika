# Cell Encoders

All encoders take `(expression: (B,N,G), gene_indices_to_mask: (B,))` and return `(B,N,D)` cell embeddings (the same shape the heads expect for `cell_embs`).

The target gene is **masked before encoding** (column zeroed out) to prevent information leakage when predicting gene *g* from cell expression that includes *g*.

```mermaid
flowchart LR
    E["expression<br>(B, N, G)"] --> M["mask target gene"] --> ENC["CellEncoder"]
    IDX["gene_encoder_idx<br>(B,)"] --> M
    ENC --> O["cell_embs<br>(B, N, D)"]

    style E fill:#4ade80,stroke:#16a34a,color:#000
    style IDX fill:#f9a8d4,stroke:#db2777,color:#000
    style ENC fill:#fb923c,stroke:#ea580c,color:#000
    style O fill:#c084fc,stroke:#9333ea,color:#000
```

---

## Available Encoders

### 1. Transformer (`"transformer"`)

Treats each gene as a token: the scalar expression value is projected to `d_model` and added to a per-gene identity embedding. A learnable CLS token is prepended, a standard transformer encoder processes the sequence, and the CLS output is projected to the final embedding dimension.

```mermaid
flowchart LR
    E["expression<br>(B,N,G)"] --> PROJ["expression_proj<br>Linear(1→d)"]
    PROJ --> ADD["+ gene_embedding(idx)"]
    ADD --> LN["LayerNorm"]
    CLS["cls_token<br>(learnable)"] --> CAT["prepend"]
    LN --> CAT
    CAT --> TR["TransformerEncoder<br>×n_layers"]
    TR --> CLS_OUT["CLS output"]
    CLS_OUT --> OP["output_proj<br>Linear(d→D)"]
    OP --> O["(B,N,D)"]

    style E fill:#4ade80,stroke:#16a34a,color:#000
    style CLS fill:#f9a8d4,stroke:#db2777,color:#000
    style TR fill:#fb923c,stroke:#ea580c,color:#000
    style O fill:#c084fc,stroke:#9333ea,color:#000
```