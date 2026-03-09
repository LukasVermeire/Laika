# Prediction Heads

All heads take `(trunk_emb: (B,L,C), cell_embs: (B,N,D))` and return `(B,N)` predictions.

```mermaid
flowchart LR
    T["trunk_emb<br>(B, L, C)"] --> H["Head"]
    C["cell_embs<br>(B, N, D)"] --> H
    H --> O["(B, N)"]

    style T fill:#60a5fa,stroke:#2563eb,color:#000
    style C fill:#4ade80,stroke:#16a34a,color:#000
    style H fill:#fb923c,stroke:#ea580c,color:#000
```

---

## Base Architectures


### 1. MLP (`"mlp"`)

Attention-pools the trunk to a single gene vector, concatenates it with each cell embedding, and runs a shared MLP.

```mermaid
flowchart LR
    T["trunk_emb"] --> AP["Linear+LayerNorm<br>+AttentionPool"] --> G["gene_vec"]
    G --> M["concat → MLP"]
    C["cell_embs"] --> M --> O["(B,N)"]

    style T fill:#60a5fa,stroke:#2563eb,color:#000
    style C fill:#4ade80,stroke:#16a34a,color:#000
```


### 2. Cross-Attention (`"cross_attention"`)

Cells attend to all trunk sequence positions via multi-head cross-attention.

```mermaid
flowchart LR
    T["trunk_emb"] --> TP["Linear+LayerNorm<br>[+pos_emb]"] --> KV["keys/values"]
    C["cell_embs"] --> CP["Linear+LayerNorm"] --> Q["queries"]
    KV --> CA["CrossAttn ×n"]
    Q --> CA --> MLP["MLP"] --> O["(B,N)"]

    style T fill:#60a5fa,stroke:#2563eb,color:#000
    style C fill:#4ade80,stroke:#16a34a,color:#000
```


### 3. FiLM (`"film"`)

Cell embeddings generate per-layer scale/shift parameters (γ, β) that modulate the gene representation.

```mermaid
flowchart LR
    T["trunk_emb"] --> AP["Linear+LayerNorm<br>+AttentionPool"] --> G["gene_vec"]
    C["cell_embs"] --> FC["FiLMConditioner"] --> GB["(γ,β) per layer"]
    G --> FL["FiLM Layers ×k → Linear"]
    GB --> FL --> O["(B,N)"]

    style T fill:#60a5fa,stroke:#2563eb,color:#000
    style C fill:#4ade80,stroke:#16a34a,color:#000
```


### 4. HyperConv (`"hyperconv"`)

Each cell generates its own convolutional filter via a hypernetwork MLP. This filter is applied across all DNA sequence positions (einsum dot-product), producing per-cell positional scores that are then pooled to a scalar. Inspired by [Scooby](https://github.com/gagneurlab/scooby).

```mermaid
flowchart LR
    T["trunk_emb"] --> TP["Linear+LayerNorm"] --> EIN["einsum<br>(B,N,L)"]
    C["cell_embs"] --> WG["weight generator<br>MLP"] --> EIN
    EIN --> PL["pool over L"] --> O["(B,N)"]

    style T fill:#60a5fa,stroke:#2563eb,color:#000
    style C fill:#4ade80,stroke:#16a34a,color:#000
```


### 5. Hybrid (`"hybrid"`)

Keeps FiLM as the main path and adds cross-attention as a small learned residual.

```mermaid
flowchart LR
    T["trunk_emb"] --> FP["FiLM path"] --> ADD["film + scale×ca"]
    T --> CP["CA path"] --> SC["× residual_scale"] --> ADD
    C["cell_embs"] --> FP
    C --> CP
    ADD --> MLP["output MLP"] --> O["(B,N)"]

    style T fill:#60a5fa,stroke:#2563eb,color:#000
    style C fill:#4ade80,stroke:#16a34a,color:#000
```


---

## Wrapper Architectures

These heads wrap any base head above and augment it with additional structure.


### 1. Decomposed (`"decomposed"`)

**Motivation:** Disentangles the gene's average expression level (driven by DNA) from cell-type-specific variation.

Additively decomposes prediction into a DNA-only baseline and a cell-specific residual: `y = μ(gene) + interaction(gene, cell)`.

```mermaid
flowchart LR
    T["trunk_emb"] --> B["Linear+LayerNorm<br>+AttentionPool+Linear"] --> S["sum"]
    T --> IH["⟨base head⟩"]
    C["cell_embs"] --> IH --> S --> ACT["activation"] --> O["(B,N)"]

    style T fill:#60a5fa,stroke:#2563eb,color:#000
    style C fill:#4ade80,stroke:#16a34a,color:#000
    style IH fill:#fb923c,stroke:#ea580c,color:#000
```


### 2. Hurdle (`"hurdle"`)

**Motivation:** Spatial transcriptomics data is highly sparse (~70% zeros). The hurdle model separates the two problems.

Independently predicts whether a gene is expressed (binary gate) and how much (regression): `y = sigmoid(gate) × softplus(expression)`.

```mermaid
flowchart LR
    T["trunk_emb"] --> EXP["⟨base head⟩"] --> SP["softplus"] --> MUL["×"]
    T --> GT["Linear+LayerNorm<br>+AttentionPool"] --> GN["concat → MLP"]
    C["cell_embs"] --> EXP
    C --> GC["Linear"] --> GN --> SIG["sigmoid"] --> MUL --> O["(B,N)"]

    style T fill:#60a5fa,stroke:#2563eb,color:#000
    style C fill:#4ade80,stroke:#16a34a,color:#000
    style EXP fill:#fb923c,stroke:#ea580c,color:#000
```
