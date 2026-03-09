# Laika

Predict spatial gene expression from DNA sequences and single-cell embeddings using a pretrained genomic trunk (e.g., Borzoi Prime) and task-specific heads.

## Installation

```bash
cd Laika
pip install -e .
```

## Architecture

```mermaid
flowchart TD
    A["DNA sequence (524 kbp)"] --> B["Borzoi Prime trunk<br>(extraction layer)"]
    B --> C["sequence embedding"]
    C --> D["Head"]
    E["single-cell<br>spatial embedding"] --> D
    D --> F["scalar prediction<br>(gene expression)"]

    classDef dna     fill:#4ade80,stroke:#16a34a,color:#000
    classDef trunk   fill:#60a5fa,stroke:#2563eb,color:#000
    classDef emb     fill:#c084fc,stroke:#9333ea,color:#000
    classDef head    fill:#fb923c,stroke:#ea580c,color:#000
    classDef out     fill:#f1f5f9,stroke:#94a3b8,color:#000

    class A dna
    class B trunk
    class C,E emb
    class D head
    class F out
```

## Overview

- **Trunk**: Borzoi-based sequence encoder with optional LoRA fine-tuning
- **Data**: Sequence-based and precomputed-embedding data pipelines
- **Training**: Configurable experiment runner with custom losses and W&B logging
- **Evaluation**: Per-gene correlation metrics and plots

## Heads

Multiple architectures for spatial gene expression prediction. Details: [heads/README.md](src/laika/heads/README.md)

## Quick start

Examples: [`/examples`](/examples/)

```python
import laika

# Full experiment from config
result = laika.run_experiment(config)

# Or components individually
model = laika.Laika(config.model)
trainer = laika.Trainer(model, config.training)
predictor = laika.Predictor(model)
```

## Documentation

Full API documentation: [https://lukasvermeire.github.io/Laika/laika.html](https://lukasvermeire.github.io/Laika/laika.html)

## Dependencies

`torch`, `numpy`, `scipy`, `anndata`, `crested`, `matplotlib`, `tqdm`, `loguru`

Optional: `wandb` — `pip install -e ".[logging]"`
