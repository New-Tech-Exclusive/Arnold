# Arnold Model Architecture

This document provides a high-level overview of the neural tiers and data
flow in the Arnold model.  The design is intentionally biological‑inspired with
nine major processing stages.

```mermaid
flowchart LR
    A[User Input] --> B[Router (router)]
    B --> C[Transformer (predictive coding)]
    C --> D[Basal Ganglia (habits)]
    C --> E[Corrector (forward model)]
    C --> F[Generation Interface]
    F --> G[Output Tokens]
    G --> H[Memory (episodic buffer)]
    H --> I[Consolidation Engine / DMN / Sleep]
    I --> C
    subgraph "Supporting Systems"
      J[Encoder (prior layer)]
      K[Astrocyte Layer]
      L[Limbic System / Neuromodulators]
    end
    A --> J
    J --> C
    K --> C
    L --> C
```

Each box in the diagram corresponds to a Python module under `model/`.  For
complete details consult the source code and docstrings.
