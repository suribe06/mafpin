# MAFPIN domain glossary

Terms used by the codebase and architecture reviews. Prefer these names over
ad-hoc module nicknames.

| Term | Meaning |
| --- | --- |
| **MAFPIN** | Matrix Factorization with Properties of Inferred Networks |
| **Cascade** | Per-item ordered `(user, time)` adoption sequence for NetInf |
| **Alpha (α)** | Diffusion transmission-rate grid point → one inferred network |
| **NetInf** | External binary; ML structure learner from cascades |
| **CMF** | Collective Matrix Factorization (`cmfrec.CMF`); baseline vs enhanced (`U` side-info) |
| **LPH** | Local Pluralistic Homophily; boundary users |
| **Social regularization** | Phase 6 L-BFGS graph penalty on user factors |
| **NetworkArtifacts** | Path locator module for inferred-network / centrality / community files |
| **Hyperparam campaign** | Shared Optuna search workflow for baseline + enhanced/social CMF |
| **Warm-eval core** | Shared filter → warm-split → scale → fit → metrics loop for CMF |
| **Variants / routes** | M1–M4d ladder; Route B = beyond-accuracy follow-up |
| **Network selection** | Freeze `(diffusion_model, alpha_index)` on CV, then one global test |
| **Cold-start** | Stratified / leave-k / trust zero-shot diagnostic suite |
