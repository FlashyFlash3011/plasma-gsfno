# plasma-gsfno

FNO surrogate for the Grad-Shafranov MHD equilibrium equation, replacing EFIT (10–100 ms) with sub-millisecond inference.

Built on [Solaris](../Solaris) — a physics AI framework for neural operators on AMD ROCm.

## Quick Start

```bash
# Activate the Solaris venv (shared)
source ~/projects/research/Solaris/.venv/bin/activate

# Install in editable mode
pip install -e ".[dev]"
```

To include real-machine data dependencies (FreeGS equilibrium solver and MDS+ data access):

```bash
pip install -e ".[dev,gw]"
```

## Architecture

The model is a forward Grad-Shafranov solver surrogate. It maps vacuum flux, normalized geometry, and pressure/current-profile inputs to the dimensionless total poloidal flux ψ(R,Z):

```
Input: (B, 5, 65, 65) — [ψ_vac, R_norm, Z_norm, p'(ψ_N) lifted, ff'(ψ_N) lifted]
FNO(in=5, out=1, hidden=64, layers=4, modes=16, dim=2)
Output: ψ_total(R,Z) — dimensionless poloidal flux
```

Training data is generated via FreeGS free-boundary forward equilibrium solves with a GS-residual validation gate. The model uses global dimensionless scaling (no per-sample normalization). The ψ_vac channel is computed via coil/vacuum Green's functions—no answer leak into the geometry inputs.

## License

Apache 2.0
