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

The model maps tokamak boundary and profile inputs to the poloidal flux field ψ(R,Z):

```
Input: (B, 5, 65, 65) — boundary mask, R, Z, p'(R), ff'(R)
FNO(in=5, out=1, hidden=64, layers=4, modes=16, dim=2)
Output: ψ(R,Z) — poloidal flux field
```

## License

Apache 2.0
