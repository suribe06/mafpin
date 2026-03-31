# Installation

## Prerequisites

| Requirement | Version |
| --- | --- |
| Python | 3.9 (required for snap-stanford compatibility) |
| snap-stanford | 6.0.0 |
| networkx | ≥ 3.0 |
| cdlib | 0.4.0 |
| cmfrec | 3.5.1 |
| pandas | ≥ 2.0 |
| numpy | ≥ 1.24 |
| scikit-learn | ≥ 1.3 |
| matplotlib | ≥ 3.7 |
| seaborn | ≥ 0.12 |

## Install Dependencies

```bash
pip install -r requirements.txt
```

## SNAP-Stanford Note

`snap-stanford` is only available for **Python 3.9** on PyPI.  
Use a dedicated virtual environment:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On **macOS** with Apple Silicon, SNAP wheels are not available.  
Build from source or use a Linux environment / Docker image.

## NetInf Binary

The `networks/netinf` binary is bundled with the repository (pre-compiled for Linux x86-64).  
If it is not executable, run:

```bash
chmod +x networks/netinf
```

To rebuild from source, clone the SNAP repository and compile `examples/netinf`:

```bash
git clone https://github.com/snap-stanford/snap.git
cd snap/examples/netinf
make
```

## Verify Installation

```python
from snap import snap
G = snap.TUNGraph.New()
print("SNAP OK — nodes:", G.GetNodes())
```

```python
import cdlib
print("cdlib version:", cdlib.__version__)
```
