# BSS-Test Examples

Example scripts demonstrating the BSS-Test framework.

## Quick Start

```bash
python examples/quick_start.py
```

Demonstrates: CWRU data loading → preprocessing → CWT → SOBI BSS → evaluation & visualization.
Falls back to synthetic data if CWRU dataset is not available.

## Running

All examples run from the project root:

```bash
$env:PYTHONPATH = "src"   # PowerShell
python examples/quick_start.py
```

## Data Requirements

- **CWRU**: Download from https://zenodo.org/records/10987113 → `data/cwru/`
- **PHM 2010**: Download from PHM Society → `data/phm2010_milling/`

If data is unavailable, examples use synthetic data from `bss_test.utils.synthetic`.

## Output

Results saved to `outputs/examples/` with descriptive subdirectories.
