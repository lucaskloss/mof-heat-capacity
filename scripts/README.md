# Command entry points

Run these commands from the repository root.

| Command | Responsibility |
| --- | --- |
| `prepare_loaded_campaign.py` | Generate independent loaded structures and classical-NPT TOMLs on the enthalpy temperature grid. |
| `submit_loaded_md.sh` | Validate and submit loaded classical-NPT jobs. |
| `submit_analysis.sh` | Validate and submit trajectory diagnostics. |
| `izar_analysis_job.sh` | Execute trajectory diagnostics in an Izar allocation. |
| `submit_heat_capacity.sh` | Quench loaded configurations and compute loaded and empty-reference AD Hessians. |
| `submit_hybrid_analysis.sh` | Differentiate classical enthalpies and add the loaded-Hessian quantum correction. |
| `izar_job.sh` | Execute MD, relaxation, Hessian, or hybrid-analysis stages on Izar. |
| `install_sadmof.sh` | Install the SADMOF/PET-JAX Hessian stack. |

The normal sequence is:

```bash
python scripts/prepare_loaded_campaign.py --model pet-mad --loading 100 \
  --replicas 5 --dry-run
./scripts/submit_loaded_md.sh --model pet-mad --loading 100 \
  --replicas 5 --dry-run
./scripts/submit_analysis.sh --model pet-mad --loading 100 \
  --replicas 1,2,3,4,5 --dry-run
./scripts/submit_heat_capacity.sh --model pet-mad --loading 100 \
  --source-temperature 300 --replicas 1,2,3 --dry-run
./scripts/submit_hybrid_analysis.sh --model pet-mad --loading 100 \
  --replicas 1,2,3,4,5 --dry-run
```

Remove `--dry-run` only after inspecting each complete command matrix. Empty
MOF-5 is intentionally absent from MD preparation: `submit_heat_capacity.sh`
uses its equilibrated structure directly.
