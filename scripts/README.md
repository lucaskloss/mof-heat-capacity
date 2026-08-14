# Command entry points

Run these commands from the repository root. The scripts are intentionally
thin entry points; reusable Python implementation lives in the
`mof_heat_capacity/` package.

| Command | Responsibility |
| --- | --- |
| `prepare_paper_protocol.py` | Generate independent structures and TOML files for the paper protocol. |
| `submit_paper_protocol.sh` | Validate and submit a paper-protocol job matrix to Slurm. |
| `izar_job.sh` | Execute one MD or heat-capacity stage inside an Izar allocation. |
| `analyze_all_results.py` | Produce convergence, structural, and heat-capacity reports. |
| `install_sadmof.sh` | Install the pinned SADMOF/PET-JAX analysis stack. |

The root `run.py` command starts one configuration-driven MD calculation.
Run standalone structure preparation and harmonic analysis with:

```bash
python -m mof_heat_capacity.structures.methane --help
python -m mof_heat_capacity.analysis.harmonic --help
```
