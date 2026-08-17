# Command entry points

Run these commands from the repository root. The scripts are intentionally
thin entry points; reusable Python implementation lives in the
`mof_heat_capacity/` package.

| Command | Responsibility |
| --- | --- |
| `prepare_paper_protocol.py` | Generate independent structures and TOML files for the paper protocol. |
| `submit_paper_protocol.sh` | Validate and submit a paper-protocol job matrix to Slurm. |
| `izar_job.sh` | Execute one MD or heat-capacity stage inside an Izar allocation. |
| `submit_analysis.sh` | Validate and submit CPU-based trajectory analysis to Slurm. |
| `izar_analysis_job.sh` | Execute trajectory analysis inside its allocation. |
| `submit_heat_capacity.sh` | Submit selected-time SADMOF calculations as a Slurm array. |
| `analyze_all_results.py` | Produce convergence, structural, and heat-capacity reports. |
| `install_sadmof.sh` | Install the pinned SADMOF/PET-JAX analysis stack. |

The root `run.py` command starts one configuration-driven MD calculation.
Run standalone structure preparation and harmonic analysis with:

```bash
python -m mof_heat_capacity.structures.methane --help
python -m mof_heat_capacity.analysis.harmonic --help
```

Submit trajectory analysis for both MLIPs and one or more methane loadings
with selectors rather than handwritten run globs:

```bash
./scripts/submit_analysis.sh --model both --loading 0 --dry-run
./scripts/submit_analysis.sh --model both --loading 100 --time 01:15:00 --dry-run
./scripts/submit_analysis.sh --model both --loading 0,100 --time 01:30:00 \
  --analysis-dir output/analysis-0ch4-100ch4 --dry-run
```

`--temperatures` and `--replicas` also accept comma-separated lists. Remove
`--dry-run` only after every requested completed trajectory has been validated.

SADMOF must run on cluster compute nodes. Preview a one-frame debug job, then
the three-frame production array with:

```bash
./scripts/submit_heat_capacity.sh --debug --dry-run
./scripts/submit_heat_capacity.sh --dry-run
```

The defaults select 200, 350, and 500 ps for both potentials, loadings 0 and
100, and all five MD temperatures. The wrapper itself performs only lightweight
file validation; all trajectory loading, JAX initialization, and Hessian work
occurs inside Slurm.
