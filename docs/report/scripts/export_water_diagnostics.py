"""Export water convergence checks from existing logs and sampled dump frames.

Run from the repository root with
``PYTHONPATH=. python docs/report/scripts/export_water_diagnostics.py``.

No MD or model inference is run. Window comparisons retain the archived harmonic
correction and change only the classical enthalpy averaging window. Dump samples
are sparse spot checks, not a complete bond-event or phase analysis.
"""

import csv
import io
import json
from pathlib import Path

import numpy as np
from ase.geometry import find_mic, get_distances
from ase.io import read, write

from mof_heat_capacity.analysis.lammps import read_lammps_thermo
from mof_heat_capacity.analysis.statistics import AMU_TO_G, EV_TO_J
from mof_heat_capacity.config import load_run_config

REPORT = Path(__file__).resolve().parents[1]
ROOT = REPORT.parents[1]
MODEL = 'pet-mad-1.5-s-40nn'
TEMPERATURES = np.arange(175, 426, 25)
BAR_A3_TO_EV = 6.241509074e-7


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sampled_frames(path):
    """Read the first/last and seven byte-spaced frames without loading a dump."""
    frames = []
    with path.open('rb') as handle:
        for fraction in np.linspace(0, 1, 9):
            if fraction == 1:
                handle.seek(max(0, path.stat().st_size - 200_000))
            else:
                handle.seek(int(fraction * path.stat().st_size))
            if fraction:
                handle.readline()
            candidates = []
            while line := handle.readline():
                if line != b'ITEM: TIMESTEP\n':
                    continue
                lines = [line, handle.readline(), handle.readline(), handle.readline()]
                count = int(lines[3])
                lines.extend(handle.readline() for _ in range(5 + count))
                candidates.append((int(lines[1]), b''.join(lines)))
                if fraction != 1:
                    break
            if candidates:
                step, frame = candidates[-1]
                if not frames or step != frames[-1][0]:
                    frames.append((step, read(io.StringIO(frame.decode()), format='lammps-dump-text')))
    return frames


def main():
    destination = REPORT / 'data'
    archive = ROOT / 'output/post-processing/water-diagnostics'
    archive.mkdir(parents=True, exist_ok=True)
    diagnostics, geometry, provenance = [], [], []
    for host, short in [('mof5', 'mof5'), ('mof303_2x2x2', 'mof303')]:
        water_means = {name: [] for name in ('full', 'early', 'late')}
        archived_means = []
        hybrid_path = ROOT / f'output/post-processing/harmonic-correction/{host}/{MODEL}/50h2o/heat-capacity.exploratory-discard-imaginary.csv'
        hybrid = np.genfromtxt(hybrid_path, delimiter=',', names=True)
        metadata = json.loads(hybrid_path.with_suffix('.json').read_text())
        provenance.append(str(hybrid_path.relative_to(ROOT)))
        for guest in ('50h2o', '50co2', '50ch4'):
            for temperature in TEMPERATURES:
                run_dir = ROOT / f'output/post-processing/trajectory-analysis/{host}/{MODEL}/{guest}/{temperature}K/rep01'
                if not run_dir.exists() and host == 'mof5':
                    run_dir = ROOT / f'output/post-processing/trajectory-analysis/{MODEL}/{guest}/{temperature}K/rep01'
                summary = json.loads((run_dir / 'summary.json').read_text())
                thermo_path = Path(summary['simulation_provenance']['thermodynamic_log'])
                thermo = read_lammps_thermo(thermo_path)
                times = thermo['time_ps']
                # All selected archived campaigns use target pressure 1 bar.
                config = load_run_config(Path(summary['config']))
                assert config.pressure_bar == 1.0
                enthalpy = thermo['total_energy_eV'] + config.pressure_bar * thermo['volume_A3'] * BAR_A3_TO_EV
                masks = {'full': (times >= 100) & (times <= 500),
                         'early': (times >= 100) & (times < 300),
                         'late': (times >= 300) & (times <= 500)}
                means = {name: float(enthalpy[mask].mean()) for name, mask in masks.items()}
                st = summary['statistics']['enthalpy_eV']
                diagnostics.append(dict(host=short, guest=guest, temperature_K=int(temperature),
                    temperature_mean_K=float(thermo['temperature_K'][masks['full']].mean()),
                    H_full_eV=means['full'], H_early_eV=means['early'], H_late_eV=means['late'],
                    H_late_minus_early_eV=means['late'] - means['early'],
                    archived_split_z=st['split_stationarity_z'],
                    volume_change_percent=100 * (thermo['volume_A3'][masks['late']].mean() / thermo['volume_A3'][masks['early']].mean() - 1)))
                provenance.append(str(thermo_path.relative_to(ROOT)))
                selected_temperatures = (200,) if short == 'mof5' else (200, 225, 250)
                if temperature in selected_temperatures and guest == '50h2o':
                    first_block = (times >= 100) & (times < 125)
                    baseline = enthalpy[first_block].mean()
                    baseline_volume = thermo['volume_A3'][first_block].mean()
                    blocks = []
                    for start in np.arange(100, 500, 25):
                        mask = (times >= start) & (times < start + 25)
                        blocks.append(dict(
                            time_ps=float(start + 12.5),
                            temperature_K=float(thermo['temperature_K'][mask].mean()),
                            delta_enthalpy_eV=float(enthalpy[mask].mean() - baseline),
                            delta_volume_percent=float(100 * (thermo['volume_A3'][mask].mean() / baseline_volume - 1)),
                        ))
                    write_rows(destination / f'water-diagnostic-{short}-{temperature}K.csv', blocks)
                if guest != '50h2o':
                    continue
                mass = summary['metadata']['total_mass_amu']
                for name, mean in means.items():
                    water_means[name].append(mean)
                archived_mean = next(row['mean_enthalpy_eV'] for row in metadata['classical_md_records'] if row['temperature_K'] == temperature)
                archived_means.append(archived_mean)
                h = enthalpy[masks['full']]
                legacy_mean = (h.sum() + h[0]) / (len(h) + 1)
                assert min(abs(means['full'] - archived_mean), abs(legacy_mean - archived_mean)) < 1e-6
                trajectory_path = Path(summary['trajectory'])
                provenance.append(str(trajectory_path.relative_to(ROOT)))
                frames = sampled_frames(trajectory_path)
                for step, atoms in frames:
                    host_atoms = summary['metadata']['host_atoms']
                    positions = atoms.positions[host_atoms:].reshape(50, 3, 3)
                    assert atoms.get_chemical_symbols()[host_atoms:] == ['O', 'H', 'H'] * 50
                    vectors = positions[:, 1:] - positions[:, :1]
                    vectors = find_mic(vectors.reshape(-1, 3), atoms.cell, atoms.pbc)[0].reshape(50, 2, 3)
                    lengths = np.linalg.norm(vectors, axis=2)
                    angles = np.degrees(np.arccos(np.clip(np.sum(vectors[:, 0] * vectors[:, 1], axis=1) / np.prod(lengths, axis=1), -1, 1)))
                    oxygen_indices = np.flatnonzero(np.array(atoms.get_chemical_symbols()) == 'O')
                    hydrogen_indices = np.arange(host_atoms, len(atoms)).reshape(50,3)[:,1:].ravel()
                    distances = get_distances(atoms.positions[hydrogen_indices], atoms.positions[oxygen_indices], cell=atoms.cell, pbc=atoms.pbc)[1]
                    original_oxygen = np.repeat(np.arange(host_atoms, len(atoms), 3), 2)
                    swapped = oxygen_indices[distances.argmin(axis=1)] != original_oxygen
                    geometry.append(dict(host=short, temperature_K=int(temperature), step=step, time_ps=step * config.timestep_fs / 1000,
                        min_OH_A=float(lengths.min()), max_OH_A=float(lengths.max()),
                        min_HOH_deg=float(angles.min()), max_HOH_deg=float(angles.max()),
                        OH_above_1p3_A=int((lengths > 1.3).sum()), H_nearest_other_oxygen=int(swapped.sum())))
                if temperature in selected_temperatures:
                    write(archive / f'{short}-{temperature}K-snapshots.extxyz', [a for _,a in frames])
        conversion = EV_TO_J / (mass * AMU_TO_G)
        cp_windows = {name: np.gradient(values, TEMPERATURES, edge_order=2) * conversion for name, values in water_means.items()}
        archived_cp = np.gradient(archived_means, TEMPERATURES, edge_order=2) * conversion
        assert np.allclose(archived_cp[1:-1], hybrid['classical_anharmonic_cp_J_per_gK'], rtol=0, atol=1e-8)
        print(short, 'maximum duplicate-record correction to cP:', float(np.max(np.abs(cp_windows['full'][1:-1] - archived_cp[1:-1]))))
        rows = []
        for index, temperature in enumerate(TEMPERATURES[1:-1], start=1):
            correction = float(hybrid['harmonic_quantum_correction_J_per_gK'][index-1])
            rows.append(dict(temperature_K=int(temperature), **{name: float(cp[index] + correction) for name, cp in cp_windows.items()}))
        write_rows(destination / f'water-diagnostic-{short}-cp-windows.csv', rows)
    write_rows(destination / 'water-diagnostic-stationarity.csv', diagnostics)
    write_rows(destination / 'water-diagnostic-geometry.csv', geometry)
    (archive / 'provenance.json').write_text(json.dumps(dict(inputs=provenance, window_definitions_ps={'full': '[100,500]', 'early': '[100,300)', 'late': '[300,500]'},
        geometry_sampling='Nine byte-spaced frames per water trajectory, including first and last; no exhaustive event search.',
        legacy_note='Archived means include a duplicated thermo record at 100 ps; current parser removes it. Window diagnostics use unique timesteps.',
        cp_comparison='Central derivatives on 175:425:25 K grid; fixed archived harmonic correction; no refitting or model inference.'), indent=2)+'\n')
    print('Exported water diagnostics. Archived classical heat capacities reproduced within 1e-8; window diagnostics use unique timesteps.')
    print('Geometry ranges:', min(r['min_OH_A'] for r in geometry), max(r['max_OH_A'] for r in geometry))
    print('Stretched OH / changed nearest O:', sum(r['OH_above_1p3_A'] for r in geometry), sum(r['H_nearest_other_oxygen'] for r in geometry))


if __name__ == '__main__':
    main()
