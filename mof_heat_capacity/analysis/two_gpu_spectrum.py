"""Review one completed experimental Hessian without modifying its raw matrix."""
import argparse
import json
from pathlib import Path
import numpy as np


def review_matrix(hessian, masses):
    from .harmonic import signed_frequencies_from_hessian
    if hessian.shape != (3 * len(masses), 3 * len(masses)):
        raise ValueError('Matrix shape does not match structure')
    if not np.isfinite(hessian).all():
        raise ValueError('Nonfinite Hessian entries')
    norm = float(np.linalg.norm(hessian))
    symmetry = float(np.linalg.norm(hessian-hessian.T) / max(norm, 1e-300))
    translation = float(abs(hessian.reshape(len(masses),3,len(masses),3).sum(axis=0)).max())
    freqs = signed_frequencies_from_hessian(hessian, masses, enforce_asr=True)
    imaginary = int(np.count_nonzero(freqs < -1))
    temperatures = np.arange(200., 401., 25.)
    # Positive vibrational modes only; unstable spectra are not assigned C_v.
    cv = None
    if imaginary == 0:
        x = 1.438776877 * freqs[freqs > 1, None] / temperatures[None, :]
        factors = x*x*np.exp(-x)/(-np.expm1(-x))**2
        cv = (8.31446261815324 / np.sum(masses) * factors.sum(axis=0)).tolist()
    report = dict(relative_asymmetry=symmetry,
                  max_translation_sum_eV_A2=translation,
                  imaginary_modes_below_minus_1_cm1=imaginary,
                  near_zero_modes_within_1_cm1=int(np.count_nonzero(abs(freqs)<=1)),
                  minimum_frequency_cm1=float(freqs.min()),
                  temperatures_K=temperatures.tolist(), harmonic_cv_J_gK=cv,
                  spectrum_convention='symmetrized, acoustic sum rule projected; signed frequencies retained')
    return report, freqs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.matrix.with_suffix('.json').read_text())
    if not metadata['complete'] or not metadata['full_hessian']:
        raise ValueError('Requires a completed full Hessian')
    from ase.io import read
    atoms = read(metadata['structure'], index=-1)
    report, frequencies = review_matrix(np.load(args.matrix, mmap_mode='r'), atoms.get_masses())
    provenance = Path(metadata['structure']).with_suffix('.relax.json')
    report['relaxation'] = json.loads(provenance.read_text()) if provenance.exists() else None
    report['member'] = metadata['member']
    report['source_metadata'] = str(args.matrix.with_suffix('.json').resolve())
    report['full_frequency_convergence_established'] = False
    np.savez(args.matrix.with_name('spectrum.npz'), frequencies_cm1=frequencies)
    args.matrix.with_name('review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2), flush=True)


if __name__ == '__main__':
    main()
