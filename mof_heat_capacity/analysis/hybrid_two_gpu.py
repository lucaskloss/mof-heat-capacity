"""Validate experimental two-GPU spectra and assemble exploratory hybrid data."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np


def collect_spectra(root):
    """Require complete, ordered spectra with matching geometry and numerical settings."""
    identity = ('structure_sha256', 'config', 'atom_count', 'dtype', 'no_shadow',
                'selection_mode', 'derivative', 'fd_step_A')
    spectra, records, reference, llpr_hash = [], [], None, None
    for index in range(65):
        label = 'central' if index == 0 else f'member-{index-1:02d}'
        folder = root / label
        record = json.loads((folder/'hessian.json').read_text())
        expected_member = None if index == 0 else index-1
        if record['member'] != expected_member:
            raise ValueError(f'Unexpected member identity: {folder}')
        if not record['complete'] or not record['full_hessian']:
            raise ValueError(f'Incomplete Hessian: {folder}')
        expected_modes = 3*record['atom_count']
        if record['columns_completed'] != expected_modes or record['columns_requested'] != expected_modes:
            raise ValueError(f'Incomplete coordinate coverage: {folder}')
        if reference is None:
            reference = record
        elif any(reference[k] != record[k] for k in identity):
            raise ValueError(f'Geometry or numerical settings differ: {folder}')
        if index:
            if not record.get('llpr_sha256'):
                raise ValueError(f'Missing LLPR identity: {folder}')
            if llpr_hash is None:
                llpr_hash = record['llpr_sha256']
            elif llpr_hash != record['llpr_sha256']:
                raise ValueError(f'LLPR checkpoint mismatch: {folder}')
        with np.load(folder/'spectrum.npz', allow_pickle=False) as archive:
            frequencies = archive['frequencies_cm1']
        if frequencies.shape != (expected_modes,) or not np.isfinite(frequencies).all():
            raise ValueError(f'Invalid spectrum: {folder}')
        review = json.loads((folder/'review.json').read_text())
        if review['imaginary_modes_below_minus_1_cm1'] != int((frequencies < -1).sum()):
            raise ValueError(f'Review does not match spectrum: {folder}')
        spectra.append(frequencies)
        records.append(dict(label=label, imaginary_modes=int((frequencies < -1).sum()),
                            minimum_frequency_cm1=float(frequencies.min()),
                            relative_asymmetry=review['relative_asymmetry']))
    source = Path(reference['structure'])
    if hashlib.sha256(source.read_bytes()).hexdigest() != reference['structure_sha256']:
        raise ValueError('Reference geometry has changed')
    return np.asarray(spectra), reference, llpr_hash, records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ensemble',type=Path,required=True)
    parser.add_argument('--analysis-archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model-label',default='pet-mad-1.5-s-40nn')
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--no-experimental-reference', action='store_true')
    args=parser.parse_args()
    if any(args.output.with_suffix(s).exists() for s in ('.npz','.png','.json','.csv','.pdf')):
        raise FileExistsError('Choose a new hybrid output prefix')
    spectra, metadata, llpr_hash, records=collect_spectra(args.ensemble)
    with np.load(args.analysis_archive,allow_pickle=False) as archive:
        if str(archive['model_sha256'].item()) != llpr_hash:
            raise ValueError('Trajectory and Hessian LLPR checkpoints differ')
        if archive['cea_cp_by_member_J_per_gK'].shape[0] != 64:
            raise ValueError('Trajectory archive must contain 64 persistent members')
    from ..config import load_run_config
    from ..campaign import system_directory
    config=load_run_config(Path(metadata['config']))
    root=args.ensemble/'hybrid-inputs'
    loaded=root/system_directory(args.model_label,config.loading,config.mof,config.guest)
    minimum=loaded/'minima/400K/rep01'
    archive_path=loaded/'hessians/400K/rep01/hessian.npz'
    if archive_path.exists():
        raise FileExistsError(f'Experimental adapter archive already exists: {archive_path}')
    minimum.mkdir(parents=True,exist_ok=True)
    source=Path(metadata['structure'])
    shutil.copy2(source,minimum/'optimized.extxyz')
    shutil.copy2(source.with_suffix('.relax.json'),minimum/'optimized.relax.json')
    archive_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez(archive_path, frequencies_cm1=spectra[:1], llpr_frequencies_cm1=spectra[1:][None,...],
             trajectory=str(source.resolve()), llpr_checkpoint_sha256=llpr_hash,
             metadata=json.dumps(dict(frequency_convention='signed', derivative='finite-difference',
                                      fd_step_A=metadata['fd_step_A'], no_shadow=metadata['no_shadow'],
                                      selection_mode=metadata['selection_mode'],
                                      source_ensemble=str(args.ensemble.resolve()),
                                      member_specific_relaxation=False)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    (args.output.parent/'ensemble-validation.fd-two-gpu.json').write_text(json.dumps(
        dict(central_and_members=records,llpr_sha256=llpr_hash,settings=metadata,
             validated_complete_spectra=65),indent=2)+'\n')
    command=[sys.executable,'-m','mof_heat_capacity.analysis.hybrid',
             '--mof',config.mof,'--guest',config.guest,'--loading',str(config.loading),
             '--model-label',args.model_label,'--replicas','1','--hybrid-dir',str(root),
             '--hessian-source-temperature','400','--model-uncertainty',str(args.analysis_archive),
             '--discard-imaginary-modes','--output',str(args.output)]
    if args.reference_root is not None:
        command.extend(['--reference-root', str(args.reference_root)])
    if args.no_experimental_reference:
        command.append('--no-experimental-reference')
    subprocess.run(command,check=True)
    print(f'Exploratory hybrid data saved beside {args.output}',flush=True)


if __name__ == '__main__':
    main()
