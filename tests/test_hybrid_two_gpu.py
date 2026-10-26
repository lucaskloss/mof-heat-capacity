import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from mof_heat_capacity.analysis.hybrid_two_gpu import collect_spectra


class HybridAdapterTests(unittest.TestCase):
    def test_requires_matching_complete_ordered_ensemble(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            source=root/'geometry.extxyz'
            source.write_text('test identity')
            base=dict(structure=str(source),structure_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      config='config.toml',atom_count=1,dtype='float64',no_shadow=True,
                      selection_mode='fixed',derivative='finite-difference',fd_step_A=5e-5,
                      complete=True,full_hessian=True,columns_completed=3,columns_requested=3,
                      llpr_sha256='same-checkpoint')
            for i in range(65):
                folder=root/('central' if i==0 else f'member-{i-1:02d}')
                folder.mkdir()
                record=dict(base,member=None if i==0 else i-1)
                (folder/'hessian.json').write_text(json.dumps(record))
                np.savez(folder/'spectrum.npz',frequencies_cm1=[-2.,0.,2.])
                (folder/'review.json').write_text(json.dumps(dict(
                    imaginary_modes_below_minus_1_cm1=1,relative_asymmetry=0.)))
            spectra, _, identity, records=collect_spectra(root)
            self.assertEqual(spectra.shape,(65,3))
            self.assertEqual(identity,'same-checkpoint')
            path=root/'member-63/hessian.json'
            record=json.loads(path.read_text());record['member']=0
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'member identity'):
                collect_spectra(root)
            record['member']=63;record['fd_step_A']=1e-4
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'numerical settings'):
                collect_spectra(root)
            record['fd_step_A']=5e-5;record['complete']=False
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'Incomplete Hessian'):
                collect_spectra(root)
