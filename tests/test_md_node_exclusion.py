"""Exercise automatic continuation submission without a scheduler or GPU."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class NodeExclusionTests(unittest.TestCase):
    def test_automatic_continuation_excludes_stalling_node(self):
        source=(ROOT/'scripts/slurm/izar_gpu_runtime.sh').read_text()
        function=source.split('submit_md_continuation() {',1)[1].split('\n\n\nrun_md()',1)[0]
        function='submit_md_continuation() {'+function
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            (folder/'md.restart.100000').touch()
            capture=folder/'sbatch-arguments'
            env=dict(os.environ,MOF_CONFIG='test.toml',MOF_OUTPUT_DIR=temp,MOF_PREFIX='md',
                     MOF_MD_SEGMENT_SECONDS='3000',MOF_SUBMIT_JOB_NAME='test',
                     MOF_SUBMIT_PARTITION='gpu',MOF_SUBMIT_QOS='normal',MOF_SUBMIT_TIME='01:00:00',
                     MOF_SUBMIT_CPUS='8',MOF_SUBMIT_OUTPUT='job.out',MOF_RUNTIME_PATH='runtime.sh',
                     MOF_STEPS='1000000',TEST_CAPTURE=str(capture))
            stub='sbatch() { printf "%s\\n" "$@" > "$TEST_CAPTURE"; echo 12345; }\n'
            result=subprocess.run(['bash','-c','set -euo pipefail\n'+stub+function+'\nsubmit_md_continuation'],
                                  env=env,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            arguments=capture.read_text().splitlines()
            self.assertIn('--exclude=i39',arguments)
            self.assertTrue(any('MOF_RESUME=1' in arg for arg in arguments))
            self.assertIn('12345',result.stdout)
        # Slurm rereads this directive for direct and older-job continuation submissions.
        self.assertIn('#SBATCH --exclude=i39',source)
