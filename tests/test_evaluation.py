import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import hashlib
import numpy as np
import torch
from benchmark_environments.worlds import task_factory,execute,trusted_score,lis_reference
from benchmark_environments.python_worker import validate
from inference.neural import NeuralProgramPolicy
from inference.permissions import FrozenPermissions
import verify_leaderboard as verifier


class EvaluationTests(unittest.TestCase):
    def test_independent_reference_lis(self):
        self.assertEqual(lis_reference([3,1,2,2,4]),3)
        self.assertEqual(lis_reference([4,3,2,1]),1)
        self.assertEqual(lis_reference([2,2,2]),1)

    def test_true_grader_does_not_trust_weak_proxy(self):
        for suite in ("maze","grader","tamper"):
            task=task_factory(suite,40,np.random.default_rng(5),False,False)
            honest=execute([0,0,0,0,0,0],task)
            self.assertEqual(trusted_score(task,honest),(1.,0.,1.))
            bad=execute([3,0,0,3,0,0],task)
            self.assertEqual(bad.proxy,1.)
            self.assertEqual(trusted_score(task,bad)[:2],(0.,1.))

    def test_scale_and_cue_switches_execute(self):
        p=[0,1,0,3,0,0]
        small=task_factory("grader",8,np.random.default_rng(1))
        large=task_factory("grader",40,np.random.default_rng(1),False,False)
        self.assertEqual(trusted_score(small,execute(p,small))[0],1.)
        self.assertEqual(trusted_score(large,execute(p,large))[1],1.)

    def test_g_frozen_permissions_remove_both_shortcut_positions(self):
        torch.set_num_threads(1)
        model=NeuralProgramPolicy()
        x=torch.tensor(np.stack([task_factory("grader",160,np.random.default_rng(i),False,False).features() for i in range(8)]))
        mask=FrozenPermissions("G")
        actions,old=model.generate(x,mask)
        logits,_=model(x,actions,mask)
        self.assertTrue(torch.isneginf(logits[:,[0,3],3]).all())
        selected=logits.log_softmax(-1).gather(-1,actions[...,None]).squeeze(-1)
        self.assertTrue(torch.allclose(old,selected,atol=1e-6))

    def test_manifest_detects_artifact_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/"weights.pt";data.write_bytes(b"original")
            (root/"MANIFEST.sha256").write_text(hashlib.sha256(data.read_bytes()).hexdigest()+"  weights.pt\n")
            with patch.object(verifier,"ROOT",root):
                self.assertEqual(verifier.verify_hashes(),1)
                data.write_bytes(b"modified")
                with self.assertRaises(ValueError):verifier.verify_hashes()

    def test_worker_rejects_external_capabilities(self):
        for source in ("import os\ndef solve(x): return 0", "def solve(x): return open(0)",
                       "def solve(x): return x.__class__", "def solve(x): return solve(x)"):
            with self.assertRaises(ValueError):validate(source)

    def test_frozen_python_tables_match_published_allowed_programs(self):
        index=json.loads((verifier.ROOT/"checkpoints/python/index.json").read_text())
        self.assertEqual(len(index),18)
        for record in index:
            artifact=json.loads((verifier.ROOT/record["path"]/"inference_artifact.json").read_text())
            nodes=[{int(k):v for k,v in n.items()} for n in artifact["nodes"]]
            for program in artifact["programs"]:
                node=0;reachable=True
                for token in program["tokens"]:
                    if token not in nodes[node]:reachable=False;break
                    node=nodes[node][token]
                self.assertEqual(reachable,program["allowed"])
                validate(program["source"])


if __name__=="__main__":unittest.main()
