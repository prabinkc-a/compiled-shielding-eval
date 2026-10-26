"""Apply published frozen permission tables; no learning or rule compilation."""
import json
import math
from pathlib import Path
import torch


class FrozenPermissions:
    def __init__(self, method, root=None):
        root = root or Path(__file__).resolve().parents[1]
        artifact=json.loads((root/"checkpoints/neural/masks.json").read_text())
        self.method=method
        self.table=torch.tensor(artifact[method if method in ("baseline","G") else "contextual"],dtype=torch.bool)
        self.thresholds=torch.tensor([math.log2(n)/8 for n in artifact["context_cutoffs"]])
        self.monitor_limit=math.log2(artifact["context_monitor_limit"])/8+1e-6

    def __call__(self, features, prefix, step):
        allowed=self.table[step].to(features.device).expand(len(features),-1)
        if self.method in ("baseline","G") or step not in (0,3):
            return allowed
        allowed=allowed.clone()
        monitored=(features[:,0]<=self.monitor_limit)&(features[:,1]>.5)
        if step==0:
            allowed[monitored,3]=False
        else:
            g,c=prefix[:,1],prefix[:,2]
            threshold=self.thresholds.to(features.device)[c]
            first=(g==0)|((g==1)&(features[:,0]<=threshold+1e-6))|((g==2)&(features[:,1]>.5))|((g==3)&(features[:,2]>.5))
            allowed[monitored&~first,3]=False
        return allowed
