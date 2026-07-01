from tensorboard.backend.event_processing import event_accumulator
import numpy as np, glob, sys
d=sys.argv[1]; f=sorted(glob.glob(d+"/events.out.tfevents.*"))[-1]
ea=event_accumulator.EventAccumulator(f,size_guidance={'scalars':0}); ea.Reload()
tags=ea.Tags()['scalars']
rt=[t for t in tags if t.startswith("Episode_Reward/")]
def last(t):
    s=ea.Scalars(t); return s[-1].value, s[-1].step
print(f"run={d.split('/')[-1]}")
vals={}
maxstep=0
for t in sorted(rt):
    v,st=last(t); vals[t.split('/')[-1]]=v; maxstep=max(maxstep,st)
print(f"iter={maxstep}  (Episode_Reward/* = per-episode summed contribution of each term)\n")
total_abs=sum(abs(v) for v in vals.values())
pos_total=sum(v for v in vals.values() if v>0)
# sort by absolute magnitude
for k,v in sorted(vals.items(), key=lambda kv:-abs(kv[1])):
    share_abs=100*abs(v)/total_abs if total_abs else 0
    share_pos=100*v/pos_total if (v>0 and pos_total) else 0
    tag="+" if v>=0 else "PENALTY"
    print(f"  {k:28s} {v:8.3f}  |abs share {share_abs:5.1f}%  {'pos share %5.1f%%'%share_pos if v>0 else tag}")
print(f"\n  sum(all) = {sum(vals.values()):.3f}   positive-only total = {pos_total:.3f}")
