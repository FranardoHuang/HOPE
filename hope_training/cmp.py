from tensorboard.backend.event_processing import event_accumulator
import numpy as np, glob, sys
def load(d):
    f=sorted(glob.glob(d+"/events.out.tfevents.*"))[-1]
    ea=event_accumulator.EventAccumulator(f,size_guidance={'scalars':0}); ea.Reload()
    return ea, set(ea.Tags()['scalars'])
def series(ea,T,t,n=7):
    if t not in T: return "MISSING"
    s=ea.Scalars(t); v=[x.value for x in s]; st=[x.step for x in s]
    idx=np.linspace(0,len(s)-1,min(n,len(s))).astype(int)
    return " ".join(f"{st[i]}:{v[i]:.3f}" for i in idx)
ea4,T4=load(sys.argv[1])
keys=[
("Live/Reward/total","mean_reward/step"),
("Live/Env/episode_length","episode_length"),
("Live/racket_target/base_pos_error_pre_strike","base_pos_error(trimmed weight)"),
("Live/motion/error_body_pos","imit_error_body_pos"),
("Live/motion/error_joint_pos","imit_error_joint_pos"),
("Live/Reward/base_position","R:base_position"),
("Live/Reward/motion_body_pos","R:motion_body_pos"),
("Live/Reward/racket_position","R:racket_position"),
("Live/Reward/racket_velocity","R:racket_velocity"),
("Live/Reward/racket_normal","R:racket_normal"),
("Live/Reward/joint_limit","R:joint_limit(penalty)"),
("Live/Reward/action_rate_l2","R:action_rate(penalty)"),
("Live/racket_target/joint_pos_near_limit_frac","joint_near_limit_frac"),
("Live/racket_target/joint_torque_abs_max","joint_torque_max"),
("Live/Termination/terminated_rate","terminated_rate"),
]
print("=== v4 (rebalanced) trajectory ===")
for t,lab in keys:
    print(f"  {lab:32s} {series(ea4,T4,t)}")
