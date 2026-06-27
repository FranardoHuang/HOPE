from tensorboard.backend.event_processing import event_accumulator
import numpy as np, glob, sys
d=sys.argv[1]; f=sorted(glob.glob(d+"/events.out.tfevents.*"))[-1]
ea=event_accumulator.EventAccumulator(f,size_guidance={'scalars':0}); ea.Reload()
tags=set(ea.Tags()['scalars'])
def tr(t,n=8):
    if t not in tags: return "MISSING"
    s=ea.Scalars(t); v=[x.value for x in s]; st=[x.step for x in s]
    idx=np.linspace(0,len(s)-1,min(n,len(s))).astype(int)
    return " ".join(f"{st[i]}:{v[i]:.3f}" for i in idx)
for k in ["strike_composite_success_exact","strike_pos_pass_exact","strike_vel_pass_exact","strike_normal_pass_exact",
"exact_strike_hit_rate","exact_strike_sample_count_decayed","strike_window_hit_rate",
"racket_pos_error_exact_strike","racket_vel_error_exact_strike","racket_normal_error_deg_exact_strike",
"racket_speed_at_strike","racket_target_speed_at_strike","time_to_strike_s","pre_strike_flag"]:
    print(f"{k:34s} {tr('Live/racket_target/'+k)}")
