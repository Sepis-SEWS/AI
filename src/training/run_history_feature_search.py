"""SEWS v2-16: causal cumulative patient history and personalized baseline features.
No Test reads, no large cache files. Run from repository root.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import traceback
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from run_tree_refine import fit_stats, make, cfg, build
from run_adaptive_search import ROOT, MANIFEST, BASELINE, ALL, read_raw, endpoints, metric, atomic_json, transform, tabular

from run_ordered_feature_search import construct as ordered_construct

OUT=ROOT/"reports/tree_v2_16"
BASE=cfg(arch="xgb",depth=3,weight=3,lr=.04,rounds=600,reg=5)
SLOW=cfg(arch="xgb",depth=3,weight=1,lr=.015,rounds=1000,reg=5)
LGB=cfg()
CAT=cfg(arch="cat",depth=6,reg=3)
MODES=("control","personal","history","combined","history_all")
TRIALS=[dict(mode=m,base=b) for m in MODES for b in (BASE,SLOW,LGB,CAT)]
# Keep temporal channels compact: measured/age/last value, not all redundant channels.
VITALS=("HR","O2Sat","Temp","SBP","MAP","DBP","Resp","FiO2","Lactate","Creatinine","WBC","Platelets","pH","BUN","Bilirubin_total")
IDX=np.array([ALL.index(v) for v in VITALS])
def key(c): return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:16]
def history_names(mode):
    variables=ALL[:34] if mode=="history_all" else VITALS
    stats=[]
    if mode in ("personal","combined"):
        stats.extend(("first_value","first_age","delta_first","initial6_mean","delta_initial6","initial6_count","personal_valid"))
    if mode in ("history","combined","history_all"):
        stats.extend(("last_value","last_age","running_mean","running_std","running_min","running_max",
                      "delta_mean","count_log","measurement_rate","previous24_mean","delta_previous24","previous24_valid"))
    return [f"{stat}__{var}" for stat in stats for var in variables]

def construct(raw,offsets,med,scale,mode):
    # Preserve the best v2-15 6-hour segment representation for every candidate.
    base=ordered_construct(raw,offsets,med,scale,"segments")
    if mode=="control": return base
    ix=np.array([ALL.index(v) for v in (ALL[:34] if mode=="history_all" else VITALS)])
    d=len(ix)
    extra=history_names(mode)
    out=np.empty((len(base),base.shape[1]+len(extra)),dtype=np.float32)
    start=base.shape[1]
    out[:,:start]=base
    del base
    cursor=0
    for a,b in zip(offsets[:-1],offsets[1:]):
        n=b-a
        if n<24: continue
        t=np.arange(n)[:,None]
        x=raw[a:b][:,ix].astype(np.float64)
        obs=np.isfinite(x)
        z=np.where(obs,(x-med[ix])/scale[ix],0.)
        # All prefix statistics use current and earlier rows only.
        count=np.cumsum(obs,axis=0)
        total=np.cumsum(z,axis=0)
        sq=np.cumsum(z*z,axis=0)
        avg=total/np.maximum(count,1)
        std=np.sqrt(np.maximum(sq/np.maximum(count,1)-avg*avg,0))
        latest=np.maximum.accumulate(np.where(obs,t,-1),axis=0)
        first=np.minimum.accumulate(np.where(obs,t,n),axis=0)
        last_val=z[np.maximum(latest,0),np.arange(d)]
        first_val=z[np.minimum(first,n-1),np.arange(d)]
        valid=count>0
        last_val=np.where(valid,last_val,0.)
        first_val=np.where(valid,first_val,0.)
        blocks=[]
        if mode in ("personal","combined"):
            # Initial six hours are only used when already elapsed (scoring starts at row 23).
            init_count=count[5]
            init_avg=total[5]/np.maximum(init_count,1)
            blocks.extend((first_val,np.where(valid,np.log1p(np.maximum(t-first,0)),0),
                           np.where(valid,last_val-first_val,0),
                           np.broadcast_to(init_avg,(n,d)),
                           np.where(valid & (init_count>0),last_val-init_avg,0),
                           np.broadcast_to(init_count/6,(n,d)),valid))
        if mode in ("history","combined","history_all"):
            lo=np.minimum.accumulate(np.where(obs,z,np.inf),axis=0)
            hi=np.maximum.accumulate(np.where(obs,z,-np.inf),axis=0)
            # Previous history excludes current 24-row window.
            prev=np.arange(n)-24
            prev_count=np.where(prev[:,None]>=0,count[np.maximum(prev,0)],0)
            prev_sum=np.where(prev[:,None]>=0,total[np.maximum(prev,0)],0)
            prev_avg=prev_sum/np.maximum(prev_count,1)
            blocks.extend((last_val,np.where(valid,np.log1p(t-latest),0),avg,std,
                           np.where(valid,lo,0),np.where(valid,hi,0),np.where(valid,last_val-avg,0),
                           np.log1p(count),count/(t+1),prev_avg,
                           np.where(valid & (prev_count>0),last_val-prev_avg,0),prev_count>0))
        features=np.concatenate(blocks,axis=1)[23:].astype(np.float32)
        if features.shape[1]!=len(extra) or not np.isfinite(features).all():
            raise ValueError("Invalid history features")
        out[cursor:cursor+len(features),start:]=features
        cursor+=len(features)
    if cursor!=len(out): raise ValueError("History endpoint mismatch")
    return out

def model(c,device):
    b=c["base"]
    m=make(b)
    if b["arch"]=="xgb" and device=="cuda":
        m.set_params(device="cuda",tree_method="hist")
    if b["arch"]=="cat" and device=="cuda":
        m.set_params(task_type="GPU",devices="0")
    return m
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    p.add_argument("--retry-failed",action="store_true")
    args=p.parse_args()
    groups=json.loads(MANIFEST.read_text(encoding="utf-8"))["split"]
    if tuple(len(groups[k]) for k in ("train","validation","test"))!=(25814,6454,8068) or len(set(groups["train"]+groups["validation"]+groups["test"]))!=40336: raise ValueError("Locked split changed")
    if args.dry_run:
        print(len(TRIALS),"settings; device",args.device,"Test HELD OUT")
        for c in TRIALS: print(c)
        return
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(version=1,base_representation="v2-15 segments",device=args.device,trials=TRIALS,manifest=hashlib.sha256(MANIFEST.read_bytes()).hexdigest())
    path=OUT/"protocol.json"
    if path.exists() and json.loads(path.read_text())!=protocol: raise ValueError("Existing protocol differs")
    if not path.exists(): atomic_json(path,protocol)
    raw,labels,off=read_raw(groups["train"])
    rr,_=endpoints(off); y=labels[rr]
    patient=np.repeat(np.arange(len(groups["train"])),np.maximum(np.diff(off)-23,0))
    raw_patient=np.repeat(np.arange(len(groups["train"])),np.diff(off))
    strata=[pid.split("/")[0]+"/"+str(int(labels[a:b].max())) for pid,a,b in zip(groups["train"],off[:-1],off[1:])]
    fi,ti=train_test_split(np.arange(len(strata)),test_size=.2,random_state=42,stratify=strata)
    fit=np.flatnonzero(np.isin(patient,fi)); tune=np.flatnonzero(np.isin(patient,ti))
    if len(y)!=430905: raise ValueError("Train endpoints changed")
    med,scale=fit_stats(raw[np.isin(raw_patient,fi)])
    results=[]
    for mode in MODES:
        x=None
        for c in [c for c in TRIALS if c["mode"]==mode]:
            folder=OUT/"trials"/key(c); folder.mkdir(parents=True,exist_ok=True)
            done=folder/"done.json"
            if done.exists():
                r=json.loads(done.read_text())
                if r["config"]!=c: raise ValueError("Resume config mismatch")
            else:
                if (folder/"error.txt").exists() and not args.retry_failed: continue
                try:
                    if x is None:
                        print("Building",mode,flush=True)
                        x=construct(raw,off,med,scale,mode)
                    m=model(c,args.device)
                    a=x[fit]; m.fit(a,y[fit]); del a
                    b=x[tune]; pred=m.predict_proba(b)[:,1]; del b,m
                    r=dict(id=key(c),config=c,**metric(y[tune],pred))
                    atomic_json(done,r); (folder/"error.txt").unlink(missing_ok=True)
                except Exception:
                    (folder/"error.txt").write_text(traceback.format_exc(),encoding="utf-8")
                    print("FAILED",key(c),flush=True); continue
                finally: gc.collect()
            results.append(r)
            atomic_json(OUT/"inner_leaderboard.json",sorted(results,key=lambda r:-r["ap"]))
            print("Inner",r,flush=True)
        del x; gc.collect()
    if not results: raise RuntimeError("No successful trials")
    ranked=sorted(results,key=lambda r:-r["ap"])
    # Freeze best history candidate per architecture, plus best segment control.
    selected=[]
    for arch in ("xgb","lgb","cat"):
        r=next((r for r in ranked if r["config"]["mode"]!="control" and r["config"]["base"]["arch"]==arch),None)
        if r: selected.append(r)
    control=next((r for r in ranked if r["config"]["mode"]=="control"),None)
    if control: selected.append(control)
    atomic_json(OUT/"selected.json",selected)
    med,scale=fit_stats(raw)
    vraw,vy,vo=read_raw(groups["validation"]); vr,_=endpoints(vo)
    with np.load(BASELINE,allow_pickle=False) as a:
        ids=np.repeat(groups["validation"],np.maximum(np.diff(vo)-23,0))
        if not np.array_equal(ids,a["patient_ids"].astype(str)) or not np.array_equal(vy[vr],a["labels"]) or not np.array_equal(vraw[vr,-1],a["icul_os"]): raise ValueError("Validation alignment mismatch")
    finals=[]; preds=[]
    for r in selected:
        c=r["config"]; folder=OUT/"final"/r["id"]; folder.mkdir(parents=True,exist_ok=True)
        done=folder/"done.json"
        if done.exists():
            if json.loads(done.read_text())["config"]!=c: raise ValueError("Final config mismatch")
            pred=np.load(folder/"scores.npy",allow_pickle=False)
        else:
            x=construct(raw,off,med,scale,c["mode"])
            m=model(c,args.device); m.fit(x,y); del x; gc.collect()
            v=construct(vraw,vo,med,scale,c["mode"])
            pred=m.predict_proba(v)[:,1]; del v
            joblib.dump(dict(model=m,config=c,median=med,scale=scale,ordered_variables=VITALS,history_features=history_names(c["mode"]) if c["mode"]!="control" else [],base_representation="v2-15 segments"),folder/"model.joblib")
            np.save(folder/"scores.npy",pred); atomic_json(done,dict(config=c,**metric(vy[vr],pred)))
            del m; gc.collect()
        if len(pred)!=len(vr) or not np.isfinite(pred).all(): raise ValueError("Invalid scores")
        finals.append(dict(id=r["id"],config=c,inner_ap=r["ap"],**metric(vy[vr],pred))); preds.append(pred)
    summary=dict(scope="inner Train patient selection, reused locked Validation common endpoints",
      test_patients_evaluated=0,target_ap=.4,inner_trials=ranked,finalists=finals,
      equal_blend=metric(vy[vr],np.mean(preds,axis=0)),failures=[p.parent.name for p in (OUT/"trials").glob("*/error.txt")])
    atomic_json(OUT/"summary.json",summary)
    table=pd.DataFrame([dict(id=r["id"],inner_ap=r["inner_ap"],ap=r["ap"],auroc=r["auroc"],mode=r["config"]["mode"],**r["config"]["base"]) for r in finals])
    table.to_csv(OUT/"leaderboard.csv",index=False)
    print(table.to_string(index=False)); print("Equal blend",summary["equal_blend"],"Test HELD OUT")
if __name__=="__main__": main()
