"""v2-13: restore v2-11 representation, select features and tune boosting on inner Train.
Run: python src/training/run_tree_refine.py
No Test reads. New output only. Large matrices remain in RAM, not Git.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from run_adaptive_search import ROOT, MANIFEST, BASELINE, ALL, read_raw, endpoints, metric, atomic_json, transform, tabular

OUT=ROOT/"reports/tree_v2_13"
OLD=ROOT/"reports/adaptive_v2_11"
def cfg(arch="lgb",k=0,weight=3,depth=-1,leaves=31,lr=.04,rounds=400,reg=0,child=20,seed=42,window=24):
    return dict(arch=arch,k=k,weight=weight,depth=depth,leaves=leaves,lr=lr,rounds=rounds,reg=reg,child=child,seed=seed,window=window)
INITIAL=[cfg(),cfg(arch="xgb",depth=5,reg=1),cfg(arch="cat",depth=6,reg=3),
         *[cfg(k=k) for k in (64,128,256,512)],
         *[cfg(arch="xgb",depth=5,reg=1,k=k) for k in (128,256,512)],
         cfg(weight=1),cfg(weight=5),cfg(leaves=15,child=80,reg=5),
         cfg(leaves=63,child=40,reg=5),cfg(lr=.02,rounds=800,reg=3),
         cfg(arch="xgb",depth=3,weight=1,lr=.04,rounds=600,reg=5),
         cfg(arch="xgb",depth=4,weight=3,lr=.02,rounds=800,reg=5),
         cfg(arch="xgb",depth=6,weight=1,reg=5),
         cfg(window=6),cfg(window=12)]
def key(c): return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:16]
def fit_stats(raw):
    med=np.nanmedian(raw,axis=0)
    scale=np.nanpercentile(raw,75,axis=0)-np.nanpercentile(raw,25,axis=0)
    return np.where(np.isfinite(med),med,0),np.where(np.isfinite(scale)&(scale>1e-12),scale,1)
def names():
    channels=[f"{v}__{s}" for v in ALL for s in ("value","mask","age","never")]
    channels += [f"{v}__{s}" for s in ("delta6","delta_valid","count6") for v in ALL]
    channels += ["shock","shock_valid"]
    return [f"{s}__{v}" for s in ("last","mean","std","delta") for v in channels]
def build(raw,offsets,med,scale,window):
    m=transform(raw,offsets,ALL,med,scale,"original",True)
    rr,rs=endpoints(offsets)
    x=tabular(m,rr,rs,window)
    if x.shape[1]!=len(names()): raise ValueError("Feature schema mismatch")
    return x
def make(c):
    if c["arch"]=="lgb":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=c["rounds"],learning_rate=c["lr"],num_leaves=c["leaves"],
          max_depth=c["depth"],min_child_samples=c["child"],reg_lambda=c["reg"],scale_pos_weight=c["weight"],
          n_jobs=4,random_state=c["seed"],verbosity=-1)
    if c["arch"]=="xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=c["rounds"],max_depth=c["depth"],learning_rate=c["lr"],
          subsample=.8,colsample_bytree=.8,tree_method="hist",scale_pos_weight=c["weight"],reg_lambda=c["reg"],
          n_jobs=4,random_state=c["seed"])
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=c["rounds"],depth=c["depth"],learning_rate=c["lr"],
      l2_leaf_reg=c["reg"],scale_pos_weight=c["weight"],random_seed=c["seed"],thread_count=4,verbose=False,allow_writing_files=False)
def rank_features(x,y):
    from lightgbm import LGBMClassifier
    m=LGBMClassifier(n_estimators=200,num_leaves=31,learning_rate=.05,n_jobs=4,verbosity=-1,random_state=42,importance_type="gain")
    m.fit(x,y)
    return np.argsort(-m.feature_importances_,kind="stable")
def take(x,rows,cols): return x[np.ix_(rows,cols)]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--adaptive",type=int,default=8)
    p.add_argument("--retry-failed",action="store_true")
    args=p.parse_args()
    if args.adaptive<0: p.error("Invalid budget")
    groups=json.loads(MANIFEST.read_text(encoding="utf-8"))["split"]
    if tuple(len(groups[k]) for k in ("train","validation","test"))!=(25814,6454,8068) or len(set(groups["train"]+groups["validation"]+groups["test"]))!=40336: raise ValueError("Locked split changed")
    if args.dry_run:
        print("Initial",len(INITIAL),"adaptive",args.adaptive,"feature count",len(names()),"Test HELD OUT")
        for c in INITIAL: print(c)
        return
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(version=1,initial=INITIAL,adaptive=args.adaptive,manifest=hashlib.sha256(MANIFEST.read_bytes()).hexdigest())
    path=OUT/"protocol.json"
    if path.exists() and json.loads(path.read_text())!=protocol: raise ValueError("Existing protocol differs")
    if not path.exists(): atomic_json(path,protocol)
    raw,ty,to=read_raw(groups["train"])
    trr,_=endpoints(to); y=ty[trr]
    patient=np.repeat(np.arange(len(groups["train"])),np.maximum(np.diff(to)-23,0))
    strata=[pid.split("/")[0]+"/"+str(int(ty[a:b].max())) for pid,a,b in zip(groups["train"],to[:-1],to[1:])]
    fit_ids,tune_ids=train_test_split(np.arange(len(strata)),test_size=.2,random_state=42,stratify=strata)
    fit=np.flatnonzero(np.isin(patient,fit_ids)); tune=np.flatnonzero(np.isin(patient,tune_ids))
    raw_patient=np.repeat(np.arange(len(strata)),np.diff(to))
    med,scale=fit_stats(raw[np.isin(raw_patient,fit_ids)])
    if len(y)!=430905: raise ValueError("Train endpoint count mismatch")
    rows=[]; seen=set(); matrix=None; current=None; ranks={}
    def get(w):
        nonlocal matrix,current
        if w!=current:
            matrix=None; gc.collect()
            matrix=build(raw,to,med,scale,w); current=w
        return matrix
    def trial(c):
        k=key(c)
        if k in seen: return
        seen.add(k)
        folder=OUT/"trials"/k; folder.mkdir(parents=True,exist_ok=True)
        done=folder/"done.json"
        if done.exists():
            r=json.loads(done.read_text())
            if r["config"]!=c: raise ValueError("Config mismatch")
        else:
            if (folder/"error.txt").exists() and not args.retry_failed: return
            try:
                x=get(c["window"]); n=x.shape[1]
                if c["k"]:
                    if c["window"] not in ranks: ranks[c["window"]]=rank_features(x[fit],y[fit])
                    cols=np.sort(ranks[c["window"]][:c["k"]])
                else: cols=np.arange(n)
                m=make(c); a=take(x,fit,cols); m.fit(a,y[fit]); del a
                b=take(x,tune,cols); pred=m.predict_proba(b)[:,1]; del b,m
                r=dict(id=k,config=c,**metric(y[tune],pred))
                atomic_json(done,r); (folder/"error.txt").unlink(missing_ok=True)
            except Exception:
                (folder/"error.txt").write_text(traceback.format_exc(),encoding="utf-8")
                print("FAILED",k,flush=True); return
            finally: gc.collect()
        rows.append(r)
        atomic_json(OUT/"inner_leaderboard.json",sorted(rows,key=lambda v:-v["ap"]))
        print("Inner",k,r["ap"],c,flush=True)
    for c in INITIAL: trial(c)
    follow=[]
    for r in sorted(rows,key=lambda v:-v["ap"])[:4]:
        for changes in (dict(weight=1),dict(weight=3),dict(k=256),dict(lr=.015,rounds=1000),dict(reg=10),dict(seed=43)):
            c=dict(r["config"],**changes)
            if key(c) not in seen and key(c) not in {key(v) for v in follow}: follow.append(c)
    for c in follow[:args.adaptive]: trial(c)
    if not rows: raise RuntimeError("No successful trials")
    ranked=sorted(rows,key=lambda v:-v["ap"])
    # Select best per architecture, plus best overall distinct setting.
    selected=[]
    for a in ("lgb","xgb","cat"):
        match=next((r for r in ranked if r["config"]["arch"]==a),None)
        if match: selected.append(match)
    for r in ranked:
        if r not in selected: selected.append(r); break
    atomic_json(OUT/"selected.json",selected)
    matrix=None; gc.collect()
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
            pred=np.load(folder/"scores.npy",allow_pickle=False)
            if len(pred)!=len(vr) or not np.isfinite(pred).all(): raise ValueError("Invalid saved scores")
        else:
            x=build(raw,to,med,scale,c["window"])
            cols=np.sort(rank_features(x,y)[:c["k"]]) if c["k"] else np.arange(x.shape[1])
            m=make(c)
            if c["k"]:
                a=x[:,cols]; del x; gc.collect(); m.fit(a,y); del a
            else: m.fit(x,y); del x
            gc.collect()
            v=build(vraw,vo,med,scale,c["window"])
            pred=m.predict_proba(v[:,cols] if c["k"] else v)[:,1]; del v
            joblib.dump(dict(model=m,config=c,cols=cols,names=names(),median=med,scale=scale),folder/"model.joblib")
            np.save(folder/"scores.npy",pred); atomic_json(done,dict(config=c,**metric(vy[vr],pred)))
            del m; gc.collect()
        finals.append(dict(id=r["id"],config=c,inner_ap=r["ap"],**metric(vy[vr],pred))); preds.append(pred)
    summary=dict(scope="inner Train selection, reused locked Validation common endpoints",test_patients_evaluated=0,
        target_ap=.4,inner_trials=ranked,finalists=finals,equal_blend=metric(vy[vr],np.mean(preds,axis=0)),
        failures=[p.parent.name for p in (OUT/"trials").glob("*/error.txt")])
    atomic_json(OUT/"summary.json",summary)
    pd.DataFrame([dict(id=r["id"],inner_ap=r["inner_ap"],ap=r["ap"],auroc=r["auroc"],**r["config"]) for r in finals]).to_csv(OUT/"leaderboard.csv",index=False)
    print(pd.read_csv(OUT/"leaderboard.csv").to_string(index=False))
    print("Equal blend",summary["equal_blend"],"Test HELD OUT")
    print("Saved",OUT/"summary.json")
if __name__=="__main__": main()
