"""v2-14: patient-balanced weights, cross-fitted hard negatives and patient bagging.
One command, inner Train selection only; locked Validation for frozen finalists.
No Test data reads; no large feature cache files.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from run_tree_refine import fit_stats, build, make, cfg, names
from run_adaptive_search import ROOT, MANIFEST, BASELINE, read_raw, endpoints, metric, atomic_json

OUT=ROOT/"reports/tree_v2_14"
BASES={
 "xgb_shallow":cfg(arch="xgb",depth=3,weight=3,lr=.04,rounds=600,reg=5),
 "xgb_slow":cfg(arch="xgb",depth=3,weight=1,lr=.015,rounds=1000,reg=5),
 "lgb":cfg()
}
# Baseline controls plus changes to the effective training distribution.
TRIALS=[dict(base=b,policy=p) for b in BASES for p in
        ("control","patient_sqrt","patient_equal","hard2","hard4","patient_hard","bagging")]
def key(c): return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:16]
def patient_weights(y,patient,power):
    # Normalize within each class so class weight remains separately controlled.
    w=np.ones(len(y),dtype=np.float32)
    for label in (0,1):
        mask=y==label
        counts=np.bincount(patient[mask],minlength=int(patient.max())+1)
        a=counts[patient[mask]].astype(float)**(-power)
        w[mask]=a/a.mean()
    return w
def hard_weights(y,oof,factor):
    if not np.isfinite(oof).all(): raise ValueError("Missing cross-fitted score")
    neg=y==0
    q=np.quantile(oof[neg],.9)
    w=np.ones(len(y),dtype=np.float32)
    w[neg & (oof>=q)]=factor
    w[neg]/=w[neg].mean()
    return w
def crossfit(raw,labels,offsets,patient,raw_patient,ids,strata):
    """Every difficulty score comes from a model and stats excluding that patient."""
    rr,_=endpoints(offsets)
    y=labels[rr]
    scores=np.full(len(y),np.nan,dtype=np.float32)
    # Fix one common mining teacher for all downstream candidates.
    c=cfg(arch="xgb",depth=3,weight=1,lr=.04,rounds=300,reg=5)
    for f,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=42).split(ids,np.asarray(strata)[ids]),1):
        train_ids,test_ids=ids[a],ids[b]
        if set(train_ids)&set(test_ids): raise ValueError("Crossfit overlap")
        tr=np.flatnonzero(np.isin(patient,train_ids)); te=np.flatnonzero(np.isin(patient,test_ids))
        med,scale=fit_stats(raw[np.isin(raw_patient,train_ids)])
        x=build(raw,offsets,med,scale,24)
        m=make(c)
        xx=x[tr]; m.fit(xx,y[tr]); del xx
        scores[te]=m.predict_proba(x[te])[:,1]
        del x,m; gc.collect()
        print("Crossfit mining fold",f,flush=True)
    required=np.isin(patient,ids)
    if not np.isfinite(scores[required]).all() or np.isfinite(scores[~required]).any():
        raise ValueError("Crossfit scope mismatch")
    return scores
def train_predict(c,x,y,patient,v,oof):
    policy=c["policy"]
    model_cfg=BASES[c["base"]]
    models=[]; preds=[]
    if policy=="bagging":
        ids=np.unique(patient)
        for seed in (42,43,44):
            rng=np.random.default_rng(seed)
            chosen=rng.choice(ids,len(ids),replace=True)
            counts=np.bincount(chosen,minlength=int(patient.max())+1)
            w=counts[patient].astype(np.float32)
            keep=np.flatnonzero(w>0)
            m=make(dict(model_cfg,seed=seed))
            m.fit(x[keep],y[keep],sample_weight=w[keep])
            preds.append(m.predict_proba(v)[:,1]); models.append(m)
    else:
        w=np.ones(len(y),dtype=np.float32)
        if policy in ("patient_sqrt","patient_hard"): w*=patient_weights(y,patient,.5)
        if policy=="patient_equal": w*=patient_weights(y,patient,1.)
        if policy in ("hard2","hard4","patient_hard"):
            w*=hard_weights(y,oof,4 if policy=="hard4" else 2)
        for label in (0,1):
            mask=y==label; w[mask]/=w[mask].mean()
        m=make(model_cfg); m.fit(x,y,sample_weight=w)
        preds.append(m.predict_proba(v)[:,1]); models.append(m)
    return models,np.mean(preds,axis=0)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--finalists",type=int,default=4)
    args=p.parse_args()
    if args.finalists<1 or args.finalists>len(TRIALS): p.error("Invalid finalists")
    groups=json.loads(MANIFEST.read_text(encoding="utf-8"))["split"]
    if tuple(len(groups[k]) for k in ("train","validation","test"))!=(25814,6454,8068) or len(set(groups["train"]+groups["validation"]+groups["test"]))!=40336: raise ValueError("Locked split changed")
    if args.dry_run:
        print(len(TRIALS),"settings, 3-fold mining, finalists",args.finalists,"Test HELD OUT")
        for c in TRIALS: print(c)
        return
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(version=1,bases=BASES,trials=TRIALS,finalists=args.finalists,manifest=hashlib.sha256(MANIFEST.read_bytes()).hexdigest())
    protocol_path=OUT/"protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol: raise ValueError("Existing protocol differs")
    if not protocol_path.exists(): atomic_json(protocol_path,protocol)
    raw,labels,offsets=read_raw(groups["train"])
    rr,_=endpoints(offsets); y=labels[rr]
    patient=np.repeat(np.arange(len(groups["train"])),np.maximum(np.diff(offsets)-23,0))
    raw_patient=np.repeat(np.arange(len(groups["train"])),np.diff(offsets))
    strata=[pid.split("/")[0]+"/"+str(int(labels[a:b].max())) for pid,a,b in zip(groups["train"],offsets[:-1],offsets[1:])]
    fit_ids,tune_ids=train_test_split(np.arange(len(strata)),test_size=.2,random_state=42,stratify=strata)
    fit=np.flatnonzero(np.isin(patient,fit_ids)); tune=np.flatnonzero(np.isin(patient,tune_ids))
    if len(y)!=430905: raise ValueError("Train count mismatch")
    # Reuse completed metrics only; crossfit is recomputed if unfinished hard trials remain.
    needs=any(c["policy"] in ("hard2","hard4","patient_hard") and not (OUT/"trials"/key(c)/"done.json").exists() for c in TRIALS)
    oof=crossfit(raw,labels,offsets,patient,raw_patient,fit_ids,strata) if needs else None
    med,scale=fit_stats(raw[np.isin(raw_patient,fit_ids)])
    matrix=build(raw,offsets,med,scale,24)
    x=matrix[fit]; v=matrix[tune]; del matrix; gc.collect()
    results=[]
    for c in TRIALS:
        folder=OUT/"trials"/key(c); folder.mkdir(parents=True,exist_ok=True)
        done=folder/"done.json"
        if done.exists():
            r=json.loads(done.read_text())
            if r["config"]!=c: raise ValueError("Resume config mismatch")
        else:
            models,pred=train_predict(c,x,y[fit],patient[fit],v,None if oof is None else oof[fit])
            r=dict(id=key(c),config=c,**metric(y[tune],pred))
            atomic_json(done,r); del models,pred; gc.collect()
        results.append(r)
        atomic_json(OUT/"inner_leaderboard.json",sorted(results,key=lambda r:-r["ap"]))
        print("Inner",r,flush=True)
    del x,v,oof; gc.collect()
    ranked=sorted(results,key=lambda r:-r["ap"])
    selected=ranked[:args.finalists]
    atomic_json(OUT/"selected.json",selected)
    need_full=any(r["config"]["policy"] in ("hard2","hard4","patient_hard") and not (OUT/"final"/r["id"]/"done.json").exists() for r in selected)
    full_oof=crossfit(raw,labels,offsets,patient,raw_patient,np.arange(len(strata)),strata) if need_full else None
    med,scale=fit_stats(raw)
    x=build(raw,offsets,med,scale,24); del raw; gc.collect()
    vraw,vy,vo=read_raw(groups["validation"]); vr,_=endpoints(vo)
    with np.load(BASELINE,allow_pickle=False) as a:
        ids=np.repeat(groups["validation"],np.maximum(np.diff(vo)-23,0))
        if not np.array_equal(ids,a["patient_ids"].astype(str)) or not np.array_equal(vy[vr],a["labels"]) or not np.array_equal(vraw[vr,-1],a["icul_os"]):
            raise ValueError("Validation alignment mismatch")
    v=build(vraw,vo,med,scale,24); del vraw; gc.collect()
    finals=[]; preds=[]
    for r in selected:
        c=r["config"]; folder=OUT/"final"/r["id"]; folder.mkdir(parents=True,exist_ok=True)
        done=folder/"done.json"
        if done.exists():
            old=json.loads(done.read_text())
            if old["config"]!=c: raise ValueError("Final config mismatch")
            pred=np.load(folder/"scores.npy",allow_pickle=False)
        else:
            models,pred=train_predict(c,x,y,patient,v,full_oof)
            joblib.dump(dict(models=models,config=c,base=BASES[c["base"]],median=med,scale=scale,names=names()),folder/"model.joblib")
            np.save(folder/"scores.npy",pred)
            atomic_json(done,dict(config=c,**metric(vy[vr],pred)))
            del models; gc.collect()
        if len(pred)!=len(vr) or not np.isfinite(pred).all(): raise ValueError("Invalid scores")
        finals.append(dict(id=r["id"],config=c,inner_ap=r["ap"],**metric(vy[vr],pred))); preds.append(pred)
    summary=dict(scope="inner Train patient selection, common locked Validation endpoints; Validation previously reused",test_patients_evaluated=0,
        target_ap=.4,inner_trials=ranked,finalists=finals,equal_blend=metric(vy[vr],np.mean(preds,axis=0)))
    atomic_json(OUT/"summary.json",summary)
    pd.DataFrame([dict(id=r["id"],inner_ap=r["inner_ap"],ap=r["ap"],auroc=r["auroc"],**r["config"]) for r in finals]).to_csv(OUT/"leaderboard.csv",index=False)
    print(pd.read_csv(OUT/"leaderboard.csv").to_string(index=False))
    print("Equal blend",summary["equal_blend"],"Test HELD OUT")
if __name__=="__main__": main()
