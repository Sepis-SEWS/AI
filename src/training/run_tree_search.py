"""v2-12: causal multi-scale tree search; inner Train patient holdout only for selection.
Run from repository root. Locked Validation scored only for selected final candidates.
No Test raw files read. Separate outputs; resume completed trials.
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
from run_adaptive_search import ROOT, MANIFEST, BASELINE, ALL, FEATURES, read_raw, endpoints, metric, atomic_json

OUT = ROOT / "reports/tree_v2_12"
DYNAMIC = ALL[:34]
WINDOWS = (3, 6, 12, 24)
VITALS = ALL[:8]

def feature_names():
    names = list(ALL)
    for col in DYNAMIC:
        names.extend([f"{col}__last", f"{col}__age", f"{col}__delta"])
    for w in WINDOWS:
        for stat in ("mean", "min", "max", "std", "count"):
            names.extend(f"{col}__{stat}{w}" for col in DYNAMIC)
    names.extend(["shock_index", "pulse_pressure", "spo2_fio2", "bun_creatinine"])
    return names

def features(raw, offsets, dest):
    names = feature_names()
    rows, _ = endpoints(offsets)
    x = np.lib.format.open_memmap(dest, mode="w+", dtype=np.float32, shape=(len(rows), len(names)))
    cursor = 0
    for i,(a,b) in enumerate(zip(offsets[:-1], offsets[1:]),1):
        if b-a < 24: continue
        d = pd.DataFrame(raw[a:b,:34],columns=DYNAMIC)
        last = d.ffill()
        t = np.arange(len(d))[:,None]
        obs = d.notna().to_numpy()
        last_idx = np.maximum.accumulate(np.where(obs,t,-1),axis=0)
        age = np.where(last_idx>=0,t-last_idx,np.nan)
        delta = last-last.shift(6)
        block = [raw[a:b],np.stack([last.to_numpy(),age,delta.to_numpy()],axis=2).reshape(len(d),-1)]
        for w in WINDOWS:
            roll = d.rolling(w,min_periods=1)
            block.extend([roll.mean().to_numpy(),roll.min().to_numpy(),roll.max().to_numpy(),
                          roll.std(ddof=0).to_numpy(),d.notna().rolling(w,min_periods=1).sum().to_numpy()])
        def div(a,b):
            a,b = last[a].to_numpy(),last[b].to_numpy()
            return np.divide(a,b,out=np.full(len(d),np.nan),where=np.isfinite(a)&np.isfinite(b)&(b>0))
        block.append(np.column_stack([div("HR","SBP"),last["SBP"]-last["DBP"],div("O2Sat","FiO2"),div("BUN","Creatinine")]))
        m=np.concatenate(block,axis=1)[23:].astype(np.float32)
        m[~np.isfinite(m)]=np.nan
        x[cursor:cursor+len(m)]=m
        cursor+=len(m)
        if i%5000==0: print(f"Features {i}/{len(offsets)-1}",flush=True)
    if cursor!=len(rows): raise ValueError("Feature row count mismatch")
    x.flush()
    return x

def columns(group):
    names=feature_names()
    def include(n):
        base=n.split("__")[0]
        derived=n in ("shock_index","pulse_pressure","spo2_fio2","bun_creatinine")
        if group=="all": return True
        if group=="short": return not any(n.endswith("__"+s+str(w)) for w in (12,24) for s in ("mean","min","max","std","count"))
        if group=="dynamic": return base in DYNAMIC or derived
        if group=="base": return base in FEATURES or derived
        if group=="no_context": return base not in ("Unit1","Unit2","HospAdmTime","ICULOS")
        if group=="values": return "__" not in n or ("__mean" in n) or ("__last" in n) or derived
        raise ValueError(group)
    return np.array([i for i,n in enumerate(names) if include(n)])

def cfg(arch="lgb",group="all",depth=5,weight=3,lr=.04,rounds=400,reg=3,seed=42):
    return dict(arch=arch,group=group,depth=depth,weight=weight,lr=lr,rounds=rounds,reg=reg,seed=seed)

INITIAL=[cfg(group=g) for g in ("all","short","dynamic","base","no_context","values")]
INITIAL += [cfg(arch=a) for a in ("xgb","cat")]
INITIAL += [cfg(arch=a,depth=d,weight=w,lr=l,rounds=r,reg=z) for a in ("lgb","xgb")
            for d,w,l,r,z in [(3,1,.06,600,5),(4,1,.03,800,5),(6,1,.04,400,8),
                               (5,5,.03,600,8),(7,3,.025,600,10)]]

def key(c): return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:16]

def make(c):
    common=dict(random_state=c["seed"],learning_rate=c["lr"])
    if c["arch"]=="lgb":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**common,n_estimators=c["rounds"],max_depth=c["depth"],
            num_leaves=min(2**c["depth"]-1,63),min_child_samples=60,reg_lambda=c["reg"],
            scale_pos_weight=c["weight"],subsample=.85,subsample_freq=1,colsample_bytree=.8,n_jobs=4,verbosity=-1)
    if c["arch"]=="xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(**common,n_estimators=c["rounds"],max_depth=c["depth"],reg_lambda=c["reg"],
            min_child_weight=8,scale_pos_weight=c["weight"],subsample=.85,colsample_bytree=.8,
            tree_method="hist",n_jobs=4,eval_metric="logloss")
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=c["rounds"],depth=c["depth"],learning_rate=c["lr"],
        l2_leaf_reg=c["reg"],scale_pos_weight=c["weight"],random_seed=c["seed"],thread_count=4,verbose=False,allow_writing_files=False)

def take(x, rows, cols):
    # Single output allocation rather than a full-width intermediate.
    return x[np.ix_(rows,cols)]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--adaptive",type=int,default=8)
    parser.add_argument("--finalists",type=int,default=4)
    parser.add_argument("--retry-failed",action="store_true")
    args=parser.parse_args()
    if args.adaptive<0 or args.finalists<1: parser.error("Invalid budget")
    groups=json.loads(MANIFEST.read_text(encoding="utf-8"))["split"]
    if tuple(len(groups[k]) for k in ("train","validation","test"))!=(25814,6454,8068) or len(set(groups["train"]+groups["validation"]+groups["test"]))!=40336:
        raise ValueError("Locked split changed")
    if args.dry_run:
        print("Initial",len(INITIAL),"adaptive",args.adaptive,"finalists",args.finalists)
        print("Feature count",len(feature_names()),"Test HELD OUT")
        for c in INITIAL: print(c)
        return
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(version=1,manifest=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),initial=INITIAL,adaptive=args.adaptive,finalists=args.finalists)
    path=OUT/"protocol.json"
    if path.exists() and json.loads(path.read_text())!=protocol: raise ValueError("Existing protocol differs")
    if not path.exists(): atomic_json(path,protocol)
    traw,ty,to=read_raw(groups["train"])
    trr,_=endpoints(to)
    patient=np.repeat(np.arange(len(groups["train"])),np.maximum(np.diff(to)-23,0))
    # Stratify by source and patient-level original label, never split a patient.
    strata=[pid.split("/")[0]+"/"+str(int(ty[a:b].max())) for pid,a,b in zip(groups["train"],to[:-1],to[1:])]
    fit_ids,tune_ids=train_test_split(np.arange(len(strata)),test_size=.2,random_state=42,stratify=strata)
    fit=np.flatnonzero(np.isin(patient,fit_ids)); tune=np.flatnonzero(np.isin(patient,tune_ids))
    y=ty[trr]
    if len(y)!=430905 or not len(fit) or not len(tune): raise ValueError("Train rows changed")
    print(f"Inner Train patients: fit={len(fit_ids)} tune={len(tune_ids)}",flush=True)
    # Cache regenerated deterministically each run, only in new experiment directory.
    tr=features(traw,to,OUT/"train_features.npy")
    del traw; gc.collect()
    results=[]; seen=set()
    def trial(c):
        k=key(c)
        if k in seen: return
        seen.add(k)
        folder=OUT/"trials"/k; folder.mkdir(parents=True,exist_ok=True)
        done=folder/"done.json"
        if done.exists():
            r=json.loads(done.read_text())
            if r["config"]!=c: raise ValueError("Trial config mismatch")
        else:
            if (folder/"error.txt").exists() and not args.retry_failed: return
            try:
                print("Trial",k,c,flush=True)
                cols=columns(c["group"])
                m=make(c)
                a=take(tr,fit,cols)
                m.fit(a,y[fit]); del a
                b=take(tr,tune,cols)
                p=m.predict_proba(b)[:,1]; del b
                r=dict(id=k,config=c,**metric(y[tune],p))
                atomic_json(done,r)
                (folder/"error.txt").unlink(missing_ok=True)
            except Exception:
                (folder/"error.txt").write_text(traceback.format_exc(),encoding="utf-8")
                print("FAILED",k,flush=True); return
            finally: gc.collect()
        results.append(r)
        atomic_json(OUT/"inner_leaderboard.json",sorted(results,key=lambda x:-x["ap"]))
        print("Inner AP",r["ap"],flush=True)
    for c in INITIAL: trial(c)
    follow=[]
    for r in sorted(results,key=lambda x:-x["ap"])[:3]:
        c=r["config"]
        for changes in (dict(weight=1),dict(weight=5),dict(depth=3,reg=10),dict(depth=7,reg=10),
                        dict(lr=.02,rounds=1000),dict(seed=43)):
            new=dict(c,**changes)
            if key(new) not in seen and key(new) not in {key(x) for x in follow}: follow.append(new)
    for c in follow[:args.adaptive]: trial(c)
    if not results: raise RuntimeError("No successful trials")
    ranked=sorted(results,key=lambda x:-x["ap"])
    selected=ranked[:args.finalists]
    atomic_json(OUT/"selected.json",selected)
    # Only now read locked Validation. Same endpoints as prior experiments.
    vraw,vy,vo=read_raw(groups["validation"]); vr,_=endpoints(vo)
    with np.load(BASELINE,allow_pickle=False) as a:
        ids=np.repeat(groups["validation"],np.maximum(np.diff(vo)-23,0))
        if not np.array_equal(ids,a["patient_ids"].astype(str)) or not np.array_equal(vy[vr],a["labels"]) or not np.array_equal(vraw[vr,-1],a["icul_os"]):
            raise ValueError("Validation archive alignment mismatch")
    va=features(vraw,vo,OUT/"validation_features.npy"); del vraw; gc.collect()
    finals=[]; preds=[]
    for r in selected:
        c=r["config"]; folder=OUT/"final"/r["id"]; folder.mkdir(parents=True,exist_ok=True)
        done=folder/"done.json"
        if done.exists():
            p=np.load(folder/"scores.npy",allow_pickle=False)
            if len(p)!=len(vr) or not np.isfinite(p).all(): raise ValueError("Invalid final scores")
        else:
            cols=columns(c["group"])
            m=make(c); a=take(tr,np.arange(len(y)),cols); m.fit(a,y); del a
            b=take(va,np.arange(len(vr)),cols); p=m.predict_proba(b)[:,1]; del b
            joblib.dump(dict(model=m,columns=[feature_names()[i] for i in cols],config=c),folder/"model.joblib")
            np.save(folder/"scores.npy",p)
            atomic_json(done,dict(config=c,**metric(vy[vr],p)))
            del m; gc.collect()
        finals.append(dict(id=r["id"],config=c,inner_ap=r["ap"],**metric(vy[vr],p))); preds.append(p)
    # Equal blend fixed by inner ranking, not tuned on locked Validation.
    ensemble=metric(vy[vr],np.mean(preds,axis=0))
    summary=dict(scope="Train inner holdout selection; locked Validation common rows only; previously reused Validation",
        target_ap=.4,test_patients_evaluated=0,inner_trials=ranked,finalists=finals,
        equal_blend=ensemble,failures=[p.parent.name for p in (OUT/"trials").glob("*/error.txt")])
    atomic_json(OUT/"summary.json",summary)
    pd.DataFrame([dict(id=r["id"],inner_ap=r["inner_ap"],ap=r["ap"],auroc=r["auroc"],**r["config"]) for r in finals]).to_csv(OUT/"leaderboard.csv",index=False)
    print(pd.read_csv(OUT/"leaderboard.csv").to_string(index=False))
    print("Fixed equal blend:",ensemble,"Test HELD OUT")
    print("Saved:",OUT/"summary.json")

if __name__=="__main__": main()
