"""SEWS v2-17: binary vs global pairwise ranking objectives.
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

OUT=ROOT/"reports/tree_v2_17"
BASE=cfg(arch="xgb",depth=3,weight=3,lr=.04,rounds=600,reg=5)
SLOW=cfg(arch="xgb",depth=3,weight=1,lr=.015,rounds=1000,reg=5)
LGB=cfg()
CAT=cfg(arch="cat",depth=6,reg=3)
MODES=("control","personal")
TRIALS=[]
for mode in MODES:
    for base in (BASE,SLOW):
        TRIALS.append(dict(mode=mode,base=base,objective="binary",pair_method="mean",pairs=4))
    for objective in ("rank:pairwise","rank:map"):
        for pairs in (2,8):
            TRIALS.append(dict(mode=mode,base=SLOW,objective=objective,pair_method="mean",pairs=pairs))
        TRIALS.append(dict(mode=mode,base=BASE,objective=objective,pair_method="mean",pairs=4))

# Keep temporal channels compact: measured/age/last value, not all redundant channels.
VITALS=("HR","O2Sat","Temp","SBP","MAP","DBP","Resp","FiO2","Lactate","Creatinine","WBC","Platelets","pH","BUN","Bilirubin_total")
IDX=np.array([ALL.index(v) for v in VITALS])
def key(c): return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:16]
from run_history_feature_search import construct

def model(c,device):
    from xgboost import XGBClassifier, XGBRanker
    b=c["base"]
    params=dict(n_estimators=b["rounds"],max_depth=b["depth"],learning_rate=b["lr"],
        reg_lambda=b["reg"],subsample=.8,colsample_bytree=.8,tree_method="hist",
        random_state=b["seed"],n_jobs=4,device=device)
    if c["objective"]=="binary":
        return XGBClassifier(**params,scale_pos_weight=b["weight"])
    # A single ranking group represents the global hourly risk ordering.
    # Sampled pairs avoid enumerating every positive-negative combination.
    return XGBRanker(**params,objective=c["objective"],
        lambdarank_pair_method=c["pair_method"],lambdarank_num_pair_per_sample=c["pairs"])

def train(m,c,x,y):
    if c["objective"]=="binary": m.fit(x,y)
    else: m.fit(x,y,group=[len(y)])

def score(m,c,x):
    if c["objective"]=="binary": return m.predict_proba(x)[:,1]
    return m.predict(x)

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
    protocol=dict(version=1,base_representation="v2-16",device=args.device,trials=TRIALS,manifest=hashlib.sha256(MANIFEST.read_bytes()).hexdigest())
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
                    a=x[fit]; train(m,c,a,y[fit]); del a
                    b=x[tune]; pred=score(m,c,b); del b,m
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
    # Select by inner AP only, one finalist per objective.
    selected=[]
    for objective in ("binary","rank:pairwise","rank:map"):
        r=next((r for r in ranked if r["config"]["objective"]==objective),None)
        if r: selected.append(r)
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
            pred=score(m,c,v); del v
            joblib.dump(dict(model=m,config=c,median=med,scale=scale,score_type="probability" if c["objective"]=="binary" else "ranking_score",base_representation="v2-16"),folder/"model.joblib")
            np.save(folder/"scores.npy",pred); atomic_json(done,dict(config=c,**metric(vy[vr],pred)))
            del m; gc.collect()
        if len(pred)!=len(vr) or not np.isfinite(pred).all(): raise ValueError("Invalid scores")
        finals.append(dict(id=r["id"],config=c,inner_ap=r["ap"],**metric(vy[vr],pred))); preds.append(pred)
    summary=dict(scope="inner Train patient selection, reused locked Validation common endpoints",
      test_patients_evaluated=0,target_ap=.4,inner_trials=ranked,finalists=finals,
      equal_rank_blend=metric(vy[vr],np.mean([pd.Series(p).rank(method="average",pct=True).to_numpy() for p in preds],axis=0)),failures=[p.parent.name for p in (OUT/"trials").glob("*/error.txt")])
    atomic_json(OUT/"summary.json",summary)
    table=pd.DataFrame([dict(id=r["id"],inner_ap=r["inner_ap"],ap=r["ap"],auroc=r["auroc"],mode=r["config"]["mode"],objective=r["config"]["objective"],pairs=r["config"]["pairs"],**r["config"]["base"]) for r in finals])
    table.to_csv(OUT/"leaderboard.csv",index=False)
    print(table.to_string(index=False)); print("Equal percentile-rank blend (offline only)",summary["equal_rank_blend"],"Test HELD OUT")
if __name__=="__main__": main()
