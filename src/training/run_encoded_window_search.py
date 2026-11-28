"""SEWS v2-21: strict 16-variable within-window causal encoding and boosting.
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

OUT=ROOT/"reports/tree_v2_21"
from train_lstm import FEATURES
from preprocess import FFILL_MAX_AGE_HOURS
BASE=cfg(arch="xgb",depth=3,weight=3,lr=.04,rounds=600,reg=5)
SLOW=cfg(arch="xgb",depth=3,weight=1,lr=.015,rounds=1000,reg=5)
LGB=cfg()
CAT=cfg(arch="cat",depth=6,reg=3)
MODES=("encoded","segments","trends","combined")
TRIALS=[dict(mode=m,base=b) for m in MODES for b in (BASE,SLOW,dict(SLOW,weight=3),LGB,CAT)]
# Keep temporal channels compact: measured/age/last value, not all redundant channels.
VITALS=tuple(FEATURES)
IDX=np.array([ALL.index(v) for v in VITALS])
def key(c): return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:16]

def construct(raw,offsets,med,scale,mode):
    # Recompute all ages and fills INSIDE each 24-hour window.
    # No time-since-admission or measurements before window start.
    rr,_=endpoints(offsets)
    med=med[IDX]; scale=scale[IDX]
    limits=np.array([FFILL_MAX_AGE_HOURS.get(v) or 0 for v in VITALS])
    t=np.arange(24)[None,:,None]
    out=None
    for a in range(0,len(rr),1024):
        rows=rr[a:a+1024]
        x=raw[rows[:,None]-np.arange(23,-1,-1)[None,:]][:,:,IDX].astype(np.float64)
        mask=np.isfinite(x)
        last=np.maximum.accumulate(np.where(mask,t,-1),axis=1)
        never=last<0
        age=t-last
        latest=np.take_along_axis(x,np.maximum(last,0),axis=1)
        latest=np.where(never,med,latest)
        val=np.where(~never & (age<=limits),latest,med)
        z=(val-med)/scale
        channels=np.stack((z,mask,np.where(never,0,age/48),never),axis=-1).reshape(len(rows),24,64)
        # Same 4 summary families used by successful expanded v2 models.
        blocks=[channels[:,-1],channels.mean(1),channels.std(1),channels[:,-1]-channels[:,0]]
        if mode in ("segments","combined"):
            carry=np.concatenate(((latest-med)/scale,mask,np.where(never,1.,age/48)),axis=2)
            for j in range(4):
                q=carry[:,6*j:6*j+6]
                blocks.extend((q.mean(1),q.std(1)))
        if mode in ("trends","combined"):
            past=np.maximum(np.arange(24)-6,0)
            valid=~never & ~never[:,past]
            delta=np.where(valid,(latest-latest[:,past])/scale,0.)
            count=np.concatenate((np.zeros((len(rows),1,16)),np.cumsum(mask,axis=1)),axis=1)
            count6=(count[:,1:]-count[:,np.maximum(np.arange(1,25)-6,0)])/6
            trend=np.concatenate((delta,valid,count6),axis=2)
            blocks.extend((trend[:,-1],trend.mean(1),trend.std(1),trend[:,-1]-trend[:,0]))
        block=np.concatenate(blocks,axis=1).astype(np.float32)
        if not np.isfinite(block).all(): raise ValueError("Nonfinite features")
        if out is None: out=np.empty((len(rr),block.shape[1]),np.float32)
        out[a:a+len(rows)]=block
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
    protocol=dict(version=1,features=list(VITALS),device=args.device,trials=TRIALS,manifest=hashlib.sha256(MANIFEST.read_bytes()).hexdigest())
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
    # Freeze best per architecture and best encoded control.
    selected=[]
    for arch in ("xgb","lgb","cat"):
        r=next((r for r in ranked if r["config"]["mode"]!="encoded" and r["config"]["base"]["arch"]==arch),None)
        if r: selected.append(r)
    control=next((r for r in ranked if r["config"]["mode"]=="encoded"),None)
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
            joblib.dump(dict(model=m,config=c,median=med,scale=scale,ordered_variables=VITALS,window_rows=24,within_window_only=True),folder/"model.joblib")
            np.save(folder/"scores.npy",pred); atomic_json(done,dict(config=c,**metric(vy[vr],pred)))
            del m; gc.collect()
        if len(pred)!=len(vr) or not np.isfinite(pred).all(): raise ValueError("Invalid scores")
        finals.append(dict(id=r["id"],config=c,inner_ap=r["ap"],**metric(vy[vr],pred))); preds.append(pred)
    summary=dict(scope="16 variables, within-window causal encoding, inner Train selection, reused Validation",
      test_patients_evaluated=0,target_ap=.4,inner_trials=ranked,finalists=finals,
      equal_blend=metric(vy[vr],np.mean(preds,axis=0)),failures=[p.parent.name for p in (OUT/"trials").glob("*/error.txt")])
    atomic_json(OUT/"summary.json",summary)
    table=pd.DataFrame([dict(id=r["id"],inner_ap=r["inner_ap"],ap=r["ap"],auroc=r["auroc"],mode=r["config"]["mode"],**r["config"]["base"]) for r in finals])
    table.to_csv(OUT/"leaderboard.csv",index=False)
    print(table.to_string(index=False)); print("Equal blend",summary["equal_blend"],"Test HELD OUT")
if __name__=="__main__": main()
