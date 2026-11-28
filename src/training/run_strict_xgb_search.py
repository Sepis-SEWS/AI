"""SEWS v2-19: strict 16-variable, 24-hour XGBoost feature search.
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

OUT=ROOT/"reports/tree_v2_19"
from train_xgboost import VARS as VITALS, FEATURES as ORIGINAL_NAMES
import warnings
IDX=np.array([ALL.index(v) for v in VITALS])
MODES=("control","measurement","segments","clinical","combined")
BASE=cfg(arch="xgb",depth=3,weight=3,lr=.04,rounds=600,reg=5)
SLOW=cfg(arch="xgb",depth=3,weight=1,lr=.015,rounds=1000,reg=5)
TRIALS=[dict(mode=m,base=b) for m in MODES for b in (BASE,SLOW,dict(SLOW,weight=3),dict(BASE,depth=4,weight=1))]
def key(c): return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:16]
def feature_names(mode):
    out=list(ORIGINAL_NAMES)
    if mode!="control":
        out += [f"{v}_{stat}" for stat in ("count","age","latest","first","delta") for v in VITALS]
    if mode in ("segments","combined"):
        out += [f"block{j}_{stat}_{v}" for j in range(4) for stat in ("mean","std","count","latest") for v in VITALS]
    if mode in ("clinical","combined"):
        out += ["shock_index","pulse_pressure","spo2_fio2","bun_creatinine","qsofa_rr_observed","qsofa_sbp_observed","rr_ge22","sbp_le100"]
    return out
def construct(raw,offsets,med,scale,mode):
    # Raw 24-row windows ONLY. No demographics, prior-stay history or ICULOS input.
    rr,_=endpoints(offsets)
    names=feature_names(mode)
    out=np.empty((len(rr),len(names)),np.float32)
    t=np.arange(24,dtype=float)[None,:,None]
    d=len(IDX)
    def divide(a,b):
        return np.divide(a,b,out=np.full(np.broadcast_shapes(np.shape(a),np.shape(b)),np.nan),where=np.isfinite(b)&(np.abs(b)>1e-12))
    for a in range(0,len(rr),1024):
        rows=rr[a:a+1024]
        x=raw[rows[:,None]-np.arange(23,-1,-1)[None,:]][:,:,IDX].astype(float)
        ok=np.isfinite(x); n=ok.sum(1)
        z=np.where(ok,x,0)
        mean=divide(z.sum(1),n)
        diff=np.where(ok,x-mean[:,None,:],0)
        std=np.sqrt(divide((diff**2).sum(1),n))
        std=np.where(n>=2,std,np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",RuntimeWarning)
            lo=np.nanmin(x,axis=1); hi=np.nanmax(x,axis=1)
            q=np.nanpercentile(x,[25,50,75],axis=1)
        skew=divide(divide((diff**3).sum(1),n),std**3)
        skew=np.where(n>=3,skew,np.nan)
        tm=divide(np.where(ok,t,0).sum(1),n)
        td=np.where(ok,t-tm[:,None,:],0)
        slope=divide((td*diff).sum(1),(td**2).sum(1))
        last=np.max(np.where(ok,t,-1),axis=1).astype(int)
        first=np.min(np.where(ok,t,24),axis=1).astype(int)
        latest=np.take_along_axis(x,np.maximum(last,0)[:,None,:],axis=1)[:,0,:]
        earliest=np.take_along_axis(x,np.minimum(first,23)[:,None,:],axis=1)[:,0,:]
        latest=np.where(n>0,latest,np.nan); earliest=np.where(n>0,earliest,np.nan)
        prev=np.maximum.accumulate(np.where(ok,t,-1),axis=1)[:,:-1,:].astype(int)
        prev_val=np.take_along_axis(x,np.maximum(prev,0),axis=1)
        pairs=ok[:,1:,:]&(prev>=0)
        rate=divide(np.where(pairs,(x[:,1:,:]-prev_val)/(t[:,1:,:]-prev),0).sum(1),pairs.sum(1))
        cv=divide(std,np.abs(mean))
        base=np.stack((mean,std,lo,hi,q[1],q[2]-q[0],skew,slope,rate,cv),axis=2).reshape(len(rows),-1)
        blocks=[base]
        if mode!="control":
            blocks.extend([n,np.where(n>0,23-last,np.nan),latest,earliest,latest-earliest])
        if mode in ("segments","combined"):
            for j in range(4):
                w=x[:,j*6:j*6+6]; m=np.isfinite(w); count=m.sum(1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore",RuntimeWarning)
                    avg=np.nanmean(w,axis=1); sd=np.nanstd(w,axis=1)
                last_w=np.max(np.where(m,np.arange(6)[None,:,None],-1),axis=1)
                val=np.take_along_axis(w,np.maximum(last_w,0)[:,None,:],axis=1)[:,0,:]
                blocks.extend([avg,np.where(count>=2,sd,np.nan),count,np.where(count>0,val,np.nan)])
        if mode in ("clinical","combined"):
            # Current row only, no invented SOFA values. SpO2/FiO2 is not PaO2/FiO2.
            hr,sbp,dbp,rrate,spo2,fio2,bun,cr=[x[:,-1,VITALS.index(v)] for v in ("HR","SBP","DBP","Resp","O2Sat","FiO2","BUN","Creatinine")]
            blocks.append(np.column_stack([divide(hr,np.where(sbp>0,sbp,np.nan)),sbp-dbp,
                 divide(spo2,np.where((fio2>0)&(fio2<=1),fio2,np.nan)),
                 divide(bun,np.where(cr>0,cr,np.nan)),np.isfinite(rrate),np.isfinite(sbp),
                 np.where(np.isfinite(rrate),rrate>=22,np.nan),np.where(np.isfinite(sbp),sbp<=100,np.nan)]))
        block=np.concatenate(blocks,axis=1)
        if block.shape[1]!=len(names): raise ValueError("Feature schema mismatch")
        block[~np.isfinite(block)]=np.nan
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
    protocol=dict(version=1,features=VITALS,device=args.device,trials=TRIALS,manifest=hashlib.sha256(MANIFEST.read_bytes()).hexdigest())
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
    med,scale=None,None
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
    # Freeze best control and top two distinct expanded modes using inner AP only.
    selected=[]
    control=next((r for r in ranked if r["config"]["mode"]=="control"),None)
    if control: selected.append(control)
    for r in ranked:
        if r["config"]["mode"]!="control" and all(q["config"]["mode"]!=r["config"]["mode"] for q in selected):
            selected.append(r)
        if len(selected)>=3: break
    atomic_json(OUT/"selected.json",selected)
    med,scale=None,None
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
            joblib.dump(dict(model=m,config=c,median=med,scale=scale,ordered_variables=VITALS,feature_names=feature_names(c["mode"]),window_rows=24),folder/"model.joblib")
            np.save(folder/"scores.npy",pred); atomic_json(done,dict(config=c,**metric(vy[vr],pred)))
            del m; gc.collect()
        if len(pred)!=len(vr) or not np.isfinite(pred).all(): raise ValueError("Invalid scores")
        finals.append(dict(id=r["id"],config=c,inner_ap=r["ap"],**metric(vy[vr],pred))); preds.append(pred)
    summary=dict(scope="strict 16 variables, raw 24-hour windows, inner Train selection, reused locked Validation",
      test_patients_evaluated=0,target_ap=.4,inner_trials=ranked,finalists=finals,
      equal_blend=metric(vy[vr],np.mean(preds,axis=0)),failures=[p.parent.name for p in (OUT/"trials").glob("*/error.txt")])
    atomic_json(OUT/"summary.json",summary)
    table=pd.DataFrame([dict(id=r["id"],inner_ap=r["inner_ap"],ap=r["ap"],auroc=r["auroc"],mode=r["config"]["mode"],**r["config"]["base"]) for r in finals])
    table.to_csv(OUT/"leaderboard.csv",index=False)
    print(table.to_string(index=False)); print("Equal blend",summary["equal_blend"],"Test HELD OUT")
if __name__=="__main__": main()
