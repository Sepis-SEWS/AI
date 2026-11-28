"""SEWS v2-22: controlled variable-group ablation and second-order path features.
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

OUT=ROOT/"reports/tree_v2_22"
from train_lstm import FEATURES
from run_ordered_feature_search import construct as segment_construct, VITALS
from run_history_feature_search import construct as history_construct
from run_encoded_window_search import construct as strict_construct

MODES=("segments16","segments34","segments39","segments40","personal40",
       "personal40_signature","strict16","strict16_signature")
SLOW=cfg(arch="xgb",depth=3,weight=1,lr=.015,rounds=1000,reg=5)
TRIALS=[dict(mode=m,base=SLOW) for m in MODES]
def key(c): return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:16]

def segment_subset(x,names):
    # v2-15: four summaries of 282 channels + eight summaries of 45 segment channels.
    # All 15 original segment variables belong to the specified 16 variables.
    selected=set(names); d=len(ALL)
    channels=[4*i+j for i,v in enumerate(ALL) if v in selected for j in range(4)]
    channels += [4*d+k*d+i for k in range(3) for i,v in enumerate(ALL) if v in selected]
    channels += [7*d,7*d+1]  # shock index + validity
    cols=[k*(7*d+2)+i for k in range(4) for i in channels]
    cols.extend(range(4*(7*d+2),x.shape[1]))
    if x.shape[1]!=1488 or len(VITALS)!=15 or not set(VITALS).issubset(selected):
        raise ValueError("Unexpected v2-15 segment schema")
    return x[:,cols]

# Compact second-order path signatures, not an exact reproduction of the challenge winner.
# Use only observed-window values, carry forward inside the window and retain missing indicators.
PATH_VARS=("HR","MAP","SBP","Resp","O2Sat","Lactate","Creatinine","BUN")
PATH_IDX=np.array([ALL.index(v) for v in PATH_VARS])
PAIR_I,PAIR_J=np.triu_indices(len(PATH_VARS),1)
def signature(raw,offsets,med,scale):
    rr,_=endpoints(offsets)
    d=len(PATH_IDX); n=3*(2*d+len(PAIR_I))
    out=np.empty((len(rr),n),np.float32)
    t=np.arange(24)[None,:,None]
    for a in range(0,len(rr),1024):
        rows=rr[a:a+1024]
        x=raw[rows[:,None]-np.arange(23,-1,-1)[None,:]][:,:,PATH_IDX]
        mask=np.isfinite(x)
        last=np.maximum.accumulate(np.where(mask,t,-1),axis=1)
        val=np.take_along_axis(x,np.maximum(last,0),axis=1)
        z=np.clip((np.where(last>=0,val,med[PATH_IDX])-med[PATH_IDX])/scale[PATH_IDX],-10,10)
        blocks=[]
        for w in (6,12,24):
            q=z[:,-w:,:]
            increments=np.diff(q,axis=1)
            prefix=q[:,:-1]-q[:,0:1]
            # Antisymmetric second-order path area for each variable pair.
            area=.5*np.sum(prefix[:,:,PAIR_I]*increments[:,:,PAIR_J]
                           -prefix[:,:,PAIR_J]*increments[:,:,PAIR_I],axis=1)
            blocks.extend((q[:,-1]-q[:,0],np.abs(increments).sum(1),area))
        out[a:a+len(rows)]=np.concatenate(blocks,axis=1)
    if not np.isfinite(out).all(): raise ValueError("Invalid signature")
    return out

def construct(raw,offsets,med,scale,mode):
    if mode.startswith("segments"):
        x=segment_construct(raw,offsets,med,scale,"segments")
        if mode=="segments40": return x
        names={"segments16":FEATURES,"segments34":ALL[:34],
               "segments39":[v for v in ALL if v!="ICULOS"]}[mode]
        return segment_subset(x,names)
    if mode.startswith("personal40"):
        x=history_construct(raw,offsets,med,scale,"personal")
    elif mode.startswith("strict16"):
        x=strict_construct(raw,offsets,med,scale,"combined")
    else:
        raise ValueError(mode)
    if mode.endswith("_signature"):
        z=signature(raw,offsets,med,scale)
        x=np.concatenate((x,z),axis=1)
    return x

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
    protocol=dict(version=1,design="fixed XGB variable ablation and signature",device=args.device,trials=TRIALS,manifest=hashlib.sha256(MANIFEST.read_bytes()).hexdigest())
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
    # All predefined candidates are reported for the controlled comparison.
    # No Validation-based selection or threshold changes.
    selected=ranked
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
            joblib.dump(dict(model=m,config=c,median=med,scale=scale,representation=c["mode"],raw_columns=ALL,window_rows=24),folder/"model.joblib")
            np.save(folder/"scores.npy",pred); atomic_json(done,dict(config=c,**metric(vy[vr],pred)))
            del m; gc.collect()
        if len(pred)!=len(vr) or not np.isfinite(pred).all(): raise ValueError("Invalid scores")
        finals.append(dict(id=r["id"],config=c,inner_ap=r["ap"],**metric(vy[vr],pred))); preds.append(pred)
    summary=dict(scope="fixed XGB settings; expanded causal patient-history and strict 24h branches; reused Validation; exploratory ablation",
      test_patients_evaluated=0,target_ap=.4,inner_trials=ranked,finalists=finals,
      failures=[p.parent.name for p in (OUT/"trials").glob("*/error.txt")])
    atomic_json(OUT/"summary.json",summary)
    table=pd.DataFrame([dict(id=r["id"],inner_ap=r["inner_ap"],ap=r["ap"],auroc=r["auroc"],mode=r["config"]["mode"],**r["config"]["base"]) for r in finals])
    table.to_csv(OUT/"leaderboard.csv",index=False)
    print(table.to_string(index=False)); print("Test HELD OUT")
if __name__=="__main__": main()
