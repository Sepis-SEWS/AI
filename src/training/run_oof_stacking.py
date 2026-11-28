"""SEWS v2-20: fixed 16-variable LSTM + XGBoost, patient 5-fold OOF stacking.
Fold-specific preprocessing; no Test reads or Validation-based selection.
OOF meta fitting is development-only; locked Validation evaluates frozen stack.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import random
import numpy as np
import pandas as pd
import torch
import joblib
from torch import nn
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from run_adaptive_search import ROOT, MANIFEST, BASELINE, FEATURES, read_raw, endpoints, metric, atomic_json, transform
from run_lstm_multitask import stats, windows, predict
from run_strict_xgb_search import construct, model, feature_names, SLOW
from train_lstm import RiskLSTM

OUT=ROOT/"reports/stacking_v2_20"
XCFG=dict(mode="segments",base=dict(SLOW,weight=3))
# Fixed before running; v2-18 inner control selected weight 1 and epoch 2.
# XGB segments selected by v2-19 inner AP, not Validation AP.
LCFG=dict(weight=1.,epochs=2,seed=42,hidden=64)
def fit_lstm(matrix,labels,rows,med,scale,folder,device,batch,seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type=="cuda": torch.cuda.manual_seed_all(seed)
    net=RiskLSTM(hidden=64,layers=1,dropout=.2).to(device)
    opt=torch.optim.Adam(net.parameters(),lr=.001)
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(LCFG["weight"],device=device))
    history=[]
    for ep in range(1,LCFG["epochs"]+1):
        net.train(); total=0.
        order=np.random.default_rng(seed+ep).permutation(rows)
        for a in range(0,len(order),batch):
            r=order[a:a+batch]
            x=torch.from_numpy(windows(matrix,r)).to(device)
            y=torch.from_numpy(labels[r].astype(np.float32)).to(device)
            loss=loss_fn(net(x),y)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.); opt.step()
            total+=loss.item()*len(r)
        history.append(dict(epoch=ep,loss=total/len(rows)))
        print("LSTM epoch",ep,history[-1],flush=True)
    torch.save(dict(state=net.state_dict(),config=LCFG,seed=seed,features=FEATURES,
                    median=med.tolist(),scale=scale.tolist(),input_dim=64),folder/"lstm.pt")
    atomic_json(folder/"lstm_history.json",history)
    return net

def logit(p):
    p=np.clip(p,1e-6,1-1e-6)
    return np.log(p/(1-p))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    p.add_argument("--batch",type=int,default=512)
    p.add_argument("--dry-run",action="store_true")
    a=p.parse_args()
    if a.batch<1: p.error("Positive batch required")
    groups=json.loads(MANIFEST.read_text(encoding="utf-8"))["split"]
    if tuple(len(groups[k]) for k in ("train","validation","test"))!=(25814,6454,8068) or len(set(groups["train"]+groups["validation"]+groups["test"]))!=40336: raise ValueError("Locked split mismatch")
    if a.dry_run:
        print("5-fold OOF",XCFG,LCFG,"Test HELD OUT"); return
    device=torch.device(a.device)
    if device.type=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(version=1,xgb=XCFG,lstm=LCFG,folds=5,seed=42,device=a.device,batch=a.batch,
                  meta="logistic regression C=1 on clipped logits",manifest=hashlib.sha256(MANIFEST.read_bytes()).hexdigest())
    path=OUT/"protocol.json"
    if path.exists() and json.loads(path.read_text())!=protocol: raise ValueError("Existing protocol differs")
    if not path.exists(): atomic_json(path,protocol)
    raw,labels,off=read_raw(groups["train"])
    rr,_=endpoints(off); y=labels[rr]
    patient=np.repeat(np.arange(len(groups["train"])),np.diff(off))
    epatient=patient[rr]
    strata=[pid.split("/")[0]+"/"+str(int(labels[s:e].max())) for pid,s,e in zip(groups["train"],off[:-1],off[1:])]
    if len(rr)!=430905: raise ValueError("Train endpoint mismatch")
    # Raw-window tree features are parameter-free and patient-local, so safe to reuse across folds.
    x=construct(raw,off,None,None,XCFG["mode"])
    oof=np.full((len(rr),2),np.nan,dtype=np.float64)
    fold_info=[]
    for fold,(fi,ti) in enumerate(StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(np.arange(len(strata)),strata),1):
        fit=np.flatnonzero(np.isin(epatient,fi)); tune=np.flatnonzero(np.isin(epatient,ti))
        if set(fi)&set(ti) or np.isfinite(oof[tune]).any(): raise ValueError("Fold overlap")
        folder=OUT/f"fold{fold}"; folder.mkdir(parents=True,exist_ok=True)
        done=folder/"done.json"
        if done.exists():
            info=json.loads(done.read_text())
            z=np.load(folder/"oof_scores.npy",allow_pickle=False)
            saved=np.load(folder/"oof_indices.npy",allow_pickle=False)
            if not np.array_equal(saved,tune) or z.shape!=(len(tune),2): raise ValueError("Resume fold mismatch")
        else:
            med,scale=stats(raw[np.isin(patient,fi)])
            matrix=transform(raw,off,FEATURES,med,scale,"original",False)
            net=fit_lstm(matrix,labels,rr[fit],med,scale,folder,device,a.batch,42+fold)
            pl=predict(net,matrix,rr[tune],device,a.batch)
            del net,matrix; gc.collect()
            if device.type=="cuda": torch.cuda.empty_cache()
            m=model(XCFG,a.device)
            m.fit(x[fit],y[fit])
            px=m.predict_proba(x[tune])[:,1]
            joblib.dump(dict(model=m,config=XCFG,feature_names=feature_names(XCFG["mode"])),folder/"xgb.joblib")
            del m; gc.collect()
            z=np.column_stack((pl,px))
            info=dict(fold=fold,fit_patients=len(fi),holdout_patients=len(ti),samples=len(tune),
                      lstm=metric(y[tune],pl),xgb=metric(y[tune],px))
            np.save(folder/"oof_scores.npy",z)
            np.save(folder/"oof_indices.npy",tune)
            atomic_json(done,info)
        if not np.isfinite(z).all(): raise ValueError("Nonfinite OOF scores")
        oof[tune]=z; fold_info.append(info); print("Fold",info,flush=True)
    if not np.isfinite(oof).all(): raise ValueError("Missing OOF scores")
    meta=LogisticRegression(C=1.,max_iter=1000)
    meta.fit(logit(oof),y)
    # Fit probability calibration for individual models using OOF only.
    calibrators=[LogisticRegression(C=1.,max_iter=1000).fit(logit(oof[:,j:j+1]),y) for j in range(2)]
    joblib.dump(dict(meta=meta,calibrators=calibrators,columns=["lstm","xgb"],transform="clip_logit_1e-6"),OUT/"meta.joblib")
    def scores(z):
        l=calibrators[0].predict_proba(logit(z[:,0:1]))[:,1]
        x=calibrators[1].predict_proba(logit(z[:,1:2]))[:,1]
        return {"lstm":z[:,0],"xgb":z[:,1],"lstm_calibrated":l,"xgb_calibrated":x,
            "soft_50_50":(l+x)/2,"soft_75_25":.75*l+.25*x,
            "stack":meta.predict_proba(logit(z))[:,1]}
    # Meta model OOF-fit metrics are in-sample for the meta learner, not unbiased CV.
    def metrics(yy,pp): return dict(**metric(yy,pp),brier=float(brier_score_loss(yy,pp)))
    oof_base={name:metric(y,oof[:,j]) for j,name in enumerate(("lstm","xgb"))}
    atomic_json(OUT/"fold_metrics.json",fold_info)
    med,scale=stats(raw)
    matrix=transform(raw,off,FEATURES,med,scale,"original",False)
    folder=OUT/"full"; folder.mkdir(exist_ok=True)
    net=fit_lstm(matrix,labels,rr,med,scale,folder,device,a.batch,42)
    del raw,matrix; gc.collect()
    vr,vy,vo=read_raw(groups["validation"]); rv,_=endpoints(vo)
    with np.load(BASELINE,allow_pickle=False) as b:
        ids=np.repeat(groups["validation"],np.maximum(np.diff(vo)-23,0))
        if not np.array_equal(ids,b["patient_ids"].astype(str)) or not np.array_equal(vy[rv],b["labels"]) or not np.array_equal(vr[rv,-1],b["icul_os"]): raise ValueError("Validation alignment mismatch")
    vm=transform(vr,vo,FEATURES,med,scale,"original",False)
    pl=predict(net,vm,rv,device,a.batch)
    del net,vm; gc.collect()
    if device.type=="cuda": torch.cuda.empty_cache()
    m=model(XCFG,a.device); m.fit(x,y); del x; gc.collect()
    xv=construct(vr,vo,None,None,XCFG["mode"]); del vr; gc.collect()
    px=m.predict_proba(xv)[:,1]
    joblib.dump(dict(model=m,config=XCFG,feature_names=feature_names(XCFG["mode"])),folder/"xgb.joblib")
    result=scores(np.column_stack((pl,px)))
    validation={k:metrics(vy[rv],v) for k,v in result.items()}
    np.savez_compressed(OUT/"validation_scores.npz",**result)
    summary=dict(scope="fixed 16 variables; 5-fold patient OOF base predictions; full Train refit; reused Validation",test_patients_evaluated=0,
                 folds=fold_info,oof_base=oof_base,meta_coef=meta.coef_.tolist(),meta_intercept=meta.intercept_.tolist(),validation=validation,
                 limitations=["OOF base scores are out-of-fold; meta learner fitted on all OOF scores","Configuration selected in earlier development experiments","Full-refit base scores may differ from OOF score distributions","Validation repeatedly used in earlier experiments"])
    atomic_json(OUT/"summary.json",summary)
    table=pd.DataFrame([dict(model=k,**v) for k,v in validation.items()])
    table.to_csv(OUT/"leaderboard.csv",index=False)
    print(table.to_string(index=False)); print("Test HELD OUT")
if __name__=="__main__": main()
