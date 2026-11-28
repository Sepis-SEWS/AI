"""SEWS v2-18: strict 16-variable 24x64 LSTM, auxiliary forecasting/pretraining.
Train-only auxiliary targets; no future inputs, no extra label shift, no Test reads.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from run_adaptive_search import ROOT, MANIFEST, BASELINE, ALL, FEATURES, read_raw, endpoints, metric, atomic_json, transform
from run_tree_refine import fit_stats
from train_lstm import RiskLSTM

OUT=ROOT/"reports/lstm_v2_18"
IDX=[ALL.index(f) for f in FEATURES]
CONFIGS=[dict(mode=m,weight=w,seed=42,aux=a,pre=p) for m,w,a,p in (
    ("control",3,0.,0),("control",1,0.,0),
    ("forecast",3,.1,0),("forecast",3,.3,0),
    ("transition",3,.1,0),("transition",3,.3,0),
    ("joint",3,.1,0),("pretrain",3,0.,2),
    ("pretrain_joint",3,.1,2))]
class MultiLSTM(RiskLSTM):
    def __init__(self):
        super().__init__(hidden=64,layers=1,dropout=.2)
        self.forecast=nn.Linear(64,16)
        self.transition=nn.Linear(64,1)
    def outputs(self,x):
        h=self.lstm(x)[0][:,-1]
        return self.head(h).squeeze(-1),self.forecast(h),self.transition(h).squeeze(-1)

def key(c): return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:16]
def stats(raw):
    med,scale=fit_stats(raw[:,IDX].astype(np.float64))
    return med,scale
def windows(matrix,rows):
    return matrix[rows[:,None]-np.arange(23,-1,-1)[None,:]]
def targets(raw,labels,off,med,scale):
    # Auxiliary labels only: next-hour observed values and transition within 6h.
    # Exclude targets beyond each patient's available follow-up.
    n=len(raw)
    v=np.zeros((n,16),np.float32); mask=np.zeros_like(v)
    transition=np.zeros(n,np.float32); valid=np.zeros(n,np.float32)
    for a,b in zip(off[:-1],off[1:]):
        if b-a>1:
            x=raw[a+1:b][:,IDX]
            ok=np.isfinite(x)
            v[a:b-1]=np.where(ok,(x-med)/scale,0)
            mask[a:b-1]=ok
        if b-a>6:
            z=labels[a:b]
            transition[a:b-6]=np.maximum.reduce([z[j:j+b-a-6] for j in range(1,7)])
            valid[a:b-6]=(z[:-6]==0)
    return v,mask,transition,valid
def predict(net,matrix,rows,device,batch):
    net.eval(); p=[]
    with torch.inference_mode():
        for a in range(0,len(rows),batch):
            x=torch.from_numpy(windows(matrix,rows[a:a+batch])).to(device)
            p.append(torch.sigmoid(net(x)).cpu().numpy())
    return np.concatenate(p)
def train(c,matrix,labels,rr,train_rows,tune_rows,aux,device,batch,epochs,folder,med,scale,refit=False):
    seed=c["seed"]; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type=="cuda": torch.cuda.manual_seed_all(seed)
    net=MultiLSTM().to(device)
    opt=torch.optim.Adam(net.parameters(),lr=.001)
    bce=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(c["weight"]),device=device))
    val,mask,trans,valid=aux
    def pass_epoch(ep,pre):
        net.train()
        order=np.random.default_rng(seed+ep+(1000 if pre else 0)).permutation(train_rows)
        total=0.
        for a in range(0,len(order),batch):
            r=order[a:a+batch]
            x=torch.from_numpy(windows(matrix,r)).to(device)
            y=torch.from_numpy(labels[r].astype(np.float32)).to(device)
            logits,f,t=net.outputs(x)
            loss=bce(logits,y) if not pre else logits.sum()*0
            if pre or c["mode"] in ("forecast","joint","pretrain_joint"):
                target=torch.from_numpy(val[r]).to(device); m=torch.from_numpy(mask[r]).to(device)
                fl=(nn.functional.smooth_l1_loss(f,target,reduction="none")*m).sum()/m.sum().clamp_min(1)
                loss=loss+(1. if pre else c["aux"])*fl
            if not pre and c["mode"] in ("transition","joint","pretrain_joint"):
                target=torch.from_numpy(trans[r]).to(device); m=torch.from_numpy(valid[r]).to(device)
                tl=(nn.functional.binary_cross_entropy_with_logits(t,target,reduction="none",pos_weight=torch.tensor(3.,device=device))*m).sum()/m.sum().clamp_min(1)
                loss=loss+c["aux"]*tl
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.); opt.step()
            total+=loss.item()*len(r)
        return total/len(order)
    for ep in range(1,c["pre"]+1):
        print("Pretrain",ep,pass_epoch(ep,True),flush=True)
    if c["pre"]: opt=torch.optim.Adam(net.parameters(),lr=.001)
    history=[]; best=-1.; best_ep=0; bad=0
    for ep in range(1,epochs+1):
        loss=pass_epoch(ep,False)
        if refit:
            print("Refit epoch",ep,"loss",loss,flush=True)
            history.append(dict(epoch=ep,loss=loss))
            continue
        p=predict(net,matrix,tune_rows,device,batch)
        m=metric(labels[tune_rows],p)
        history.append(dict(epoch=ep,loss=loss,**m))
        print(c["mode"],"epoch",ep,m,flush=True)
        if m["ap"]>best:
            best=m["ap"]; best_ep=ep; bad=0
            best_metric=m
        else: bad+=1
        atomic_json(folder/"history.json",history)
        if bad>=3: break
    if refit:
        torch.save(dict(state=net.state_dict(),config=c,features=FEATURES,median=med.tolist(),scale=scale.tolist(),epoch=epochs,input_dim=64),folder/"model.pt")
        atomic_json(folder/"history.json",history)
        return net
    return dict(best_epoch=best_ep,**best_metric)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    p.add_argument("--epochs",type=int,default=12)
    p.add_argument("--batch",type=int,default=512)
    p.add_argument("--dry-run",action="store_true")
    a=p.parse_args()
    if a.epochs<1 or a.batch<1: p.error("Positive epochs and batch required")
    groups=json.loads(MANIFEST.read_text(encoding="utf-8"))["split"]
    if tuple(len(groups[k]) for k in ("train","validation","test"))!=(25814,6454,8068) or len(set(groups["train"]+groups["validation"]+groups["test"]))!=40336: raise ValueError("Locked split mismatch")
    if len(FEATURES)!=16: raise ValueError("16 features required")
    if a.dry_run:
        print("24x64 LSTM",len(CONFIGS),"settings; Test HELD OUT")
        for c in CONFIGS: print(c)
        return
    device=torch.device(a.device)
    if a.device=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    print("Device",device,torch.cuda.get_device_name(0) if device.type=="cuda" else "",flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(version=1,configs=CONFIGS,epochs=a.epochs,batch=a.batch,device=a.device,features=FEATURES,manifest=hashlib.sha256(MANIFEST.read_bytes()).hexdigest())
    path=OUT/"protocol.json"
    if path.exists() and json.loads(path.read_text())!=protocol: raise ValueError("Existing protocol differs")
    if not path.exists(): atomic_json(path,protocol)
    raw,y,off=read_raw(groups["train"])
    rr,_=endpoints(off)
    patient=np.repeat(np.arange(len(groups["train"])),np.diff(off))
    strata=[pid.split("/")[0]+"/"+str(int(y[x:z].max())) for pid,x,z in zip(groups["train"],off[:-1],off[1:])]
    fi,ti=train_test_split(np.arange(len(strata)),test_size=.2,random_state=42,stratify=strata)
    fit=rr[np.isin(patient[rr],fi)]; tune=rr[np.isin(patient[rr],ti)]
    if len(rr)!=430905: raise ValueError("Train endpoint mismatch")
    med,scale=stats(raw[np.isin(patient,fi)])
    matrix=transform(raw,off,FEATURES,med,scale,"original",False)
    if matrix.shape[1]!=64: raise ValueError("Input mismatch")
    aux=targets(raw,y,off,med,scale)
    rows=[]
    for c in CONFIGS:
        folder=OUT/"trials"/key(c); folder.mkdir(parents=True,exist_ok=True)
        done=folder/"done.json"
        if done.exists(): r=json.loads(done.read_text())
        else:
            r=dict(id=key(c),config=c,**train(c,matrix,y,rr,fit,tune,aux,device,a.batch,a.epochs,folder,med,scale))
            atomic_json(done,r)
        rows.append(r); print("Inner",r,flush=True)
    ranked=sorted(rows,key=lambda r:-r["ap"])
    control=next(r for r in ranked if r["config"]["mode"]=="control")
    improved=next(r for r in ranked if r["config"]["mode"]!="control")
    selected=[control,improved]
    atomic_json(OUT/"selected.json",selected)
    del matrix,aux; gc.collect()
    med,scale=stats(raw)
    matrix=transform(raw,off,FEATURES,med,scale,"original",False)
    aux=targets(raw,y,off,med,scale)
    del raw; gc.collect()
    vr,vy,vo=read_raw(groups["validation"]); rowsv,_=endpoints(vo)
    with np.load(BASELINE,allow_pickle=False) as b:
        ids=np.repeat(groups["validation"],np.maximum(np.diff(vo)-23,0))
        if not np.array_equal(ids,b["patient_ids"].astype(str)) or not np.array_equal(vy[rowsv],b["labels"]) or not np.array_equal(vr[rowsv,-1],b["icul_os"]): raise ValueError("Validation alignment mismatch")
    vm=transform(vr,vo,FEATURES,med,scale,"original",False); del vr; gc.collect()
    finals=[]
    for r in selected:
        c=r["config"]; folder=OUT/"final"/r["id"]; folder.mkdir(parents=True,exist_ok=True)
        done=folder/"done.json"
        if done.exists(): final=json.loads(done.read_text())
        else:
            net=train(c,matrix,y,rr,rr,None,aux,device,a.batch,r["best_epoch"],folder,med,scale,refit=True)
            pred=predict(net,vm,rowsv,device,a.batch)
            np.save(folder/"scores.npy",pred)
            final=dict(id=r["id"],config=c,inner_ap=r["ap"],epoch=r["best_epoch"],**metric(vy[rowsv],pred))
            atomic_json(done,final); del net; gc.collect()
        finals.append(final)
    summary=dict(scope="strict 16 variables, 24x64 LSTM, Train inner patient split, reused Validation",test_patients_evaluated=0,target_ap=.4,inner_trials=ranked,finalists=finals)
    atomic_json(OUT/"summary.json",summary)
    table=pd.DataFrame([dict(id=r["id"],inner_ap=r["inner_ap"],ap=r["ap"],auroc=r["auroc"],epoch=r["epoch"],**r["config"]) for r in finals])
    table.to_csv(OUT/"leaderboard.csv",index=False)
    print(table.to_string(index=False)); print("Test HELD OUT")
if __name__=="__main__": main()
