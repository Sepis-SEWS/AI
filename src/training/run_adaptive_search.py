"""SEWS v2-11 automated exploratory model search. No Test data access.
Run from root: python src/training/run_adaptive_search.py
New artifacts only. Resume completed trials; failed trials are logged.
All APs use the same end-of-window rows (row >= 23), regardless of window size.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import average_precision_score, roc_auc_score
from train_lstm import FEATURES, patient_path
from preprocess import FFILL_MAX_AGE_HOURS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/adaptive_v2_11"
DATA = ROOT / "data/data-sepsis/training"
MANIFEST = ROOT / "data/splits/patient_splits.json"
BASELINE = ROOT / "reports/v2_ensemble_validation_scores.npz"
ALL = ("HR O2Sat Temp SBP MAP DBP Resp EtCO2 BaseExcess HCO3 FiO2 pH "
       "PaCO2 SaO2 AST BUN Alkalinephos Calcium Chloride Creatinine Bilirubin_direct "
       "Glucose Lactate Magnesium Phosphate Potassium Bilirubin_total TroponinI Hct Hgb "
       "PTT WBC Fibrinogen Platelets Age Gender Unit1 Unit2 HospAdmTime ICULOS").split()
SEED = 42

def atomic_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)

def config(**kw):
    return dict(dict(arch="lstm", features="all", fill="original", trend=False,
                     window=24, hidden=64, layers=1, dropout=.2, lr=.001,
                     weight=3, repeat=1, neg_ratio=0, focal=0, seed=SEED,
                     early=False, sampler="none"), **kw)

INITIAL = [
    config(features="base"),
    config(),
    config(features="dynamic"),
    config(fill="carry"),
    config(fill="none"),
    config(trend=True),
    config(window=6, early=True),
    config(window=12, early=True),
    config(window=48, early=True),
    config(arch="gru"),
    config(arch="tcn"),
    config(arch="attention"),
    config(hidden=128, layers=2),
    config(focal=2, weight=1),
    config(repeat=2),
    config(neg_ratio=10, repeat=2, weight=1),
    config(arch="xgb", trend=True, weight=3),
    config(arch="xgb", trend=True, weight=1, sampler="under"),
    config(arch="xgb", trend=True, weight=1, sampler="over"),
    config(arch="xgb", trend=True, weight=1, sampler="smote", features="dynamic"),
    config(arch="lightgbm", trend=True),
    config(arch="catboost", trend=True),
]

def key(c):
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:16]

def read_raw(ids):
    xs, ys, offsets = [], [], [0]
    for i, pid in enumerate(ids, 1):
        f = pd.read_csv(patient_path(DATA, pid), sep="|")
        f = f.sort_values("ICULOS", kind="stable")
        x = f[ALL].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
        y = pd.to_numeric(f["SepsisLabel"], errors="raise").to_numpy()
        if not len(y) or not np.isin(y, [0,1]).all() or not np.isfinite(x[:,-1]).all() or not np.all(np.diff(x[:,-1]) == 1):
            raise ValueError(f"Invalid timeline: {pid}")
        x[~np.isfinite(x)] = np.nan
        xs.append(x); ys.append(y.astype(np.int8)); offsets.append(offsets[-1]+len(y))
        if i % 5000 == 0:
            print(f"Raw patients {i}/{len(ids)}", flush=True)
    return np.concatenate(xs), np.concatenate(ys), np.array(offsets, dtype=np.int64)

def endpoints(offsets, early=False):
    starts = np.repeat(offsets[:-1], np.diff(offsets))
    rows = np.arange(offsets[-1], dtype=np.int64)
    if not early:
        rows = rows[rows-starts >= 23]
    return rows, starts[rows]

def transform(raw, offsets, names, med, scale, fill, trend):
    idx = [ALL.index(n) for n in names]
    n, d = len(raw), len(idx)
    extra = 3*d+2 if trend else 0
    out = np.empty((n, 4*d+extra), dtype=np.float32)
    limits = np.array([FFILL_MAX_AGE_HOURS.get(name) or 0 for name in names])
    if fill == "none":
        limits[:] = 0
    elif fill == "carry":
        limits[:] = 24
        for j, name in enumerate(names):
            if name in ("Age","Gender","Unit1","Unit2","HospAdmTime"):
                limits[j] = 1e9
    for a,b in zip(offsets[:-1], offsets[1:]):
        x = raw[a:b, idx]
        mask = np.isfinite(x)
        t = np.arange(b-a)[:,None]
        last = np.maximum.accumulate(np.where(mask,t,-1),axis=0)
        never = last < 0
        age = t-last
        latest = x[np.maximum(last,0), np.arange(d)]
        value = np.where(~never & (age<=limits), latest, med)
        z = (value-med)/scale
        # Same channel order and no future observations.
        out[a:b,:4*d] = np.stack((z,mask,np.where(never,0,np.minimum(age,48)/48),never),axis=-1).reshape(b-a,-1)
        if trend:
            past = np.maximum(t-6,0).ravel()
            cumulative = np.vstack((np.zeros((1,d)),np.cumsum(mask,axis=0)))
            count = cumulative[np.arange(1,b-a+1)]-cumulative[np.maximum(np.arange(1,b-a+1)-6,0)]
            last6 = np.where(np.isfinite(latest),latest,med)
            delta = (last6-last6[past])/scale
            observed = (~never & ~never[past]).astype(np.float32)
            delta = np.where(observed>0,delta,0)
            hr = raw[a:b,ALL.index("HR")]
            sbp = raw[a:b,ALL.index("SBP")]
            valid = np.isfinite(hr)&np.isfinite(sbp)&(sbp>0)
            shock = np.divide(hr,sbp,out=np.zeros_like(hr),where=valid)
            out[a:b,4*d:] = np.column_stack((delta,observed,count/6,shock,valid))
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite transformed features")
    return out

def batch_windows(matrix, rows, starts, width):
    ix = rows[:,None] - np.arange(width-1,-1,-1)[None,:]
    valid = ix>=starts[:,None]
    x = matrix[np.maximum(ix,starts[:,None])].copy()
    x[~valid] = 0
    return np.concatenate((x,valid[:,:,None].astype(np.float32)),axis=2)

class Net(nn.Module):
    def __init__(self, dim, c):
        super().__init__()
        self.arch = c["arch"]
        h = c["hidden"]
        if self.arch in ("lstm","gru"):
            cls = nn.LSTM if self.arch=="lstm" else nn.GRU
            self.encoder = cls(dim,h,c["layers"],batch_first=True,dropout=c["dropout"] if c["layers"]>1 else 0)
        elif self.arch=="tcn":
            self.encoder = nn.Sequential(nn.Conv1d(dim,h,3,padding=2),nn.ReLU(),
                nn.Conv1d(h,h,3,padding=2,dilation=2),nn.ReLU())
        else:
            self.project = nn.Linear(dim,h)
            self.position = nn.Parameter(torch.zeros(1,c["window"],h))
            self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(h,4,h*2,c["dropout"],batch_first=True),1)
        self.head = nn.Sequential(nn.Dropout(c["dropout"]),nn.Linear(h,1))

    def forward(self,x):
        if self.arch in ("lstm","gru"):
            z,_=self.encoder(x)
            z=z[:,-1]
        elif self.arch=="tcn":
            z=self.encoder(x.transpose(1,2))[:,:,x.shape[1]-1]
        else:
            z=self.encoder(self.project(x)+self.position)[:,-1]
        return self.head(z).squeeze(-1)

def sample(y,c,rng):
    pos=np.flatnonzero(y==1); neg=np.flatnonzero(y==0)
    if c["neg_ratio"]:
        neg=rng.choice(neg,min(len(neg),len(pos)*c["neg_ratio"]),replace=False)
    return rng.permutation(np.concatenate((neg,np.tile(pos,c["repeat"]))))

def metric(y,p):
    return dict(ap=float(average_precision_score(y,p)),auroc=float(roc_auc_score(y,p)))

def predict_net(net,matrix,rows,starts,width,device):
    net.eval(); p=[]
    with torch.no_grad():
        for a in range(0,len(rows),256):
            x=batch_windows(matrix,rows[a:a+256],starts[a:a+256],width)
            p.append(torch.sigmoid(net(torch.from_numpy(x).to(device))).cpu().numpy())
    return np.concatenate(p)

def neural(c,tr,va,ty,vy,to,vo,device,epochs,folder):
    trr,trs=endpoints(to,c["early"])
    vr,vs=endpoints(vo)
    y=ty[trr]
    random.seed(c["seed"]); np.random.seed(c["seed"]); torch.manual_seed(c["seed"])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(c["seed"])
    net=Net(tr.shape[1]+1,c).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=c["lr"],weight_decay=0.0001)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=.5,patience=1)
    history=[]; best=-1; bad=0; best_p=None
    for ep in range(1,epochs+1):
        net.train()
        order=sample(y,c,np.random.default_rng(c["seed"]+ep))
        loss_sum=0
        for a in range(0,len(order),256):
            ix=order[a:a+256]
            x=torch.from_numpy(batch_windows(tr,trr[ix],trs[ix],c["window"])).to(device)
            label=torch.from_numpy(y[ix].astype(np.float32)).to(device)
            logits=net(x)
            loss=nn.functional.binary_cross_entropy_with_logits(logits,label,reduction="none",pos_weight=torch.tensor(float(c["weight"]),device=device))
            if c["focal"]:
                p=torch.sigmoid(logits)
                pt=torch.where(label>0.5,p,1-p)
                loss=loss*(1-pt).pow(c["focal"])
            loss=loss.mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
            loss_sum+=float(loss.item())*len(ix)
        p=predict_net(net,va,vr,vs,c["window"],device)
        m=metric(vy[vr],p)
        history.append(dict(epoch=ep,loss=loss_sum/len(order),**m))
        print(f"Epoch {ep}: AP={m['ap']:.6f} AUROC={m['auroc']:.6f}",flush=True)
        sched.step(m["ap"])
        if m["ap"]>best:
            best=m["ap"]; bad=0; best_p=p
            torch.save({"state":net.state_dict(),"config":c,"input_dim":tr.shape[1]+1,"epoch":ep},folder/"model.pt")
        else:
            bad+=1
        atomic_json(folder/"history.json",history)
        if bad>=3: break
    return best_p

def tabular(matrix,rows,starts,width,continuous=False):
    blocks=[]
    # For SMOTE use only continuous robust-scaled value channels, no masks or ages.
    for a in range(0,len(rows),2048):
        x=batch_windows(matrix,rows[a:a+2048],starts[a:a+2048],width)[:,:,:-1]
        if continuous: x=x[:,:,::4]
        blocks.append(np.concatenate((x[:,-1],x.mean(1),x.std(1),x.min(1),x.max(1),x[:,-1]-x[:,0]),axis=1))
    return np.concatenate(blocks)

def trees(c,tr,va,ty,vy,to,vo,folder):
    import joblib
    rr,rs=endpoints(to); vr,vs=endpoints(vo)
    # SMOTE is a deliberately separate continuous-only representation.
    # Exclude trend block and static variables for this experiment.
    cont=c["sampler"]=="smote"
    if cont:
        d=34
        tr=tr[:,:4*d]; va=va[:,:4*d]
    x=tabular(tr,rr,rs,c["window"],cont)
    v=tabular(va,vr,vs,c["window"],cont)
    y=ty[rr]
    rng=np.random.default_rng(c["seed"])
    if c["sampler"]=="under":
        p=np.flatnonzero(y==1); n=np.flatnonzero(y==0)
        ix=rng.permutation(np.concatenate((p,rng.choice(n,min(len(n),10*len(p)),replace=False))))
        x,y=x[ix],y[ix]
    elif c["sampler"]=="over":
        p=np.flatnonzero(y==1); ix=rng.permutation(np.concatenate((np.arange(len(y)),p)))
        x,y=x[ix],y[ix]
    elif cont:
        from imblearn.over_sampling import SMOTE
        # Synthetic statistics are exploratory, not synthetic clinical patients.
        x,y=SMOTE(sampling_strategy=.1,random_state=c["seed"]).fit_resample(x,y)
    if c["arch"]=="xgb":
        from xgboost import XGBClassifier
        model=XGBClassifier(n_estimators=400,max_depth=5,learning_rate=.04,subsample=.8,
            colsample_bytree=.8,tree_method="hist",scale_pos_weight=c["weight"],n_jobs=4,random_state=c["seed"])
    elif c["arch"]=="lightgbm":
        from lightgbm import LGBMClassifier
        model=LGBMClassifier(n_estimators=400,learning_rate=.04,num_leaves=31,
            scale_pos_weight=c["weight"],n_jobs=4,random_state=c["seed"],verbosity=-1)
    else:
        from catboost import CatBoostClassifier
        model=CatBoostClassifier(iterations=400,depth=6,learning_rate=.04,
            scale_pos_weight=c["weight"],random_seed=c["seed"],thread_count=4,verbose=False,allow_writing_files=False)
    model.fit(x,y)
    p=model.predict_proba(v)[:,1]
    joblib.dump({"model":model,"config":c},folder/"model.joblib")
    return p

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--epochs",type=int,default=12)
    parser.add_argument("--adaptive",type=int,default=8)
    parser.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    args=parser.parse_args()
    if args.epochs<1 or args.adaptive<0: parser.error("Invalid search budget")
    groups=json.loads(MANIFEST.read_text(encoding="utf-8"))["split"]
    if tuple(len(groups[k]) for k in ("train","validation","test"))!=(25814,6454,8068) or len(set(groups["train"]+groups["validation"]+groups["test"]))!=40336:
        raise ValueError("Locked patient split changed")
    if args.dry_run:
        for i,c in enumerate(INITIAL,1): print(i,c)
        print(f"Initial {len(INITIAL)} trials + up to {args.adaptive} adaptive trials; Test HELD OUT.")
        return
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(epochs=args.epochs,version=1,initial=INITIAL,manifest_hash=hashlib.sha256(MANIFEST.read_bytes()).hexdigest())
    protocol_path=OUT/"protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:
        raise ValueError("Existing search protocol differs; use original epochs.")
    atomic_json(protocol_path,protocol)
    device="cuda" if args.device=="auto" and torch.cuda.is_available() else ("cpu" if args.device=="auto" else args.device)
    traw,ty,to=read_raw(groups["train"])
    vraw,vy,vo=read_raw(groups["validation"])
    vr,vs=endpoints(vo)
    if len(vr)!=108748 or len(endpoints(to)[0])!=430905:
        raise ValueError("Common cohort row counts changed")
    with np.load(BASELINE,allow_pickle=False) as a:
        ids=np.repeat(groups["validation"],np.maximum(np.diff(vo)-23,0))
        if not np.array_equal(ids,a["patient_ids"].astype(str)) or not np.array_equal(vy[vr],a["labels"]) or not np.array_equal(vraw[vr,-1],a["icul_os"]):
            raise ValueError("Baseline archive alignment mismatch")
        base={"old_lstm":a["lstm_v1_scores"].astype(float),"old_v2":a["lstm_v2_scores"].astype(float),
              "old_xgb":a["xgboost_scores"].astype(float)}
    med=np.nanmedian(traw,axis=0); scale=np.nanpercentile(traw,75,axis=0)-np.nanpercentile(traw,25,axis=0)
    med=np.where(np.isfinite(med),med,0); scale=np.where(np.isfinite(scale)&(scale>1e-12),scale,1)
    rows=[]; pred=dict(base); configs={}
    def run(c):
        k=key(c); configs[k]=c
        if k in pred: return
        folder=OUT/k; folder.mkdir(exist_ok=True)
        done=folder/"done.json"
        if done.exists():
            result=json.loads(done.read_text()); p=np.load(folder/"scores.npy",allow_pickle=False)
            if result["config"]!=c or len(p)!=len(vr) or not np.isfinite(p).all():
                raise ValueError(f"Invalid resume artifact {k}")
        else:
            # Preserve any incomplete attempt before retrying.
            existing=[p for p in folder.iterdir() if p.name!="attempts"]
            if existing:
                attempts=folder/"attempts"; attempts.mkdir(exist_ok=True)
                dest=attempts/str(len(list(attempts.iterdir()))+1); dest.mkdir()
                for p in existing: p.rename(dest/p.name)
            try:
                names=list(FEATURES) if c["features"]=="base" else (ALL[:34] if c["features"]=="dynamic" else ALL)
                ix=[ALL.index(n) for n in names]
                tr=transform(traw,to,names,med[ix],scale[ix],c["fill"],c["trend"])
                va=transform(vraw,vo,names,med[ix],scale[ix],c["fill"],c["trend"])
                atomic_json(folder/"preprocessing.json",dict(names=names,median=med[ix].tolist(),scale=scale[ix].tolist(),config=c))
                print(f"Trial {k}: {c}",flush=True)
                if c["arch"] in ("xgb","lightgbm","catboost"):
                    p=trees(c,tr,va,ty,vy,to,vo,folder)
                else:
                    p=neural(c,tr,va,ty,vy,to,vo,device,args.epochs,folder)
                if not np.isfinite(p).all() or len(p)!=len(vr): raise ValueError("Invalid predictions")
                result=dict(id=k,config=c,**metric(vy[vr],p))
                np.save(folder/"scores.npy",p)
                atomic_json(done,result)
            except Exception:
                (folder/"error.txt").write_text(traceback.format_exc(),encoding="utf-8")
                print(f"FAILED {k}: see error.txt",flush=True)
                return
            finally:
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        pred[k]=p; rows.append(result)
        atomic_json(OUT/"leaderboard.json",sorted(rows,key=lambda x:-x["ap"]))
        print(f"Completed {k}: AP={result['ap']:.6f}",flush=True)
    for c in INITIAL: run(c)
    # Adapt only after initial results: expand the two strongest neural candidates.
    best=sorted((r for r in rows if r["config"]["arch"] in ("lstm","gru","tcn","attention")),key=lambda r:-r["ap"])[:2]
    follow=[]
    for r in best:
        c=r["config"]
        for changes in (dict(lr=.0003),dict(weight=1),dict(seed=43),dict(hidden=128,dropout=.35),
                        dict(focal=2,weight=1),dict(repeat=2,neg_ratio=20)):
            new=dict(c,**changes)
            if key(new) not in configs and key(new) not in {key(x) for x in follow}: follow.append(new)
    for c in follow[:args.adaptive]: run(c)
    # Soft-voting only among strongest candidates; same Validation used for selection.
    pred["old_75_25"]=.75*base["old_v2"]+.25*base["old_xgb"]
    pred["old_90_10"]=.9*base["old_v2"]+.1*base["old_xgb"]
    ranked=sorted(pred,key=lambda k:-metric(vy[vr],pred[k])["ap"])[:5]
    ensembles=[]
    for i,a in enumerate(ranked):
        for b in ranked[i+1:]:
            for w in (.5,.75,.9):
                p=w*pred[a]+(1-w)*pred[b]
                ensembles.append(dict(a=a,b=b,weight_a=w,**metric(vy[vr],p)))
    ensembles.sort(key=lambda x:-x["ap"])
    summary=dict(scope="exploratory reused Validation, common rows >=24",test_patients_evaluated=0,
        target_ap=.4,baseline={k:metric(vy[vr],v) for k,v in base.items()},
        trials=sorted(rows,key=lambda x:-x["ap"]),ensembles=ensembles,
        failures=[p.parent.name for p in OUT.glob("*/error.txt")])
    atomic_json(OUT/"summary.json",summary)
    print(pd.DataFrame(summary["trials"]).drop(columns=["config"],errors="ignore").to_string(index=False))
    print("Top ensembles:",ensembles[:5])
    print("Saved:",OUT/"summary.json","; Test HELD OUT.")

if __name__=="__main__":
    main()
