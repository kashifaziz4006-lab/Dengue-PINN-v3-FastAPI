"""
=====================================================================
DENGUE PINN v3 — FastAPI Backend (13-Marker Model)
Thesis: Pakistan 2023, N=268

PROBLEM FIXED:
  Original dengue_pinn.py trained on only 8 features.
  This version trains on ALL 13 WHO-required markers:
    age, sex, ALT, AST, ALP, Urea, Creat, tday,
    Platelet, Hematocrit, WBC, Albumin, Bilirubin

SETUP:
  pip install fastapi uvicorn numpy torch scikit-learn joblib

RUN:
  python dengue_api.py          # trains model on first run, then serves
  uvicorn dengue_api:app --reload   # after model is cached

API DOCS: http://localhost:8000/docs
=====================================================================
"""

from __future__ import annotations
import os, time, warnings
from typing import Literal, Optional
import numpy as np
warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Dengue PINN API v3",
    description=(
        "**13-Marker PINN** for dengue severity (DF/DHF/DSS).\n\n"
        "Pakistan 2023 cohort (N=268). WHO 2009 hard-gate criteria applied post-inference."
    ),
    version="3.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Constants ────────────────────────────────────────────────────────
COHORT_MEANS = {
    "DF":  {"ALT":77.68, "AST":90.4,  "ALP":218.6,  "Urea":30.8,"Creat":1.1,
            "Plt":120.0, "HCT":42.0,  "WBC":4.2,    "Alb":4.1,  "Bil":0.7,  "n":168},
    "DHF": {"ALT":110.17,"AST":155.5, "ALP":281.9,  "Urea":34.1,"Creat":1.2,
            "Plt":68.0,  "HCT":48.0,  "WBC":3.1,    "Alb":3.4,  "Bil":1.4,  "n":83},
    "DSS": {"ALT":232.0, "AST":314.6, "ALP":682.63, "Urea":31.4,"Creat":1.0,
            "Plt":28.0,  "HCT":54.0,  "WBC":2.2,    "Alb":2.9,  "Bil":2.8,  "n":17},
    "CTRL":{"ALT":22.81, "AST":21.2,  "ALP":193.4,  "Urea":27.6,"Creat":0.9,
            "Plt":220.0, "HCT":42.0,  "WBC":6.5,    "Alb":4.5,  "Bil":0.5},
}
WHO = {
    "PLT_DHF":100.0,"PLT_DSS":50.0,"HCT_RISE":0.20,
    "HCT_NORMAL_M":44.0,"HCT_NORMAL_F":40.0,
    "ALB_DSS":3.5,"BIL_DHF":1.5,"BIL_DSS":2.0,"WBC_LEUKO":5.0,
}
NORMAL_RANGES = {
    "ALT":(9,41),"AST":(10,35),"ALP":(70,306),"Urea":(20,50),"Creat":(0.7,1.1),
    "Plt":(150,400),"HCT":(37,52),"WBC":(4,11),"Alb":(3.5,5.2),"Bil":(0.1,1.2),
}
LABELS    = ["DF","DHF","DSS"]
SEV_NAMES = {"DF":"Dengue Fever","DHF":"Dengue Hemorrhagic Fever","DSS":"Dengue Shock Syndrome"}
MODEL_PATH  = "dengue_pinn13_best.pth"
SCALER_PATH = "dengue_scaler13.pkl"

_model = _scaler = None

# ── Model architecture ───────────────────────────────────────────────
def _build_model():
    import torch, torch.nn as nn
    class ODE13(nn.Module):
        def __init__(self):
            super().__init__()
            for k,v in [("b",.30),("d",.40),("r",.10),("a1",.82),
                        ("a2",.61),("a3",.29),("g",.20),("kp",.12),("ka",.08)]:
                setattr(self,f"_{k}",nn.Parameter(torch.tensor(float(v))))
        def p(self,n): return torch.abs(getattr(self,f"_{n}"))
        def forward(self,s):
            V,T = s[:,0:1],s[:,1:2]
            AST,ALT,ALP,Plt,Alb = s[:,2:3],s[:,3:4],s[:,4:5],s[:,5:6],s[:,6:7]
            return torch.cat([
                self.p("b")*T*V - self.p("d")*V,
               -self.p("b")*T*V*0.001 + self.p("r")*T,
                self.p("a1")*V - self.p("g")*AST,
                self.p("a2")*V - self.p("g")*ALT,
                self.p("a3")*V*0.55 - self.p("g")*ALP,
               -self.p("kp")*V*Plt/100,
               -self.p("ka")*V,
            ], dim=1)

    class PINN13(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(13,128),nn.Tanh(),
                nn.Linear(128,128),nn.Tanh(),
                nn.Linear(128,64),nn.Tanh(),
                nn.Linear(64,32),nn.Tanh(),
            )
            self.sh_head  = nn.Sequential(nn.Linear(32,16),nn.Tanh(),nn.Linear(16,7),nn.Sigmoid())
            self.sev_head = nn.Sequential(nn.Linear(32,16),nn.Tanh(),nn.Linear(16,3))
            self.ode = ODE13()
        def forward(self,x):
            h = self.trunk(x)
            return self.sh_head(h)*800.0, self.sev_head(h)
    return PINN13()

# ── Training data generator ──────────────────────────────────────────
def _generate_data(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    def nc(m,s,lo,hi,n): return np.clip(rng.normal(m,s,n),lo,hi).astype(np.float32)
    rows, labels = [], []
    counts = {"DF":int(n*.626),"DHF":int(n*.31)}
    counts["DSS"] = n - counts["DF"] - counts["DHF"]
    for ci,(sev,cnt) in enumerate(counts.items()):
        c = COHORT_MEANS[sev]; sp=.22
        age  = nc(40+ci*2,14,10,99,cnt)
        sex  = (rng.random(cnt)<.35).astype(np.float32)
        tday = nc(4+ci*1.2,1.5,1,12,cnt)
        alt  = np.maximum(1, rng.normal(c["ALT"], c["ALT"]*.20, cnt)).astype(np.float32)
        ast  = np.maximum(1, rng.normal(c["AST"], c["AST"]*sp,  cnt)).astype(np.float32)
        alp  = np.maximum(50,rng.normal(c["ALP"], c["ALP"]*.28, cnt)).astype(np.float32)
        urea = np.maximum(5, rng.normal(c["Urea"],c["Urea"]*.20,cnt)).astype(np.float32)
        creat= np.maximum(.3,rng.normal(c["Creat"],c["Creat"]*.18,cnt)).astype(np.float32)
        plt  = np.maximum(5, rng.normal(c["Plt"], c["Plt"]*.30, cnt)).astype(np.float32)
        hbase= np.where(sex==1,WHO["HCT_NORMAL_F"],WHO["HCT_NORMAL_M"]).astype(np.float32)
        hrise= [0.0, 0.22, 0.27][ci]
        hct  = np.clip(hbase*(1+hrise)+rng.normal(0,2,cnt),25,65).astype(np.float32)
        wbc  = np.maximum(.5, rng.normal(c["WBC"], c["WBC"]*.25, cnt)).astype(np.float32)
        alb  = np.clip(rng.normal(c["Alb"],.4,cnt),1.5,5.5).astype(np.float32)
        bil  = np.maximum(.1, rng.normal(c["Bil"], c["Bil"]*.30, cnt)).astype(np.float32)
        X = np.stack([age,sex,alt,ast,alp,urea,creat,tday,plt,hct,wbc,alb,bil],axis=1)
        rows.append(X); labels.extend([ci]*cnt)
    return np.vstack(rows), np.array(labels,dtype=np.int64)

# ── Train ────────────────────────────────────────────────────────────
def _train(epochs=10000, bs=64, lr=1e-3):
    import torch, torch.nn as nn, joblib
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report

    print("="*62)
    print("DENGUE PINN v3 — TRAINING  (13 markers, WHO-aware)")
    print("="*62)

    X_raw,y = _generate_data(2000)
    Xtr_raw,Xte_raw,ytr,yte = train_test_split(X_raw,y,test_size=.2,stratify=y,random_state=42)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr_raw).astype(np.float32)
    Xte = scaler.transform(Xte_raw).astype(np.float32)
    joblib.dump(scaler, SCALER_PATH)
    print(f"  Train:{len(Xtr)}  Test:{len(Xte)}")

    model  = _build_model()
    cw     = torch.tensor([1.,2.,9.])
    ce     = nn.CrossEntropyLoss(weight=cw)
    opt    = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    Xt,yt  = torch.tensor(Xtr), torch.tensor(ytr,dtype=torch.long)
    Xte_t  = torch.tensor(Xte)
    N      = len(Xt); best=0.0

    def phys_loss(m, xb_t):
        t_col = xb_t[:,7:8].clone().detach().requires_grad_(True)
        x_inp = torch.cat([xb_t[:,:7], t_col, xb_t[:,8:]],dim=1)
        states,_ = m(x_inp)
        ode_res  = m.ode(states[:,:7]/800.0)
        loss = torch.tensor(0.)
        for i in range(7):
            g = torch.autograd.grad(states[:,i].sum(),t_col,create_graph=True,retain_graph=True)[0]
            loss = loss + torch.mean((g - ode_res[:,i:i+1])**2)
        return loss/7.

    for ep in range(1,epochs+1):
        model.train(); opt.zero_grad()
        idx = torch.randperm(N)[:bs]
        xb,yb = Xt[idx],yt[idx]
        states,logits = model(xb)
        sv  = ce(logits,yb)
        dl  = torch.mean(((states[:,2:7]/800.)-xb[:,2:7])**2)
        pl  = phys_loss(model,Xt[idx]) if ep>800 else torch.tensor(0.)
        loss= sv*1.5 + dl*0.4 + pl*0.08
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step(); sched.step()

        if ep%2000==0 or ep==1:
            model.eval()
            with torch.no_grad():
                _,tl=model(Xte_t); preds=tl.argmax(1).numpy()
                acc=(preds==yte).mean()
            if acc>best: best=acc; torch.save(model.state_dict(),MODEL_PATH)
            print(f"  Ep{ep:6,d} | loss={loss.item():.4f} | sv={sv.item():.4f} | acc={acc*100:.1f}% | best={best*100:.1f}%")
            model.train()

    model.load_state_dict(torch.load(MODEL_PATH,map_location="cpu"))
    model.eval()
    with torch.no_grad():
        _,tl=model(Xte_t); preds=tl.argmax(1).numpy()
    print(f"\n  FINAL ACCURACY: {(preds==yte).mean()*100:.1f}%")
    print(classification_report(yte,preds,target_names=LABELS))
    print(f"  Saved → {MODEL_PATH}\n{'='*62}")
    return model,scaler

def _load_or_train():
    import torch, joblib
    global _model,_scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        print(f"[PINN] Loading cached model…")
        _scaler = joblib.load(SCALER_PATH)
        m = _build_model()
        m.load_state_dict(torch.load(MODEL_PATH,map_location="cpu"))
        m.eval(); _model=m
    else:
        m,sc = _train()
        m.eval(); _model=m; _scaler=sc
    print("[PINN] Ready ✓")

# ── WHO hard gate ────────────────────────────────────────────────────
def who_gate(probs,plt,hct,sex,alb,bil,alp,wbc):
    pDF,pDHF,pDSS = probs.copy().astype(float)
    hn = WHO["HCT_NORMAL_F"] if sex=="female" else WHO["HCT_NORMAL_M"]
    rise = (hct-hn)/hn
    dhf  = plt<WHO["PLT_DHF"] and rise>=WHO["HCT_RISE"]
    # DSS: classic WHO criteria OR severe organ involvement (hepatic failure + severe hypoalbuminaemia)
    dss  = (plt<WHO["PLT_DSS"] and alb<WHO["ALB_DSS"]) or \
           (alb<2.8 and bil>WHO["BIL_DSS"])
    if not dhf and not dss:
        ov=pDHF*.75+pDSS*.80; pDF+=ov*.85; pDHF*=.25; pDSS*=.20
    elif not dhf and dss:
        # severe organ DSS without platelet criterion — still suppress DHF but trust DSS
        pDHF*=.30; pDF*=.40
    if not dss and pDSS>.35:
        pDF+=pDSS*.50; pDHF+=pDSS*.30; pDSS*=.20
    if alp>450 and (dhf or dss):
        b=min(.20,(alp-450)/1000*.3); pDSS+=b; pDF-=b*.6; pDHF-=b*.4
    if bil>WHO["BIL_DSS"] and (dhf or dss):
        pDSS+=.08; pDF-=.06; pDHF-=.02
    elif bil>WHO["BIL_DHF"]:
        pDHF+=.05; pDF-=.05
    if alb<WHO["ALB_DSS"] and (dhf or dss):
        pDSS+=.10; pDF-=.08; pDHF-=.02
    if wbc<2.5: pDSS+=.04; pDF-=.04
    a=np.array([max(0,pDF),max(0,pDHF),max(0,pDSS)])
    return a/a.sum()

# ── Core prediction ──────────────────────────────────────────────────
def run_prediction(p: "PatientInput") -> "PredictionResult":
    import torch
    t0=time.perf_counter()
    feat=np.array([[p.age,1. if p.sex=="female" else 0.,
                    p.alt,p.ast,p.alp,p.urea,p.creat,float(p.tday),
                    p.plt,p.hct,p.wbc,p.alb,p.bil]],dtype=np.float32)
    sc=_scaler.transform(feat).astype(np.float32)
    import torch
    with torch.no_grad():
        _,logits=_model(torch.tensor(sc))
        raw=torch.softmax(logits,dim=1)[0].numpy()
    adj=who_gate(raw,p.plt,p.hct,p.sex,p.alb,p.bil,p.alp,p.wbc)
    idx=int(adj.argmax()); sev=LABELS[idx]
    hn=WHO["HCT_NORMAL_F"] if p.sex=="female" else WHO["HCT_NORMAL_M"]
    rise=(p.hct-hn)/hn
    who_met=p.plt<WHO["PLT_DHF"] and rise>=WHO["HCT_RISE"]
    flags=[
        MarkerFlag(marker="Platelet",  value=p.plt,unit="×10³/μL",flagged=p.plt<WHO["PLT_DHF"],
            message="⚠ <100k → WHO DHF criterion" if p.plt<WHO["PLT_DHF"] else "Normal"),
        MarkerFlag(marker="Hematocrit",value=p.hct,unit="%",flagged=rise>=WHO["HCT_RISE"],
            message=f"⚠ +{rise*100:.0f}% rise → plasma leak" if rise>=WHO["HCT_RISE"] else "Normal"),
        MarkerFlag(marker="WBC",       value=p.wbc,unit="×10³/μL",flagged=p.wbc<WHO["WBC_LEUKO"],
            message="Leukopenia" if p.wbc<WHO["WBC_LEUKO"] else "Normal"),
        MarkerFlag(marker="Albumin",   value=p.alb,unit="g/dL",   flagged=p.alb<WHO["ALB_DSS"],
            message="⚠ <3.5 → plasma leak" if p.alb<WHO["ALB_DSS"] else "Normal"),
        MarkerFlag(marker="Bilirubin", value=p.bil,unit="mg/dL",  flagged=p.bil>WHO["BIL_DHF"],
            message=f"⚠ >{WHO['BIL_DHF']} → hepatic" if p.bil>WHO["BIL_DHF"] else "Normal"),
        MarkerFlag(marker="AST",       value=p.ast,unit="IU/L",   flagged=p.ast>35,
            message=f"{p.ast/35:.1f}× normal" if p.ast>35 else "Normal"),
        MarkerFlag(marker="ALT",       value=p.alt,unit="IU/L",   flagged=p.alt>41,
            message=f"{p.alt/41:.1f}× normal" if p.alt>41 else "Normal"),
        MarkerFlag(marker="ALP",       value=p.alp,unit="IU/L",   flagged=p.alp>450,
            message="⚠ DSS-level" if p.alp>450 else "Normal"),
    ]
    return PredictionResult(
        severity=sev, severity_name=SEV_NAMES[sev],
        confidence=round(float(adj[idx]),4),
        raw_probs={k:round(float(v),4) for k,v in zip(LABELS,raw)},
        probs={k:round(float(v),4) for k,v in zip(LABELS,adj)},
        hepatic_index=round(.45*(p.ast/35)+.35*(p.alt/41)+.20*(p.alp/306),3),
        renal_index=round(.55*(p.creat/1.1)+.45*(p.urea/50),3),
        hema_index=round(1-p.plt/400+(p.hct-42)/100,3),
        who_dhf_met=who_met, hct_rise_pct=round(rise*100,1),
        marker_flags=flags, model_version="3.0.0",
        processing_ms=round((time.perf_counter()-t0)*1000,2),
    )

# ── Schemas ──────────────────────────────────────────────────────────
class PatientInput(BaseModel):
    age:  int   = Field(...,ge=5,  le=99)
    sex:  Literal["male","female"]
    tday: int   = Field(...,ge=1,  le=14)
    alt:  float = Field(...,ge=1,  le=2000, description="ALT IU/L (normal 9–41)")
    ast:  float = Field(...,ge=1,  le=3000, description="AST IU/L (normal 10–35) ★ most sensitive")
    alp:  float = Field(...,ge=1,  le=5000, description="ALP IU/L (normal 70–306) ★ sig. DSS only")
    bil:  float = Field(...,ge=0.1,le=30,   description="Total Bilirubin mg/dL (>1.5 = DHF/DSS)")
    alb:  float = Field(...,ge=1.0,le=6.0,  description="Albumin g/dL (<3.5 = DSS risk)")
    plt:  float = Field(...,ge=1,  le=800,  description="Platelets ×10³/μL (<100 = DHF criterion)")
    hct:  float = Field(...,ge=15, le=70,   description="Haematocrit % (≥20% rise = DHF criterion)")
    wbc:  float = Field(...,ge=0.1,le=30,   description="WBC ×10³/μL (<5 = leukopenia)")
    urea: float = Field(...,ge=5,  le=300,  description="Urea mg/dL (normal 20–50)")
    creat:float = Field(...,ge=0.1,le=15,   description="Creatinine mg/dL (normal 0.7–1.1)")

class MarkerFlag(BaseModel):
    marker:str; value:float; unit:str; flagged:bool; message:str

class PredictionResult(BaseModel):
    severity:      Literal["DF","DHF","DSS"]
    severity_name: str
    confidence:    float
    raw_probs:     dict[str,float]
    probs:         dict[str,float]
    hepatic_index: float
    renal_index:   float
    hema_index:    float
    who_dhf_met:   bool
    hct_rise_pct:  float
    marker_flags:  list[MarkerFlag]
    model_version: str
    processing_ms: float
    model_trained: bool = True

class BatchInput(BaseModel):
    patients: list[PatientInput] = Field(...,min_length=1,max_length=50)

class BatchResult(BaseModel):
    total:int; results:list[PredictionResult]; summary:dict[str,int]

# ── Startup ──────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup(): _load_or_train()

# ── Routes ───────────────────────────────────────────────────────────
@app.get("/health", tags=["Meta"])
def health():
    return {"status":"ok","model":"dengue-pinn-v3","n_features":13,
            "trained":_model is not None,"loaded":_model is not None}

@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
def predict(patient: PatientInput):
    """Single patient → DF / DHF / DSS with WHO-gated probabilities."""
    if _model is None: raise HTTPException(503,"Model not ready")
    try: return run_prediction(patient)
    except Exception as e: raise HTTPException(500,str(e))

@app.post("/predict/mock", response_model=PredictionResult, tags=["Prediction"])
def predict_mock(patient: PatientInput):
    """Rule-based mock prediction (no ML model required). Uses WHO criteria + cohort means."""
    t0 = time.perf_counter()
    p  = patient
    hn = WHO["HCT_NORMAL_F"] if p.sex == "female" else WHO["HCT_NORMAL_M"]
    rise = (p.hct - hn) / hn

    # ── Score DSS risk ────────────────────────────────────────────────
    dss_score = 0.0
    if p.plt < WHO["PLT_DSS"]:        dss_score += 0.40
    if p.alb < WHO["ALB_DSS"]:        dss_score += 0.25
    if p.bil > WHO["BIL_DSS"]:        dss_score += 0.15
    if p.alp > 450:                   dss_score += 0.10
    if p.wbc < 2.5:                   dss_score += 0.05
    if rise  >= WHO["HCT_RISE"]:      dss_score += 0.05
    dss_score = min(dss_score, 0.92)

    # ── Score DHF risk ────────────────────────────────────────────────
    dhf_score = 0.0
    if p.plt < WHO["PLT_DHF"]:        dhf_score += 0.35
    if rise  >= WHO["HCT_RISE"]:      dhf_score += 0.25
    if p.bil > WHO["BIL_DHF"]:        dhf_score += 0.15
    if p.alb < 3.8:                   dhf_score += 0.10
    if p.ast > COHORT_MEANS["DHF"]["AST"]: dhf_score += 0.08
    if p.wbc < WHO["WBC_LEUKO"]:      dhf_score += 0.07
    dhf_score = min(dhf_score, 0.90)

    # ── Build probability vector & normalise ─────────────────────────
    raw_dss  = round(dss_score * 0.80, 4)
    raw_dhf  = round(dhf_score * (1 - raw_dss) * 0.85, 4)
    raw_df   = round(max(0.0, 1.0 - raw_dss - raw_dhf), 4)
    raw_probs = {"DF": raw_df, "DHF": raw_dhf, "DSS": raw_dss}

    # normalise
    total = raw_df + raw_dhf + raw_dss or 1.0
    probs = {k: round(v / total, 4) for k, v in raw_probs.items()}

    idx = max(probs, key=probs.get)
    who_met = p.plt < WHO["PLT_DHF"] and rise >= WHO["HCT_RISE"]

    flags = [
        MarkerFlag(marker="Platelet",   value=p.plt,  unit="×10³/μL",
            flagged=p.plt < WHO["PLT_DHF"],
            message="⚠ <100k → WHO DHF criterion" if p.plt < WHO["PLT_DHF"] else "Normal"),
        MarkerFlag(marker="Hematocrit", value=p.hct,  unit="%",
            flagged=rise >= WHO["HCT_RISE"],
            message=f"⚠ +{rise*100:.0f}% rise → plasma leak" if rise >= WHO["HCT_RISE"] else "Normal"),
        MarkerFlag(marker="WBC",        value=p.wbc,  unit="×10³/μL",
            flagged=p.wbc < WHO["WBC_LEUKO"],
            message="Leukopenia" if p.wbc < WHO["WBC_LEUKO"] else "Normal"),
        MarkerFlag(marker="Albumin",    value=p.alb,  unit="g/dL",
            flagged=p.alb < WHO["ALB_DSS"],
            message="⚠ <3.5 → plasma leak" if p.alb < WHO["ALB_DSS"] else "Normal"),
        MarkerFlag(marker="Bilirubin",  value=p.bil,  unit="mg/dL",
            flagged=p.bil > WHO["BIL_DHF"],
            message=f"⚠ >{WHO['BIL_DHF']} → hepatic" if p.bil > WHO["BIL_DHF"] else "Normal"),
        MarkerFlag(marker="AST",        value=p.ast,  unit="IU/L",
            flagged=p.ast > 35,
            message=f"{p.ast/35:.1f}× normal" if p.ast > 35 else "Normal"),
        MarkerFlag(marker="ALT",        value=p.alt,  unit="IU/L",
            flagged=p.alt > 41,
            message=f"{p.alt/41:.1f}× normal" if p.alt > 41 else "Normal"),
        MarkerFlag(marker="ALP",        value=p.alp,  unit="IU/L",
            flagged=p.alp > 450,
            message="⚠ DSS-level" if p.alp > 450 else "Normal"),
    ]
    return PredictionResult(
        severity=idx, severity_name=SEV_NAMES[idx],
        confidence=round(probs[idx], 4),
        raw_probs=raw_probs, probs=probs,
        hepatic_index=round(.45*(p.ast/35)+.35*(p.alt/41)+.20*(p.alp/306), 3),
        renal_index=round(.55*(p.creat/1.1)+.45*(p.urea/50), 3),
        hema_index=round(1-p.plt/400+(p.hct-42)/100, 3),
        who_dhf_met=who_met, hct_rise_pct=round(rise*100, 1),
        marker_flags=flags, model_version="3.0.0-mock",
        processing_ms=round((time.perf_counter()-t0)*1000, 2),
        model_trained=False,
    )

@app.post("/predict/batch", response_model=BatchResult, tags=["Prediction"])
def predict_batch(body: BatchInput):
    """Up to 50 patients in one request."""
    if _model is None: raise HTTPException(503,"Model not ready")
    results=[]
    for pt in body.patients:
        try: results.append(run_prediction(pt))
        except Exception as e: raise HTTPException(422,str(e))
    summary={k:sum(1 for r in results if r.severity==k) for k in LABELS}
    return BatchResult(total=len(results),results=results,summary=summary)

@app.get("/cohort/stats", tags=["Reference"])
def cohort_stats():
    return {"source":"Pakistan 2023 Thesis","n_total":268,
            "breakdown":{"DF":168,"DHF":83,"DSS":17},
            "means":COHORT_MEANS,"normal_ranges":NORMAL_RANGES,"who":WHO}

@app.get("/criteria/who", tags=["Reference"])
def who_criteria_route():
    return {
        "thresholds": WHO,
        "criteria":{
            "DF": "Fever + 2 of: nausea, rash, aches, leukopenia, +tourniquet",
            "DHF":"DF + Plt<100k + HCT≥20% rise + haemorrhagic signs",
            "DSS":"DHF + circulatory shock (pulse pressure ≤20 mmHg)",
        },
        "key_markers":[
            {"marker":"Platelet",  "rank":"★★★","note":"<100k = WHO DHF criterion"},
            {"marker":"Hematocrit","rank":"★★★","note":"≥20% rise = plasma leakage"},
            {"marker":"AST",       "rank":"★★★","note":"Most sensitive hepatic marker (thesis)"},
            {"marker":"Albumin",   "rank":"★★", "note":"<3.5 g/dL = severe plasma leak (DSS)"},
            {"marker":"Bilirubin", "rank":"★★", "note":">1.5 = hepatic involvement"},
            {"marker":"ALP",       "rank":"★",  "note":"Sig. ONLY at DSS p≤0.05, mean 682.63 IU/L"},
        ],
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
