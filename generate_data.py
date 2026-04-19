import os
import sys
import time
import shutil
import traceback

THIS_DIR = os.path.dirname(os.path.abspath(__file__))  
sys.path.insert(0, THIS_DIR)                           


_data_beside = os.path.join(THIS_DIR, "data")          
_data_parent = os.path.join(THIS_DIR, "..", "data")  

if os.path.isdir(_data_beside):
    DATA_DIR = _data_beside
elif os.path.isdir(os.path.abspath(_data_parent)):
    DATA_DIR = os.path.abspath(_data_parent)
else:
   
    DATA_DIR = _data_beside
    os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_DATASET = os.path.join(DATA_DIR, "transactions.csv")

if not os.path.exists(DEFAULT_DATASET):
    print("⚠️  transactions.csv not found. Generating sample dataset now...")
    os.makedirs(DATA_DIR, exist_ok=True)

    import pandas as pd, numpy as np, random
    from datetime import datetime, timedelta

    random.seed(42); np.random.seed(42)

    PRODUCTS = {
        "Electronics": ["Laptop","Mouse","Keyboard","Monitor","Webcam","Headphones","USB Hub","Laptop Stand","Charger","Speaker"],
        "Office":      ["Notebook","Pen Set","Desk Organizer","Stapler","Sticky Notes","File Folder","Whiteboard","Printer Paper","Scissors","Tape"],
        "Beverages":   ["Coffee","Green Tea","Protein Shake","Energy Drink","Water Bottle","Juice Pack","Herbal Tea","Cold Brew","Kombucha","Coconut Water"],
        "Snacks":      ["Granola Bar","Mixed Nuts","Dark Chocolate","Popcorn","Trail Mix","Rice Cakes","Protein Bar","Dried Fruit","Crackers","Yogurt"],
        "Health":      ["Vitamins","Omega-3","Protein Powder","Probiotics","Melatonin","Zinc","Vitamin D","Magnesium","Collagen","Turmeric"],
        "Fitness":     ["Resistance Band","Yoga Mat","Foam Roller","Jump Rope","Dumbbells","Water Bottle","Gym Gloves","Workout Towel","Ankle Weights","Balance Board"],
    }
    ASSOC = {
        "Laptop":        ["Mouse","Keyboard","Laptop Stand","USB Hub","Monitor"],
        "Coffee":        ["Dark Chocolate","Granola Bar","Notebook"],
        "Protein Shake": ["Protein Bar","Gym Gloves","Resistance Band"],
        "Yoga Mat":      ["Foam Roller","Resistance Band","Water Bottle"],
        "Vitamins":      ["Omega-3","Probiotics","Protein Powder"],
        "Headphones":    ["Laptop","Speaker","USB Hub"],
    }
    SEGS = {
        "tech_professional": {"cats":["Electronics","Office","Beverages"],        "w":0.25},
        "health_enthusiast": {"cats":["Health","Fitness","Beverages","Snacks"],   "w":0.30},
        "office_worker":     {"cats":["Office","Beverages","Snacks"],             "w":0.25},
        "student":           {"cats":["Electronics","Office","Snacks","Beverages"],"w":0.20},
    }
    seg_names = list(SEGS.keys())
    seg_w     = [SEGS[s]["w"] for s in seg_names]
    start     = datetime(2023, 1, 1)
    records   = []

    for tid in range(1, 2001):
        date = start + timedelta(days=random.randint(0, 364))
        seg  = random.choices(seg_names, weights=seg_w)[0]
        cid  = f"CUST_{random.randint(1,500):04d}"
        cats = SEGS[seg]["cats"]
        basket = list(set([random.choice(PRODUCTS[random.choice(cats)]) for _ in range(random.randint(2,8))]))
        for p in basket[:]:
            if p in ASSOC and random.random() < 0.6:
                basket.append(random.choice(ASSOC[p]))
        for prod in set(basket):
            records.append({
                "transaction_id":   f"TXN_{tid:05d}",
                "customer_id":      cid,
                "product":          prod,
                "date":             date.strftime("%Y-%m-%d"),
                "month":            date.month,
                "day_of_week":      date.strftime("%A"),
                "customer_segment": seg,
                "quantity":         random.randint(1, 3),
                "price":            round(random.uniform(5, 200), 2),
            })

    pd.DataFrame(records).to_csv(DEFAULT_DATASET, index=False)
    print(f"✅ Dataset generated → {DEFAULT_DATASET}")

from ml_engine import (
    load_transactions, run_full_analysis, RecommendationEngine,
    rank_rules, build_product_graph, segment_customers,
    analyze_seasonality, compare_models, FPGrowth,
)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional

app = FastAPI(
    title="Market Basket Analysis API",
    description="AI-powered upselling and cross-selling recommendation system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_fe_candidates = [
    os.path.join(THIS_DIR, "frontend"),           
    os.path.join(THIS_DIR, "..", "frontend"),   
    os.path.join(THIS_DIR, "..", "..", "frontend"),
]
FRONTEND_DIR = next(
    (os.path.abspath(p) for p in _fe_candidates if os.path.isdir(p)),
    None
)
FRONTEND_HTML = os.path.join(FRONTEND_DIR, "index.html") if FRONTEND_DIR else None

if FRONTEND_DIR and os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
    print(f"🎨 Frontend found: {FRONTEND_DIR}")


_cache: Dict = {
    "analysis_results": None,
    "rec_engine":       None,
    "last_trained":     None,
    "training_status":  "idle",  
    "training_progress": 0,
    "dataset_path":     DEFAULT_DATASET,
}


class TrainRequest(BaseModel):
    min_support:    float = 0.01
    min_confidence: float = 0.20
    max_len:        int   = 3

class RecommendRequest(BaseModel):
    cart:              List[str]
    n_recommendations: int = 5

class ProductRequest(BaseModel):
    product:           str
    n_recommendations: int = 5



def _ensure_trained():
    if _cache["analysis_results"] is None:
        raise HTTPException(
            status_code=400,
            detail="Model not trained yet. POST /api/train first."
        )

@app.get("/", response_class=HTMLResponse)
def root():
    """Serve the React frontend UI at localhost:8000"""
   
    if FRONTEND_HTML and os.path.exists(FRONTEND_HTML):
        with open(FRONTEND_HTML, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())

   
    return HTMLResponse(content="""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>BasketAI — Market Basket Intelligence</title>
<script src="https://unpkg.com/react@18/umd/react.development.js"></script>
<script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
<script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/recharts@2.12.7/umd/Recharts.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet"/>
<style>
:root{--bg:#050810;--surface:#0c1120;--surface2:#111827;--border:rgba(99,179,255,0.12);
--accent:#3b82f6;--accent2:#06b6d4;--accent3:#8b5cf6;--gold:#f59e0b;--green:#10b981;
--text:#e2e8f0;--muted:#64748b;--glass:rgba(15,23,42,0.7)}
.light{--bg:#f0f4ff;--surface:#fff;--surface2:#f8faff;--border:rgba(59,130,246,0.15);
--text:#0f172a;--muted:#64748b;--glass:rgba(255,255,255,0.8)}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden;transition:background .3s,color .3s}
h1,h2,h3,h4{font-family:'Syne',sans-serif}
.grid-bg{position:fixed;inset:0;pointer-events:none;z-index:0;
background-image:linear-gradient(rgba(59,130,246,.04) 1px,transparent 1px),linear-gradient(90deg,rgba(59,130,246,.04) 1px,transparent 1px);
background-size:40px 40px}
.orb{position:fixed;border-radius:50%;filter:blur(80px);pointer-events:none;z-index:0;opacity:.3}
.orb1{width:400px;height:400px;background:#3b82f6;top:-100px;left:-100px}
.orb2{width:300px;height:300px;background:#8b5cf6;bottom:-50px;right:-50px}
.app{position:relative;z-index:1;display:flex;flex-direction:column;min-height:100vh}
.navbar{display:flex;align-items:center;justify-content:space-between;padding:0 32px;height:64px;
background:var(--glass);backdrop-filter:blur(20px);border-bottom:1px solid var(--border);
position:sticky;top:0;z-index:100}
.logo{font-family:'Syne',sans-serif;font-weight:800;font-size:1.4rem;
background:linear-gradient(135deg,#3b82f6,#06b6d4);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.nav-links{display:flex;gap:4px}
.nav-btn{padding:6px 16px;border-radius:8px;border:none;cursor:pointer;
font-family:'DM Sans',sans-serif;font-size:.85rem;font-weight:500;
background:transparent;color:var(--muted);transition:all .2s}
.nav-btn:hover{color:var(--text);background:var(--surface2)}
.nav-btn.active{color:var(--accent);background:rgba(59,130,246,.12)}
.theme-btn{width:36px;height:36px;border-radius:10px;border:1px solid var(--border);
background:var(--surface2);color:var(--text);cursor:pointer;font-size:1rem;
display:flex;align-items:center;justify-content:center;transition:all .2s}
.main{flex:1;padding:32px;max-width:1400px;margin:0 auto;width:100%}
.card{background:var(--glass);backdrop-filter:blur(20px);border:1px solid var(--border);
border-radius:16px;padding:24px;transition:border-color .2s;margin-bottom:20px}
.card:hover{border-color:rgba(59,130,246,.3)}
.card-title{font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;
color:var(--text);margin-bottom:16px;display:flex;align-items:center;gap:8px}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-bottom:24px}
.stat-card{background:var(--surface2);border:1px solid var(--border);border-radius:14px;padding:20px;position:relative;overflow:hidden}
.stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
background:linear-gradient(90deg,var(--accent),var(--accent2))}
.stat-label{font-size:.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}
.stat-value{font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:var(--text);margin:4px 0}
.stat-sub{font-size:.75rem;color:var(--muted)}
.badge{display:inline-flex;align-items:center;gap:4px;padding:2px 10px;border-radius:20px;font-size:.72rem;font-weight:600}
.badge-blue{background:rgba(59,130,246,.15);color:#60a5fa;border:1px solid rgba(59,130,246,.3)}
.badge-green{background:rgba(16,185,129,.15);color:#34d399;border:1px solid rgba(16,185,129,.3)}
.badge-purple{background:rgba(139,92,246,.15);color:#a78bfa;border:1px solid rgba(139,92,246,.3)}
.badge-gold{background:rgba(245,158,11,.15);color:#fbbf24;border:1px solid rgba(245,158,11,.3)}
.badge-cyan{background:rgba(6,182,212,.15);color:#22d3ee;border:1px solid rgba(6,182,212,.3)}
.btn{padding:10px 20px;border-radius:10px;border:none;cursor:pointer;
font-family:'DM Sans',sans-serif;font-weight:600;font-size:.9rem;
transition:all .2s;display:inline-flex;align-items:center;gap:8px}
.btn-primary{background:linear-gradient(135deg,#3b82f6,#06b6d4);color:white;
box-shadow:0 4px 20px rgba(59,130,246,.3)}
.btn-primary:hover{transform:translateY(-1px);box-shadow:0 6px 25px rgba(59,130,246,.4)}
.btn-secondary{background:var(--surface2);color:var(--text);border:1px solid var(--border)}
.btn-secondary:hover{border-color:var(--accent);color:var(--accent)}
.btn:disabled{opacity:.5;cursor:not-allowed;transform:none!important}
.input{background:var(--surface2);border:1px solid var(--border);border-radius:10px;
padding:10px 14px;color:var(--text);font-family:'DM Sans',sans-serif;
font-size:.9rem;width:100%;outline:none;transition:border-color .2s}
.input:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(59,130,246,.1)}
.input::placeholder{color:var(--muted)}
.range-input{width:100%;accent-color:var(--accent)}
.rules-table{width:100%;border-collapse:collapse;font-size:.82rem}
.rules-table th{text-align:left;padding:10px 12px;border-bottom:1px solid var(--border);
color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.06em;font-size:.72rem}
.rules-table td{padding:10px 12px;border-bottom:1px solid rgba(99,179,255,.05)}
.rules-table tr:hover td{background:rgba(59,130,246,.04)}
.lift-bar{height:4px;background:var(--surface2);border-radius:2px;overflow:hidden;min-width:60px}
.lift-fill{height:100%;border-radius:2px;background:linear-gradient(90deg,#3b82f6,#06b6d4)}
.progress-bar{height:6px;background:var(--surface2);border-radius:3px;overflow:hidden}
.progress-fill{height:100%;border-radius:3px;background:linear-gradient(90deg,#3b82f6,#06b6d4);transition:width .5s}
.cart-area{min-height:120px;border:2px dashed var(--border);border-radius:14px;
padding:16px;display:flex;flex-wrap:wrap;gap:8px;align-content:flex-start;transition:border-color .2s,background .2s}
.cart-area.drag-over{border-color:var(--accent);background:rgba(59,130,246,.05)}
.product-chip{padding:6px 14px;border-radius:20px;font-size:.82rem;font-weight:600;
cursor:grab;user-select:none;transition:all .15s;display:flex;align-items:center;gap:6px}
.product-chip:hover{transform:scale(1.05)}
.chip-blue{background:rgba(59,130,246,.2);color:#93c5fd;border:1px solid rgba(59,130,246,.3)}
.chip-purple{background:rgba(139,92,246,.2);color:#c4b5fd;border:1px solid rgba(139,92,246,.3)}
.chip-cyan{background:rgba(6,182,212,.2);color:#67e8f9;border:1px solid rgba(6,182,212,.3)}
.chip-green{background:rgba(16,185,129,.2);color:#6ee7b7;border:1px solid rgba(16,185,129,.3)}
.chip-gold{background:rgba(245,158,11,.2);color:#fcd34d;border:1px solid rgba(245,158,11,.3)}
.rec-card{border:1px solid var(--border);border-radius:12px;padding:16px;
background:var(--surface2);transition:all .2s;display:flex;flex-direction:column;gap:8px}
.rec-card:hover{border-color:var(--accent);transform:translateY(-2px)}
.rec-card .rec-title{font-family:'Syne',sans-serif;font-weight:700;font-size:1rem}
.rec-card .rec-explain{font-size:.78rem;color:var(--muted);line-height:1.5}
.metric-row{display:flex;gap:8px;flex-wrap:wrap}
.graph-container{position:relative;width:100%;height:400px;overflow:hidden;border-radius:12px;background:var(--surface2)}
.graph-svg{width:100%;height:100%}
.tabs{display:flex;gap:4px;border-bottom:1px solid var(--border);margin-bottom:20px}
.tab{padding:8px 18px;border:none;background:transparent;cursor:pointer;
font-family:'DM Sans',sans-serif;font-size:.88rem;font-weight:500;
color:var(--muted);border-bottom:2px solid transparent;transition:all .2s;margin-bottom:-1px}
.tab.active{color:var(--accent);border-bottom-color:var(--accent)}
.tab:hover:not(.active){color:var(--text)}
.algo-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}
.algo-card{border:1px solid var(--border);border-radius:14px;padding:20px;
background:var(--surface2);position:relative;overflow:hidden}
.algo-card .algo-name{font-family:'Syne',sans-serif;font-weight:800;font-size:1.1rem;margin-bottom:12px}
.algo-metric{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.algo-metric-label{font-size:.78rem;color:var(--muted)}
.algo-metric-value{font-weight:700;font-size:.9rem}
.algo-glow{position:absolute;top:0;right:0;width:80px;height:80px;border-radius:50%;filter:blur(40px);opacity:.3}
.upload-zone{border:2px dashed var(--border);border-radius:16px;padding:48px;
text-align:center;cursor:pointer;transition:all .2s}
.upload-zone:hover{border-color:var(--accent);background:rgba(59,130,246,.04)}
.upload-zone.drag-over{border-color:var(--accent);background:rgba(59,130,246,.08)}
.upload-icon{font-size:3rem;margin-bottom:16px}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.anim-fade{animation:fadeIn .4s ease forwards}
.spinner{animation:spin 1s linear infinite}
.pulse{animation:pulse 2s infinite}
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px}
@media(max-width:900px){.grid-2,.grid-3{grid-template-columns:1fr}}
.seg-bar{display:flex;align-items:center;gap:12px;margin-bottom:12px}
.seg-label{font-size:.82rem;min-width:140px;color:var(--text)}
.seg-track{flex:1;height:8px;background:var(--surface2);border-radius:4px;overflow:hidden}
.seg-fill{height:100%;border-radius:4px;transition:width .8s ease}
.seg-count{font-size:.78rem;color:var(--muted);min-width:30px;text-align:right}
.status-dot{width:8px;height:8px;border-radius:50%;display:inline-block}
.dot-green{background:#10b981;box-shadow:0 0 8px #10b981;animation:pulse 2s infinite}
.dot-yellow{background:#f59e0b;animation:pulse 1s infinite}
.dot-red{background:#ef4444}
.dot-gray{background:var(--muted)}
.param-row{display:flex;flex-direction:column;gap:6px;margin-bottom:16px}
.param-label{font-size:.82rem;color:var(--muted);display:flex;justify-content:space-between}
.param-value{font-weight:700;color:var(--accent)}
.chip-remove{background:none;border:none;cursor:pointer;color:var(--muted);font-size:.8rem;padding:0 2px}
.chip-remove:hover{color:#ef4444}
select.input{cursor:pointer}
</style>
</head>
<body>
<div id="root"></div>
<script type="text/babel">
const { useState, useEffect, useRef } = React;
const { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
        Tooltip, Legend, ResponsiveContainer, AreaChart, Area } = Recharts;

const API = "";   // same origin — no CORS needed when served from FastAPI

const fmt = (n,d=2) => typeof n==='number' ? n.toFixed(d) : n;
const chipColors = ['chip-blue','chip-purple','chip-cyan','chip-green','chip-gold'];
const chipFor = i => chipColors[i % chipColors.length];
const algoColors = { Apriori:'#3b82f6', 'FP-Growth':'#06b6d4', ECLAT:'#8b5cf6' };

function useApi(path, deps=[]) {
  const [data,setData]=useState(null);
  const [loading,setLoading]=useState(false);
  useEffect(()=>{
    if(!path) return;
    setLoading(true);
    fetch(API+path).then(r=>r.json()).then(d=>{setData(d);setLoading(false);}).catch(()=>setLoading(false));
  }, deps);
  return {data,loading};
}

function StatusBadge({status}) {
  const map={ready:['dot-green','Ready'],training:['dot-yellow','Training...'],idle:['dot-gray','Idle'],error:['dot-red','Error']};
  const [cls,label]=map[status]||map.idle;
  return <span style={{display:'flex',alignItems:'center',gap:6,fontSize:'.8rem',color:'var(--muted)'}}>
    <span className={`status-dot ${cls}`}/>{label}</span>;
}

function LiftBar({value,max=5}) {
  const pct=Math.min((value/max)*100,100);
  const color=value>=3?'#10b981':value>=2?'#3b82f6':'#64748b';
  return <div className="lift-bar"><div className="lift-fill" style={{width:`${pct}%`,background:color}}/></div>;
}

function UploadPage({onTrained}) {
  const [file,setFile]=useState(null);
  const [uploadMsg,setUploadMsg]=useState(null);
  const [training,setTraining]=useState(false);
  const [params,setParams]=useState({min_support:0.01,min_confidence:0.2,max_len:3});
  const [trainResult,setTrainResult]=useState(null);
  const [dragging,setDragging]=useState(false);
  const fileRef=useRef();

  const handleFile=async(f)=>{
    setFile(f);
    const fd=new FormData(); fd.append('file',f);
    try {
      const r=await fetch(`${API}/api/upload`,{method:'POST',body:fd});
      setUploadMsg(await r.json());
    } catch(e){setUploadMsg({error:e.message});}
  };

  const handleTrain=async()=>{
    setTraining(true); setTrainResult(null);
    try {
      const r=await fetch(`${API}/api/train`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(params)});
      const d=await r.json(); setTrainResult(d);
      if(d.status==='ready') onTrained();
    } catch(e){setTrainResult({error:e.message});}
    setTraining(false);
  };

  return <div className="anim-fade">
    <h2 style={{fontFamily:'Syne',fontWeight:800,fontSize:'1.8rem',marginBottom:8}}>Dataset Setup</h2>
    <p style={{color:'var(--muted)',marginBottom:28}}>Upload your transaction data or use the built-in demo dataset</p>
    <div className="grid-2">
      <div>
        <div className="card">
          <div className="card-title">📁 Upload Dataset</div>
          <div className={`upload-zone ${dragging?'drag-over':''}`}
            onDragOver={e=>{e.preventDefault();setDragging(true)}}
            onDragLeave={()=>setDragging(false)}
            onDrop={e=>{e.preventDefault();setDragging(false);const f=e.dataTransfer.files[0];if(f)handleFile(f);}}
            onClick={()=>fileRef.current.click()}>
            <div className="upload-icon">📊</div>
            <div style={{fontFamily:'Syne',fontWeight:700,fontSize:'1.1rem',marginBottom:8}}>{file?file.name:'Drop CSV file here'}</div>
            <div style={{color:'var(--muted)',fontSize:'.85rem'}}>Columns: transaction_id, product, date, customer_id</div>
            <input ref={fileRef} type="file" accept=".csv" style={{display:'none'}} onChange={e=>handleFile(e.target.files[0])}/>
          </div>
          {uploadMsg&&<div style={{marginTop:16,padding:12,borderRadius:10,
            background:uploadMsg.error?'rgba(239,68,68,.1)':'rgba(16,185,129,.1)',
            border:`1px solid ${uploadMsg.error?'rgba(239,68,68,.3)':'rgba(16,185,129,.3)'}`,fontSize:'.82rem'}}>
            {uploadMsg.error?<span style={{color:'#f87171'}}>❌ {uploadMsg.error}</span>
              :<span style={{color:'#34d399'}}>✅ Uploaded: {uploadMsg.rows} rows, {uploadMsg.columns?.length} columns</span>}
          </div>}
        </div>
        <div className="card" style={{background:'rgba(16,185,129,.05)',borderColor:'rgba(16,185,129,.2)'}}>
          <div className="card-title" style={{color:'#34d399'}}>✅ Demo Dataset Ready</div>
          <p style={{fontSize:'.83rem',color:'var(--muted)',marginBottom:12}}>2,000 transactions · 59 products · 6 categories · 12-month seasonality</p>
          <div style={{display:'flex',gap:8,flexWrap:'wrap'}}>
            {['Electronics','Office','Beverages','Snacks','Health','Fitness'].map(c=><span key={c} className="badge badge-green">{c}</span>)}
          </div>
        </div>
      </div>
      <div className="card">
        <div className="card-title">⚙️ Training Parameters</div>
        {[['Min Support',0.005,0.1,0.005,'min_support'],['Min Confidence',0.1,0.9,0.05,'min_confidence']].map(([label,min,max,step,key])=>(
          <div className="param-row" key={key}>
            <div className="param-label">{label} <span className="param-value">{params[key]}</span></div>
            <input type="range" className="range-input" min={min} max={max} step={step} value={params[key]}
              onChange={e=>setParams(p=>({...p,[key]:parseFloat(e.target.value)}))}/>
          </div>
        ))}
        <div className="param-row">
          <div className="param-label">Max Itemset Length <span className="param-value">{params.max_len}</span></div>
          <input type="range" className="range-input" min={2} max={5} step={1} value={params.max_len}
            onChange={e=>setParams(p=>({...p,max_len:parseInt(e.target.value)}))}/>
        </div>
        <button className="btn btn-primary" style={{width:'100%',justifyContent:'center',marginTop:8}}
          onClick={handleTrain} disabled={training}>
          {training?<><span className="spinner">⟳</span> Training Models...</>:'🚀 Train All Models'}
        </button>
        {training&&<div style={{marginTop:16}}>
          <div style={{fontSize:'.78rem',color:'var(--muted)',marginBottom:6}}>Running Apriori → FP-Growth → ECLAT...</div>
          <div className="progress-bar"><div className="progress-fill pulse" style={{width:'70%'}}/></div>
        </div>}
        {trainResult&&<div style={{marginTop:16,padding:14,borderRadius:10,
          background:trainResult.error?'rgba(239,68,68,.1)':'rgba(59,130,246,.1)',
          border:`1px solid ${trainResult.error?'rgba(239,68,68,.3)':'rgba(59,130,246,.3)'}`,fontSize:'.83rem'}}>
          {trainResult.error?<span style={{color:'#f87171'}}>❌ {trainResult.error}</span>
            :<><div style={{color:'#60a5fa',fontWeight:700,marginBottom:8}}>✅ Training Complete!</div>
              <div style={{color:'var(--muted)'}}>Rules found: <b style={{color:'var(--text)'}}>{trainResult.n_rules}</b></div>
              {trainResult.training_time&&Object.entries(trainResult.training_time).map(([a,t])=>(
                <div key={a} style={{color:'var(--muted)'}}>{a}: <b style={{color:'var(--text)'}}>{t}s</b></div>))}
            </>}
        </div>}
      </div>
    </div>
  </div>;
}

function DashboardPage() {
  const {data:summary}=useApi('/api/summary',[]);
  const {data:comparison}=useApi('/api/model-comparison',[]);
  const {data:season}=useApi('/api/seasonality',[]);
  const {data:segments}=useApi('/api/segments',[]);
  const [rulesTab,setRulesTab]=useState('all');
  const [rulesData,setRulesData]=useState(null);
  useEffect(()=>{fetch(`${API}/api/rules?limit=30`).then(r=>r.json()).then(setRulesData);},[]);
  const segColors=['#3b82f6','#06b6d4','#8b5cf6','#f59e0b'];
  return <div className="anim-fade">
    <h2 style={{fontFamily:'Syne',fontWeight:800,fontSize:'1.8rem',marginBottom:8}}>Insights Dashboard</h2>
    <p style={{color:'var(--muted)',marginBottom:24}}>Real-time analysis of purchase patterns and associations</p>
    {summary&&<div className="stat-grid">
      {[{label:'Transactions',value:summary.total_transactions?.toLocaleString(),sub:'total baskets',icon:'🛒'},
        {label:'Customers',value:summary.total_customers?.toLocaleString(),sub:'unique buyers',icon:'👥'},
        {label:'Products',value:summary.total_products,sub:'unique items',icon:'📦'},
        {label:'Rules Found',value:summary.total_rules,sub:'association rules',icon:'🔗'},
        {label:'Avg Basket',value:summary.avg_basket_size,sub:'items per basket',icon:'📊'},
      ].map((s,i)=>(
        <div className="stat-card" key={i}>
          <div style={{fontSize:'1.5rem',marginBottom:8}}>{s.icon}</div>
          <div className="stat-label">{s.label}</div>
          <div className="stat-value">{s.value}</div>
          <div className="stat-sub">{s.sub}</div>
        </div>
      ))}
    </div>}
    {comparison&&<div className="card">
      <div className="card-title">🤖 Algorithm Performance Comparison</div>
      <div className="algo-grid">
        {Object.entries(comparison).map(([algo,data])=>(
          <div className="algo-card" key={algo}>
            <div className="algo-glow" style={{background:algoColors[algo]}}/>
            <div className="algo-name" style={{color:algoColors[algo]}}>{algo}</div>
            {[['Frequent Itemsets',data.n_frequent_itemsets],['Rules',data.n_rules],
              ['Train Time',`${data.training_time_sec}s`],['Avg Lift',fmt(data.avg_lift)],
              ['Max Lift',fmt(data.max_lift)]].map(([l,v])=>(
              <div className="algo-metric" key={l}>
                <span className="algo-metric-label">{l}</span>
                <span className="algo-metric-value" style={{color:algoColors[algo]}}>{v}</span>
              </div>))}
          </div>))}
      </div>
      <div style={{marginTop:24,height:200}}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={Object.entries(comparison).map(([algo,d])=>({algo,rules:d.n_rules,itemsets:d.n_frequent_itemsets}))}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,179,255,.08)"/>
            <XAxis dataKey="algo" tick={{fill:'var(--muted)',fontSize:11}} axisLine={false}/>
            <YAxis tick={{fill:'var(--muted)',fontSize:10}} axisLine={false}/>
            <Tooltip contentStyle={{background:'var(--surface)',border:'1px solid var(--border)',borderRadius:8}}/>
            <Legend wrapperStyle={{fontSize:'.8rem'}}/>
            <Bar dataKey="rules" name="Rules" fill="#3b82f6" radius={[4,4,0,0]}/>
            <Bar dataKey="itemsets" name="Itemsets" fill="#06b6d4" radius={[4,4,0,0]}/>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>}
    <div className="grid-2">
      {season?.trend_data&&<div className="card">
        <div className="card-title">📅 Monthly Sales Trends</div>
        <div style={{height:220}}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={season.trend_data}>
              <defs>
                {['#3b82f6','#06b6d4','#8b5cf6'].map((c,i)=>(
                  <linearGradient key={i} id={`g${i}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={c} stopOpacity={.3}/>
                    <stop offset="95%" stopColor={c} stopOpacity={0}/>
                  </linearGradient>))}
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,179,255,.08)"/>
              <XAxis dataKey="month" tick={{fill:'var(--muted)',fontSize:10}} axisLine={false}/>
              <YAxis tick={{fill:'var(--muted)',fontSize:10}} axisLine={false}/>
              <Tooltip contentStyle={{background:'var(--surface)',border:'1px solid var(--border)',borderRadius:8,fontSize:'.8rem'}}/>
              {season.top_products?.slice(0,3).map((p,i)=>(
                <Area key={p} type="monotone" dataKey={p} stroke={['#3b82f6','#06b6d4','#8b5cf6'][i]}
                  fill={`url(#g${i})`} strokeWidth={2} dot={false} name={p}/>))}
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>}
      {segments?.segments&&<div className="card">
        <div className="card-title">👥 Customer Segments</div>
        {Object.entries(segments.segments).map(([seg,data],i)=>{
          const maxCount=Math.max(...Object.values(segments.segments).map(d=>d.count));
          return <div key={seg}>
            <div className="seg-bar">
              <span className="seg-label">{seg}</span>
              <div className="seg-track"><div className="seg-fill" style={{width:`${(data.count/maxCount)*100}%`,background:segColors[i]}}/></div>
              <span className="seg-count">{data.count}</span>
            </div>
            <div style={{fontSize:'.72rem',color:'var(--muted)',marginBottom:12,marginLeft:152}}>
              Avg spend: ${fmt(data.avg_spend,0)} · {fmt(data.avg_transactions,1)} orders
            </div>
          </div>;
        })}
      </div>}
    </div>
    {rulesData&&<div className="card">
      <div className="card-title">🔗 Association Rules</div>
      <div className="tabs">
        {['all','Apriori','FP-Growth','ECLAT'].map(t=>(
          <button key={t} className={`tab ${rulesTab===t?'active':''}`} onClick={()=>setRulesTab(t)}>
            {t==='all'?'All Rules':t}</button>))}
      </div>
      <div style={{overflowX:'auto'}}>
        <table className="rules-table">
          <thead><tr>
            <th>IF (Antecedent)</th><th>THEN (Consequent)</th>
            <th>Support</th><th>Confidence</th><th>Lift</th><th>Score</th><th>Algorithm</th>
          </tr></thead>
          <tbody>
            {rulesData.rules.filter(r=>rulesTab==='all'||r.algorithm===rulesTab).map((rule,i)=>(
              <tr key={i}>
                <td>{rule.antecedents.map((a,j)=><span key={j} className="badge badge-blue" style={{marginRight:4}}>{a}</span>)}</td>
                <td>{rule.consequents.map((c,j)=><span key={j} className="badge badge-green" style={{marginRight:4}}>{c}</span>)}</td>
                <td style={{fontWeight:600}}>{fmt(rule.support*100,1)}%</td>
                <td style={{fontWeight:600}}>{fmt(rule.confidence*100,1)}%</td>
                <td><div style={{display:'flex',alignItems:'center',gap:8}}>
                  <span style={{fontWeight:700,color:rule.lift>=3?'#10b981':rule.lift>=2?'#3b82f6':'var(--muted)'}}>{fmt(rule.lift)}</span>
                  <LiftBar value={rule.lift}/>
                </div></td>
                <td><div style={{width:32,height:32,borderRadius:'50%',
                  background:`conic-gradient(#3b82f6 ${(rule.interestingness||0)*360}deg,var(--surface2) 0)`,
                  display:'flex',alignItems:'center',justifyContent:'center',fontSize:'.6rem',fontWeight:700}}>
                  {fmt((rule.interestingness||0)*100,0)}</div></td>
                <td><span className="badge" style={{background:algoColors[rule.algorithm]+'22',color:algoColors[rule.algorithm],
                  border:`1px solid ${algoColors[rule.algorithm]}44`}}>{rule.algorithm}</span></td>
              </tr>))}
          </tbody>
        </table>
      </div>
    </div>}
  </div>;
}

function RecommendPage() {
  const {data:productsData}=useApi('/api/products',[]);
  const {data:graphData}=useApi('/api/graph?top_n=40',[]);
  const [cart,setCart]=useState([]);
  const [dragOver,setDragOver]=useState(false);
  const [dragging,setDragging]=useState(null);
  const [recs,setRecs]=useState(null);
  const [loading,setLoading]=useState(false);
  const [searchTerm,setSearchTerm]=useState('');
  const products=productsData?.products||[];
  const filtered=products.filter(p=>p.toLowerCase().includes(searchTerm.toLowerCase())&&!cart.includes(p));
  const addToCart=p=>{if(!cart.includes(p))setCart(c=>[...c,p]);};
  const removeFromCart=p=>setCart(c=>c.filter(x=>x!==p));
  const getRecs=async()=>{
    if(!cart.length) return; setLoading(true);
    try {
      const r=await fetch(`${API}/api/recommend`,{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({cart,n_recommendations:6})});
      setRecs(await r.json());
    } catch(e){console.error(e);}
    setLoading(false);
  };
  useEffect(()=>{if(cart.length>0)getRecs();else setRecs(null);},[cart]);
  const svgRef=useRef();
  useEffect(()=>{
    if(!graphData||!svgRef.current) return;
    const {nodes,edges}=graphData; if(!nodes.length) return;
    const W=svgRef.current.clientWidth||700,H=380;
    const pos={};
    nodes.forEach((n,i)=>{
      const angle=(i/nodes.length)*2*Math.PI, r=130+Math.random()*50;
      pos[n.id]={x:W/2+r*Math.cos(angle),y:H/2+r*Math.sin(angle),...n};
    });
    for(let it=0;it<50;it++){
      nodes.forEach(a=>nodes.forEach(b=>{if(a.id===b.id)return;
        const dx=pos[a.id].x-pos[b.id].x,dy=pos[a.id].y-pos[b.id].y;
        const d=Math.sqrt(dx*dx+dy*dy)+.1,f=800/(d*d);
        pos[a.id].x+=dx/d*f;pos[a.id].y+=dy/d*f;}));
      edges.forEach(e=>{if(!pos[e.source]||!pos[e.target])return;
        const dx=pos[e.target].x-pos[e.source].x,dy=pos[e.target].y-pos[e.source].y;
        const d=Math.sqrt(dx*dx+dy*dy)+.1;
        pos[e.source].x+=dx*.05;pos[e.source].y+=dy*.05;
        pos[e.target].x-=dx*.05;pos[e.target].y-=dy*.05;});
      Object.values(pos).forEach(p=>{p.x=Math.max(40,Math.min(W-40,p.x));p.y=Math.max(30,Math.min(H-30,p.y));});
    }
    const maxConn=Math.max(...nodes.map(n=>n.connections||1));
    const maxLift=Math.max(...edges.map(e=>e.weight||1));
    const svg=svgRef.current; svg.innerHTML=''; svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
    edges.forEach(e=>{if(!pos[e.source]||!pos[e.target])return;
      const line=document.createElementNS('http://www.w3.org/2000/svg','line');
      line.setAttribute('x1',pos[e.source].x);line.setAttribute('y1',pos[e.source].y);
      line.setAttribute('x2',pos[e.target].x);line.setAttribute('y2',pos[e.target].y);
      line.setAttribute('stroke','#3b82f6');line.setAttribute('stroke-width',1+(e.weight/maxLift)*2);
      line.setAttribute('stroke-opacity',.2+(e.weight/maxLift)*.6);svg.appendChild(line);});
    nodes.forEach(n=>{const p=pos[n.id];const r=6+(n.connections/maxConn)*14;
      const g=document.createElementNS('http://www.w3.org/2000/svg','g');g.style.cursor='pointer';
      const circle=document.createElementNS('http://www.w3.org/2000/svg','circle');
      circle.setAttribute('cx',p.x);circle.setAttribute('cy',p.y);circle.setAttribute('r',r);
      const hue=n.connections/maxConn;
      circle.setAttribute('fill',hue>.7?'#06b6d4':hue>.4?'#3b82f6':'#8b5cf6');
      circle.setAttribute('fill-opacity','.8');circle.setAttribute('stroke','#3b82f6');
      circle.setAttribute('stroke-width','1.5');circle.setAttribute('stroke-opacity','.5');
      const text=document.createElementNS('http://www.w3.org/2000/svg','text');
      text.setAttribute('x',p.x);text.setAttribute('y',p.y+r+12);text.setAttribute('text-anchor','middle');
      text.setAttribute('font-size','9');text.setAttribute('fill','rgba(226,232,240,.7)');
      text.setAttribute('font-family','DM Sans');text.textContent=n.label;
      g.appendChild(circle);g.appendChild(text);svg.appendChild(g);});
  },[graphData]);
  return <div className="anim-fade">
    <h2 style={{fontFamily:'Syne',fontWeight:800,fontSize:'1.8rem',marginBottom:8}}>Recommendation Engine</h2>
    <p style={{color:'var(--muted)',marginBottom:24}}>Drag products into your cart to get AI-powered upsell & cross-sell suggestions</p>
    <div className="grid-2" style={{marginBottom:20}}>
      <div className="card">
        <div className="card-title">📦 Product Catalog</div>
        <input className="input" placeholder="Search products..." style={{marginBottom:12}}
          value={searchTerm} onChange={e=>setSearchTerm(e.target.value)}/>
        <div style={{display:'flex',flexWrap:'wrap',gap:8,maxHeight:260,overflowY:'auto'}}>
          {filtered.map((p,i)=>(
            <div key={p} className={`product-chip ${chipFor(i)}`} draggable
              onDragStart={()=>setDragging(p)} onDragEnd={()=>setDragging(null)} onClick={()=>addToCart(p)}>
              {p} <span style={{opacity:.6,fontSize:'.7rem'}}>+</span>
            </div>))}
        </div>
      </div>
      <div className="card">
        <div className="card-title">🛒 Your Cart {cart.length>0&&<span className="badge badge-blue">{cart.length} items</span>}</div>
        <div className={`cart-area ${dragOver?'drag-over':''}`}
          onDragOver={e=>{e.preventDefault();setDragOver(true)}}
          onDragLeave={()=>setDragOver(false)}
          onDrop={e=>{e.preventDefault();setDragOver(false);if(dragging)addToCart(dragging);setDragging(null);}}>
          {!cart.length?<span style={{color:'var(--muted)',fontSize:'.85rem',margin:'auto'}}>Drag & drop products here</span>
            :cart.map((p,i)=>(
              <div key={p} className={`product-chip ${chipFor(i)}`}>{p}
                <button className="chip-remove" onClick={()=>removeFromCart(p)}>✕</button>
              </div>))}
        </div>
        {cart.length>0&&<button className="btn btn-primary" style={{marginTop:12,width:'100%',justifyContent:'center'}}
          onClick={getRecs} disabled={loading}>
          {loading?<><span className="spinner">⟳</span> Analyzing...</>:'🔮 Get Recommendations'}
        </button>}
      </div>
    </div>
    {recs&&<div className="anim-fade">
      {recs.upsell?.length>0&&<div className="card" style={{marginBottom:16}}>
        <div className="card-title">⬆️ Upsell Suggestions <span className="badge badge-gold">High Value</span></div>
        <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(280px,1fr))',gap:12}}>
          {recs.upsell.map((r,i)=>(
            <div key={i} className="rec-card">
              <div style={{display:'flex',justifyContent:'space-between',alignItems:'flex-start'}}>
                <div className="rec-title">{r.product}</div>
                <button className="btn btn-secondary" style={{padding:'4px 10px',fontSize:'.75rem'}} onClick={()=>addToCart(r.product)}>+ Add</button>
              </div>
              <div className="metric-row">
                <span className="badge badge-gold">Lift: {fmt(r.lift)}</span>
                <span className="badge badge-blue">Conf: {fmt(r.confidence*100,0)}%</span>
                <span className="badge badge-green">Sup: {fmt(r.support*100,1)}%</span>
              </div>
              <div className="rec-explain">{r.explanation}</div>
            </div>))}
        </div>
      </div>}
      {recs.cross_sell?.length>0&&<div className="card" style={{marginBottom:16}}>
        <div className="card-title">↔️ Cross-Sell Suggestions <span className="badge badge-cyan">Complementary</span></div>
        <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(280px,1fr))',gap:12}}>
          {recs.cross_sell.map((r,i)=>(
            <div key={i} className="rec-card">
              <div style={{display:'flex',justifyContent:'space-between',alignItems:'flex-start'}}>
                <div className="rec-title">{r.product}</div>
                <button className="btn btn-secondary" style={{padding:'4px 10px',fontSize:'.75rem'}} onClick={()=>addToCart(r.product)}>+ Add</button>
              </div>
              <div className="metric-row">
                <span className="badge badge-cyan">Lift: {fmt(r.lift)}</span>
                <span className="badge badge-blue">Conf: {fmt(r.confidence*100,0)}%</span>
              </div>
              <div className="rec-explain">{r.explanation}</div>
            </div>))}
        </div>
      </div>}
      {!recs.upsell?.length&&!recs.cross_sell?.length&&<div className="card" style={{textAlign:'center',padding:40}}>
        <div style={{fontSize:'2rem',marginBottom:8}}>🤔</div>
        <div style={{color:'var(--muted)'}}>No strong associations found for this cart.</div>
        <div style={{color:'var(--muted)',fontSize:'.82rem',marginTop:4}}>Try adding more products or lowering the confidence threshold.</div>
      </div>}
    </div>}
    {graphData&&<div className="card" style={{marginTop:20}}>
      <div className="card-title">🕸️ Product Relationship Network
        <span className="badge badge-purple">{graphData.nodes?.length} nodes · {graphData.edges?.length} edges</span>
      </div>
      <p style={{fontSize:'.82rem',color:'var(--muted)',marginBottom:12}}>Node size = connection strength · Edge opacity = lift score</p>
      <div className="graph-container"><svg ref={svgRef} className="graph-svg"/></div>
    </div>}
  </div>;
}

function App() {
  const [page,setPage]=useState('upload');
  const [darkMode,setDarkMode]=useState(true);
  const [trainedStatus,setTrainedStatus]=useState('idle');
  const {data:status}=useApi('/api/status',[]);
  useEffect(()=>{if(status?.training_status){setTrainedStatus(status.training_status);}},[status]);
  useEffect(()=>{document.body.className=darkMode?'':'light';},[darkMode]);
  const onTrained=()=>{setTrainedStatus('ready');setPage('dashboard');};
  const navItems=[{id:'upload',label:'📤 Setup',always:true},{id:'dashboard',label:'📊 Dashboard'},{id:'recommend',label:'🔮 Recommendations'}];
  return <div className="app">
    <div className="grid-bg"/>
    <div className="orb orb1"/><div className="orb orb2"/>
    <nav className="navbar">
      <div className="logo">◈ BasketAI</div>
      <div className="nav-links">
        {navItems.map(item=>(
          <button key={item.id} className={`nav-btn ${page===item.id?'active':''}`}
            onClick={()=>setPage(item.id)}
            disabled={!item.always&&trainedStatus!=='ready'}
            style={{opacity:!item.always&&trainedStatus!=='ready'?.4:1}}>
            {item.label}</button>))}
      </div>
      <div style={{display:'flex',alignItems:'center',gap:12}}>
        <StatusBadge status={trainedStatus}/>
        <button className="theme-btn" onClick={()=>setDarkMode(d=>!d)}>{darkMode?'☀️':'🌙'}</button>
      </div>
    </nav>
    <main className="main">
      {page==='upload'&&<UploadPage onTrained={onTrained}/>}
      {page==='dashboard'&&trainedStatus==='ready'&&<DashboardPage/>}
      {page==='recommend'&&trainedStatus==='ready'&&<RecommendPage/>}
      {page!=='upload'&&trainedStatus!=='ready'&&<div style={{display:'flex',flexDirection:'column',alignItems:'center',
        justifyContent:'center',height:400,gap:16,color:'var(--muted)',textAlign:'center'}}>
        <div style={{fontSize:'3rem'}}>🚀</div>
        <div style={{fontFamily:'Syne',fontSize:'1.3rem',color:'var(--text)'}}>Train a model first</div>
        <button className="btn btn-primary" onClick={()=>setPage('upload')}>Go to Setup →</button>
      </div>}
    </main>
    <footer style={{padding:'16px 32px',borderTop:'1px solid var(--border)',color:'var(--muted)',
      fontSize:'.78rem',textAlign:'center',background:'var(--glass)',backdropFilter:'blur(20px)'}}>
      BasketAI · Market Basket Intelligence · Apriori + FP-Growth + ECLAT
    </footer>
  </div>;
}
ReactDOM.createRoot(document.getElementById('root')).render(<App/>);
</script>
</body>
</html>""")


@app.get("/ui", response_class=HTMLResponse)
def ui_redirect():
    """Alias — /ui also serves the frontend."""
    return root()


@app.get("/api/status")
def get_status():
    return {
        "training_status":   _cache["training_status"],
        "training_progress": _cache["training_progress"],
        "last_trained":      _cache["last_trained"],
        "has_results":       _cache["analysis_results"] is not None,
        "dataset_path":      _cache["dataset_path"],
    }


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    save_path = os.path.join(DATA_DIR, f"uploaded_{file.filename}")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    import pandas as pd
    try:
        df = pd.read_csv(save_path)
        _cache["dataset_path"]      = save_path
        _cache["analysis_results"]  = None
        _cache["rec_engine"]        = None
        return {
            "message":  "Dataset uploaded successfully",
            "filename": file.filename,
            "rows":     len(df),
            "columns":  df.columns.tolist(),
            "preview":  df.head(5).to_dict("records"),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")


@app.post("/api/train")
def train_model(req: TrainRequest):
    if _cache["training_status"] == "training":
        return {"message": "Training already in progress", "status": "training"}

    _cache["training_status"]   = "training"
    _cache["training_progress"] = 0

    try:
        params = {
            "min_support":    req.min_support,
            "min_confidence": req.min_confidence,
            "max_len":        req.max_len,
        }
        _cache["training_progress"] = 20
        results = run_full_analysis(_cache["dataset_path"], params)
        _cache["training_progress"] = 90

        _cache["analysis_results"] = {
            "summary":          results["summary"],
            "model_comparison": {
                k: {**v, "top_rules": v["top_rules"][:5]}
                for k, v in results["model_comparison"].items()
            },
            "rules":            results["rules"],
            "product_graph":    results["product_graph"],
            "seasonality":      results["seasonality"],
            "customer_segments":results["customer_segments"],
        }
        _cache["rec_engine"]        = results["recommendation_engine"]
        _cache["last_trained"]      = time.strftime("%Y-%m-%d %H:%M:%S")
        _cache["training_status"]   = "ready"
        _cache["training_progress"] = 100

        return {
            "message": "Training complete",
            "status":  "ready",
            "summary": results["summary"],
            "n_rules": len(results["rules"]),
            "training_time": {
                k: v["training_time_sec"]
                for k, v in results["model_comparison"].items()
            },
        }

    except Exception as e:
        _cache["training_status"]   = "error"
        _cache["training_progress"] = 0
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {e}\n{traceback.format_exc()}"
        )


@app.get("/api/summary")
def get_summary():
    _ensure_trained()
    return _cache["analysis_results"]["summary"]


@app.get("/api/rules")
def get_rules(limit: int = 50, algorithm: Optional[str] = None, min_lift: float = 1.0):
    _ensure_trained()
    rules = _cache["analysis_results"]["rules"]
    if algorithm:
        rules = [r for r in rules if r["algorithm"].lower() == algorithm.lower()]
    rules = [r for r in rules if r["lift"] >= min_lift]
    return {"rules": rules[:limit], "total": len(rules)}


@app.post("/api/recommend")
def get_recommendations(req: RecommendRequest):
    _ensure_trained()
    if not _cache["rec_engine"]:
        raise HTTPException(status_code=400, detail="Recommendation engine not initialised.")
    return _cache["rec_engine"].recommend(cart=req.cart, n=req.n_recommendations)


@app.post("/api/recommend/product")
def get_product_recommendations(req: ProductRequest):
    _ensure_trained()
    return _cache["rec_engine"].recommend(cart=[req.product], n=req.n_recommendations)


@app.get("/api/graph")
def get_product_graph(top_n: int = 50):
    _ensure_trained()
    rules = _cache["analysis_results"]["rules"][:top_n]
    return build_product_graph(rules, top_n=top_n)


@app.get("/api/seasonality")
def get_seasonality():
    _ensure_trained()
    return _cache["analysis_results"]["seasonality"]


@app.get("/api/segments")
def get_customer_segments():
    _ensure_trained()
    segments = _cache["analysis_results"]["customer_segments"]
    seg_summary: Dict = {}
    for cust in segments:
        seg = cust["segment"]
        if seg not in seg_summary:
            seg_summary[seg] = {"count": 0, "avg_spend": 0, "avg_transactions": 0}
        seg_summary[seg]["count"]            += 1
        seg_summary[seg]["avg_spend"]        += cust["total_spend"]
        seg_summary[seg]["avg_transactions"] += cust["n_transactions"]
    for seg in seg_summary:
        n = seg_summary[seg]["count"]
        seg_summary[seg]["avg_spend"]        = round(seg_summary[seg]["avg_spend"] / n, 2)
        seg_summary[seg]["avg_transactions"] = round(seg_summary[seg]["avg_transactions"] / n, 2)
    return {"segments": seg_summary, "customers": segments[:100]}


@app.get("/api/model-comparison")
def get_model_comparison():
    _ensure_trained()
    return _cache["analysis_results"]["model_comparison"]


@app.get("/api/products")
def get_all_products():
    _ensure_trained()
    return {"products": list(_cache["analysis_results"]["summary"]["top_products"].keys())}


@app.on_event("startup")
def startup_event():
    if os.path.exists(DEFAULT_DATASET):
        print(f"🚀 Auto-training with: {DEFAULT_DATASET}")
        try:
            results = run_full_analysis(DEFAULT_DATASET, {
                "min_support": 0.01, "min_confidence": 0.2, "max_len": 3
            })
            _cache["analysis_results"] = {
                "summary":           results["summary"],
                "model_comparison":  {
                    k: {**v, "top_rules": v["top_rules"][:5]}
                    for k, v in results["model_comparison"].items()
                },
                "rules":             results["rules"],
                "product_graph":     results["product_graph"],
                "seasonality":       results["seasonality"],
                "customer_segments": results["customer_segments"],
            }
            _cache["rec_engine"]        = results["recommendation_engine"]
            _cache["last_trained"]      = time.strftime("%Y-%m-%d %H:%M:%S")
            _cache["training_status"]   = "ready"
            _cache["training_progress"] = 100
            print(f"✅ Auto-training complete — {len(results['rules'])} rules found.")
        except Exception as e:
            print(f"⚠️  Auto-training failed: {e}")
    else:
        print(f"⚠️  Dataset not found at {DEFAULT_DATASET}. POST /api/train to begin.")

if __name__ == "__main__":
    import uvicorn
    print(f"\n📂 App directory : {THIS_DIR}")
    print(f"📊 Data directory: {DATA_DIR}")
    print(f"📄 Dataset       : {DEFAULT_DATASET}")
    print(f"\n🌐 Open the UI    → http://localhost:8000")
    print(f"📖 API docs       → http://localhost:8000/docs\n")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)