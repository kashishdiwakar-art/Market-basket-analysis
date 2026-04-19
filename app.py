"""
Market Basket Analysis — FastAPI Backend + Self-Contained UI
============================================================
✅ Pure vanilla HTML/CSS/JS frontend — zero CDN, zero black screen
✅ Open http://localhost:8000  →  full dashboard appears instantly
✅ Auto-finds ml_engine.py from any working directory
✅ Auto-generates transactions.csv if missing

HOW TO RUN:
    cd MARKET_ANALYSIS
    python app.py
    → open http://localhost:8000
"""

import os, sys, time, shutil, traceback

# ── Path setup ────────────────────────────────────────────────
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

# ── Auto-generate dataset if missing ─────────────────────────
if not os.path.exists(DEFAULT_DATASET):
    print("Generating sample dataset...")
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
        "Laptop":["Mouse","Keyboard","Laptop Stand","USB Hub","Monitor"],
        "Coffee":["Dark Chocolate","Granola Bar","Notebook"],
        "Protein Shake":["Protein Bar","Gym Gloves","Resistance Band"],
        "Yoga Mat":["Foam Roller","Resistance Band","Water Bottle"],
        "Vitamins":["Omega-3","Probiotics","Protein Powder"],
        "Headphones":["Laptop","Speaker","USB Hub"],
    }
    SEGS = {
        "tech_professional":{"cats":["Electronics","Office","Beverages"],"w":0.25},
        "health_enthusiast":{"cats":["Health","Fitness","Beverages","Snacks"],"w":0.30},
        "office_worker":{"cats":["Office","Beverages","Snacks"],"w":0.25},
        "student":{"cats":["Electronics","Office","Snacks","Beverages"],"w":0.20},
    }
    sn=list(SEGS.keys()); sw=[SEGS[s]["w"] for s in sn]
    start=datetime(2023,1,1); records=[]
    for tid in range(1,2001):
        date=start+timedelta(days=random.randint(0,364))
        seg=random.choices(sn,weights=sw)[0]
        cid=f"CUST_{random.randint(1,500):04d}"
        basket=list(set([random.choice(PRODUCTS[random.choice(SEGS[seg]["cats"])]) for _ in range(random.randint(2,8))]))
        for p in basket[:]:
            if p in ASSOC and random.random()<0.6: basket.append(random.choice(ASSOC[p]))
        for prod in set(basket):
            records.append({"transaction_id":f"TXN_{tid:05d}","customer_id":cid,"product":prod,
                "date":date.strftime("%Y-%m-%d"),"month":date.month,"day_of_week":date.strftime("%A"),
                "customer_segment":seg,"quantity":random.randint(1,3),"price":round(random.uniform(5,200),2)})
    pd.DataFrame(records).to_csv(DEFAULT_DATASET,index=False)
    print(f"Dataset saved to {DEFAULT_DATASET}")

# ── ML Engine import ──────────────────────────────────────────
from ml_engine import (
    load_transactions, run_full_analysis, RecommendationEngine,
    rank_rules, build_product_graph, segment_customers,
    analyze_seasonality, compare_models, FPGrowth,
)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Optional

app = FastAPI(title="BasketAI API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

_cache: Dict = {
    "analysis_results": None, "rec_engine": None, "last_trained": None,
    "training_status": "idle", "training_progress": 0,
    "dataset_path": DEFAULT_DATASET,
}

class TrainRequest(BaseModel):
    min_support: float = 0.01
    min_confidence: float = 0.20
    max_len: int = 3

class RecommendRequest(BaseModel):
    cart: List[str]
    n_recommendations: int = 5

class ProductRequest(BaseModel):
    product: str
    n_recommendations: int = 5

def _ensure_trained():
    if _cache["analysis_results"] is None:
        raise HTTPException(status_code=400, detail="Model not trained yet.")

# ══════════════════════════════════════════════════════════════
#  SELF-CONTAINED UI  —  pure HTML + CSS + vanilla JS
#  No React, No Babel, No CDN — renders instantly in any browser
# ══════════════════════════════════════════════════════════════
UI_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>BasketAI</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#05080f;--s1:#0c1120;--s2:#111827;--bdr:rgba(99,179,255,.13);
--acc:#3b82f6;--acc2:#06b6d4;--pur:#8b5cf6;--gold:#f59e0b;--grn:#10b981;--red:#ef4444;
--tx:#e2e8f0;--mu:#64748b;--glass:rgba(12,17,32,.8)}
body{font-family:system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--tx);min-height:100vh;overflow-x:hidden}
.grid-bg{position:fixed;inset:0;pointer-events:none;z-index:0;
background-image:linear-gradient(var(--bdr) 1px,transparent 1px),linear-gradient(90deg,var(--bdr) 1px,transparent 1px);
background-size:44px 44px}
.orb{position:fixed;border-radius:50%;filter:blur(90px);pointer-events:none;z-index:0;opacity:.22}
.o1{width:440px;height:440px;background:#3b82f6;top:-120px;left:-120px}
.o2{width:320px;height:320px;background:#8b5cf6;bottom:-60px;right:-60px}
.app{display:flex;flex-direction:column;min-height:100vh;position:relative;z-index:1}
nav{display:flex;align-items:center;justify-content:space-between;padding:0 28px;height:60px;
background:var(--glass);backdrop-filter:blur(18px);border-bottom:1px solid var(--bdr);
position:sticky;top:0;z-index:100}
.logo{font-weight:800;font-size:1.3rem;background:linear-gradient(135deg,#3b82f6,#06b6d4);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-.5px}
.nav-tabs{display:flex;gap:4px}
.tb{padding:6px 15px;border-radius:8px;border:none;cursor:pointer;font-size:.84rem;font-weight:500;
background:transparent;color:var(--mu);transition:all .18s}
.tb:hover{color:var(--tx);background:var(--s2)}
.tb.active{color:var(--acc);background:rgba(59,130,246,.13)}
.tb:disabled{opacity:.35;cursor:default}
.sp{display:flex;align-items:center;gap:6px;font-size:.78rem;color:var(--mu)}
.dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.d-idle{background:var(--mu)}.d-train{background:var(--gold);animation:blink 1s infinite}
.d-ready{background:var(--grn);box-shadow:0 0 7px var(--grn);animation:blink 2.5s infinite}
.d-err{background:var(--red)}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.4}}
main{flex:1;padding:28px 32px;max-width:1360px;width:100%;margin:0 auto}
footer{padding:14px;border-top:1px solid var(--bdr);color:var(--mu);font-size:.74rem;text-align:center;background:var(--glass)}
.page{animation:fin .3s ease}
@keyframes fin{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.card{background:var(--glass);backdrop-filter:blur(18px);border:1px solid var(--bdr);
border-radius:16px;padding:22px;margin-bottom:20px;transition:border-color .2s}
.card:hover{border-color:rgba(59,130,246,.28)}
.ct{font-weight:700;font-size:.95rem;margin-bottom:16px;display:flex;align-items:center;gap:8px}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:780px){.g2{grid-template-columns:1fr}}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:14px;margin-bottom:22px}
.stat{background:var(--s2);border:1px solid var(--bdr);border-radius:14px;padding:18px;position:relative;overflow:hidden}
.stat::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--acc),var(--acc2))}
.sl{font-size:.71rem;color:var(--mu);text-transform:uppercase;letter-spacing:.07em}
.sv{font-size:1.85rem;font-weight:800;line-height:1;margin:4px 0}
.ss{font-size:.71rem;color:var(--mu)}
.algo-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin-bottom:20px}
.ac{background:var(--s2);border:1px solid var(--bdr);border-radius:14px;padding:18px;position:relative;overflow:hidden}
.ag{position:absolute;top:-10px;right:-10px;width:80px;height:80px;border-radius:50%;filter:blur(40px);opacity:.35}
.an{font-weight:800;font-size:1rem;margin-bottom:10px}
.ar{display:flex;justify-content:space-between;margin-bottom:5px;font-size:.79rem}
.al{color:var(--mu)}.av{font-weight:700}
.bar-chart{display:flex;align-items:flex-end;gap:14px;height:150px;background:var(--s2);
border-radius:12px;padding:16px 20px;margin-top:14px}
.bg{display:flex;flex-direction:column;align-items:center;gap:6px;flex:1;height:100%}
.bw{display:flex;align-items:flex-end;gap:4px;flex:1;width:100%}
.bar{border-radius:5px 5px 0 0;flex:1;min-height:3px;transition:height .5s}
.bl{font-size:.7rem;color:var(--mu);text-align:center;line-height:1.3}
.seg-row{display:flex;align-items:center;gap:12px;margin-bottom:10px}
.sn{font-size:.81rem;min-width:148px}.st{flex:1;height:8px;background:var(--s2);border-radius:4px;overflow:hidden}
.sf{height:100%;border-radius:4px;transition:width .6s}.scount{font-size:.74rem;color:var(--mu);min-width:26px;text-align:right}
.trend-wrap{overflow-x:auto;padding-bottom:6px}
.trend-bc{display:flex;align-items:flex-end;gap:5px;height:130px;min-width:560px;padding:0 4px}
.mc{display:flex;flex-direction:column;align-items:center;gap:3px;flex:1}
.mb{display:flex;align-items:flex-end;gap:2px;height:100%;width:100%}
.mb-bar{flex:1;border-radius:3px 3px 0 0;min-height:2px}
.ml{font-size:.65rem;color:var(--mu)}
.tbl-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:.8rem}
th{text-align:left;padding:9px 11px;border-bottom:1px solid var(--bdr);color:var(--mu);
font-weight:600;text-transform:uppercase;letter-spacing:.06em;font-size:.69rem;white-space:nowrap}
td{padding:9px 11px;border-bottom:1px solid rgba(99,179,255,.05);vertical-align:middle}
tr:hover td{background:rgba(59,130,246,.04)}
.lb{height:4px;background:var(--s2);border-radius:2px;min-width:52px}
.lf{height:100%;border-radius:2px}
.tag{display:inline-flex;align-items:center;padding:2px 9px;border-radius:20px;font-size:.69rem;font-weight:600;margin:1px}
.t-b{background:rgba(59,130,246,.15);color:#60a5fa;border:1px solid rgba(59,130,246,.3)}
.t-g{background:rgba(16,185,129,.15);color:#34d399;border:1px solid rgba(16,185,129,.3)}
.t-p{background:rgba(139,92,246,.15);color:#a78bfa;border:1px solid rgba(139,92,246,.3)}
.t-gold{background:rgba(245,158,11,.15);color:#fbbf24;border:1px solid rgba(245,158,11,.3)}
.t-c{background:rgba(6,182,212,.15);color:#22d3ee;border:1px solid rgba(6,182,212,.3)}
.btn{padding:9px 18px;border-radius:10px;border:none;cursor:pointer;font-weight:600;
font-size:.87rem;transition:all .18s;display:inline-flex;align-items:center;gap:7px}
.bp{background:linear-gradient(135deg,#3b82f6,#06b6d4);color:#fff;box-shadow:0 4px 18px rgba(59,130,246,.3)}
.bp:hover{transform:translateY(-1px);box-shadow:0 6px 22px rgba(59,130,246,.4)}
.bp:disabled{opacity:.5;cursor:not-allowed;transform:none}
.bs{background:var(--s2);color:var(--tx);border:1px solid var(--bdr);padding:4px 10px;font-size:.75rem}
.bs:hover{border-color:var(--acc);color:var(--acc)}
input[type=range]{width:100%;accent-color:var(--acc);margin:4px 0}
.pr{margin-bottom:16px}
.pl{display:flex;justify-content:space-between;font-size:.8rem;color:var(--mu);margin-bottom:5px}
.pv{font-weight:700;color:var(--acc)}
input[type=text],input[type=search]{background:var(--s2);border:1px solid var(--bdr);border-radius:9px;
padding:9px 13px;color:var(--tx);font-size:.87rem;width:100%;outline:none;transition:border-color .18s;font-family:inherit}
input[type=text]:focus,input[type=search]:focus{border-color:var(--acc);box-shadow:0 0 0 3px rgba(59,130,246,.1)}
input::placeholder{color:var(--mu)}
.uz{border:2px dashed var(--bdr);border-radius:14px;padding:38px;text-align:center;cursor:pointer;transition:all .2s}
.uz:hover,.uz.over{border-color:var(--acc);background:rgba(59,130,246,.05)}
.ui{font-size:2.6rem;margin-bottom:12px}.ut{font-weight:700;font-size:1.02rem;margin-bottom:5px}.us{color:var(--mu);font-size:.82rem}
.cz{min-height:96px;border:2px dashed var(--bdr);border-radius:13px;padding:13px;
display:flex;flex-wrap:wrap;gap:8px;align-content:flex-start;transition:border-color .2s,background .2s}
.cz.over{border-color:var(--acc);background:rgba(59,130,246,.06)}
.chip{padding:5px 13px;border-radius:20px;font-size:.79rem;font-weight:600;cursor:grab;user-select:none;
display:flex;align-items:center;gap:5px;transition:transform .12s}
.chip:hover{transform:scale(1.04)}
.crm{background:none;border:none;cursor:pointer;color:var(--mu);font-size:.77rem;padding:0 2px;line-height:1}
.crm:hover{color:var(--red)}
.c0{background:rgba(59,130,246,.2);color:#93c5fd;border:1px solid rgba(59,130,246,.3)}
.c1{background:rgba(139,92,246,.2);color:#c4b5fd;border:1px solid rgba(139,92,246,.3)}
.c2{background:rgba(6,182,212,.2);color:#67e8f9;border:1px solid rgba(6,182,212,.3)}
.c3{background:rgba(16,185,129,.2);color:#6ee7b7;border:1px solid rgba(16,185,129,.3)}
.c4{background:rgba(245,158,11,.2);color:#fcd34d;border:1px solid rgba(245,158,11,.3)}
.rec-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(265px,1fr));gap:12px}
.rc{background:var(--s2);border:1px solid var(--bdr);border-radius:12px;padding:16px;
transition:all .18s;display:flex;flex-direction:column;gap:8px}
.rc:hover{border-color:var(--acc);transform:translateY(-2px)}
.rn{font-weight:700;font-size:.95rem}.re{font-size:.76rem;color:var(--mu);line-height:1.55}
.mr{display:flex;gap:6px;flex-wrap:wrap}
.graph-box{width:100%;height:390px;background:var(--s2);border-radius:12px;overflow:hidden}
.alert{padding:11px 15px;border-radius:10px;font-size:.81rem;margin-top:11px}
.a-ok{background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.3);color:#34d399}
.a-err{background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);color:#f87171}
.a-info{background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.3);color:#60a5fa}
.pb{height:6px;background:var(--s2);border-radius:3px;overflow:hidden;margin-top:10px}
.pf{height:100%;border-radius:3px;background:linear-gradient(90deg,var(--acc),var(--acc2));animation:shi 1.5s infinite}
@keyframes shi{0%{filter:brightness(1)}50%{filter:brightness(1.3)}100%{filter:brightness(1)}}
@keyframes spin{to{transform:rotate(360deg)}}
.spin{display:inline-block;animation:spin .9s linear infinite}
.empty{display:flex;flex-direction:column;align-items:center;justify-content:center;
height:280px;gap:12px;color:var(--mu);text-align:center}
.ei{font-size:2.5rem}.et{font-size:1.15rem;font-weight:700;color:var(--tx)}
</style>
</head>
<body>
<div class="grid-bg"></div>
<div class="orb o1"></div><div class="orb o2"></div>
<div class="app">

<nav>
  <div class="logo">&#9672; BasketAI</div>
  <div class="nav-tabs">
    <button class="tb active" id="btn-setup"     onclick="show('setup')">&#128228; Setup</button>
    <button class="tb" id="btn-dashboard" disabled onclick="show('dashboard')">&#128202; Dashboard</button>
    <button class="tb" id="btn-recommend" disabled onclick="show('recommend')">&#128302; Recommend</button>
  </div>
  <div class="sp"><div class="dot d-idle" id="sdot"></div><span id="stxt">Idle</span></div>
</nav>

<main>

<!-- SETUP -->
<div id="pg-setup" class="page">
  <h2 style="font-size:1.65rem;font-weight:800;margin-bottom:6px">Dataset Setup</h2>
  <p style="color:var(--mu);margin-bottom:22px">Use the built-in demo dataset or upload your own CSV, then train all three algorithms.</p>
  <div class="g2">
    <div>
      <div class="card">
        <div class="ct">&#128193; Upload Dataset</div>
        <div class="uz" id="dz" onclick="document.getElementById('fi').click()"
          ondragover="event.preventDefault();this.classList.add('over')"
          ondragleave="this.classList.remove('over')"
          ondrop="dropFile(event)">
          <div class="ui">&#128202;</div>
          <div class="ut" id="flbl">Drop CSV here or click to browse</div>
          <div class="us">Required: transaction_id, product, date</div>
        </div>
        <input type="file" id="fi" accept=".csv" style="display:none" onchange="selFile(this)"/>
        <div id="umsg"></div>
      </div>
      <div class="card" style="background:rgba(16,185,129,.05);border-color:rgba(16,185,129,.2)">
        <div class="ct" style="color:#34d399">&#9989; Demo Dataset Ready</div>
        <p style="font-size:.82rem;color:var(--mu);margin-bottom:11px">2,000 transactions &middot; 59 products &middot; 6 categories &middot; 12-month data</p>
        <div style="display:flex;flex-wrap:wrap;gap:6px">
          <span class="tag t-g">Electronics</span><span class="tag t-c">Beverages</span>
          <span class="tag t-p">Health</span><span class="tag t-b">Office</span>
          <span class="tag t-gold">Snacks</span><span class="tag t-g">Fitness</span>
        </div>
      </div>
    </div>
    <div class="card">
      <div class="ct">&#9881;&#65039; Training Parameters</div>
      <div class="pr"><div class="pl">Min Support <span class="pv" id="lsup">0.01</span></div>
        <input type="range" min=".005" max=".1" step=".005" value=".01" id="isup"
          oninput="document.getElementById('lsup').textContent=this.value"/>
        <div style="display:flex;justify-content:space-between;font-size:.67rem;color:var(--mu)"><span>0.005 more rules</span><span>0.1 fewer</span></div>
      </div>
      <div class="pr"><div class="pl">Min Confidence <span class="pv" id="lconf">0.20</span></div>
        <input type="range" min=".1" max=".9" step=".05" value=".2" id="iconf"
          oninput="document.getElementById('lconf').textContent=parseFloat(this.value).toFixed(2)"/>
      </div>
      <div class="pr"><div class="pl">Max Itemset Length <span class="pv" id="llen">3</span></div>
        <input type="range" min="2" max="5" step="1" value="3" id="ilen"
          oninput="document.getElementById('llen').textContent=this.value"/>
      </div>
      <button class="btn bp" style="width:100%;justify-content:center;margin-top:8px" id="tbtn" onclick="doTrain()">
        &#128640; Train All Models
      </button>
      <div id="tprog" style="display:none">
        <div style="font-size:.77rem;color:var(--mu);margin-top:10px">Running Apriori &#8594; FP-Growth &#8594; ECLAT&#8230;</div>
        <div class="pb"><div class="pf" style="width:65%"></div></div>
      </div>
      <div id="tmsg"></div>
    </div>
  </div>
</div>

<!-- DASHBOARD -->
<div id="pg-dashboard" class="page" style="display:none">
  <h2 style="font-size:1.65rem;font-weight:800;margin-bottom:6px">Insights Dashboard</h2>
  <p style="color:var(--mu);margin-bottom:22px">Live analysis of purchase patterns, algorithms and customer segments.</p>
  <div class="stats" id="stats"></div>
  <div class="card">
    <div class="ct">&#129504; Algorithm Performance</div>
    <div class="algo-grid" id="algog"></div>
    <div class="bar-chart" id="algobc"></div>
  </div>
  <div class="g2">
    <div class="card"><div class="ct">&#128197; Monthly Trends</div>
      <div class="trend-wrap"><div class="trend-bc" id="trend"></div></div>
      <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:10px;font-size:.73rem" id="tleg"></div>
    </div>
    <div class="card"><div class="ct">&#128101; Customer Segments</div><div id="segs"></div></div>
  </div>
  <div class="card">
    <div class="ct">&#128279; Association Rules</div>
    <div style="display:flex;gap:5px;margin-bottom:14px;flex-wrap:wrap">
      <button class="tb active" onclick="filtR('all',this)">All Rules</button>
      <button class="tb" onclick="filtR('Apriori',this)">Apriori</button>
      <button class="tb" onclick="filtR('FP-Growth',this)">FP-Growth</button>
      <button class="tb" onclick="filtR('ECLAT',this)">ECLAT</button>
    </div>
    <div class="tbl-wrap" id="rtbl"></div>
  </div>
</div>

<!-- RECOMMEND -->
<div id="pg-recommend" class="page" style="display:none">
  <h2 style="font-size:1.65rem;font-weight:800;margin-bottom:6px">Recommendation Engine</h2>
  <p style="color:var(--mu);margin-bottom:22px">Click products to build your cart, then get AI-powered upsell &amp; cross-sell suggestions.</p>
  <div class="g2" style="margin-bottom:18px">
    <div class="card">
      <div class="ct">&#128230; Product Catalog</div>
      <input type="search" placeholder="Search products&#8230;" style="margin-bottom:12px" oninput="filtP(this.value)"/>
      <div style="display:flex;flex-wrap:wrap;gap:8px;max-height:245px;overflow-y:auto" id="catalog"></div>
    </div>
    <div class="card">
      <div class="ct" id="ctitle">&#128722; Your Cart</div>
      <div class="cz" id="czone"
        ondragover="event.preventDefault();this.classList.add('over')"
        ondragleave="this.classList.remove('over')"
        ondrop="cartDrop(event)">
        <span style="color:var(--mu);font-size:.83rem;margin:auto" id="chint">Click products above or drag &amp; drop here</span>
      </div>
      <button class="btn bp" style="margin-top:12px;width:100%;justify-content:center;display:none"
        onclick="getRecs()" id="rbtn">&#128302; Get Recommendations</button>
    </div>
  </div>
  <div id="rout"></div>
  <div class="card" style="margin-top:18px">
    <div class="ct">&#128376;&#65039; Product Relationship Network <span class="tag t-p" id="gmeta"></span></div>
    <p style="font-size:.8rem;color:var(--mu);margin-bottom:12px">Node size = connection count &middot; Edge opacity = lift strength</p>
    <div class="graph-box"><svg id="nsvg" width="100%" height="390" viewBox="0 0 800 390"></svg></div>
  </div>
</div>

</main>
<footer>BasketAI &middot; Market Basket Intelligence &middot; Apriori + FP-Growth + ECLAT</footer>
</div>

<script>
const AC={'Apriori':'#3b82f6','FP-Growth':'#06b6d4','ECLAT':'#8b5cf6'};
const SC=['#3b82f6','#06b6d4','#8b5cf6','#f59e0b'];
const CC=['c0','c1','c2','c3','c4'];
let RULES=[],PRODS=[],CART=[],DRAG=null,TRAINED=false;

/* ── Nav ── */
function show(p){
  ['setup','dashboard','recommend'].forEach(n=>{
    document.getElementById('pg-'+n).style.display=n===p?'':'none';
    const b=document.getElementById('btn-'+n);
    if(b) b.classList.toggle('active',n===p);
  });
}

/* ── Status ── */
async function poll(){
  try{
    const d=await fetch('/api/status').then(r=>r.json());
    const dot=document.getElementById('sdot');
    const txt=document.getElementById('stxt');
    const cls={ready:'d-ready',training:'d-train',idle:'d-idle',error:'d-err'};
    dot.className='dot '+(cls[d.training_status]||'d-idle');
    txt.textContent={ready:'Ready',training:'Training\u2026',idle:'Idle',error:'Error'}[d.training_status]||'Idle';
    if(d.training_status==='ready'&&!TRAINED){
      TRAINED=true;
      document.getElementById('btn-dashboard').disabled=false;
      document.getElementById('btn-recommend').disabled=false;
      loadDash();loadProds();loadGraph();
    }
  }catch(e){}
}
setInterval(poll,3000);poll();

/* ── Upload ── */
function dropFile(e){e.preventDefault();document.getElementById('dz').classList.remove('over');const f=e.dataTransfer.files[0];if(f)upFile(f);}
function selFile(i){if(i.files[0])upFile(i.files[0]);}
async function upFile(f){
  document.getElementById('flbl').textContent=f.name;
  const fd=new FormData();fd.append('file',f);
  const m=document.getElementById('umsg');
  m.innerHTML='<div class="alert a-info">Uploading\u2026</div>';
  try{
    const d=await fetch('/api/upload',{method:'POST',body:fd}).then(r=>r.json());
    if(d.detail) throw new Error(d.detail);
    m.innerHTML=`<div class="alert a-ok">&#9989; ${d.rows} rows &middot; ${d.columns.length} columns uploaded</div>`;
  }catch(e){m.innerHTML=`<div class="alert a-err">&#10060; ${e.message}</div>`;}
}

/* ── Train ── */
async function doTrain(){
  const btn=document.getElementById('tbtn');
  const prog=document.getElementById('tprog');
  const msg=document.getElementById('tmsg');
  btn.disabled=true;btn.innerHTML='<span class="spin">&#8635;</span> Training\u2026';
  prog.style.display='';msg.innerHTML='';
  const params={
    min_support:parseFloat(document.getElementById('isup').value),
    min_confidence:parseFloat(document.getElementById('iconf').value),
    max_len:parseInt(document.getElementById('ilen').value),
  };
  try{
    const d=await fetch('/api/train',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(params)}).then(r=>r.json());
    if(d.detail) throw new Error(d.detail);
    prog.style.display='none';
    const times=Object.entries(d.training_time||{}).map(([a,t])=>`${a}: ${t}s`).join(' &middot; ');
    msg.innerHTML=`<div class="alert a-ok">&#9989; ${d.n_rules} rules found &middot; ${times}</div>`;
    TRAINED=true;
    document.getElementById('btn-dashboard').disabled=false;
    document.getElementById('btn-recommend').disabled=false;
    poll();loadDash();loadProds();loadGraph();
    setTimeout(()=>show('dashboard'),700);
  }catch(e){prog.style.display='none';msg.innerHTML=`<div class="alert a-err">&#10060; ${e.message}</div>`;}
  btn.disabled=false;btn.innerHTML='&#128640; Train All Models';
}

/* ── Dashboard ── */
async function loadDash(){
  try{
    const [sum,cmp,sea,seg,rls]=await Promise.all([
      fetch('/api/summary').then(r=>r.json()),
      fetch('/api/model-comparison').then(r=>r.json()),
      fetch('/api/seasonality').then(r=>r.json()),
      fetch('/api/segments').then(r=>r.json()),
      fetch('/api/rules?limit=50').then(r=>r.json()),
    ]);
    renderStats(sum);renderAlgo(cmp);renderTrend(sea);renderSegs(seg);
    RULES=rls.rules||[];renderRules('all');
  }catch(e){console.error(e);}
}

function renderStats(s){
  document.getElementById('stats').innerHTML=[
    {i:'&#128722;',l:'Transactions',v:(s.total_transactions||0).toLocaleString(),s:'total baskets'},
    {i:'&#128101;',l:'Customers',v:(s.total_customers||0).toLocaleString(),s:'unique buyers'},
    {i:'&#128230;',l:'Products',v:s.total_products||0,s:'unique items'},
    {i:'&#128279;',l:'Rules',v:s.total_rules||0,s:'association rules'},
    {i:'&#128202;',l:'Avg Basket',v:(s.avg_basket_size||0).toFixed(2),s:'items per order'},
  ].map(x=>`<div class="stat"><div style="font-size:1.35rem;margin-bottom:7px">${x.i}</div>
    <div class="sl">${x.l}</div><div class="sv">${x.v}</div><div class="ss">${x.s}</div></div>`).join('');
}

function renderAlgo(cmp){
  const mr=Math.max(...Object.values(cmp).map(d=>d.n_rules),1);
  const mi=Math.max(...Object.values(cmp).map(d=>d.n_frequent_itemsets),1);
  document.getElementById('algog').innerHTML=Object.entries(cmp).map(([algo,d])=>`
    <div class="ac"><div class="ag" style="background:${AC[algo]}"></div>
    <div class="an" style="color:${AC[algo]}">${algo}</div>
    ${[['Itemsets',d.n_frequent_itemsets],['Rules',d.n_rules],['Time',d.training_time_sec+'s'],
       ['Avg Lift',(d.avg_lift||0).toFixed(2)],['Max Lift',(d.max_lift||0).toFixed(2)]]
      .map(([l,v])=>`<div class="ar"><span class="al">${l}</span><span class="av" style="color:${AC[algo]}">${v}</span></div>`).join('')}
    </div>`).join('');
  document.getElementById('algobc').innerHTML=Object.entries(cmp).map(([algo,d])=>{
    const rp=(d.n_rules/mr*100).toFixed(1);
    const ip=(d.n_frequent_itemsets/mi*100).toFixed(1);
    return `<div class="bg"><div class="bw">
      <div class="bar" style="height:${rp}%;background:${AC[algo]};opacity:.9" title="Rules: ${d.n_rules}"></div>
      <div class="bar" style="height:${ip}%;background:${AC[algo]};opacity:.45" title="Itemsets: ${d.n_frequent_itemsets}"></div>
    </div><div class="bl">${algo}<br><b style="color:${AC[algo]}">${d.n_rules}</b> rules</div></div>`;
  }).join('');
}

function renderTrend(sea){
  const data=sea.trend_data||[];
  const prods=(sea.top_products||[]).slice(0,3);
  const cols=['#3b82f6','#06b6d4','#8b5cf6'];
  if(!data.length){document.getElementById('trend').innerHTML='<div style="color:var(--mu);padding:16px">No data</div>';return;}
  const mx=Math.max(...data.flatMap(m=>prods.map(p=>m[p]||0)),1);
  document.getElementById('trend').innerHTML=data.map(m=>`
    <div class="mc"><div class="mb">
      ${prods.map((p,i)=>`<div class="mb-bar" style="height:${Math.max(2,((m[p]||0)/mx*100)).toFixed(1)}%;background:${cols[i]}"></div>`).join('')}
    </div><div class="ml">${m.month}</div></div>`).join('');
  document.getElementById('tleg').innerHTML=prods.map((p,i)=>
    `<span style="display:flex;align-items:center;gap:5px">
      <span style="width:9px;height:9px;border-radius:50%;background:${cols[i]};display:inline-block"></span>${p}
    </span>`).join('');
}

function renderSegs(seg){
  const segs=seg.segments||{};
  const mx=Math.max(...Object.values(segs).map(d=>d.count),1);
  document.getElementById('segs').innerHTML=Object.entries(segs).map(([name,d],i)=>`
    <div class="seg-row">
      <div class="sn">${name}</div>
      <div class="st"><div class="sf" style="width:${(d.count/mx*100).toFixed(1)}%;background:${SC[i%4]}"></div></div>
      <div class="scount">${d.count}</div>
    </div>
    <div style="font-size:.7rem;color:var(--mu);margin-bottom:11px;padding-left:160px">
      $${(d.avg_spend||0).toFixed(0)} avg &middot; ${(d.avg_transactions||0).toFixed(1)} orders</div>`).join('');
}

function filtR(algo,btn){
  document.querySelectorAll('[onclick^="filtR"]').forEach(b=>b.classList.remove('active'));
  if(btn) btn.classList.add('active');
  renderRules(algo);
}

function renderRules(algo){
  const fr=algo==='all'?RULES:RULES.filter(r=>r.algorithm===algo);
  if(!fr.length){document.getElementById('rtbl').innerHTML='<div style="color:var(--mu);padding:20px;text-align:center">No rules.</div>';return;}
  const mx=Math.max(...fr.map(r=>r.lift),1);
  document.getElementById('rtbl').innerHTML=`<table>
    <thead><tr><th>IF</th><th>THEN</th><th>Support</th><th>Confidence</th><th>Lift</th><th>Score</th><th>Algorithm</th></tr></thead>
    <tbody>${fr.slice(0,40).map(r=>{
      const lc=r.lift>=3?'#10b981':r.lift>=2?'#3b82f6':'var(--mu)';
      const lp=Math.min((r.lift/mx)*100,100).toFixed(1);
      return `<tr>
        <td>${r.antecedents.map(a=>`<span class="tag t-b">${a}</span>`).join('')}</td>
        <td>${r.consequents.map(c=>`<span class="tag t-g">${c}</span>`).join('')}</td>
        <td style="font-weight:600">${(r.support*100).toFixed(1)}%</td>
        <td style="font-weight:600">${(r.confidence*100).toFixed(1)}%</td>
        <td><div style="display:flex;align-items:center;gap:7px">
          <span style="font-weight:700;color:${lc}">${r.lift.toFixed(2)}</span>
          <div class="lb"><div class="lf" style="width:${lp}%;background:${lc}"></div></div>
        </div></td>
        <td><div style="width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;
          font-size:.6rem;font-weight:700;background:conic-gradient(var(--acc) ${((r.interestingness||0)*360).toFixed(0)}deg,var(--s2) 0)">
          ${((r.interestingness||0)*100).toFixed(0)}</div></td>
        <td><span class="tag" style="background:${AC[r.algorithm]||'#333'}22;color:${AC[r.algorithm]||'#aaa'};
          border:1px solid ${AC[r.algorithm]||'#555'}44">${r.algorithm}</span></td>
      </tr>`;
    }).join('')}</tbody></table>`;
}

/* ── Products & Cart ── */
async function loadProds(){
  try{const d=await fetch('/api/products').then(r=>r.json());PRODS=d.products||[];drawCatalog(PRODS);}catch(e){}
}
function drawCatalog(ps){
  document.getElementById('catalog').innerHTML=ps.filter(p=>!CART.includes(p))
    .map((p,i)=>`<div class="chip ${CC[i%5]}" draggable="true"
      ondragstart="DRAG='${esc(p)}'" ondragend="DRAG=null" onclick="addC('${esc(p)}')">${p} <span style="opacity:.5;font-size:.67rem">+</span></div>`).join('');
}
function filtP(q){drawCatalog(PRODS.filter(p=>p.toLowerCase().includes(q.toLowerCase())));}
function cartDrop(e){e.preventDefault();document.getElementById('czone').classList.remove('over');if(DRAG)addC(DRAG);DRAG=null;}
function addC(p){if(!CART.includes(p)){CART.push(p);drawCart();drawCatalog(PRODS);}}
function remC(p){CART=CART.filter(x=>x!==p);drawCart();drawCatalog(PRODS);if(!CART.length)document.getElementById('rout').innerHTML='';}
function esc(s){return s.replace(/'/g,"\\'");}

function drawCart(){
  const z=document.getElementById('czone');
  const h=document.getElementById('chint');
  const b=document.getElementById('rbtn');
  const t=document.getElementById('ctitle');
  t.innerHTML='&#128722; Your Cart'+(CART.length?` <span class="tag t-b">${CART.length} items</span>`:'');
  if(!CART.length){h.style.display='';b.style.display='none';z.querySelectorAll('.chip').forEach(c=>c.remove());return;}
  h.style.display='none';b.style.display='';
  z.querySelectorAll('.chip').forEach(c=>c.remove());
  CART.forEach((p,i)=>{
    const c=document.createElement('div');
    c.className=`chip ${CC[i%5]}`;
    c.innerHTML=`${p}<button class="crm" onclick="remC('${esc(p)}')">&#10005;</button>`;
    z.appendChild(c);
  });
  if(CART.length) getRecs();
}

async function getRecs(){
  const btn=document.getElementById('rbtn');
  btn.disabled=true;btn.innerHTML='<span class="spin">&#8635;</span> Analyzing\u2026';
  try{
    const d=await fetch('/api/recommend',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({cart:CART,n_recommendations:6})}).then(r=>r.json());
    drawRecs(d);
  }catch(e){document.getElementById('rout').innerHTML=`<div class="alert a-err">&#10060; ${e.message}</div>`;}
  btn.disabled=false;btn.innerHTML='&#128302; Get Recommendations';
}

function drawRecs(d){
  const up=d.upsell||[],cs=d.cross_sell||[];
  let h='';
  if(up.length) h+=`<div class="card"><div class="ct">&#11014;&#65039; Upsell Suggestions <span class="tag t-gold">High Value</span></div>
    <div class="rec-grid">${up.map(r=>`<div class="rc">
      <div style="display:flex;justify-content:space-between;align-items:flex-start">
        <div class="rn">${r.product}</div>
        <button class="btn bs" onclick="addC('${esc(r.product)}')">+ Add</button>
      </div>
      <div class="mr">
        <span class="tag t-gold">Lift ${r.lift.toFixed(2)}</span>
        <span class="tag t-b">Conf ${(r.confidence*100).toFixed(0)}%</span>
        <span class="tag t-g">Sup ${(r.support*100).toFixed(1)}%</span>
      </div>
      <div class="re">${r.explanation}</div>
      <div style="font-size:.69rem;color:var(--mu)">Trigger: ${(r.trigger||[]).join(', ')}</div>
    </div>`).join('')}</div></div>`;
  if(cs.length) h+=`<div class="card"><div class="ct">&#8596;&#65039; Cross-Sell Suggestions <span class="tag t-c">Complementary</span></div>
    <div class="rec-grid">${cs.map(r=>`<div class="rc">
      <div style="display:flex;justify-content:space-between;align-items:flex-start">
        <div class="rn">${r.product}</div>
        <button class="btn bs" onclick="addC('${esc(r.product)}')">+ Add</button>
      </div>
      <div class="mr">
        <span class="tag t-c">Lift ${r.lift.toFixed(2)}</span>
        <span class="tag t-b">Conf ${(r.confidence*100).toFixed(0)}%</span>
      </div>
      <div class="re">${r.explanation}</div>
    </div>`).join('')}</div></div>`;
  if(!up.length&&!cs.length) h=`<div class="card"><div class="empty">
    <div class="ei">&#129300;</div>
    <div class="et">No strong associations found</div>
    <div>Try adding more products or lowering the confidence threshold.</div>
  </div></div>`;
  document.getElementById('rout').innerHTML=h;
}

/* ── Network graph ── */
async function loadGraph(){
  try{
    const d=await fetch('/api/graph?top_n=45').then(r=>r.json());
    drawGraph(d);
    document.getElementById('gmeta').textContent=`${(d.nodes||[]).length} nodes \u00b7 ${(d.edges||[]).length} edges`;
  }catch(e){}
}
function drawGraph(data){
  const svg=document.getElementById('nsvg');
  const W=800,H=390;
  const nodes=data.nodes||[],edges=data.edges||[];
  if(!nodes.length){svg.innerHTML='';return;}
  const pos={};
  nodes.forEach((n,i)=>{
    const a=(i/nodes.length)*2*Math.PI,r=130+Math.random()*60;
    pos[n.id]={x:W/2+r*Math.cos(a),y:H/2+r*Math.sin(a),...n};
  });
  for(let it=0;it<60;it++){
    nodes.forEach(a=>nodes.forEach(b=>{
      if(a.id===b.id)return;
      const dx=pos[a.id].x-pos[b.id].x,dy=pos[a.id].y-pos[b.id].y;
      const d=Math.sqrt(dx*dx+dy*dy)+.1,f=900/(d*d);
      pos[a.id].x+=dx/d*f;pos[a.id].y+=dy/d*f;
    }));
    edges.forEach(e=>{
      if(!pos[e.source]||!pos[e.target])return;
      const dx=pos[e.target].x-pos[e.source].x,dy=pos[e.target].y-pos[e.source].y;
      const d=Math.sqrt(dx*dx+dy*dy)+.1;
      pos[e.source].x+=dx*.05;pos[e.source].y+=dy*.05;
      pos[e.target].x-=dx*.05;pos[e.target].y-=dy*.05;
    });
    Object.values(pos).forEach(p=>{p.x=Math.max(45,Math.min(W-45,p.x));p.y=Math.max(30,Math.min(H-30,p.y));});
  }
  const mc=Math.max(...nodes.map(n=>n.connections||1));
  const ml=Math.max(...edges.map(e=>e.weight||1));
  let s='';
  edges.forEach(e=>{
    if(!pos[e.source]||!pos[e.target])return;
    const op=(0.18+(e.weight/ml)*.65).toFixed(2);
    const sw=(1+(e.weight/ml)*2.5).toFixed(1);
    s+=`<line x1="${pos[e.source].x.toFixed(1)}" y1="${pos[e.source].y.toFixed(1)}"
      x2="${pos[e.target].x.toFixed(1)}" y2="${pos[e.target].y.toFixed(1)}"
      stroke="#3b82f6" stroke-width="${sw}" stroke-opacity="${op}"/>`;
  });
  nodes.forEach(n=>{
    const p=pos[n.id],r=6+(n.connections/mc)*15;
    const h=n.connections/mc,fill=h>.7?'#06b6d4':h>.4?'#3b82f6':'#8b5cf6';
    s+=`<g style="cursor:pointer">
      <circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="${r.toFixed(1)}"
        fill="${fill}" fill-opacity=".82" stroke="${fill}" stroke-width="1.5" stroke-opacity=".4"/>
      <text x="${p.x.toFixed(1)}" y="${(p.y+r+11).toFixed(1)}" text-anchor="middle"
        font-size="9" fill="rgba(226,232,240,.7)" font-family="system-ui">${n.label}</text>
    </g>`;
  });
  svg.innerHTML=s;
}
</script>
</body>
</html>"""

# ══════════════════════════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(content=UI_HTML)

@app.get("/ui", response_class=HTMLResponse)
def ui():
    return HTMLResponse(content=UI_HTML)

@app.get("/api/status")
def get_status():
    return {"training_status":_cache["training_status"],
            "training_progress":_cache["training_progress"],
            "last_trained":_cache["last_trained"],
            "has_results":_cache["analysis_results"] is not None}

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files supported.")
    save_path = os.path.join(DATA_DIR, f"up_{file.filename}")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    import pandas as pd
    try:
        df = pd.read_csv(save_path)
        _cache["dataset_path"] = save_path
        _cache["analysis_results"] = None
        _cache["rec_engine"] = None
        return {"message":"Uploaded","filename":file.filename,"rows":len(df),"columns":df.columns.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")

@app.post("/api/train")
def train_model(req: TrainRequest):
    if _cache["training_status"] == "training":
        return {"message":"Already training","status":"training"}
    _cache["training_status"] = "training"
    try:
        results = run_full_analysis(_cache["dataset_path"],
                    {"min_support":req.min_support,"min_confidence":req.min_confidence,"max_len":req.max_len})
        _cache["analysis_results"] = {
            "summary":results["summary"],
            "model_comparison":{k:{**v,"top_rules":v["top_rules"][:5]} for k,v in results["model_comparison"].items()},
            "rules":results["rules"],"product_graph":results["product_graph"],
            "seasonality":results["seasonality"],"customer_segments":results["customer_segments"]}
        _cache["rec_engine"]      = results["recommendation_engine"]
        _cache["last_trained"]    = time.strftime("%Y-%m-%d %H:%M:%S")
        _cache["training_status"] = "ready"
        _cache["training_progress"] = 100
        return {"message":"Done","status":"ready","summary":results["summary"],"n_rules":len(results["rules"]),
                "training_time":{k:v["training_time_sec"] for k,v in results["model_comparison"].items()}}
    except Exception as e:
        _cache["training_status"] = "error"
        raise HTTPException(status_code=500, detail=f"Training failed: {e}\n{traceback.format_exc()}")

@app.get("/api/summary")
def get_summary():
    _ensure_trained(); return _cache["analysis_results"]["summary"]

@app.get("/api/rules")
def get_rules(limit: int=50, algorithm: Optional[str]=None, min_lift: float=1.0):
    _ensure_trained()
    rules = _cache["analysis_results"]["rules"]
    if algorithm: rules=[r for r in rules if r["algorithm"].lower()==algorithm.lower()]
    rules=[r for r in rules if r["lift"]>=min_lift]
    return {"rules":rules[:limit],"total":len(rules)}

@app.post("/api/recommend")
def get_recommendations(req: RecommendRequest):
    _ensure_trained()
    return _cache["rec_engine"].recommend(cart=req.cart, n=req.n_recommendations)

@app.post("/api/recommend/product")
def get_product_recommendations(req: ProductRequest):
    _ensure_trained()
    return _cache["rec_engine"].recommend(cart=[req.product], n=req.n_recommendations)

@app.get("/api/graph")
def get_product_graph(top_n: int=50):
    _ensure_trained()
    return build_product_graph(_cache["analysis_results"]["rules"][:top_n], top_n=top_n)

@app.get("/api/seasonality")
def get_seasonality():
    _ensure_trained(); return _cache["analysis_results"]["seasonality"]

@app.get("/api/segments")
def get_customer_segments():
    _ensure_trained()
    segments=_cache["analysis_results"]["customer_segments"]
    ss: Dict={}
    for c in segments:
        sg=c["segment"]
        if sg not in ss: ss[sg]={"count":0,"avg_spend":0,"avg_transactions":0}
        ss[sg]["count"]+=1; ss[sg]["avg_spend"]+=c["total_spend"]; ss[sg]["avg_transactions"]+=c["n_transactions"]
    for sg in ss:
        n=ss[sg]["count"]; ss[sg]["avg_spend"]=round(ss[sg]["avg_spend"]/n,2); ss[sg]["avg_transactions"]=round(ss[sg]["avg_transactions"]/n,2)
    return {"segments":ss,"customers":segments[:100]}

@app.get("/api/model-comparison")
def get_model_comparison():
    _ensure_trained(); return _cache["analysis_results"]["model_comparison"]

@app.get("/api/products")
def get_all_products():
    _ensure_trained()
    return {"products":list(_cache["analysis_results"]["summary"]["top_products"].keys())}

@app.on_event("startup")
def startup_event():
    if os.path.exists(DEFAULT_DATASET):
        print(f"Auto-training with {DEFAULT_DATASET}")
        try:
            results=run_full_analysis(DEFAULT_DATASET,{"min_support":0.01,"min_confidence":0.2,"max_len":3})
            _cache["analysis_results"]={
                "summary":results["summary"],
                "model_comparison":{k:{**v,"top_rules":v["top_rules"][:5]} for k,v in results["model_comparison"].items()},
                "rules":results["rules"],"product_graph":results["product_graph"],
                "seasonality":results["seasonality"],"customer_segments":results["customer_segments"]}
            _cache["rec_engine"]      =results["recommendation_engine"]
            _cache["last_trained"]    =time.strftime("%Y-%m-%d %H:%M:%S")
            _cache["training_status"] ="ready"
            _cache["training_progress"]=100
            print(f"Auto-training complete — {len(results['rules'])} rules.")
        except Exception as e:
            print(f"Auto-training failed: {e}")

if __name__ == "__main__":
    import uvicorn
    print(f"\nFolder  : {THIS_DIR}")
    print(f"Dataset : {DEFAULT_DATASET}")
    print(f"\nUI      -> http://localhost:8000")
    print(f"API docs-> http://localhost:8000/docs\n")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
