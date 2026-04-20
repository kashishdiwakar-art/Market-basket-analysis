import os
import sys
import time
import random
import io
import matplotlib
matplotlib.use("Agg")                  
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from itertools import combinations
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


st.set_page_config(
    page_title="BasketAI — Market Basket Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


try:
    from ml_engine import (
        Apriori, FPGrowth, ECLAT,
        RecommendationEngine, rank_rules,
        segment_customers, analyze_seasonality,
        compare_models, build_product_graph,
    )
    ML_SOURCE = "ml_engine"
except ImportError:
   
    ML_SOURCE = "inline"

    class FPNode:
        def __init__(self, item, count=0, parent=None):
            self.item = item; self.count = count
            self.parent = parent; self.children = {}; self.link = None

    class Apriori:
        def __init__(self, min_support=0.02, min_confidence=0.3, max_len=4):
            self.min_support=min_support; self.min_confidence=min_confidence
            self.max_len=max_len; self.frequent_itemsets={}; self.rules=[]
        def _sup(self, fs, txs):
            return sum(1 for t in txs if fs.issubset(t)) / len(txs)
        def fit(self, basket_list):
            t0=time.time(); txs=[frozenset(b) for b in basket_list]; n=len(txs)
            ic=defaultdict(int)
            for t in txs:
                for item in t: ic[frozenset([item])]+=1
            L={fs:c/n for fs,c in ic.items() if c/n>=self.min_support}
            self.frequent_itemsets.update(L); Lk=list(L.keys()); k=2
            while Lk and k<=self.max_len:
                cands={}
                for i in range(len(Lk)):
                    for j in range(i+1,len(Lk)):
                        u=Lk[i]|Lk[j]
                        if len(u)==k:
                            s=self._sup(u,txs)
                            if s>=self.min_support: cands[u]=s
                self.frequent_itemsets.update(cands); Lk=list(cands.keys()); k+=1
            self.rules=self._rules(txs); self.training_time=time.time()-t0; return self
        def _rules(self, txs):
            rules=[]; n=len(txs)
            for fs,sup in self.frequent_itemsets.items():
                if len(fs)<2: continue
                for r in range(1,len(fs)):
                    for ant in combinations(fs,r):
                        ant=frozenset(ant); con=fs-ant
                        as_=self.frequent_itemsets.get(ant,self._sup(ant,txs))
                        cs_=self.frequent_itemsets.get(con,self._sup(con,txs))
                        if as_==0 or cs_==0: continue
                        conf=sup/as_; lift=conf/cs_
                        if conf>=self.min_confidence:
                            rules.append({"antecedents":list(ant),"consequents":list(con),
                                "support":round(sup,4),"confidence":round(conf,4),
                                "lift":round(lift,4),"leverage":round(sup-as_*cs_,4),
                                "conviction":round((1-cs_)/(1-conf+1e-9),4),"algorithm":"Apriori"})
            return sorted(rules,key=lambda x:x["lift"],reverse=True)

    class FPGrowth:
        def __init__(self, min_support=0.02, min_confidence=0.3, max_len=4):
            self.min_support=min_support; self.min_confidence=min_confidence
            self.max_len=max_len; self.frequent_itemsets={}; self.rules=[]
        def _build(self, txs, mc):
            freq=defaultdict(int)
            for t in txs:
                for i in t: freq[i]+=1
            freq={k:v for k,v in freq.items() if v>=mc}
            root=FPNode(None); header={i:None for i in freq}
            for t in txs:
                items=sorted([i for i in t if i in freq],key=lambda x:freq[x],reverse=True)
                self._ins(items,root,header)
            return root,header,freq
        def _ins(self,items,node,header):
            if not items: return
            item=items[0]
            if item in node.children: node.children[item].count+=1
            else:
                child=FPNode(item,1,node); node.children[item]=child
                curr=header[item]
                if curr is None: header[item]=child
                else:
                    while curr.link: curr=curr.link
                    curr.link=child
            self._ins(items[1:],node.children[item],header)
        def _cond(self,item,header):
            patterns=[]; node=header.get(item)
            while node:
                path=[]; p=node.parent
                while p and p.item is not None: path.append(p.item); p=p.parent
                if path: patterns.append((path,node.count))
                node=node.link
            return patterns
        def _mine(self,txs,prefix,mc,n):
            root,header,freq=self._build(txs,mc)
            for item,cnt in freq.items():
                np2=prefix+[item]; sup=cnt/n
                if sup>=self.min_support and len(np2)<=self.max_len:
                    self.frequent_itemsets[frozenset(np2)]=round(sup,4)
                    cond=[]
                    for path,count in self._cond(item,header): cond.extend([path]*count)
                    if cond: self._mine(cond,np2,mc,n)
        def fit(self,basket_list):
            t0=time.time(); n=len(basket_list); mc=int(self.min_support*n)
            self._mine(basket_list,[],mc,n)
            self.rules=self._rules(basket_list); self.training_time=time.time()-t0; return self
        def _rules(self,basket_list):
            txs=[frozenset(b) for b in basket_list]; n=len(txs); rules=[]
            for fs,sup in self.frequent_itemsets.items():
                if len(fs)<2: continue
                for r in range(1,len(fs)):
                    for ant in combinations(fs,r):
                        ant=frozenset(ant); con=fs-ant
                        as_=self.frequent_itemsets.get(ant,sum(1 for t in txs if ant.issubset(t))/n)
                        cs_=self.frequent_itemsets.get(con,sum(1 for t in txs if con.issubset(t))/n)
                        if as_==0 or cs_==0: continue
                        conf=sup/as_; lift=conf/cs_
                        if conf>=self.min_confidence:
                            rules.append({"antecedents":list(ant),"consequents":list(con),
                                "support":round(sup,4),"confidence":round(conf,4),
                                "lift":round(lift,4),"leverage":round(sup-as_*cs_,4),
                                "conviction":round((1-cs_)/(1-conf+1e-9),4),"algorithm":"FP-Growth"})
            return sorted(rules,key=lambda x:x["lift"],reverse=True)

    class ECLAT:
        def __init__(self, min_support=0.02, min_confidence=0.3, max_len=4):
            self.min_support=min_support; self.min_confidence=min_confidence
            self.max_len=max_len; self.frequent_itemsets={}; self.rules=[]
        def fit(self,basket_list):
            t0=time.time(); n=len(basket_list)
            tids=defaultdict(set)
            for tid,b in enumerate(basket_list):
                for item in b: tids[item].add(tid)
            filtered={k:v for k,v in tids.items() if len(v)/n>=self.min_support}
            self._rec([],filtered,n)
            self.rules=self._rules(basket_list); self.training_time=time.time()-t0; return self
        def _rec(self,prefix,items_tids,n):
            items=list(items_tids.keys())
            for i,item_i in enumerate(items):
                np2=prefix+[item_i]; tl=items_tids[item_i]; sup=len(tl)/n
                if sup>=self.min_support:
                    self.frequent_itemsets[frozenset(np2)]=round(sup,4)
                    if len(np2)<self.max_len:
                        ni={item_j:tl&items_tids[item_j] for item_j in items[i+1:] if len(tl&items_tids[item_j])/n>=self.min_support}
                        if ni: self._rec(np2,ni,n)
        def _rules(self,basket_list):
            txs=[frozenset(b) for b in basket_list]; n=len(txs); rules=[]
            for fs,sup in self.frequent_itemsets.items():
                if len(fs)<2: continue
                for r in range(1,len(fs)):
                    for ant in combinations(fs,r):
                        ant=frozenset(ant); con=fs-ant
                        as_=self.frequent_itemsets.get(ant,sum(1 for t in txs if ant.issubset(t))/n)
                        cs_=self.frequent_itemsets.get(con,sum(1 for t in txs if con.issubset(t))/n)
                        if as_==0 or cs_==0: continue
                        conf=sup/as_; lift=conf/cs_
                        if conf>=self.min_confidence:
                            rules.append({"antecedents":list(ant),"consequents":list(con),
                                "support":round(sup,4),"confidence":round(conf,4),
                                "lift":round(lift,4),"leverage":round(sup-as_*cs_,4),
                                "conviction":round((1-cs_)/(1-conf+1e-9),4),"algorithm":"ECLAT"})
            return sorted(rules,key=lambda x:x["lift"],reverse=True)

    class RecommendationEngine:
        def __init__(self): self.rules=[]; self.idx=defaultdict(list)
        def fit(self,rules):
            self.rules=rules
            for rule in rules:
                for item in rule["antecedents"]: self.idx[item].append(rule)
            return self
        def recommend(self,cart,n=5):
            cart_set=set(cart); scores=defaultdict(float); info={}
            for item in cart:
                for rule in self.idx.get(item,[]):
                    if set(rule["antecedents"]).issubset(cart_set):
                        for rec in rule["consequents"]:
                            if rec in cart_set: continue
                            sc=rule.get("interestingness",rule["lift"])
                            if sc>scores[rec]: scores[rec]=sc; info[rec]=rule
            ranked=sorted(scores.items(),key=lambda x:x[1],reverse=True)
            results=[]
            for item,sc in ranked[:n*2]:
                rule=info[item]; triggers=[t for t in rule["antecedents"] if t in cart]
                conf_pct=int(rule["confidence"]*100); lift=rule["lift"]
                strength="very strongly" if lift>=3 else "strongly" if lift>=2 else "commonly"
                explanation=(f"Customers who buy {' and '.join(triggers[:2])} {strength} also purchase "
                             f"{item} ({conf_pct}% of the time, {lift:.1f}× more likely).")
                results.append({"product":item,"score":round(sc,4),"confidence":rule["confidence"],
                    "lift":rule["lift"],"support":rule["support"],"trigger":rule["antecedents"],
                    "explanation":explanation})
            return {"upsell":[r for r in results if r["lift"]>=2.0][:n],
                    "cross_sell":[r for r in results if r["lift"]<2.0][:n]}

    def rank_rules(rules):
        for r in rules:
            ls=min(r["lift"]/10,1.0); lv=min(max(r["leverage"]*20,0),1.0)
            cv=min(r["conviction"]/5,1.0); ss=min(r["support"]/0.2,1.0)
            r["interestingness"]=round(0.35*ls+0.30*r["confidence"]+0.15*ss+0.10*lv+0.10*cv,4)
        return sorted(rules,key=lambda x:x["interestingness"],reverse=True)

    def compare_models(basket_list,params):
        results={}
        for AlgoClass in [Apriori,FPGrowth,ECLAT]:
            name=AlgoClass.__name__; m=AlgoClass(**params); m.fit(basket_list)
            results[name]={"algorithm":name,"n_frequent_itemsets":len(m.frequent_itemsets),
                "n_rules":len(m.rules),"training_time_sec":round(m.training_time,3),
                "avg_lift":round(np.mean([r["lift"] for r in m.rules]) if m.rules else 0,3),
                "max_lift":round(max([r["lift"] for r in m.rules]) if m.rules else 0,3),
                "top_rules":m.rules[:10]}
        return results

    def segment_customers(df):
        if "customer_id" not in df.columns: return None
        cf=df.groupby("customer_id").agg(total_spend=("price","sum"),
            n_transactions=("transaction_id","nunique"),
            n_unique_products=("product","nunique"),
            avg_price=("price","mean")).reset_index()
        X=cf[["total_spend","n_transactions","n_unique_products","avg_price"]].values
        Xn=(X-X.mean(0))/(X.std(0)+1e-9)
        np.random.seed(42); k=4
        C=Xn[np.random.choice(len(Xn),k,replace=False)]
        for _ in range(50):
            d=np.linalg.norm(Xn[:,None]-C[None,:],axis=2); L=d.argmin(1)
            nc=np.array([Xn[L==ki].mean(0) if (L==ki).any() else C[ki] for ki in range(k)])
            if np.allclose(C,nc): break
            C=nc
        cf["cluster"]=L
        sl=["Budget Shoppers","Frequent Buyers","Premium Customers","Occasional Buyers"]
        order=cf.groupby("cluster")["total_spend"].mean().sort_values().index.tolist()
        cf["segment"]=cf["cluster"].map({old:sl[new] for new,old in enumerate(order)})
        return cf

    def analyze_seasonality(df):
        df2=df.copy(); df2["month"]=pd.to_datetime(df2["date"]).dt.month
        mp=df2.groupby(["month","product"]).size().reset_index(name="count")
        top_prods=mp.groupby("product")["count"].sum().nlargest(10).index.tolist()
        pivot=mp.pivot_table(index="month",columns="product",values="count",fill_value=0)
        mn=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        trend=[]
        for m in range(1,13):
            row={"month":mn[m-1],"month_num":m}
            for p in top_prods:
                row[p]=int(pivot.loc[m,p]) if m in pivot.index and p in pivot.columns else 0
            trend.append(row)
        return {"trend_data":trend,"top_products":top_prods}

    def build_product_graph(rules,top_n=50):
        top=sorted(rules,key=lambda x:x["lift"],reverse=True)[:top_n]
        nodes={}; edges=[]
        for rule in top:
            for item in rule["antecedents"]+rule["consequents"]:
                if item not in nodes: nodes[item]={"id":item,"label":item,"connections":0}
                nodes[item]["connections"]+=1
            for ant in rule["antecedents"]:
                for con in rule["consequents"]:
                    edges.append({"source":ant,"target":con,"weight":rule["lift"],
                        "confidence":rule["confidence"],"support":rule["support"]})
        return {"nodes":list(nodes.values()),"edges":edges}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500&display=swap');
.stApp { background: #05080f !important; }
[data-testid="stSidebar"] { background: #0c1120 !important; border-right:1px solid rgba(99,179,255,.13); }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
h1,h2,h3 { font-family:'Syne',sans-serif !important; color:#e2e8f0 !important; }
h1 { background:linear-gradient(135deg,#3b82f6,#06b6d4);
     -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
[data-testid="stMetric"] { background:#0c1120; border:1px solid rgba(99,179,255,.15);
    border-radius:14px; padding:16px !important; position:relative; overflow:hidden; }
[data-testid="stMetric"]::before { content:''; position:absolute; top:0; left:0; right:0;
    height:2px; background:linear-gradient(90deg,#3b82f6,#06b6d4); }
[data-testid="stMetricValue"] { color:#e2e8f0 !important; font-size:1.9rem !important; font-weight:800 !important; }
[data-testid="stMetricLabel"] { color:#64748b !important; }
.stButton>button { background:linear-gradient(135deg,#3b82f6,#06b6d4) !important;
    color:white !important; border:none !important; border-radius:10px !important;
    font-weight:600 !important; padding:10px 24px !important; }
.stButton>button:hover { transform:translateY(-1px) !important; }
.stTabs [data-baseweb="tab-list"] { background:#0c1120; border-radius:10px; padding:4px; gap:4px; }
.stTabs [data-baseweb="tab"] { background:transparent !important; color:#64748b !important;
    border-radius:8px !important; font-weight:500 !important; }
.stTabs [aria-selected="true"] { background:rgba(59,130,246,.15) !important; color:#60a5fa !important; }
[data-testid="stExpander"] { background:#0c1120 !important;
    border:1px solid rgba(99,179,255,.13) !important; border-radius:12px !important; }
.stProgress>div>div { background:linear-gradient(90deg,#3b82f6,#06b6d4) !important; border-radius:3px !important; }
[data-testid="stFileUploader"] { background:#0c1120 !important;
    border:2px dashed rgba(99,179,255,.25) !important; border-radius:14px !important; }
.stMultiSelect [data-baseweb="tag"] { background:rgba(59,130,246,.2) !important; color:#93c5fd !important; }
.stDataFrame { border-radius:12px !important; }
div[data-testid="stVerticalBlock"] { color:#e2e8f0; }
p, li, label { color:#e2e8f0 !important; }
.stSelectbox label, .stSlider label, .stRadio label { color:#e2e8f0 !important; }
.stInfo { background:rgba(59,130,246,.1) !important; border-left:3px solid #3b82f6 !important; }
.stSuccess { background:rgba(16,185,129,.1) !important; border-left:3px solid #10b981 !important; }
.stWarning { background:rgba(245,158,11,.1) !important; border-left:3px solid #f59e0b !important; }
.stError   { background:rgba(239,68,68,.1)  !important; border-left:3px solid #ef4444 !important; }
.rec-card { background:#0c1120; border:1px solid rgba(99,179,255,.15); border-radius:14px; padding:16px; margin-bottom:12px; }
.rec-title { font-family:'Syne',sans-serif; font-weight:700; font-size:1rem; color:#e2e8f0; margin-bottom:6px; }
.rec-explain { font-size:.81rem; color:#64748b; line-height:1.55; margin-top:6px; }
</style>
""", unsafe_allow_html=True)


def generate_sample_data() -> pd.DataFrame:
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
        "tech_professional": {"cats":["Electronics","Office","Beverages"],"w":0.25},
        "health_enthusiast": {"cats":["Health","Fitness","Beverages","Snacks"],"w":0.30},
        "office_worker":     {"cats":["Office","Beverages","Snacks"],"w":0.25},
        "student":           {"cats":["Electronics","Office","Snacks","Beverages"],"w":0.20},
    }
    sn=list(SEGS.keys()); sw=[SEGS[s]["w"] for s in sn]
    start=datetime(2023,1,1); records=[]
    for tid in range(1, 2001):
        date=start+timedelta(days=random.randint(0,364))
        seg=random.choices(sn,weights=sw)[0]; cid=f"CUST_{random.randint(1,500):04d}"
        basket=list(set([random.choice(PRODUCTS[random.choice(SEGS[seg]["cats"])]) for _ in range(random.randint(2,8))]))
        for p in basket[:]:
            if p in ASSOC and random.random()<0.6: basket.append(random.choice(ASSOC[p]))
        for prod in set(basket):
            records.append({"transaction_id":f"TXN_{tid:05d}","customer_id":cid,"product":prod,
                "date":date.strftime("%Y-%m-%d"),"month":date.month,"day_of_week":date.strftime("%A"),
                "customer_segment":seg,"quantity":random.randint(1,3),"price":round(random.uniform(5,200),2)})
    return pd.DataFrame(records)


_defaults = {
    "df":None,"basket_list":None,"rules":[],"rec_engine":None,
    "comparison":None,"segments":None,"seasonality":None,"trained":False,"cart":[],
}
for k,v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


with st.sidebar:
    st.markdown("# ◈ BasketAI")
    st.markdown("**Market Basket Intelligence**")
    st.caption(f"ML source: `{ML_SOURCE}`")
    st.divider()

    st.markdown("### 📤 Data Source")
    data_source = st.radio("Choose dataset",
        ["🎯 Use Demo Dataset","📁 Upload Your CSV"],
        label_visibility="collapsed")

    if data_source == "📁 Upload Your CSV":
        uploaded = st.file_uploader("Upload transaction CSV", type=["csv"])
        if uploaded:
            try:
                df_up = pd.read_csv(uploaded)
                if {"transaction_id","product"}.issubset(set(df_up.columns)):
                    st.session_state["df"] = df_up
                    st.success(f"✅ {len(df_up):,} rows loaded")
                else:
                    st.error("Missing required columns: transaction_id, product")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        if st.button("Load Demo Dataset"):
            with st.spinner("Generating 2,000 transactions…"):
                st.session_state["df"] = generate_sample_data()
            st.success("✅ Demo dataset ready!")

    st.divider()
    st.markdown("### ⚙️ Parameters")
    min_sup  = st.slider("Min Support",    0.005, 0.10, 0.01, 0.005)
    min_conf = st.slider("Min Confidence", 0.10,  0.90, 0.20, 0.05)
    max_len  = st.slider("Max Itemset Length", 2, 5, 3, 1)
    st.divider()

    train_disabled = st.session_state["df"] is None
    if st.button("🚀 Train All Models", disabled=train_disabled):
        df = st.session_state["df"]
        basket_list = df.groupby("transaction_id")["product"].apply(list).tolist()
        params = {"min_support":min_sup,"min_confidence":min_conf,"max_len":max_len}
        with st.spinner("Training Apriori → FP-Growth → ECLAT…"):
            prog = st.progress(0, text="Comparing models…")
            comparison = compare_models(basket_list, params)
            prog.progress(40, text="Ranking rules…")
            fpg = FPGrowth(**params); fpg.fit(basket_list)
            rules = rank_rules(fpg.rules)
            prog.progress(70, text="Building recommender…")
            engine = RecommendationEngine(); engine.fit(rules)
            seg_df = segment_customers(df) if "customer_id" in df.columns else None
            season = analyze_seasonality(df)
            prog.progress(100, text="Done!")
            time.sleep(0.3); prog.empty()
        st.session_state.update({
            "basket_list":basket_list,"rules":rules,"rec_engine":engine,
            "comparison":comparison,"segments":seg_df,"seasonality":season,"trained":True,
        })
        st.success(f"✅ {len(rules)} rules found!")
        st.balloons()

    if st.session_state["trained"]:
        st.markdown("---")
        st.markdown("**Status:** 🟢 Model Ready")
        st.markdown(f"**Rules:** {len(st.session_state['rules'])}")

st.markdown("# ◈ BasketAI — Market Basket Intelligence")
st.markdown("*Apriori · FP-Growth · ECLAT · Recommendation Engine · Customer Segmentation*")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Dashboard", "🔗 Association Rules", "🔮 Recommendations", "👥 Segments & Trends"]
)

with tab1:
    if st.session_state["df"] is None:
        st.info("👈 Load a dataset and click **Train All Models** in the sidebar to begin.")
        st.markdown("""
        ### Getting Started
        1. In the sidebar, choose **Use Demo Dataset** → click **Load Demo Dataset**
        2. Adjust training parameters if desired
        3. Click **🚀 Train All Models**
        4. Explore results across all tabs
        """)
    else:
        df = st.session_state["df"]
        st.markdown("### 📈 Dataset Overview")
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("🛒 Transactions", f"{df['transaction_id'].nunique():,}")
        c2.metric("👥 Customers", f"{df['customer_id'].nunique():,}" if "customer_id" in df.columns else "N/A")
        c3.metric("📦 Products", f"{df['product'].nunique()}")
        c4.metric("🔗 Rules Found", f"{len(st.session_state['rules'])}")
        avg_b = df.groupby("transaction_id")["product"].count().mean()
        c5.metric("🧺 Avg Basket", f"{avg_b:.2f}")
        st.divider()

        if st.session_state["trained"]:
            cmp = st.session_state["comparison"]

            
            st.markdown("### 🤖 Algorithm Performance")
            cols = st.columns(3)
            clr  = {"Apriori":"🔵","FP-Growth":"🩵","ECLAT":"🟣"}
            for i,(algo,data) in enumerate(cmp.items()):
                with cols[i]:
                    st.markdown(f"**{clr.get(algo,'⚪')} {algo}**")
                    st.metric("Itemsets",    data["n_frequent_itemsets"])
                    st.metric("Rules",       data["n_rules"])
                    st.metric("Train Time",  f"{data['training_time_sec']}s")
                    st.metric("Avg Lift",    data["avg_lift"])
                    st.metric("Max Lift",    data["max_lift"])

           
            st.markdown("#### Comparison Chart")
            algos = list(cmp.keys())
            clrs  = ["#3b82f6","#06b6d4","#8b5cf6"]
            fig, axes = plt.subplots(1, 3, figsize=(13, 4))
            fig.patch.set_facecolor("#05080f")
            for ax, metric, label in zip(axes,
                    ["n_rules","avg_lift","training_time_sec"],
                    ["Rules Found","Average Lift","Train Time (s)"]):
                ax.set_facecolor("#0c1120")
                vals = [cmp[a][metric] for a in algos]
                bars = ax.bar(algos, vals, color=clrs, edgecolor="none", alpha=0.85, width=0.5)
                ax.set_title(label, color="#e2e8f0", fontsize=11, pad=8)
                ax.tick_params(colors="#64748b")
                for sp in ["top","right","left"]: ax.spines[sp].set_visible(False)
                ax.spines["bottom"].set_color("#1e293b")
                for bar,val in zip(bars,vals):
                    ax.text(bar.get_x()+bar.get_width()/2, val*1.02,
                            f"{val:.2f}" if isinstance(val,float) else str(val),
                            ha="center", color="#e2e8f0", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig); plt.close()
            st.divider()

            
            st.markdown("### 📦 Top 10 Products by Frequency")
            tp = df["product"].value_counts().head(10)
            fig2, ax2 = plt.subplots(figsize=(10,4))
            fig2.patch.set_facecolor("#05080f"); ax2.set_facecolor("#0c1120")
            bar_clrs = ["#3b82f6","#06b6d4","#8b5cf6","#f59e0b","#10b981"]*2
            ax2.barh(tp.index[::-1], tp.values[::-1], color=bar_clrs, alpha=0.85)
            ax2.tick_params(colors="#64748b")
            for sp in ["top","right"]: ax2.spines[sp].set_visible(False)
            ax2.spines["bottom"].set_color("#1e293b"); ax2.spines["left"].set_color("#1e293b")
            plt.tight_layout(); st.pyplot(fig2); plt.close()

        else:
            st.info("👈 Click **Train All Models** to see algorithm comparisons.")
            st.markdown("### 📋 Data Preview")
            st.dataframe(df.head(20), use_container_width=True)


with tab2:
    if not st.session_state["trained"]:
        st.info("👈 Train a model first.")
    else:
        rules = st.session_state["rules"]
        st.markdown(f"### 🔗 Association Rules  —  **{len(rules)}** found")

      
        f1,f2,f3,f4 = st.columns(4)
        algo_filter = f1.selectbox("Algorithm", ["All","Apriori","FP-Growth","ECLAT"])
        min_lift_f  = f2.slider("Min Lift",  1.0, 10.0, 1.0, 0.1)
        min_conf_f  = f3.slider("Min Conf",  0.0,  1.0, 0.0, 0.05)
        n_show      = f4.slider("Show top N", 5,  100,  30, 5)

        filtered = rules
        if algo_filter != "All":
            filtered = [r for r in filtered if r["algorithm"]==algo_filter]
        filtered = [r for r in filtered if r["lift"]>=min_lift_f and r["confidence"]>=min_conf_f]

        st.caption(f"Showing {min(n_show,len(filtered))} of {len(filtered)} filtered rules")

        if filtered:
            df_rules = pd.DataFrame([{
                "Antecedent (IF)":   " + ".join(r["antecedents"]),
                "Consequent (THEN)": " + ".join(r["consequents"]),
                "Support":    f"{r['support']*100:.1f}%",
                "Confidence": f"{r['confidence']*100:.1f}%",
                "Lift":       round(r["lift"],3),
                "Score":      round(r.get("interestingness",0),3),
                "Algorithm":  r["algorithm"],
            } for r in filtered[:n_show]])
            st.dataframe(df_rules, use_container_width=True, height=500)

            st.markdown("#### Lift Distribution")
            lifts = [r["lift"] for r in filtered]
            fig3,ax3 = plt.subplots(figsize=(10,3))
            fig3.patch.set_facecolor("#05080f"); ax3.set_facecolor("#0c1120")
            ax3.hist(lifts, bins=max(5,len(lifts)//3), color="#3b82f6", alpha=0.85, edgecolor="#0c1120")
            ax3.axvline(float(np.mean(lifts)), color="#06b6d4", linestyle="--",
                        label=f"Mean: {np.mean(lifts):.2f}")
            ax3.legend(labelcolor="#e2e8f0", framealpha=0)
            ax3.tick_params(colors="#64748b")
            for sp in ["top","right"]: ax3.spines[sp].set_visible(False)
            ax3.spines["bottom"].set_color("#1e293b"); ax3.spines["left"].set_color("#1e293b")
            plt.tight_layout(); st.pyplot(fig3); plt.close()

            st.divider()
            st.download_button("⬇️ Download Rules CSV",
                pd.DataFrame([{**r,"antecedents":" + ".join(r["antecedents"]),
                    "consequents":" + ".join(r["consequents"])} for r in filtered[:n_show]])
                .to_csv(index=False).encode(), "rules.csv","text/csv")
        else:
            st.warning("No rules match the current filters. Try lowering the thresholds.")


with tab3:
    if not st.session_state["trained"]:
        st.info("👈 Train a model first.")
    else:
        st.markdown("### 🔮 Recommendation Engine")
        st.markdown("Select products to add to your cart, then get upsell & cross-sell suggestions.")

        df = st.session_state["df"]
        all_prods = sorted(df["product"].unique().tolist())

        col_cat, col_cart = st.columns([1,1])
        with col_cat:
            st.markdown("**📦 Product Catalog**")
            selected = st.multiselect("Add to cart (search or click):", all_prods,
                placeholder="Type a product name…")
            if st.button("➕ Add Selected to Cart") and selected:
                for p in selected:
                    if p not in st.session_state["cart"]:
                        st.session_state["cart"].append(p)

        with col_cart:
            st.markdown("**🛒 Your Cart**")
            if st.session_state["cart"]:
                for p in list(st.session_state["cart"]):
                    cc1,cc2 = st.columns([4,1])
                    cc1.markdown(f"• **{p}**")
                    if cc2.button("✕", key=f"rm_{p}"):
                        st.session_state["cart"].remove(p)
                        st.rerun()
                if st.button("🗑️ Clear Cart"):
                    st.session_state["cart"] = []
                    st.rerun()
            else:
                st.caption("Cart is empty — add products from the catalog.")

        st.divider()

        if st.session_state["cart"] and st.session_state["rec_engine"]:
            recs = st.session_state["rec_engine"].recommend(
                st.session_state["cart"], n=6)

            up   = recs.get("upsell",[])
            cs   = recs.get("cross_sell",[])

            if up:
                st.markdown("#### ⬆️ Upsell Suggestions  🏆 High Value")
                cols = st.columns(min(len(up),3))
                for i,r in enumerate(up):
                    with cols[i % 3]:
                        st.markdown(f"""
                        <div class="rec-card">
                          <div class="rec-title">{r['product']}</div>
                          <span style="background:rgba(245,158,11,.2);color:#fcd34d;
                            border:1px solid rgba(245,158,11,.35);padding:2px 9px;
                            border-radius:20px;font-size:.72rem;font-weight:600">
                            Lift {r['lift']:.2f}</span>
                          <span style="background:rgba(59,130,246,.2);color:#93c5fd;
                            border:1px solid rgba(59,130,246,.35);padding:2px 9px;
                            border-radius:20px;font-size:.72rem;font-weight:600;margin-left:4px">
                            Conf {r['confidence']*100:.0f}%</span>
                          <div class="rec-explain">{r['explanation']}</div>
                        </div>""", unsafe_allow_html=True)

            if cs:
                st.markdown("#### ↔️ Cross-Sell Suggestions  🔗 Complementary")
                cols2 = st.columns(min(len(cs),3))
                for i,r in enumerate(cs):
                    with cols2[i % 3]:
                        st.markdown(f"""
                        <div class="rec-card">
                          <div class="rec-title">{r['product']}</div>
                          <span style="background:rgba(6,182,212,.2);color:#67e8f9;
                            border:1px solid rgba(6,182,212,.35);padding:2px 9px;
                            border-radius:20px;font-size:.72rem;font-weight:600">
                            Lift {r['lift']:.2f}</span>
                          <span style="background:rgba(59,130,246,.2);color:#93c5fd;
                            border:1px solid rgba(59,130,246,.35);padding:2px 9px;
                            border-radius:20px;font-size:.72rem;font-weight:600;margin-left:4px">
                            Conf {r['confidence']*100:.0f}%</span>
                          <div class="rec-explain">{r['explanation']}</div>
                        </div>""", unsafe_allow_html=True)

            if not up and not cs:
                st.warning("No strong associations found for this cart. Try adding more products "
                           "or lowering the Min Confidence parameter.")
        elif st.session_state["cart"]:
            st.info("Click **Train All Models** first to enable recommendations.")


with tab4:
    if not st.session_state["trained"]:
        st.info("👈 Train a model first.")
    else:
     
        seg_df = st.session_state["segments"]
        if seg_df is not None:
            st.markdown("### 👥 Customer Segmentation")
            seg_counts = seg_df["segment"].value_counts().reset_index()
            seg_counts.columns = ["Segment","Customers"]
            seg_summary = seg_df.groupby("segment").agg(
                Customers=("customer_id","count"),
                Avg_Spend=("total_spend","mean"),
                Avg_Orders=("n_transactions","mean"),
                Avg_Products=("n_unique_products","mean"),
            ).round(2).reset_index()

            s1,s2 = st.columns([1,1])
            with s1:
                fig4,ax4 = plt.subplots(figsize=(6,5))
                fig4.patch.set_facecolor("#05080f"); ax4.set_facecolor("#05080f")
                wedges,texts,autos = ax4.pie(
                    seg_counts["Customers"], labels=seg_counts["Segment"],
                    autopct="%1.1f%%",
                    colors=["#3b82f6","#06b6d4","#8b5cf6","#f59e0b"],
                    textprops={"color":"#e2e8f0","fontsize":9},
                    wedgeprops={"edgecolor":"#05080f","linewidth":2})
                for a in autos: a.set_color("#e2e8f0")
                ax4.set_title("Segment Distribution", color="#e2e8f0", pad=10)
                plt.tight_layout(); st.pyplot(fig4); plt.close()

            with s2:
                st.markdown("**Segment Profiles**")
                st.dataframe(seg_summary.rename(columns={
                    "segment":"Segment","Avg_Spend":"Avg Spend ($)",
                    "Avg_Orders":"Avg Orders","Avg_Products":"Avg Products"}),
                    use_container_width=True, hide_index=True)
                st.download_button("⬇️ Download Segments CSV",
                    seg_df.to_csv(index=False).encode(), "segments.csv","text/csv")
        else:
            st.info("Customer segmentation requires a `customer_id` column in your dataset.")

        st.divider()

       
        season = st.session_state["seasonality"]
        if season:
            st.markdown("### 📅 Monthly Product Trends")
            trend_df = pd.DataFrame(season["trend_data"])
            top_prods = season["top_products"][:5]
            month_col = "month"

            fig5,ax5 = plt.subplots(figsize=(12,4))
            fig5.patch.set_facecolor("#05080f"); ax5.set_facecolor("#0c1120")
            line_clrs = ["#3b82f6","#06b6d4","#8b5cf6","#f59e0b","#10b981"]
            for i,prod in enumerate(top_prods):
                if prod in trend_df.columns:
                    ax5.plot(trend_df[month_col], trend_df[prod],
                             color=line_clrs[i%len(line_clrs)], linewidth=2.2,
                             marker="o", markersize=4, label=prod)
            ax5.legend(labelcolor="#e2e8f0", framealpha=0.2,
                       facecolor="#0c1120", loc="upper right", fontsize=8)
            ax5.tick_params(colors="#64748b"); ax5.set_xlabel("Month", color="#64748b")
            ax5.set_ylabel("Purchase Count", color="#64748b")
            for sp in ["top","right"]: ax5.spines[sp].set_visible(False)
            ax5.spines["bottom"].set_color("#1e293b"); ax5.spines["left"].set_color("#1e293b")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout(); st.pyplot(fig5); plt.close()

            with st.expander("📋 View Monthly Data Table"):
                cols_to_show = [month_col] + [p for p in top_prods if p in trend_df.columns]
                st.dataframe(trend_df[cols_to_show], use_container_width=True, hide_index=True)

            st.markdown("#### 🗓️ Top Products by Month")
            df_main = st.session_state["df"].copy()
            df_main["month"] = pd.to_datetime(df_main["date"]).dt.month
            month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            sel_month = st.selectbox("Select month:", range(1,13),
                format_func=lambda x: month_names[x-1])
            top5 = df_main[df_main["month"]==sel_month]["product"].value_counts().head(5)
            st.dataframe(pd.DataFrame({"Product":top5.index,"Count":top5.values}),
                use_container_width=True, hide_index=True)
