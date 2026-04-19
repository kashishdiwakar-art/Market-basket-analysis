import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict
import time
import json
from typing import List, Dict, Tuple, Optional


def load_transactions(filepath: str) -> Tuple[pd.DataFrame, List[List[str]], List[str]]:
    """Load CSV and return raw df, basket list, and unique items."""
    df = pd.read_csv(filepath)
   
    baskets = df.groupby("transaction_id")["product"].apply(list).reset_index()
    basket_list = baskets["product"].tolist()
    all_items = sorted(df["product"].unique().tolist())
    return df, basket_list, all_items


def encode_baskets(basket_list: List[List[str]], all_items: List[str]) -> pd.DataFrame:
    """One-hot encode baskets into a boolean DataFrame."""
    item_set = set(all_items)
    rows = []
    for basket in basket_list:
        row = {item: (item in basket) for item in item_set}
        rows.append(row)
    return pd.DataFrame(rows, columns=all_items)



class Apriori:
    """Classic Apriori association rule mining algorithm."""

    def __init__(self, min_support: float = 0.02, min_confidence: float = 0.3, max_len: int = 4):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.max_len = max_len
        self.frequent_itemsets = {}
        self.rules = []

    def _get_support(self, itemset: frozenset, transactions: List[frozenset]) -> float:
        count = sum(1 for t in transactions if itemset.issubset(t))
        return count / len(transactions)

    def fit(self, basket_list: List[List[str]]) -> "Apriori":
        start = time.time()
        transactions = [frozenset(b) for b in basket_list]
        n = len(transactions)

        
        item_counts = defaultdict(int)
        for t in transactions:
            for item in t:
                item_counts[frozenset([item])] += 1

        L = {frozenset([item]): cnt / n
             for item, cnt in item_counts.items()
             if cnt / n >= self.min_support}
        self.frequent_itemsets.update(L)

        k = 2
        Lk = list(L.keys())

        while Lk and k <= self.max_len:
           
            candidates = {}
            for i in range(len(Lk)):
                for j in range(i + 1, len(Lk)):
                    union = Lk[i] | Lk[j]
                    if len(union) == k:
                        sup = self._get_support(union, transactions)
                        if sup >= self.min_support:
                            candidates[union] = sup

            self.frequent_itemsets.update(candidates)
            Lk = list(candidates.keys())
            k += 1

        self.rules = self._generate_rules(transactions)
        self.training_time = time.time() - start
        return self

    def _generate_rules(self, transactions: List[frozenset]) -> List[Dict]:
        rules = []
        n = len(transactions)

        for itemset, support in self.frequent_itemsets.items():
            if len(itemset) < 2:
                continue
            for r in range(1, len(itemset)):
                for antecedent in combinations(itemset, r):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    ant_sup = self.frequent_itemsets.get(antecedent, self._get_support(antecedent, transactions))
                    if ant_sup == 0:
                        continue
                    confidence = support / ant_sup
                    con_sup = self.frequent_itemsets.get(consequent, self._get_support(consequent, transactions))
                    lift = confidence / con_sup if con_sup > 0 else 0

                    if confidence >= self.min_confidence:
                        rules.append({
                            "antecedents": list(antecedent),
                            "consequents": list(consequent),
                            "support": round(support, 4),
                            "confidence": round(confidence, 4),
                            "lift": round(lift, 4),
                            "leverage": round(support - ant_sup * con_sup, 4),
                            "conviction": round((1 - con_sup) / (1 - confidence + 1e-9), 4),
                            "algorithm": "Apriori"
                        })

        return sorted(rules, key=lambda x: x["lift"], reverse=True)

class FPNode:
    """Node in an FP-Tree."""
    def __init__(self, item, count=0, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.link = None  

class FPGrowth:
    """FP-Growth algorithm - more efficient than Apriori."""

    def __init__(self, min_support: float = 0.02, min_confidence: float = 0.3, max_len: int = 4):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.max_len = max_len
        self.frequent_itemsets = {}
        self.rules = []

    def _build_tree(self, transactions, min_count):
        
        freq = defaultdict(int)
        for t in transactions:
            for item in t:
                freq[item] += 1
        freq = {k: v for k, v in freq.items() if v >= min_count}

      
        root = FPNode(None)
        header = {item: None for item in freq}

        for t in transactions:
            
            items = [i for i in t if i in freq]
            items.sort(key=lambda x: freq[x], reverse=True)
            self._insert_tree(items, root, header)

        return root, header, freq

    def _insert_tree(self, items, node, header):
        if not items:
            return
        item = items[0]
        if item in node.children:
            node.children[item].count += 1
        else:
            child = FPNode(item, 1, node)
            node.children[item] = child
           
            if header[item] is None:
                header[item] = child
            else:
                curr = header[item]
                while curr.link:
                    curr = curr.link
                curr.link = child
        self._insert_tree(items[1:], node.children[item], header)

    def _conditional_patterns(self, item, header):
        patterns = []
        node = header.get(item)
        while node:
            path = []
            parent = node.parent
            while parent and parent.item is not None:
                path.append(parent.item)
                parent = parent.parent
            if path:
                patterns.append((path, node.count))
            node = node.link
        return patterns

    def _mine(self, transactions, prefix, min_count, n_total):
        root, header, freq = self._build_tree(transactions, min_count)
        for item, cnt in freq.items():
            new_prefix = prefix + [item]
            sup = cnt / n_total
            if sup >= self.min_support and len(new_prefix) <= self.max_len:
                self.frequent_itemsets[frozenset(new_prefix)] = round(sup, 4)
               
                cond_patterns = self._conditional_patterns(item, header)
                new_transactions = []
                for path, count in cond_patterns:
                    new_transactions.extend([path] * count)
                if new_transactions:
                    self._mine(new_transactions, new_prefix, min_count, n_total)

    def fit(self, basket_list: List[List[str]]) -> "FPGrowth":
        start = time.time()
        n = len(basket_list)
        min_count = int(self.min_support * n)
        self._mine(basket_list, [], min_count, n)
        self.rules = self._generate_rules(basket_list)
        self.training_time = time.time() - start
        return self

    def _generate_rules(self, basket_list):
        transactions = [frozenset(b) for b in basket_list]
        n = len(transactions)
        rules = []

        for itemset, support in self.frequent_itemsets.items():
            if len(itemset) < 2:
                continue
            for r in range(1, len(itemset)):
                for ant in combinations(itemset, r):
                    ant = frozenset(ant)
                    con = itemset - ant
                    ant_sup = self.frequent_itemsets.get(ant, sum(1 for t in transactions if ant.issubset(t)) / n)
                    con_sup = self.frequent_itemsets.get(con, sum(1 for t in transactions if con.issubset(t)) / n)
                    if ant_sup == 0 or con_sup == 0:
                        continue
                    confidence = support / ant_sup
                    lift = confidence / con_sup

                    if confidence >= self.min_confidence:
                        rules.append({
                            "antecedents": list(ant),
                            "consequents": list(con),
                            "support": round(support, 4),
                            "confidence": round(confidence, 4),
                            "lift": round(lift, 4),
                            "leverage": round(support - ant_sup * con_sup, 4),
                            "conviction": round((1 - con_sup) / (1 - confidence + 1e-9), 4),
                            "algorithm": "FP-Growth"
                        })

        return sorted(rules, key=lambda x: x["lift"], reverse=True)

class ECLAT:
    """ECLAT uses vertical data format (tidlists) for efficient mining."""

    def __init__(self, min_support: float = 0.02, min_confidence: float = 0.3, max_len: int = 4):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.max_len = max_len
        self.frequent_itemsets = {}
        self.rules = []

    def _build_tidlists(self, basket_list):
        """Build transaction ID lists for each item."""
        tidlists = defaultdict(set)
        for tid, basket in enumerate(basket_list):
            for item in basket:
                tidlists[item].add(tid)
        return tidlists

    def _eclat_recursive(self, prefix, items_tidlists, n_total):
        """Recursively mine itemsets using set intersection."""
        items = list(items_tidlists.keys())
        for i, item_i in enumerate(items):
            new_prefix = prefix + [item_i]
            tidlist_i = items_tidlists[item_i]
            support = len(tidlist_i) / n_total

            if support >= self.min_support:
                self.frequent_itemsets[frozenset(new_prefix)] = round(support, 4)

                if len(new_prefix) < self.max_len:
                  
                    new_items = {}
                    for item_j in items[i+1:]:
                        intersection = tidlist_i & items_tidlists[item_j]
                        if len(intersection) / n_total >= self.min_support:
                            new_items[item_j] = intersection

                    if new_items:
                        self._eclat_recursive(new_prefix, new_items, n_total)

    def fit(self, basket_list: List[List[str]]) -> "ECLAT":
        start = time.time()
        n = len(basket_list)
        tidlists = self._build_tidlists(basket_list)

        filtered = {item: tids for item, tids in tidlists.items()
                    if len(tids) / n >= self.min_support}

        self._eclat_recursive([], filtered, n)
        self.rules = self._generate_rules(basket_list)
        self.training_time = time.time() - start
        return self

    def _generate_rules(self, basket_list):
        transactions = [frozenset(b) for b in basket_list]
        n = len(transactions)
        rules = []

        for itemset, support in self.frequent_itemsets.items():
            if len(itemset) < 2:
                continue
            for r in range(1, len(itemset)):
                for ant in combinations(itemset, r):
                    ant = frozenset(ant)
                    con = itemset - ant
                    ant_sup = self.frequent_itemsets.get(ant, sum(1 for t in transactions if ant.issubset(t)) / n)
                    con_sup = self.frequent_itemsets.get(con, sum(1 for t in transactions if con.issubset(t)) / n)
                    if ant_sup == 0 or con_sup == 0:
                        continue
                    confidence = support / ant_sup
                    lift = confidence / con_sup

                    if confidence >= self.min_confidence:
                        rules.append({
                            "antecedents": list(ant),
                            "consequents": list(con),
                            "support": round(support, 4),
                            "confidence": round(confidence, 4),
                            "lift": round(lift, 4),
                            "leverage": round(support - ant_sup * con_sup, 4),
                            "conviction": round((1 - con_sup) / (1 - confidence + 1e-9), 4),
                            "algorithm": "ECLAT"
                        })

        return sorted(rules, key=lambda x: x["lift"], reverse=True)


def compute_interestingness_score(rule: Dict) -> float:
    """
    Custom interestingness score combining multiple metrics.
    Score ∈ [0, 1] — higher is more interesting/actionable.
    """
 
    lift_score = min(rule["lift"] / 10, 1.0)                       
    conf_score = rule["confidence"]                            
    sup_score = min(rule["support"] / 0.2, 1.0)                  
    lev_score = min(max(rule["leverage"] * 20, 0), 1.0)           
    conv_score = min(rule["conviction"] / 5, 1.0)                  

    score = (0.35 * lift_score + 0.30 * conf_score +
             0.15 * sup_score + 0.10 * lev_score + 0.10 * conv_score)
    return round(score, 4)


def rank_rules(rules: List[Dict]) -> List[Dict]:
    """Add interestingness score and rank rules."""
    for rule in rules:
        rule["interestingness"] = compute_interestingness_score(rule)
    return sorted(rules, key=lambda x: x["interestingness"], reverse=True)



class RecommendationEngine:
    """
    Generates upsell & cross-sell suggestions based on association rules.
    """

    def __init__(self):
        self.rules: List[Dict] = []
        self.product_to_rules: Dict[str, List[Dict]] = defaultdict(list)

    def fit(self, rules: List[Dict]) -> "RecommendationEngine":
        self.rules = rules
       
        for rule in rules:
            for item in rule["antecedents"]:
                self.product_to_rules[item].append(rule)
        return self

    def recommend(self, cart: List[str], n: int = 5, exclude_cart: bool = True) -> Dict:
        """
        Given a cart, find top recommendations.
        Returns upsell (high-value, high-lift) and cross-sell (complementary).
        """
        cart_set = set(cart)
        candidate_scores = defaultdict(float)
        candidate_info = {}

        for item in cart:
            for rule in self.product_to_rules.get(item, []):
           
                if set(rule["antecedents"]).issubset(cart_set):
                    for rec_item in rule["consequents"]:
                        if exclude_cart and rec_item in cart_set:
                            continue
                        score = rule["interestingness"] if "interestingness" in rule else rule["lift"]
                        candidate_scores[rec_item] = max(candidate_scores[rec_item], score)
                        if rec_item not in candidate_info or candidate_info[rec_item]["lift"] < rule["lift"]:
                            candidate_info[rec_item] = rule


        ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for item, score in ranked[:n * 2]:
            rule = candidate_info[item]
            results.append({
                "product": item,
                "score": round(score, 4),
                "confidence": rule["confidence"],
                "lift": rule["lift"],
                "support": rule["support"],
                "trigger": rule["antecedents"],
                "explanation": self._explain(cart, item, rule)
            })

        upsell = [r for r in results if r["lift"] >= 2.0][:n]
        cross_sell = [r for r in results if r["lift"] < 2.0][:n]

        return {
            "cart": cart,
            "upsell": upsell,
            "cross_sell": cross_sell,
            "all_recommendations": results[:n]
        }

    def _explain(self, cart: List[str], rec_item: str, rule: Dict) -> str:
        """Generate human-readable explanation for a recommendation."""
        triggers = [t for t in rule["antecedents"] if t in cart]
        confidence_pct = int(rule["confidence"] * 100)
        lift = rule["lift"]

        if lift >= 3:
            strength = "very strongly"
        elif lift >= 2:
            strength = "strongly"
        else:
            strength = "commonly"

        trigger_str = " and ".join(triggers[:2])
        return (f"Customers who buy {trigger_str} {strength} tend to also purchase "
                f"{rec_item} ({confidence_pct}% of the time, {lift:.1f}x more likely).")



def segment_customers(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """
    Segment customers using K-Means based on purchase behavior.
    Features: total spend, basket size, category diversity, purchase frequency.
    """
  
    customer_features = df.groupby("customer_id").agg(
        total_spend=("price", "sum"),
        n_transactions=("transaction_id", "nunique"),
        avg_basket_size=("product", lambda x: len(x) / df[df["customer_id"] == x.name]["transaction_id"].nunique() if df[df["customer_id"] == x.name]["transaction_id"].nunique() > 0 else 0),
        n_unique_products=("product", "nunique"),
        avg_price=("price", "mean"),
    ).reset_index()


    features = ["total_spend", "n_transactions", "n_unique_products", "avg_price"]
    X = customer_features[features].values
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

    np.random.seed(42)
    centroids = X_norm[np.random.choice(len(X_norm), n_clusters, replace=False)]

    for _ in range(50):
        dists = np.linalg.norm(X_norm[:, None] - centroids[None, :], axis=2)
        labels = dists.argmin(axis=1)
        new_centroids = np.array([X_norm[labels == k].mean(axis=0) for k in range(n_clusters)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    customer_features["cluster"] = labels
    segment_labels = ["Budget Shoppers", "Frequent Buyers", "Premium Customers", "Occasional Buyers"]
    cluster_order = customer_features.groupby("cluster")["total_spend"].mean().sort_values().index.tolist()
    label_map = {old: segment_labels[new] for new, old in enumerate(cluster_order)}
    customer_features["segment"] = customer_features["cluster"].map(label_map)

    return customer_features


def analyze_seasonality(df: pd.DataFrame) -> Dict:
    """Analyze monthly purchase trends per product category."""
    df["month"] = pd.to_datetime(df["date"]).dt.month
    month_product = df.groupby(["month", "product"]).size().reset_index(name="count")


    monthly_top = month_product.sort_values("count", ascending=False).groupby("month").head(5)
    monthly_trends = {}
    for month in range(1, 13):
        month_data = month_product[month_product["month"] == month]
        monthly_trends[str(month)] = month_data.sort_values("count", ascending=False).head(10).to_dict("records")

    pivot = month_product.pivot_table(index="month", columns="product", values="count", fill_value=0)
    top_products = month_product.groupby("product")["count"].sum().nlargest(10).index.tolist()

    trend_data = []
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    for m in range(1, 13):
        row = {"month": month_names[m-1], "month_num": m}
        for prod in top_products:
            row[prod] = int(pivot.loc[m, prod]) if m in pivot.index and prod in pivot.columns else 0
        trend_data.append(row)

    return {
        "monthly_trends": monthly_trends,
        "trend_data": trend_data,
        "top_products": top_products
    }


def build_product_graph(rules: List[Dict], top_n: int = 50) -> Dict:
    """Build nodes + edges for product relationship network visualization."""

    top_rules = sorted(rules, key=lambda x: x["lift"], reverse=True)[:top_n]

    nodes = {}
    edges = []

    for rule in top_rules:
        for item in rule["antecedents"] + rule["consequents"]:
            if item not in nodes:
                nodes[item] = {"id": item, "label": item, "connections": 0}
            nodes[item]["connections"] += 1

       
        for ant in rule["antecedents"]:
            for con in rule["consequents"]:
                edges.append({
                    "source": ant,
                    "target": con,
                    "weight": rule["lift"],
                    "confidence": rule["confidence"],
                    "support": rule["support"]
                })

    return {
        "nodes": list(nodes.values()),
        "edges": edges
    }


def compare_models(basket_list: List[List[str]], params: Dict) -> Dict:
    """Train all three models and compare performance metrics."""
    results = {}

    for AlgoClass in [Apriori, FPGrowth, ECLAT]:
        name = AlgoClass.__name__
        model = AlgoClass(
            min_support=params.get("min_support", 0.02),
            min_confidence=params.get("min_confidence", 0.3),
            max_len=params.get("max_len", 3)
        )
        model.fit(basket_list)
        results[name] = {
            "algorithm": name,
            "n_frequent_itemsets": len(model.frequent_itemsets),
            "n_rules": len(model.rules),
            "training_time_sec": round(model.training_time, 3),
            "avg_lift": round(np.mean([r["lift"] for r in model.rules]) if model.rules else 0, 3),
            "max_lift": round(max([r["lift"] for r in model.rules]) if model.rules else 0, 3),
            "top_rules": model.rules[:10]
        }

    return results


def run_full_analysis(filepath: str, params: Dict = None) -> Dict:
    """Run the complete Market Basket Analysis pipeline."""
    if params is None:
        params = {"min_support": 0.02, "min_confidence": 0.3, "max_len": 3}

    print("Loading data...")
    df, basket_list, all_items = load_transactions(filepath)

    print("Comparing models...")
    model_comparison = compare_models(basket_list, params)


    primary_model = FPGrowth(**params)
    primary_model.fit(basket_list)
    ranked_rules = rank_rules(primary_model.rules)

    print("Building recommendation engine...")
    rec_engine = RecommendationEngine()
    rec_engine.fit(ranked_rules)

    print("Segmenting customers...")
    customer_segments = segment_customers(df)

    print("Analyzing seasonality...")
    seasonality = analyze_seasonality(df)

    print("Building product graph...")
    product_graph = build_product_graph(ranked_rules)

    summary = {
        "total_transactions": df["transaction_id"].nunique(),
        "total_customers": df["customer_id"].nunique(),
        "total_products": df["product"].nunique(),
        "total_rules": len(ranked_rules),
        "avg_basket_size": round(df.groupby("transaction_id")["product"].count().mean(), 2),
        "date_range": {"start": df["date"].min(), "end": df["date"].max()},
        "top_products": df["product"].value_counts().head(10).to_dict(),
        "segment_distribution": customer_segments["segment"].value_counts().to_dict()
    }

    return {
        "summary": summary,
        "model_comparison": model_comparison,
        "rules": ranked_rules[:100],
        "product_graph": product_graph,
        "seasonality": seasonality,
        "customer_segments": customer_segments[["customer_id", "total_spend", "n_transactions", "n_unique_products", "segment"]].head(200).to_dict("records"),
        "recommendation_engine": rec_engine
    }


if __name__ == "__main__":
    import os
    _here = os.path.dirname(os.path.abspath(__file__))
 
    _data1 = os.path.join(_here, "data", "transactions.csv")
    _data2 = os.path.join(_here, "..", "data", "transactions.csv")
    _csv = _data1 if os.path.exists(_data1) else os.path.abspath(_data2)

    if not os.path.exists(_csv):
        raise FileNotFoundError(
            f"transactions.csv not found.\n"
            f"Looked in:\n  {_data1}\n  {_data2}\n\n"
            f"Run this first:\n  python data/generate_data.py"
        )

    results = run_full_analysis(_csv)
    print(f"\n✅ Analysis complete!")
    print(f"  Transactions: {results['summary']['total_transactions']}")
    print(f"  Rules found: {results['summary']['total_rules']}")
    print(f"\nTop 3 rules by lift:")
    for rule in results['rules'][:3]:
        print(f"  {rule['antecedents']} → {rule['consequents']} | lift={rule['lift']} conf={rule['confidence']}")