"""End-to-end: build catalog -> simulate users -> split -> fit -> tune hybrid -> evaluate -> save results."""
import json, time
import numpy as np, pandas as pd
from src.config import RESULTS, LEVELS, EQUIPMENT_PROFILES
from src.data import build_catalog
from src.simulate import load_or_simulate
from src.content import ContentModel
from src.evaluate import rating_errors, top_k, ranking_metrics
from src.hybrid import Hybrid, grid_search
from src.rerank import constraint_mask, mmr, level_penalty
from src.cf import GlobalMean, UserMean
from src.system import (to_matrix, temporal_split, relevant_sets, fit_models, components,
                        DEFAULT_WEIGHTS)

K = 10
t0 = time.time()
cat = build_catalog()
users, inter = load_or_simulate(cat, force=True)
nU, nI = len(users), len(cat)
tr, va, te = temporal_split(inter)
print(f"catalog={nI} users={nU} interactions={len(inter)} | train/val/test = {len(tr)}/{len(va)}/{len(te)}")
content = ContentModel().fit(cat)
sim = content.similarity_matrix()
all_users = np.arange(nU)


def idx_df(df):
    return df.assign(user_idx=df.user_id, item_idx=df.item_id)


# ======================= stage 1: tune on validation (fit on train only) =======================
R, M = to_matrix(tr, nU, nI)
Rv, Mv = to_matrix(va, nU, nI)
models = fit_models(R, M, content, val=(va.user_id.values, va.item_id.values, va.rating.values.astype(float)))
comp = components(models, content, R, M, all_users)
n_tr = M.sum(1)
rel_va = relevant_sets(va)
pop_item = (M.sum(0) / nU)

def ndcg_val(S):
    return ranking_metrics(top_k(S, M, K), rel_va, k=K)[f"NDCG@{K}"]

best, w = grid_search(comp, n_tr, ndcg_val, ["content", "item_knn", "user_knn", "als", "pop"], step=0.25)
w = {k: float(v) for k, v in w.items()}
print("best weights on validation:", w, "NDCG@10 =", round(best, 4))
(RESULTS / "hybrid_weights.json").write_text(json.dumps(w, indent=2))
pd.DataFrame(models["mf"].history, columns=["epoch", "train_rmse", "val_rmse"]).to_csv(RESULTS / "mf_training_curve.csv", index=False)

# ======================= stage 2: refit on train+val, evaluate on test ========================
trva = pd.concat([tr, va])
R2, M2 = to_matrix(trva, nU, nI)
models2 = fit_models(R2, M2, content)
comp2 = components(models2, content, R2, M2, all_users)
n2 = M2.sum(1)
rel_te = relevant_sets(te)
pop2 = M2.sum(0) / nU
hyb = Hybrid(w)
H = hyb.combine(comp2, n2)

def eval_scores(name, S, recs=None):
    recs = top_k(S, M2, K) if recs is None else recs
    r = ranking_metrics(recs, rel_te, sim, pop2, nI, K)
    return {"model": name, **r}

rows = [
    eval_scores("Random", np.random.default_rng(0).random((nU, nI))),
    eval_scores("Popularity (Bayesian WR)", comp2["pop"]),
    eval_scores("Content-based (Rocchio+TF-IDF)", comp2["content"]),
    eval_scores("User-KNN", comp2["user_knn"]),
    eval_scores("Item-KNN", comp2["item_knn"]),
    eval_scores("Biased MF (SGD, rating-oriented)", comp2["mf"]),
    eval_scores("Implicit ALS", comp2["als"]),
    eval_scores("Hybrid (weighted)", H),
]
# hybrid + equipment hard-filter (users declare their equipment in a real app)
ok_eq = np.stack([constraint_mask(cat, EQUIPMENT_PROFILES[p]) for p in users.equipment_profile])
rows.append(eval_scores("Hybrid + equipment filter", np.where(ok_eq, H, -np.inf)))
recs_mmr = np.array([(mmr(np.where(ok_eq[u], H[u], -np.inf) - 1e9 * (M2[u] > 0), content.X, K, 0.7, ok_eq[u] & (M2[u] == 0))
                      + [0] * K)[:K] for u in all_users])
rows.append(eval_scores("Hybrid + filter + MMR(0.7)", None, recs_mmr))
res = pd.DataFrame(rows).set_index("model").round(4)
res.to_csv(RESULTS / "ranking_metrics.csv")
print(res.to_string())

# ---- rating prediction (RMSE/MAE) on test
te_i = idx_df(te)
acc = {}
for name, m in [("Global mean", GlobalMean().fit(R2, M2)), ("User mean", UserMean().fit(R2, M2)),
                ("User-KNN", models2["user_knn"]), ("Item-KNN", models2["item_knn"]), ("Biased MF", models2["mf"])]:
    acc[name] = rating_errors(m, te_i)
acc = pd.DataFrame(acc, index=["RMSE", "MAE"]).T.round(4)
acc.to_csv(RESULTS / "rating_prediction.csv")
print(acc.to_string())

# ---- by user activity (cold-start analysis)
bucket = pd.cut(n2, [0, 15, 30, 60, 1000], labels=["<=15", "16-30", "31-60", ">60"])
rowsb = []
for b in bucket.categories:
    sel = {u: r for u, r in rel_te.items() if bucket[u] == b}
    for name, S in [("Content", comp2["content"]), ("Item-KNN", comp2["item_knn"]), ("ALS", comp2["als"]), ("Hybrid", H)]:
        rowsb.append({"ratings_in_history": b, "model": name, "n_users": len(sel),
                      f"NDCG@{K}": ranking_metrics(top_k(S, M2, K), sel, k=K)[f"NDCG@{K}"]})
pd.DataFrame(rowsb).round(4).to_csv(RESULTS / "by_user_activity.csv", index=False)
print(pd.DataFrame(rowsb).pivot(index="ratings_in_history", columns="model", values=f"NDCG@{K}").round(4))
print(f"done in {time.time()-t0:.0f}s")
