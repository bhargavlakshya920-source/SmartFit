# SmartFit - Hybrid Gym Workout Recommender

Recommends individual exercises **and builds full weekly workout plans** from three exercise catalogs
(megaGym 2,918 rows + bodybuilding.com sheet 471 rows + Top-50 sheet), plus profile/age handling informed by the Hugging Face planner dataset using a hybrid recommender
(content-based + collaborative + popularity) with constraint-aware re-ranking.

```
python -m pip install -r requirements.txt
python run_pipeline.py          # builds catalog, simulates users, tunes + evaluates, writes results/
python -m src.eda               # figures for the report
python -m src.planner_analysis  # audit of the Coach-Plan Corpus (Hugging Face data) + leave-one-out demographic test
streamlit run app/streamlit_app.py
pytest -q tests
```

## Accounts, ratings and progress (new)
The app now has real accounts (`src/auth_db.py`, SQLite file `data/app.db`, created on first run):
- **Create account / Sign in / Sign out**, or continue as guest (guests can plan and browse, nothing is saved).
- Each user has a private profile, **star ratings**, **workout log** (weight, reps, RPE), **body-weight log** and **saved plans**.
- **Recommendations adapt per user without retraining**: the user's ratings (and logged sessions, as implicit ratings) are
  folded into the trained models (`GymRecommender.user_components`): Rocchio profile, item-KNN, user-KNN, ALS and MF fold-in.
  The personal share of the score grows with history (0 -> 0.7 at 20 rated/logged exercises); before that the profile drives it.
- Pages: Today, Plan, For you, Exercises (dropdown of all 2,906 exercises with body-part/equipment/type filters), Log, Progress, Account, Methodology.
- Security: scrypt-hashed passwords with per-user salt, parameterised SQL, 5-attempt lockout, export and delete-account.
  It is a **local demo**: no e-mail verification/reset and no HTTPS. For a public deployment use HTTPS plus a managed identity provider.
- Ratings/logs reference catalogue `item_id`s; do not re-order the catalogue (rebuilding from the same raw files is safe).
- Back up or delete `data/app.db` to keep or reset all user data.

## Architecture
```
raw CSV/XLSX --> data.py (clean, merge, dedupe) --> catalog.csv (2,906 exercises)
                                   |                         |
                       content.py (TF-IDF + one-hot)    simulate.py / strong_adapter.py  --> interactions
                                   |                         |
        popularity.py  cf.py (ItemKNN, UserKNN, Biased MF, Implicit ALS)
                                   \\                       /
                              hybrid.py (z-score blend, cold-start confidence)
                                           |
                    rerank.py (equipment/level/goal filters, MMR)  -->  workout.py (session + weekly plan)
                                           |
                    evaluate.py  <-- run_pipeline.py            app/streamlit_app.py
```

## Techniques and formulae
| Module | Technique | Formula |
|---|---|---|
| content.py | TF-IDF + one-hot, cosine | tfidf = (1+log tf)·log(N/df); cos(a,b) = a·b/(‖a‖‖b‖) |
| content.py | Rocchio user profile | p_u = Σ (r_ui − r̄_u) x_i ; score = cos(p_u, x_j) |
| popularity.py | Bayesian weighted rating | WR = v/(v+m)·R + m/(v+m)·C |
| cf.py | Item-KNN (adjusted cosine + shrinkage) | sim(i,j) = Σ(r_ui−r̄_u)(r_uj−r̄_u)/(‖·‖‖·‖); s(u,i) = Σ_{j∈N_k(i)} sim·p_uj |
| cf.py | User-KNN | s(u,i) = Σ_{v∈N_k(u)} cos(u,v)·p_vi |
| cf.py | Biased MF (SGD) | r̂ = μ + b_u + b_i + p_u·q_i ; min Σ(r−r̂)² + λ(‖p‖²+‖q‖²+b²) |
| cf.py | Implicit ALS | min Σ c_ui (p_ui − x_u·y_i)² + λ(‖X‖²+‖Y‖²), c = 1+α·r⁺ |
| hybrid.py | Weighted hybrid, grid-searched on validation NDCG | s = Σ w_m·c_m(u)·z_m(u,i), c_u = min(1, n_u/n0) |
| rerank.py | MMR diversification | MMR(i) = λ·rel(i) − (1−λ)·max_{j∈S} cos(i,j) |
| progression.py | Epley 1RM, RIR-adjusted next load | 1RM = w(1+reps/30); w_next = 1RM/(1+(reps_t+RIR)/30) |
| demographic.py | Demographic filtering (kernel-weighted neighbour vote over 24 exercise families) | K(q,p)=exp(-d²/2h²); score_f = ΣK·y_pf / ΣK |
| age_policy.py | Rule layer: rest ×, sets Δ, avoided types/titles, warm-up count, RPE target by age band | heuristic, NOT learned |
| evaluate.py | RMSE, MAE, P/R/MAP/NDCG@K, HitRate, coverage, ILD, novelty | see file docstring |

## Results (temporal split, K=10, 1,200 simulated users - see `results/`)
| Model | P@10 | NDCG@10 | HitRate@10 |
|---|---|---|---|
| Random | 0.0022 | 0.0029 | 0.022 |
| Popularity | 0.0089 | 0.0140 | 0.080 |
| Content-based | 0.0060 | 0.0088 | 0.057 |
| User-KNN | 0.0124 | 0.0211 | 0.116 |
| Item-KNN | 0.0101 | 0.0161 | 0.096 |
| Implicit ALS | 0.0118 | 0.0194 | 0.112 |
| **Hybrid** | **0.0160** | **0.0232** | **0.143** |
| Hybrid + equipment filter | 0.0169 | 0.0244 | 0.150 |

Rating prediction RMSE: global mean 1.268, user mean 1.229, **biased MF 1.082**. The hybrid wins on ranking;
MF wins on rating error - report both, they answer different questions. Re-run `run_pipeline.py` to regenerate.

## Coach-Plan Corpus (Hugging Face "Workout Planner" data, main_workout_planner.jsonl) - what we found and how it is used
*Shown in the app as "Coach-Plan Corpus". Source: Hugging Face "Workout Planner" dataset curated by Navaneeth K from several datasets and generated data - credit the original sources when reusing it. Internal file names still start with `planner_`.*
Parsed by `src/planner_data.py` into profiles (age, sex, height, weight, target weight, goal, level) and plan items.
**Audit (results/planner_audit.json, app tab "Planner dataset audit"):**
- 1,800 rows are only **100 unique** profile/plan pairs, each duplicated 18x. We de-duplicate; evaluating on the raw
  file would leak answers through identical twins.
- Plans are drawn from a pool of 24 exercises, always 3 sets, rest days always Wed/Sat/Sun. Target weights are
  inconsistent with goals except for muscle gain (+5 kg) and weight loss (-5 kg); BMI ranges from 12.9 to 39.
- The dataset card says it was created from other datasets and **generated** (ChatGPT/DeepSeek/Claude) data.
- Chi-square / ANOVA tests: exercise choice and reps do **not** depend on age band (p=0.49 / 0.59), goal, level or sex
  (all p>0.39). Leave-one-profile-out: the demographic model (NDCG@5 0.40) is statistically indistinguishable from
  global popularity (0.38) and from random (0.44); bootstrap CI of the difference to popularity includes 0.

**So it cannot teach the system how different ages train.** What it *is* used for:
1. Richer user profile in the app (age, sex, height/weight -> BMI, goal incl. "General fitness", level) and a 7-day calendar with rest days.
2. 24 canonical staple movements mapped to catalogue "families" (`family_map`), all 24 matched.
3. A demographic-filtering component (a standard recommender technique) wired in with **weight 0 by default**, because
   the test above shows it adds nothing; a slider lets you demonstrate it.
4. A reference panel showing the nearest dataset profile's plan (clearly labelled LLM-generated).

**Age-aware behaviour comes from `src/age_policy.py`**, an explicit rule layer (more rest, fewer sets, no high-impact types
or risky lifts such as behind-the-neck presses for older bands, more mobility work, lower target RPE, fewer max training days).
These are conservative coaching heuristics, not learned and not medical advice; cite a guideline (e.g. ACSM/NSCA)
of your choice in the report, and consider a real age-labelled workout log to learn them properly.

## IMPORTANT limitations (state these in your report)
1. **The datasets contain no user-exercise interactions**, so users/ratings are *simulated* (`src/simulate.py`,
   generative story in its docstring). Absolute metrics are therefore only meaningful for *comparing models*,
   not as real-world accuracy. For real data, export a Strong-app log and use `src/strong_adapter.py`.
2. `Level` is 84% "Intermediate" in megaGym, so level filtering is weak; ~73% of exercises have no rating.
3. Catalog is a merge of three sources with name matching - a few muscle labels (e.g. Top-50 deadlift -> Lats) are approximate.
4. The planner dataset is synthetic with no age signal (above); age handling is rule-based.
5. Not medical or coaching advice; progression suggestions are formula-based.

## Extending
SBERT description embeddings; BPR / LightFCN; contextual bandits for exploration; knowledge-graph features;
LLM-generated explanations; FastAPI backend + React front end; A/B test on real users.
