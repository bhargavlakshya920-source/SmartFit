import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from datetime import date, datetime
from html import escape

import numpy as np
import pandas as pd
import streamlit as st

from app.style import CSS
from src import auth_db as auth
from src.age_policy import POLICIES, policy_for
from src.config import EQUIPMENT_PROFILES, LEVELS, PLAN_GOALS, RESULTS
from src.data import load_catalog
from src.progression import epley_1rm, next_load
from src.simulate import load_or_simulate
from src.system import GymRecommender
from src.workout import build_week, calendar_week

st.set_page_config(page_title="SmartFit", page_icon="●", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)
ss = st.session_state
ss.setdefault("uid", None)
ss.setdefault("guest", False)


@st.cache_resource(show_spinner="Preparing models (first launch only)...")
def get_system():
    cat = load_catalog()
    users, inter = load_or_simulate(cat)
    return GymRecommender(cat, users, inter)


rec = get_system()
cat = rec.cat
SEX = ["female", "male"]
EQUIP = list(EQUIPMENT_PROFILES)


# ============================================================================ helpers
def tags(*items):
    return "".join(f'<span class="tag">{escape(str(i))}</span>' for i in items if i)


def label(i):
    r = cat.iloc[int(i)]
    return f"{r.title}  -  {r.bodypart}, {r.equipment}"


def user():
    return auth.get_user(ss.uid) if ss.uid else None


def sign_out():
    for k in ["uid", "guest", "nav", "plan_gen", "foryou"]:
        ss.pop(k, None)


def go_log(item_id):
    ss["nav"] = "Log"
    ss["log_item"] = int(item_id)


def exercise_picker(key, default_item=None):
    """Searchable dropdown listing EVERY exercise, narrowed by optional body-part / equipment / type filters."""
    c1, c2, c3 = st.columns(3)
    bp = c1.selectbox("Body part", ["All"] + sorted(cat.bodypart.unique()), key=f"{key}_bp")
    eq = c2.selectbox("Equipment", ["All"] + sorted(cat.equipment.unique()), key=f"{key}_eq")
    ty = c3.selectbox("Type", ["All"] + sorted(cat["type"].unique()), key=f"{key}_ty")
    d = cat
    if bp != "All": d = d[d.bodypart == bp]
    if eq != "All": d = d[d.equipment == eq]
    if ty != "All": d = d[d["type"] == ty]
    d = d.sort_values("title")
    ids = d.item_id.tolist()
    idx = ids.index(default_item) if default_item in ids else None
    st.caption(f"{len(ids):,} exercises available. Open the list and scroll, or start typing to search.")
    return st.selectbox("Exercise", ids, index=idx, format_func=label, key=f"{key}_pick",
                        placeholder="Choose an exercise...")


def stars(item_id, key):
    """Star rating bound to the signed-in user; writes only when the value changes."""
    if not ss.uid:
        return
    if "ratings_map" not in ss or ss.get("ratings_uid") != ss.uid:
        df = auth.get_ratings(ss.uid)
        ss.ratings_map, ss.ratings_uid = dict(zip(df.item_id, df.rating)), ss.uid
    cur = ss.ratings_map.get(int(item_id))
    k = f"star_{key}_{item_id}"
    if k not in ss and cur:
        ss[k] = cur - 1
    v = st.feedback("stars", key=k)
    if v is not None and v + 1 != cur:
        auth.set_rating(ss.uid, item_id, v + 1)
        ss.ratings_map[int(item_id)] = v + 1


def user_vector():
    return rec.interaction_vector(auth.get_ratings(ss.uid), auth.get_logs(ss.uid))


def show_session(plan, summ):
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated time", f"{summ['est_minutes']} min")
    c2.caption("Warm-up: " + (", ".join(summ["warmup"]) or "none"))
    c3.caption("Cool-down: " + (", ".join(summ["cooldown"]) or "none"))
    st.dataframe(plan[["order", "exercise", "muscle", "equipment", "sets", "reps", "rest_s"]].rename(
        columns={"order": "#", "rest_s": "rest (s)"}), hide_index=True, width="stretch")


def week_payload(week, params):
    days = []
    for wd, item in calendar_week(week).items():
        if item:
            name, (plan, summ) = item
            days.append(dict(weekday=wd, name=name, summary={k: summ[k] for k in ["est_minutes", "warmup", "cooldown"]},
                             rows=plan[["exercise", "muscle", "equipment", "sets", "reps", "rest_s"]].to_dict("records")))
    return dict(params=params, days=days)


# ============================================================================ header + auth gate
h1, h2 = st.columns([3, 2])
h1.markdown('<div class="brand">Smart<span>Fit</span></div><div class="muted">Personalised gym programming</div>',
            unsafe_allow_html=True)
me = user()
if ss.uid and me is None:
    sign_out(); st.rerun()
if me:
    with h2:
        a, b = st.columns([3, 2])
        a.markdown(f'<div class="muted" style="text-align:right;padding-top:10px">{escape(me["display_name"] or me["username"])}</div>',
                   unsafe_allow_html=True)
        b.button("Sign out", on_click=sign_out, width="stretch")
elif ss.guest:
    with h2:
        st.button("Sign in", on_click=lambda: ss.update(guest=False), width="stretch")
st.markdown("<hr>", unsafe_allow_html=True)

if not me and not ss.guest:
    st.markdown("## Train smarter, with a plan that learns from you")
    st.markdown('<p class="muted">Sign in to keep your ratings, workout log and progress. Your recommendations adapt '
                'as you rate exercises and log sessions.</p>', unsafe_allow_html=True)
    t_in, t_new = st.tabs(["Sign in", "Create account"])
    with t_in:
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Sign in", type="primary"):
                ok, msg, uid = auth.login(u, p)
                if ok:
                    ss.uid = uid; ss.pop("nav", None); st.rerun()
                else:
                    st.error(msg)
    with t_new:
        with st.form("register"):
            c1, c2 = st.columns(2)
            nu = c1.text_input("Username", help="3-24 letters, numbers or underscore")
            dn = c2.text_input("Display name")
            c1, c2 = st.columns(2)
            pw = c1.text_input("Password", type="password", help="At least 8 characters")
            pw2 = c2.text_input("Confirm password", type="password")
            c1, c2, c3, c4 = st.columns(4)
            age = c1.number_input("Age", 18, 90, 30)
            sex = c2.selectbox("Sex", SEX)
            hgt = c3.number_input("Height (cm)", 120, 230, 172)
            wgt = c4.number_input("Weight (kg)", 30, 250, 72)
            c1, c2, c3 = st.columns(3)
            goal = c1.selectbox("Goal", PLAN_GOALS, index=1)
            lvl = c2.selectbox("Experience", LEVELS, index=1)
            eqp = c3.selectbox("Equipment", EQUIP, index=2)
            if st.form_submit_button("Create account", type="primary"):
                if pw != pw2:
                    st.error("Passwords do not match.")
                else:
                    ok, res = auth.create_user(nu, pw, display_name=dn or nu, age=int(age), sex=sex, height_cm=float(hgt),
                                               weight_kg=float(wgt), goal=goal, level=lvl, equipment_profile=eqp)
                    if ok:
                        ss.uid = res; ss.pop("nav", None); st.rerun()
                    else:
                        st.error(res)
    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button("Continue as guest"):
        ss.guest = True; st.rerun()
    st.caption("Guests can build plans and browse exercises, but nothing is saved.")
    st.stop()

PAGES = (["Today", "Plan", "For you", "Exercises", "Log", "Progress", "Account", "Methodology"] if me
         else ["Plan", "Exercises", "Methodology"])
nav = st.segmented_control("Navigation", PAGES, default=PAGES[0], key="nav", label_visibility="collapsed")
nav = nav if nav in PAGES else PAGES[0]
st.write("")


# ============================================================================ pages
def page_today():
    hour = datetime.now().hour
    st.markdown(f"## Good {'morning' if hour < 12 else 'afternoon' if hour < 18 else 'evening'}, "
                f"{escape(me['display_name'] or me['username'])}")
    logs, rt = auth.get_logs(ss.uid), auth.get_ratings(ss.uid)
    if len(logs):
        logs["day"] = pd.to_datetime(logs["day"]); logs["vol"] = logs.weight * logs.reps
        wk = logs[logs.day >= pd.Timestamp(date.today()) - pd.Timedelta(days=date.today().weekday())]
        m30 = logs[logs.day >= pd.Timestamp(date.today()) - pd.Timedelta(days=30)]
    else:
        wk = m30 = logs
    c = st.columns(4)
    c[0].metric("Sessions this week", f"{wk.day.nunique() if len(wk) else 0} / {me['days']}")
    c[1].metric("Sets, last 30 days", f"{len(m30):,}")
    c[2].metric("Volume, last 30 days", f"{(m30.vol.sum() if len(m30) else 0):,.0f} kg")
    c[3].metric("Exercises rated", f"{len(rt)}")
    st.write("")
    st.markdown("### Today's session")
    plans = auth.list_plans(ss.uid)
    wd = date.today().strftime("%A")
    if not plans:
        st.info("You have no saved plan yet. Build one in the Plan tab and save it to see today's session here.")
    else:
        pl = json.loads(plans[0]["payload"])
        today = next((d for d in pl["days"] if d["weekday"] == wd), None)
        st.caption(f"Active plan: {plans[0]['name']}")
        if today is None:
            st.markdown(f"**{wd}** is a rest day in your plan. Light walking or mobility work is a good option.")
        else:
            with st.container(border=True):
                st.markdown(f"**{today['name']}**  -  about {today['summary']['est_minutes']} min")
                st.dataframe(pd.DataFrame(today["rows"]).rename(columns={"rest_s": "rest (s)"}), hide_index=True, width="stretch")
                st.caption("Warm-up: " + ", ".join(today["summary"]["warmup"]) + "   |   Cool-down: " + ", ".join(today["summary"]["cooldown"]))
    if len(logs):
        st.markdown("### Recent activity")
        r = logs.groupby(["day", "item_id"]).agg(sets=("set_no", "count"), top=("weight", "max"), reps=("reps", "max")).reset_index()
        r["exercise"] = r.item_id.map(cat.set_index("item_id").title)
        r["day"] = r.day.dt.strftime("%d %b")
        st.dataframe(r.sort_values("day", ascending=False).head(8)[["day", "exercise", "sets", "top", "reps"]]
                     .rename(columns={"top": "top weight (kg)", "reps": "best reps"}), hide_index=True, width="stretch")


def page_plan():
    st.markdown("## Build a weekly plan")
    d = me or {}
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        goal = c1.selectbox("Goal", PLAN_GOALS, index=PLAN_GOALS.index(d.get("goal") or "Hypertrophy"))
        level = c2.selectbox("Experience", LEVELS, index=LEVELS.index(d.get("level") or "Intermediate"))
        eq_name = c3.selectbox("Equipment", EQUIP, index=EQUIP.index(d.get("equipment_profile") or EQUIP[2]))
        if me:
            age, sex, bmi = d["age"], d["sex"], d["weight_kg"] / (d["height_cm"] / 100) ** 2
        else:
            c1, c2, c3 = st.columns(3)
            age = c1.number_input("Age", 18, 80, 35); sex = c2.selectbox("Sex", SEX)
            bmi = c3.number_input("BMI", 15.0, 40.0, 24.0, 0.5)
        c1, c2 = st.columns(2)
        days = c1.slider("Training days per week", 2, 6, int(d.get("days") or 3))
        minutes = c2.slider("Minutes per session", 30, 90, int(d.get("minutes") or 60), step=15)
        kw = st.text_input("Preferences (optional)", placeholder="e.g. explosive, shoulder friendly, unilateral")
        pol = policy_for(age)
        st.caption(f"Age band {pol.band}: rest x{pol.rest_mult}, up to {pol.max_days} training days a week, "
                   f"{pol.warmup_n} warm-up moves" + (", high-impact exercises excluded" if pol.avoid_types else "") + ".")
        go = st.button("Generate plan", type="primary")
    if go:
        fn = rec.account_score_fn(*user_vector(), me) if me else None
        wk = build_week(rec, goal, level, EQUIPMENT_PROFILES[eq_name], days, minutes, None, kw, age=age, sex=sex, bmi=bmi, score_fn=fn)
        ss.plan_gen = dict(week=wk, params=dict(goal=goal, level=level, equipment=eq_name, days=len(wk), minutes=minutes, age=age))
    gen = ss.get("plan_gen")
    if not gen:
        return
    if len(gen["week"]) < days:
        st.info(f"For your age band the plan is capped at {len(gen['week'])} training days.")
    for wd, item in calendar_week(gen["week"]).items():
        if item is None:
            st.markdown(f'<div class="muted" style="padding:6px 2px">{wd} - rest</div>', unsafe_allow_html=True)
            continue
        name, (plan, summ) = item
        with st.expander(f"{wd}  -  {name.split(' - ')[-1]}  ({summ['est_minutes']} min)"):
            show_session(plan, summ)
    if me:
        st.write("")
        c1, c2 = st.columns([3, 1])
        nm = c1.text_input("Plan name", value=f"{gen['params']['goal']} {gen['params']['days']}-day", label_visibility="collapsed")
        if c2.button("Save as active plan", width="stretch"):
            auth.save_plan(ss.uid, nm, week_payload(gen["week"], gen["params"]))
            st.success("Saved. It now appears on your Today page.")
    else:
        st.caption("Sign in to save this plan.")
    if me:
        saved = auth.list_plans(ss.uid)
        if saved:
            with st.expander(f"Saved plans ({len(saved)})"):
                for p in saved:
                    c1, c2 = st.columns([4, 1])
                    c1.write(f"{p['name']}  -  {p['created_at'][:10]}")
                    if c2.button("Delete", key=f"dp{p['id']}"):
                        auth.delete_plan(ss.uid, p["id"]); st.rerun()


def page_foryou():
    st.markdown("## Recommended for you")
    r, m = user_vector()
    n = int(m.sum())
    st.markdown(f'<p class="muted">{"Based on your profile. Rate exercises or log workouts and these will adapt." if n < 3 else f"Learned from {n} exercises you rated or train."}</p>',
                unsafe_allow_html=True)
    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        muscles = c1.multiselect("Focus muscles (optional)", sorted(cat.bodypart.unique()))
        k = c2.number_input("How many", 5, 20, 10)
        lam = st.slider("Variety", 0.3, 1.0, 0.7, help="Lower = more varied, higher = closest match")
        refresh = st.button("Update recommendations", type="primary")
    key = (tuple(muscles), k, lam)
    if refresh or ss.get("foryou", {}).get("key") != key:
        ss.foryou = dict(key=key, view=rec.recommend_account(r, m, me, me["goal"], me["level"], EQUIPMENT_PROFILES[me["equipment_profile"]],
                                                              int(k), muscles or None, lam < 1.0, lam))
    for _, row in ss.foryou["view"].iterrows():
        with st.container(border=True):
            c1, c2 = st.columns([5, 3])
            c1.markdown(f'<div class="title">{escape(row.title)}</div>{tags(row.bodypart, row.equipment, row.level, row.type)}'
                        + (f'<div class="reason">{escape(row.because)}</div>' if row.because else ""), unsafe_allow_html=True)
            with c2:
                stars(row.item_id, "fy")
                st.button("Log it", key=f"lg{row.item_id}", on_click=go_log, args=(row.item_id,))


def page_exercises():
    st.markdown("## Exercise library")
    pick = exercise_picker("lib", ss.get("lib_default"))
    if pick is None:
        st.markdown('<p class="muted">Pick an exercise to see details and similar movements.</p>', unsafe_allow_html=True)
        return
    r = cat.iloc[int(pick)]
    with st.container(border=True):
        c1, c2 = st.columns([3, 2])
        c1.markdown(f'<div class="title" style="font-size:1.2rem">{escape(r.title)}</div>{tags(r.bodypart, r.equipment, r.level, r.type)}',
                    unsafe_allow_html=True)
        c1.write(r.desc if r.desc else "No description is available for this exercise.")
        if r.benefit: c1.caption(f"Benefit: {r.benefit}")
        if pd.notna(r.rating): c1.caption(f"Catalogue rating {r.rating:.1f} / 10")
        if isinstance(r.image_url, str) and r.image_url.startswith("http"):
            c2.image(r.image_url, width="stretch")
        if me:
            c1, c2, _ = st.columns([2, 2, 3])
            with c1: stars(pick, "lib")
            c2.button("Log this exercise", on_click=go_log, args=(pick,))
    st.markdown("### Similar exercises")
    mode = st.radio("Similarity", ["hybrid", "content"], horizontal=True, label_visibility="collapsed")
    sim = rec.similar_items(int(pick), 8, mode)
    st.dataframe(sim[["title", "bodypart", "equipment", "level"]], hide_index=True, width="stretch")


def page_log():
    st.markdown("## Log a workout")
    pick = exercise_picker("log", ss.pop("log_item", None))
    c1, c2 = st.columns([1, 3])
    day = c1.date_input("Date", date.today(), max_value=date.today())
    if pick is None:
        st.markdown('<p class="muted">Choose an exercise to log your sets.</p>', unsafe_allow_html=True)
    else:
        logs = auth.get_logs(ss.uid)
        mine = logs[logs.item_id == int(pick)]
        if len(mine):
            last = mine[mine.day == mine.day.max()].sort_values("weight").iloc[-1]
            tgt = policy_for(me["age"]).target_rpe
            c2.markdown(f'<div class="muted" style="padding-top:28px">Last time ({last.day}): {last.weight:g} kg x {int(last.reps)}. '
                        f'Suggested next top set: <b>{next_load(last.weight, last.reps, last.rpe if pd.notna(last.rpe) else None, 8, tgt):g} kg</b> for 8 reps.</div>',
                        unsafe_allow_html=True)
        sets = st.data_editor(pd.DataFrame({"weight_kg": [0.0] * 3, "reps": [0] * 3, "rpe": [None] * 3}), num_rows="dynamic",
                              width="stretch", key=f"sets_{pick}",
                              column_config={"weight_kg": st.column_config.NumberColumn("Weight (kg)", min_value=0.0, max_value=600.0, step=0.5),
                                             "reps": st.column_config.NumberColumn("Reps", min_value=0, max_value=200, step=1),
                                             "rpe": st.column_config.NumberColumn("RPE (optional)", min_value=1.0, max_value=10.0, step=0.5)})
        if st.button("Save sets", type="primary"):
            n = auth.add_sets(ss.uid, day.isoformat(), pick, sets.itertuples(index=False, name=None))
            st.success(f"Saved {n} set(s).") if n else st.warning("Enter at least one set with reps above 0.")
    logs = auth.get_logs(ss.uid)
    if len(logs):
        st.markdown("### History")
        h = logs.groupby(["day", "item_id"]).agg(sets=("set_no", "count"), top_kg=("weight", "max"), best_reps=("reps", "max")).reset_index()
        h["exercise"] = h.item_id.map(cat.set_index("item_id").title)
        st.dataframe(h.sort_values("day", ascending=False).head(30)[["day", "exercise", "sets", "top_kg", "best_reps"]], hide_index=True, width="stretch")
        with st.expander("Delete an entry"):
            opts = h.sort_values("day", ascending=False).head(30)
            sel = st.selectbox("Entry", opts.index, format_func=lambda i: f"{opts.loc[i, 'day']}  -  {opts.loc[i, 'exercise']}")
            if st.button("Delete entry"):
                auth.delete_logs(ss.uid, opts.loc[sel, "day"], opts.loc[sel, "item_id"]); st.rerun()


def page_progress():
    st.markdown("## Progress")
    logs = auth.get_logs(ss.uid)
    if logs.empty:
        st.info("Log a few workouts and your strength, volume and consistency charts will appear here.")
    else:
        logs["day"] = pd.to_datetime(logs["day"])
        logs["name"] = logs.item_id.map(cat.set_index("item_id").title)
        logs["vol"] = logs.weight * logs.reps
        logs["e1rm"] = np.where((logs.weight > 0) & (logs.reps <= 20), epley_1rm(logs.weight, logs.reps), np.nan)
        c = st.columns(3)
        c[0].metric("Workout days", logs.day.nunique()); c[1].metric("Total sets", f"{len(logs):,}")
        c[2].metric("Total volume", f"{logs.vol.sum():,.0f} kg")
        st.markdown("### Strength trend")
        weighted = logs.dropna(subset=["e1rm"])
        if len(weighted):
            ex = st.selectbox("Exercise", sorted(weighted.name.unique()))
            s = weighted[weighted.name == ex].groupby("day").e1rm.max()
            st.line_chart(s, y_label="Estimated 1RM (kg)")
        else:
            st.caption("Log sets with a weight to see estimated 1RM.")
        st.markdown("### Weekly volume and sessions")
        wk = logs.groupby(logs.day.dt.to_period("W").dt.start_time).agg(volume_kg=("vol", "sum"), sessions=("day", "nunique"))
        a, b = st.columns(2)
        a.bar_chart(wk.volume_kg, y_label="kg lifted"); b.bar_chart(wk.sessions, y_label="sessions")
        st.markdown("### Personal bests")
        pb = weighted.loc[weighted.groupby("name").e1rm.idxmax()][["name", "day", "weight", "reps", "e1rm"]].sort_values("e1rm", ascending=False).head(10)
        pb["day"] = pb.day.dt.strftime("%d %b %Y"); pb["e1rm"] = pb.e1rm.round(1)
        st.dataframe(pb.rename(columns={"name": "exercise", "e1rm": "est. 1RM"}), hide_index=True, width="stretch")
    st.markdown("### Body weight")
    c1, c2, c3 = st.columns([2, 2, 1])
    bd = c1.date_input("Date", date.today(), key="bwd", max_value=date.today())
    bk = c2.number_input("Weight (kg)", 30.0, 250.0, float(me["weight_kg"] or 70), 0.1)
    c3.write(""); c3.write("")
    if c3.button("Add"):
        auth.add_bodyweight(ss.uid, bd.isoformat(), bk); st.rerun()
    bw = auth.get_bodyweight(ss.uid)
    if len(bw):
        st.line_chart(bw.assign(day=pd.to_datetime(bw.day)).set_index("day").kg, y_label="kg")


def page_account():
    st.markdown("## Account")
    with st.form("profile"):
        c1, c2 = st.columns(2)
        dn = c1.text_input("Display name", me["display_name"] or "")
        age = c2.number_input("Age", 18, 90, int(me["age"] or 30))
        c1, c2, c3 = st.columns(3)
        sex = c1.selectbox("Sex", SEX, index=SEX.index(me["sex"] or "female"))
        hgt = c2.number_input("Height (cm)", 120, 230, int(me["height_cm"] or 170))
        wgt = c3.number_input("Weight (kg)", 30, 250, int(me["weight_kg"] or 70))
        c1, c2, c3 = st.columns(3)
        goal = c1.selectbox("Goal", PLAN_GOALS, index=PLAN_GOALS.index(me["goal"] or "Hypertrophy"))
        lvl = c2.selectbox("Experience", LEVELS, index=LEVELS.index(me["level"] or "Intermediate"))
        eqp = c3.selectbox("Equipment", EQUIP, index=EQUIP.index(me["equipment_profile"] or EQUIP[2]))
        c1, c2 = st.columns(2)
        days = c1.slider("Training days per week", 1, 7, int(me["days"] or 3))
        mins = c2.slider("Minutes per session", 15, 180, int(me["minutes"] or 60), step=5)
        if st.form_submit_button("Save profile", type="primary"):
            ok, msg = auth.update_profile(ss.uid, display_name=dn, age=int(age), sex=sex, height_cm=float(hgt), weight_kg=float(wgt),
                                          goal=goal, level=lvl, equipment_profile=eqp, days=int(days), minutes=int(mins))
            st.success("Profile updated.") if ok else st.error(msg)
    with st.expander("Change password"):
        with st.form("pw"):
            o = st.text_input("Current password", type="password")
            n1 = st.text_input("New password", type="password")
            if st.form_submit_button("Update password"):
                ok, msg = auth.change_password(ss.uid, o, n1)
                st.success("Password updated.") if ok else st.error(msg)
    with st.expander("Your data"):
        data = auth.export_all(ss.uid)
        st.download_button("Download everything (JSON)", json.dumps({k: v.to_dict("records") for k, v in data.items()}, indent=1, default=str),
                           file_name="smartfit_data.json", mime="application/json")
        st.caption("Delete your account and all of its ratings, logs and plans permanently.")
        conf = st.text_input("Type your username to confirm", key="delconf")
        if st.button("Delete my account") and conf == me["username"]:
            auth.delete_user(ss.uid); sign_out(); st.rerun()


def page_method():
    st.markdown("## Methodology")
    t_models, t_data, t_math = st.tabs(["Models", "Data", "Mathematics"])
    with t_models:
        st.markdown('<p class="muted">Offline evaluation on a temporal split (K = 10). User histories used for training are simulated '
                    '(see Data), so compare models with each other rather than reading absolute values as real-world accuracy.</p>', unsafe_allow_html=True)
        f = RESULTS / "ranking_metrics.csv"
        if f.exists():
            st.dataframe(pd.read_csv(f, index_col=0), width="stretch")
            for img in ["model_comparison.png", "mf_training_curve.png"]:
                if (RESULTS / img).exists(): st.image(str(RESULTS / img))
            for csv, t in [("rating_prediction.csv", "Rating prediction error"), ("by_user_activity.csv", "NDCG@10 by amount of history")]:
                if (RESULTS / csv).exists():
                    st.markdown(f"### {t}"); st.dataframe(pd.read_csv(RESULTS / csv), hide_index=True, width="stretch")
        else:
            st.warning("Run `python run_pipeline.py` first.")
    with t_data:
        st.markdown("### Exercise catalogue")
        st.markdown(f"{len(cat):,} exercises merged from three sources: megaGymDataset, a bodybuilding.com exercise sheet and a Top-50 exercise list.")
        if (RESULTS / "eda_overview.png").exists(): st.image(str(RESULTS / "eda_overview.png"))
        st.markdown("### Coach-Plan Corpus (external reference data)")
        st.caption("Source: Hugging Face 'Workout Planner' dataset (main_workout_planner.jsonl), curated by Navaneeth K from several "
                   "datasets and generated data. Credit the original data sources when you reuse it.")
        af = RESULTS / "planner_audit.json"
        if af.exists():
            au = json.loads(af.read_text())
            c = st.columns(3)
            c[0].metric("Rows in file", au["raw_rows"]); c[1].metric("Unique profiles", au["unique_profiles"]); c[2].metric("Distinct exercises", au["n_exercises"])
            st.warning(au["conclusion"])
            st.markdown("**Significance tests** (large p-value: no detectable dependence)")
            st.dataframe(pd.Series(au["tests"], name="p-value").round(3).to_frame(), width="stretch")
            st.markdown("**Leave-one-profile-out test: can demographics predict a profile's exercises?**")
            st.dataframe(pd.read_csv(RESULTS / "planner_demographic_eval.csv", index_col=0), width="stretch")
            if (RESULTS / "planner_audit.png").exists(): st.image(str(RESULTS / "planner_audit.png"))
        st.markdown("### Age rules")
        st.caption("Heuristic rules, editable in src/age_policy.py. They are not learned from data and are not medical advice.")
        st.dataframe(pd.DataFrame([vars(p) for p in POLICIES.values()]).set_index("band"), width="stretch")
    with t_math:
        L = st.latex
        L(r"\mathrm{tfidf}(t,d)=(1+\log \mathrm{tf})\cdot\log\frac{N}{\mathrm{df}(t)}\qquad \cos(a,b)=\frac{a\cdot b}{\|a\|\|b\|}")
        L(r"p_u=\sum_i (r_{ui}-\bar r_u)\,x_i,\qquad s_{CB}(u,j)=\cos(p_u,x_j)")
        L(r"\mathrm{WR}_i=\frac{v_i}{v_i+m}R_i+\frac{m}{v_i+m}C")
        L(r"s(u,i)=\sum_{j\in N_k(i)}\cos(i,j)\,p_{uj},\quad p_{uj}=\max(r_{uj}-2,0)")
        L(r"\hat r_{ui}=\mu+b_u+b_i+p_u^{\top}q_i,\quad \min \sum (r_{ui}-\hat r_{ui})^2+\lambda(\|p_u\|^2+\|q_i\|^2+b_u^2+b_i^2)")
        L(r"\min \sum c_{ui}(p_{ui}-x_u^{\top}y_i)^2+\lambda(\|X\|^2+\|Y\|^2),\; c_{ui}=1+\alpha\,r^{+}_{ui}")
        L(r"s(u,i)=\sum_m w_m\, c_m(u)\, z_m(u,i),\quad c_u=\min(1,n_u/n_0)")
        L(r"\mathrm{MMR}(i)=\lambda\,\mathrm{rel}(i)-(1-\lambda)\max_{j\in S}\cos(x_i,x_j)")
        L(r"\mathrm{NDCG@K}=\frac{\sum_{p=1}^K \mathrm{rel}_p/\log_2(p+1)}{\mathrm{IDCG@K}},\quad \mathrm{RMSE}=\sqrt{\tfrac1n\sum (r-\hat r)^2}")
        L(r"1RM=w\left(1+\tfrac{\mathrm{reps}}{30}\right),\quad w_{next}=\frac{1RM}{1+(\mathrm{reps}_{target}+RIR)/30}")


{"Today": page_today, "Plan": page_plan, "For you": page_foryou, "Exercises": page_exercises, "Log": page_log,
 "Progress": page_progress, "Account": page_account, "Methodology": page_method}[nav]()
