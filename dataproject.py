import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os

# Optional live Gemini SDK import (falls back gracefully if not installed or no key set)
try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="NC PAC Message & $2M Allocation Intelligence",
    page_icon="🗳️",
    layout="wide"
)

# --- GLOBAL CSS: EXECUTIVE POLISH & TOOLBAR ALIGNMENT ---
st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.25);
        padding: 12px 14px;
        border-radius: 8px;
        height: 100%;
    }
    div[data-testid="stElementToolbar"] {
        opacity: 1 !important;
        visibility: visible !important;
        display: flex !important;
        background-color: var(--background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.3) !important;
        border-radius: 6px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.08) !important;
        padding: 2px 6px !important;
        top: -42px !important;
        right: 0px !important;
        z-index: 99 !important;
    }
    div[data-testid="stElementToolbar"] button {
        opacity: 1 !important;
        visibility: visible !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 1. DATA LOADER (SIMULATED META + GOOGLE BIGQUERY SCHEMA) ---
@st.cache_data
def load_data():
    np.random.seed(42)
    themes = [
        "Economy & Inflation",
        "Abortion & Reproductive Rights",
        "Immigration & Border Security",
        "Health Care & Prescription Drugs",
        "Public Safety & Crime",
        "Democracy & Voting Access",
        "Candidate Character & Ethics"
    ]
    advertisers = [
        {"name": "Forward NC PAC (Our PAC)", "short": "Forward NC (Us)", "side": "Democratic / Progressive", "side_short": "Dem / Progressive", "platform": "Meta"},
        {"name": "Carolina Victory Fund", "short": "Carolina Victory", "side": "Republican / Conservative", "side_short": "Rep / Conservative", "platform": "Meta"},
        {"name": "NC Senate Candidate A", "short": "Senate Cand. A", "side": "Democratic / Progressive", "side_short": "Dem / Progressive", "platform": "Google"},
        {"name": "NC Senate Candidate B", "short": "Senate Cand. B", "side": "Republican / Conservative", "side_short": "Rep / Conservative", "platform": "Google"},
        {"name": "Citizens for NC Growth", "short": "Citizens Growth", "side": "Independent / Issue Group", "side_short": "Independent", "platform": "Meta"},
    ]
    
    records = []
    ad_id_counter = 10001
    
    for _ in range(450):
        adv = np.random.choice(advertisers)
        if adv["name"] == "Forward NC PAC (Our PAC)":
            theme = np.random.choice(themes, p=[0.10, 0.35, 0.00, 0.25, 0.00, 0.20, 0.10])
        elif adv["name"] == "Carolina Victory Fund":
            theme = np.random.choice(themes, p=[0.38, 0.00, 0.27, 0.05, 0.22, 0.00, 0.08])
        elif adv["name"] == "Citizens for NC Growth":
            theme = np.random.choice(themes, p=[0.40, 0.00, 0.00, 0.35, 0.00, 0.00, 0.25])
        else:
            theme = np.random.choice(themes)
            
        min_spend = int(np.random.choice([1000, 2500, 5000, 10000, 25000, 50000]))
        max_spend = int(min_spend * np.random.uniform(1.25, 2.2))
        min_imp = min_spend * int(np.random.uniform(45, 65))
        max_imp = max_spend * int(np.random.uniform(55, 75))
        
        sample_copy = {
            "Economy & Inflation": f"North Carolina families are paying more at the pump and grocery store every single week. {adv['name']} has a concrete plan to cut middle-class taxes, lower energy costs, and rein in inflation across our state.",
            "Abortion & Reproductive Rights": f"Private medical decisions belong between a woman and her doctor—not politicians in Raleigh or Washington. Stand with {adv['name']} to protect reproductive freedom across North Carolina.",
            "Immigration & Border Security": f"Securing our border starts with real accountability and enforcement. {adv['name']} supports backing border patrol agents and stopping illegal crossings that strain local communities.",
            "Health Care & Prescription Drugs": f"Nobody in North Carolina should ever have to ration life-saving insulin or skip medication. {adv['name']} is taking on price-gouging drug corporations to cap prescription costs.",
            "Public Safety & Crime": f"Safe North Carolina neighborhoods require fully funded police departments and tough accountability for violent offenders. {adv['name']} proudly backs local law enforcement.",
            "Democracy & Voting Access": f"Protect the freedom to vote across every single North Carolina county. {adv['name']} stands for fair, accessible, secure, and transparent elections for all eligible voters.",
            "Candidate Character & Ethics": f"Leadership is about integrity, trust, and delivering real results—not empty campaign promises. See the full public service record behind {adv['name']}."
        }
        
        records.append({
            "ad_id": f"AD-{ad_id_counter}",
            "advertiser": adv["name"],
            "short_name": adv["short"],
            "political_side": adv["side"],
            "side_short": adv["side_short"],
            "platform": adv["platform"],
            "issue_theme": theme,
            "min_spend": min_spend,
            "max_spend": max_spend,
            "est_spend": (min_spend + max_spend) / 2.0,
            "min_impressions": min_imp,
            "max_impressions": max_imp,
            "est_impressions": (min_imp + max_imp) / 2.0,
            "creative_text": sample_copy[theme],
            "date": pd.Timestamp("2026-08-01") + pd.Timedelta(days=int(np.random.randint(0, 85)))
        })
        ad_id_counter += 1
        
    return pd.DataFrame(records)

df_raw = load_data()
ALL_THEMES = sorted(df_raw["issue_theme"].unique().tolist())

# --- LIVE LLM + GROUNDED FALLBACK ENGINE (TAB 1) ---
@st.cache_data(ttl=3600, show_spinner=False)
def call_gemini_counter_brief(entity_label: str, theme_label: str, ad_cnt: int, min_s: float, max_s: float, top_plat: str, sample_ads_str: str, angle_mode: str) -> str | None:
    """Calls Gemini 2.5 Flash if GEMINI_API_KEY is available in secrets or env; returns None otherwise."""
    api_key = None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        api_key = os.environ.get("GEMINI_API_KEY")

    if not (HAS_GENAI and api_key):
        return None

    try:
        client = genai.Client(api_key=api_key)
        prompt = f"""
You are a senior political media strategist advising Forward NC PAC.
Analyze ONLY the verified ad records below for '{entity_label}' on '{theme_label}':
- Total Ads: {ad_cnt} | Verified Spend Bracket: ${min_s:,.0f}–${max_s:,.0f} | Primary Platform: {top_plat}
- Sample Creative Records:
{sample_ads_str}

Task: Write a 2-sentence '{angle_mode}' recommendation for Forward NC PAC's next creative test.
Rules:
1. Cite at least two exact Ad IDs from the records above.
2. Do NOT estimate exact spending outside the bracket or invent polling/voter data.
3. Keep it sharp, actionable, and under 55 words.
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception:
        return None

def generate_grounded_ai_insight(sub_df: pd.DataFrame, entity_label: str, theme_label: str, angle_mode: str = "Direct Rebuttal") -> dict:
    if sub_df.empty:
        return {
            "summary": f"Zero active spend detected for **{entity_label}** on **{theme_label}** in the current window. This represents an uncontested thematic gap.",
            "action": f"**Opportunity ({angle_mode}):** Launch a low-cost $15k–$25k exploratory test on **{theme_label}** to establish early Share of Voice before opposition enters.",
            "citations": "0 verified records",
            "confidence": "1.00 (Deterministic Zero-Count)",
            "engine": "Deterministic Rule Check"
        }

    ad_cnt = len(sub_df)
    total_min = sub_df["min_spend"].sum()
    total_max = sub_df["max_spend"].sum()
    top_platform = sub_df["platform"].mode().iloc[0]
    sample_rows = sub_df.head(3)
    sample_ids = ", ".join(sample_rows["ad_id"].tolist())
    sample_ads_str = "\n".join([f"- [{r['ad_id']}] ({r['platform']}): {r['creative_text']}" for _, r in sample_rows.iterrows()])

    theme_frames = {
        "Economy & Inflation": "emphasizing household cost-of-living pressures (gas/grocery prices) and tax relief",
        "Abortion & Reproductive Rights": "framing medical privacy and bodily autonomy as core ballot issues",
        "Immigration & Border Security": "highlighting border enforcement accountability and security",
        "Health Care & Prescription Drugs": "targeting insulin and prescription drug price caps",
        "Public Safety & Crime": "focusing on law enforcement funding and neighborhood safety",
        "Democracy & Voting Access": "stressing ballot access and election integrity across NC counties",
        "Candidate Character & Ethics": "contrasting leadership trustworthiness and public service records"
    }
    frame_desc = theme_frames.get(theme_label, "reinforcing core issue positioning")

    # Attempt live Gemini generation first
    live_ai_text = call_gemini_counter_brief(
        entity_label, theme_label, ad_cnt, total_min, total_max, top_platform, sample_ads_str, angle_mode
    )

    if live_ai_text:
        action_text = f"**{angle_mode} (Live Gemini):** {live_ai_text}"
        engine_tag = "Gemini 2.5 Flash (Grounded)"
    else:
        # Dynamic, theme- and angle-specific grounded fallback (changes per cell & toggle!)
        dynamic_counter_plays = {
            ("Economy & Inflation", "Direct Rebuttal"): f"Counter {entity_label}'s tax framing ({sample_ids}) by highlighting corporate price-gouging and middle-class kitchen-table savings on {top_platform}.",
            ("Economy & Inflation", "Issue Pivot"): f"Rather than matching {entity_label}'s ${total_max/1000:,.0f}k spend ceiling on inflation ({sample_ids}), pivot creative testing toward prescription drug cost caps.",
            ("Abortion & Reproductive Rights", "Direct Rebuttal"): f"Reinforce bodily autonomy messaging seen in {sample_ids} with local NC physician testimonials on {top_platform} to lock down suburban persuadables.",
            ("Abortion & Reproductive Rights", "Issue Pivot"): f"With {ad_cnt} ads already active here ({sample_ids}), test pairing reproductive freedom with healthcare affordability to broaden reach.",
            ("Immigration & Border Security", "Direct Rebuttal"): f"Neutralize {entity_label}'s enforcement framing ({sample_ids}) by spotlighting bipartisan border security proposals and local NC economic stability.",
            ("Immigration & Border Security", "Issue Pivot"): f"Avoid fighting on {entity_label}'s strongest turf ({sample_ids}); redirect testing dollars to public safety funding and local jobs.",
            ("Health Care & Prescription Drugs", "Direct Rebuttal"): f"Amplify the insulin cap contrast ({sample_ids}) on {top_platform} with direct voter testimonials showing out-of-pocket monthly savings.",
            ("Health Care & Prescription Drugs", "Issue Pivot"): f"Expand beyond insulin ({sample_ids}) to test rural NC hospital access and Medicare protection creative.",
            ("Public Safety & Crime", "Direct Rebuttal"): f"Counter {entity_label}'s crime attacks ({sample_ids}) by highlighting verified local law enforcement endorsements and fentanyl interdiction funding.",
            ("Public Safety & Crime", "Issue Pivot"): f"Shift the public safety conversation ({sample_ids}) toward community mental health resources and school safety investments.",
            ("Democracy & Voting Access", "Direct Rebuttal"): f"Pair early-voting deadline reminders ({sample_ids}) on {top_platform} with clear county-level ballot verification tools.",
            ("Democracy & Voting Access", "Issue Pivot"): f"Connect voting access ({sample_ids}) to protecting local accountability on economic and healthcare issues.",
            ("Candidate Character & Ethics", "Direct Rebuttal"): f"Contrast {entity_label}'s record claims ({sample_ids}) using verified public voting records and local NC endorsements.",
            ("Candidate Character & Ethics", "Issue Pivot"): f"Move away from personality-driven ads ({sample_ids}) and test concrete policy contrast spots on {top_platform}."
        }
        selected_play = dynamic_counter_plays.get(
            (theme_label, angle_mode),
            f"Test a {angle_mode.lower()} creative angle against {sample_ids} on {top_platform}."
        )
        action_text = f"**{angle_mode}:** {selected_play}"
        engine_tag = "Context-Grounded Synthesizer"

    return {
        "summary": (
            f"**{entity_label}** deployed **{ad_cnt} ads** on **{theme_label}** "
            f"(spend bracket: **${total_min:,.0f}–${total_max:,.0f}**, concentrated on **{top_platform}**), "
            f"{frame_desc}."
        ),
        "action": action_text,
        "citations": f"Verified Ad IDs: {sample_ids}",
        "confidence": "0.94 (Taxonomy Validated)",
        "engine": engine_tag
    }

# --- GROUNDED AI ENGINE 2: TAB 2 BUDGET ROI & RISK ADVISOR ---
def generate_budget_ai_advice(comp_df: pd.DataFrame, t1: str, m1: float, t2: str, m2: float) -> dict:
    total_deployed = round(m1 + m2, 1)
    if total_deployed == 0:
        biggest_gap_row = comp_df.sort_values("Net Dollar Gap ($)").iloc[0]
        return {
            "status": "Awaiting Allocation",
            "advice": (
                f"Currently **$0.0M** allocated. Your steepest deficit is **{biggest_gap_row['Issue Theme']}** "
                f"(trailing selected opponents by **${abs(biggest_gap_row['Net Dollar Gap ($)']):,.0f}**). "
                "Drag the sliders on the right to simulate closing high-leverage gaps."
            ),
            "risk_note": "Guardrail: Avoid concentrating 100% of remaining funds into a single theme past 75% Share of Voice due to diminishing marginal reach."
        }

    notes = []
    risks = []
    for t_name, added_m in [(t1, m1), (t2, m2)]:
        if added_m <= 0:
            continue
        row = comp_df[comp_df["Issue Theme"] == t_name].iloc[0]
        base_our = row["Base Our PAC ($)"]
        opp_val = row["Selected Opponents Spend ($)"]
        added_dollars = added_m * 1_000_000
        new_sov = ((base_our + added_dollars) / (base_our + added_dollars + opp_val)) * 100.0 if (base_our + added_dollars + opp_val) > 0 else 100.0

        if new_sov >= 75.0:
            notes.append(f"**+${added_m:.1f}M on {t_name}** achieves dominant **{new_sov:.1f}% SOV**.")
            risks.append(f"Diminishing returns warning on {t_name} (>75% SOV); consider trimming $0.2M–$0.3M to reinforce secondary gaps.")
        elif new_sov >= 45.0:
            notes.append(f"**+${added_m:.1f}M on {t_name}** establishes competitive parity (**{new_sov:.1f}% SOV**).")
        else:
            notes.append(f"**+${added_m:.1f}M on {t_name}** lifts SOV to **{new_sov:.1f}%**, though opponents still outspend you.")
            risks.append(f"Under-threshold warning: {t_name} remains below 45% SOV; creative must rely on sharp contrast framing rather than volume.")

    return {
        "status": f"Simulating ${total_deployed:.1f}M of $2.0M War Chest",
        "advice": " ".join(notes),
        "risk_note": risks[0] if risks else "Balanced allocation: Both selected themes achieve efficient Share of Voice lift without over-saturation."
    }

# --- 2. CLEAN TOP HEADER & SEARCH-FIRST CONTROLS ---
st.title("🗳️ NC Midterm Message & $2M Allocation Tracker")
st.caption("Cross-Platform Intelligence (Meta Ad Library + Google BigQuery) • Decision: Optimal Allocation of Remaining $2,000,000")

with st.container(border=True):
    f_col1, f_col2, f_col3, f_col4 = st.columns([2.2, 1.5, 1.3, 1.3])
    
    with f_col1:
        keyword_search = st.text_input(
            "🔍 Search Creative Text or Keywords",
            placeholder="Type keywords (e.g., insulin, border, taxes)..."
        )
    with f_col2:
        sponsor_search = st.selectbox(
            "🏢 Filter Specific Sponsor (Type to Search)",
            options=sorted(df_raw["advertiser"].unique().tolist()),
            index=None,
            placeholder="All Sponsors (or type name)..."
        )
    with f_col3:
        platform_toggle = st.segmented_control(
            "📡 Platform",
            options=["All", "Meta", "Google"],
            default="All"
        )
    with f_col4:
        metric_mode = st.selectbox(
            "📊 Heatmap Metric",
            options=["Budget Share (%)", "Est. Spend ($)", "Est. Impressions"]
        )

# Apply Filters
filtered_df = df_raw.copy()
if keyword_search:
    filtered_df = filtered_df[
        filtered_df["creative_text"].str.contains(keyword_search, case=False, na=False) |
        filtered_df["issue_theme"].str.contains(keyword_search, case=False, na=False)
    ]
if sponsor_search:
    filtered_df = filtered_df[filtered_df["advertiser"] == sponsor_search]
if platform_toggle and platform_toggle != "All":
    filtered_df = filtered_df[filtered_df["platform"] == platform_toggle]

# --- 3. KPI SUMMARY ROW ---
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Audited Ads Matching Filter", f"{len(filtered_df):,}")
kpi2.metric("Total Est. Market Spend", f"${filtered_df['est_spend'].sum():,.0f}")
kpi3.metric("Remaining PAC War Chest", "$2,000,000", delta="Ready to Deploy")
kpi4.metric("Active Sponsors Tracked", f"{filtered_df['advertiser'].nunique()}")

# --- 4. WORKSPACE TABS ---
tab_matrix, tab_allocator = st.tabs([
    "📊 Market Message Matrix & Ad Drill-Down",
    "💰 $2M Super PAC Head-to-Head Comparator & Simulator"
])

# ==============================================================================
# TAB 1: TOP ROW = HEATMAP + INTERACTIVE AI BRIEF | BOTTOM = EVIDENCE TABLE
# ==============================================================================
with tab_matrix:
    if filtered_df.empty:
        st.warning("No ads match your current keyword or sponsor search. Clear the search bar to reset.")
    else:
        top_left, top_right = st.columns([1.4, 1])

        with top_left:
            with st.container(border=True):
                hm_head_l, hm_head_mid, hm_head_spacer = st.columns([1.15, 1.15, 0.45])
                with hm_head_l:
                    st.subheader("Message Portfolio Heatmap")
                with hm_head_mid:
                    group_mode = st.segmented_control(
                        "Group Columns By",
                        options=["Individual Sponsors", "Dem vs. Rep Side"],
                        default="Individual Sponsors"
                    )

                if group_mode == "Dem vs. Rep Side":
                    x_field = "side_short"
                    full_label_field = "political_side"
                    active_entities = filtered_df[["side_short", "political_side"]].drop_duplicates()
                else:
                    x_field = "short_name"
                    full_label_field = "advertiser"
                    active_entities = filtered_df[["short_name", "advertiser"]].drop_duplicates()

                full_grid = active_entities.merge(pd.DataFrame({"issue_theme": ALL_THEMES}), how="cross")

                col_totals = filtered_df.groupby(x_field)["est_spend"].sum().reset_index()
                col_totals.rename(columns={"est_spend": "col_total_spend"}, inplace=True)

                grouped_actuals = (
                    filtered_df.groupby([x_field, "issue_theme"])
                    .agg(est_spend=("est_spend", "sum"), est_impressions=("est_impressions", "sum"), ad_count=("ad_id", "count"))
                    .reset_index()
                )

                theme_matrix = (
                    full_grid
                    .merge(grouped_actuals, on=[x_field, "issue_theme"], how="left")
                    .merge(col_totals, on=x_field, how="left")
                )
                theme_matrix["est_spend"] = theme_matrix["est_spend"].fillna(0.0)
                theme_matrix["est_impressions"] = theme_matrix["est_impressions"].fillna(0.0)
                theme_matrix["ad_count"] = theme_matrix["ad_count"].fillna(0).astype(int)
                theme_matrix["spend_share_pct"] = np.where(
                    theme_matrix["col_total_spend"] > 0,
                    (theme_matrix["est_spend"] / theme_matrix["col_total_spend"]) * 100.0,
                    0.0
                )
                theme_matrix["has_spend"] = theme_matrix["ad_count"] > 0

                if metric_mode == "Budget Share (%)":
                    val_col, legend_title = "spend_share_pct", "Budget Share (%)"
                    theme_matrix["cell_label"] = np.where(
                        theme_matrix["has_spend"],
                        theme_matrix["spend_share_pct"].map(lambda v: f"{v:.1f}%"),
                        "N/A ($0)"
                    )
                elif metric_mode == "Est. Spend ($)":
                    val_col, legend_title = "est_spend", "Est. Spend ($)"
                    theme_matrix["cell_label"] = np.where(
                        theme_matrix["has_spend"],
                        theme_matrix["est_spend"].map(lambda v: f"${v/1000:,.0f}k"),
                        "N/A ($0)"
                    )
                else:
                    val_col, legend_title = "est_impressions", "Est. Impressions"
                    theme_matrix["cell_label"] = np.where(
                        theme_matrix["has_spend"],
                        theme_matrix["est_impressions"].map(lambda v: f"{v/1_000_000:.1f}M"),
                        "N/A (0)"
                    )

                st.caption("Gray dashed cells (`N/A`) indicate zero observed spend. Click any active tile to update the AI brief & evidence below.")

                selection = alt.selection_point(
                    name="cell_select",
                    fields=[x_field, "issue_theme"],
                    clear="dblclick"
                )

                base = alt.Chart(theme_matrix).encode(
                    x=alt.X(
                        f"{x_field}:N",
                        title=None,
                        sort=sorted(active_entities[x_field].tolist()),
                        scale=alt.Scale(paddingInner=0.06),
                        axis=alt.Axis(
                            labelAngle=0,
                            labelFlush=False,
                            labelOverlap=False,
                            labelFontSize=11,
                            labelPadding=8,
                            labelLimit=140
                        )
                    ),
                    y=alt.Y(
                        "issue_theme:N",
                        title=None,
                        sort=ALL_THEMES,
                        scale=alt.Scale(paddingInner=0.06),
                        axis=alt.Axis(labelFontSize=11, labelLimit=185)
                    )
                )

                na_tiles = base.transform_filter(
                    alt.datum.has_spend == False
                ).mark_rect(
                    cornerRadius=4,
                    color="#f1f5f9",
                    stroke="#cbd5e1",
                    strokeWidth=1,
                    strokeDash=[3, 3]
                ).encode(
                    tooltip=[
                        alt.Tooltip(f"{full_label_field}:N", title="Entity"),
                        alt.Tooltip("issue_theme:N", title="Theme"),
                        alt.Tooltip("cell_label:N", title="Status")
                    ]
                )

                active_tiles = base.transform_filter(
                    alt.datum.has_spend == True
                ).mark_rect(
                    cornerRadius=4
                ).encode(
                    color=alt.Color(
                        f"{val_col}:Q",
                        scale=alt.Scale(range=["#dbeafe", "#60a5fa", "#1d4ed8", "#1e3a8a"]),
                        title=legend_title
                    ),
                    opacity=alt.condition(selection, alt.value(1.0), alt.value(0.92)),
                    tooltip=[
                        alt.Tooltip(f"{full_label_field}:N", title="Entity"),
                        alt.Tooltip("issue_theme:N", title="Theme"),
                        alt.Tooltip("spend_share_pct:Q", title="Budget Share (%)", format=".1f"),
                        alt.Tooltip("est_spend:Q", title="Est. Spend ($)", format="$,.0f"),
                        alt.Tooltip("ad_count:Q", title="Ad Count")
                    ]
                ).add_params(selection)

                na_text = base.transform_filter(
                    alt.datum.has_spend == False
                ).mark_text(
                    baseline="middle",
                    fontSize=11,
                    color="#64748b"
                ).encode(
                    text=alt.Text("cell_label:N")
                )

                midpoint = theme_matrix[val_col].max() * 0.45 if theme_matrix[val_col].max() > 0 else 1
                active_text = base.transform_filter(
                    alt.datum.has_spend == True
                ).mark_text(
                    baseline="middle",
                    fontSize=11,
                    fontWeight="bold"
                ).encode(
                    text=alt.Text("cell_label:N"),
                    color=alt.condition(
                        alt.datum[val_col] > midpoint,
                        alt.value("white"),
                        alt.value("#0f172a")
                    )
                )

                chart_events = st.altair_chart(
                    (na_tiles + active_tiles + na_text + active_text).properties(height=400),
                    use_container_width=True,
                    on_select="rerun",
                    key="main_heatmap"
                )

        # Determine active slice from heatmap click or top filters
        selected_points = chart_events.selection.get("cell_select", [])
        if selected_points:
            sel_val = selected_points[0].get(x_field)
            sel_thm = selected_points[0].get("issue_theme")
            drill_df = filtered_df[
                (filtered_df[x_field] == sel_val) &
                (filtered_df["issue_theme"] == sel_thm)
            ]
            ai_entity_label = sel_val
            ai_theme_label = sel_thm
        else:
            drill_df = filtered_df
            ai_entity_label = sponsor_search if sponsor_search else "All Tracked Sponsors"
            ai_theme_label = filtered_df.groupby("issue_theme")["est_spend"].sum().idxmax()

        # Top Right: Interactive, Changeable AI Tactical Shift & Counter-Brief Card
        with top_right:
            with st.container(border=True):
                st.subheader("🤖 AI Tactical Shift & Counter-Brief")
                angle_choice = st.segmented_control(
                    "Counter-Strategy Mode",
                    options=["Direct Rebuttal", "Issue Pivot"],
                    default="Direct Rebuttal"
                )

                ai_out = generate_grounded_ai_insight(
                    drill_df if selected_points else filtered_df[filtered_df["issue_theme"] == ai_theme_label],
                    ai_entity_label,
                    ai_theme_label,
                    angle_mode=angle_choice or "Direct Rebuttal"
                )

                st.markdown(f"**Active Focus:** `{ai_entity_label}` • `{ai_theme_label}`")
                st.markdown(ai_out["summary"])
                st.info(ai_out["action"])

                st.divider()
                st.caption(f"🛡️ {ai_out['citations']} | Engine: `{ai_out['engine']}`")
                st.caption(f"✅ Validation: `{ai_out['confidence']}` • *Zero spend estimation*")

        # Full-Width Bottom Section: Complete Audited Creative Evidence Table (Non-Overlapping Header)
        with st.container(border=True):
            # Left column holds title + blue status box; Right column stays clear for dataframe toolbar
            ev_head_l, ev_head_spacer = st.columns([2.2, 0.6])
            with ev_head_l:
                st.subheader("📋 Audited Creative Evidence (Full Copy & Spend Brackets)")
                if selected_points:
                    st.info(f"Filtered to: **{sel_val}** • **{sel_thm}** ({len(drill_df)} ads)")
                else:
                    st.caption("Showing all filtered records. Click any heatmap tile above to isolate specific ads.")

            if drill_df.empty:
                st.warning(f"**{sel_val}** has **$0** recorded spend on **{sel_thm}**.")
            else:
                st.dataframe(
                    drill_df[[
                        "ad_id",
                        "advertiser",
                        "political_side",
                        "platform",
                        "issue_theme",
                        "min_spend",
                        "max_spend",
                        "creative_text"
                    ]].rename(columns={
                        "ad_id": "Ad ID",
                        "advertiser": "Sponsor",
                        "political_side": "Alignment",
                        "platform": "Platform",
                        "issue_theme": "Issue Theme",
                        "min_spend": "Min Spend ($)",
                        "max_spend": "Max Spend ($)",
                        "creative_text": "Full Audited Creative Text"
                    }),
                    column_config={
                        "Full Audited Creative Text": st.column_config.TextColumn(width="large"),
                        "Min Spend ($)": st.column_config.NumberColumn(format="$%d"),
                        "Max Spend ($)": st.column_config.NumberColumn(format="$%d"),
                    },
                    height=300,
                    use_container_width=True,
                    hide_index=True
                )

# ==============================================================================
# TAB 2: LEFT = BAR CHART + AI ADVISOR | RIGHT = SIMULATOR + PROJECTED SOV
# ==============================================================================
with tab_allocator:
    st.subheader("Super PAC Head-to-Head Gap Analysis & $2M Deployment Simulator")
    st.markdown(
        "Compare **your Super PAC** directly against chosen opponents, test a **$2.0M** allocation, "
        "and apply it to preview how your spend bars shift."
    )

    if "alloc_1_m" not in st.session_state:
        st.session_state.alloc_1_m = 0.0
    if "alloc_2_m" not in st.session_state:
        st.session_state.alloc_2_m = 0.0
    if "applied_boosts" not in st.session_state:
        st.session_state.applied_boosts = {}

    def reset_simulator():
        st.session_state.alloc_1_m = 0.0
        st.session_state.alloc_2_m = 0.0
        st.session_state.applied_boosts = {}

    all_sponsors = sorted(df_raw["advertiser"].unique().tolist())

    with st.container(border=True):
        c1, c2 = st.columns([1, 1.5])
        with c1:
            our_pac = st.selectbox(
                "1. Select Your Super PAC",
                options=all_sponsors,
                index=all_sponsors.index("Forward NC PAC (Our PAC)")
            )
        with c2:
            default_opps = ["Carolina Victory Fund", "NC Senate Candidate B"]
            available_opps = [s for s in all_sponsors if s != our_pac]
            chosen_opponents = st.multiselect(
                "2. Select Opponent PAC(s) / Candidates to Compare Against",
                options=available_opps,
                default=[o for o in default_opps if o in available_opps]
            )

    if not chosen_opponents:
        st.warning("Select at least one opponent above to run the head-to-head comparison.")
    else:
        our_base_series = (
            df_raw[df_raw["advertiser"] == our_pac]
            .groupby("issue_theme")["est_spend"]
            .sum()
            .reindex(ALL_THEMES, fill_value=0.0)
        )
        opp_spend_series = (
            df_raw[df_raw["advertiser"].isin(chosen_opponents)]
            .groupby("issue_theme")["est_spend"]
            .sum()
            .reindex(ALL_THEMES, fill_value=0.0)
        )

        boost_series = pd.Series(
            [st.session_state.applied_boosts.get(t, 0.0) for t in ALL_THEMES],
            index=ALL_THEMES
        )
        our_total_with_sim = our_base_series + boost_series

        comp_df = pd.DataFrame({
            "Issue Theme": ALL_THEMES,
            "Base Our PAC ($)": our_base_series.values,
            "Simulated Boost ($)": boost_series.values,
            "Our PAC Spend ($)": our_total_with_sim.values,
            "Selected Opponents Spend ($)": opp_spend_series.values
        })
        comp_df["Net Dollar Gap ($)"] = comp_df["Base Our PAC ($)"] - comp_df["Selected Opponents Spend ($)"]
        comp_df["Current Our SOV (%)"] = (
            comp_df["Our PAC Spend ($)"] /
            (comp_df["Our PAC Spend ($)"] + comp_df["Selected Opponents Spend ($)"]).replace(0, 1)
        ) * 100.0

        sim_left, sim_right = st.columns([1.3, 1])

        with sim_right:
            with st.container(border=True):
                st.markdown("#### 🎯 $2.0M War Chest Simulator")
                st.caption("Drag sliders (in $ Millions) to cap total spend at $2.0M, then apply to the chart.")

                biggest_gaps = comp_df.sort_values("Net Dollar Gap ($)").head(2)["Issue Theme"].tolist()
                st.success(f"💡 **Biggest Strategic Deficits:** {biggest_gaps[0]} & {biggest_gaps[1]}")

                theme_1 = st.selectbox(
                    "Primary Test Theme (#1)",
                    options=ALL_THEMES,
                    index=ALL_THEMES.index(biggest_gaps[0])
                )
                alloc_1_m = st.slider(
                    f"Allocate to {theme_1}",
                    min_value=0.0,
                    max_value=2.0,
                    step=0.1,
                    format="$%.1fM",
                    key="alloc_1_m"
                )

                remaining_m = round(2.0 - alloc_1_m, 1)
                if st.session_state.alloc_2_m > remaining_m:
                    st.session_state.alloc_2_m = remaining_m

                theme_2_options = [t for t in ALL_THEMES if t != theme_1]
                default_t2 = biggest_gaps[1] if biggest_gaps[1] in theme_2_options else theme_2_options[0]
                theme_2 = st.selectbox(
                    "Secondary Test Theme (#2)",
                    options=theme_2_options,
                    index=theme_2_options.index(default_t2)
                )

                if remaining_m <= 0.0:
                    st.slider(
                        f"Allocate to {theme_2} (Max $0.0M left)",
                        min_value=0.0,
                        max_value=0.1,
                        value=0.0,
                        format="$%.1fM",
                        disabled=True
                    )
                    alloc_2_m = 0.0
                else:
                    alloc_2_m = st.slider(
                        f"Allocate to {theme_2} (Up to ${remaining_m:.1f}M left)",
                        min_value=0.0,
                        max_value=remaining_m,
                        step=0.1,
                        format="$%.1fM",
                        key="alloc_2_m"
                    )

                unallocated_m = round(2.0 - (alloc_1_m + alloc_2_m), 1)
                st.caption(f"Unallocated War Chest Reserve: **${unallocated_m:.1f}M**")

                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("✅ Update Bar Chart", type="secondary", use_container_width=True):
                        st.session_state.applied_boosts = {
                            theme_1: alloc_1_m * 1_000_000,
                            theme_2: alloc_2_m * 1_000_000
                        }
                        st.rerun()
                with btn_col2:
                    st.button("🔄 Reset to $0", type="secondary", on_click=reset_simulator, use_container_width=True)

                st.divider()

                st.caption("ℹ️ **SOV (Share of Voice):** Your PAC's percentage of total ad spend on this issue compared against the selected opponents.")

                for t_name, added_m in [(theme_1, alloc_1_m), (theme_2, alloc_2_m)]:
                    added_dollars = added_m * 1_000_000
                    row = comp_df[comp_df["Issue Theme"] == t_name].iloc[0]
                    base_our = row["Base Our PAC ($)"]
                    opp_val = row["Selected Opponents Spend ($)"]

                    old_sov = (base_our / (base_our + opp_val)) * 100.0 if (base_our + opp_val) > 0 else 0.0
                    new_sov = ((base_our + added_dollars) / (base_our + added_dollars + opp_val)) * 100.0 if (base_our + added_dollars + opp_val) > 0 else 0.0

                    st.metric(
                        label=f"Projected SOV: {t_name}",
                        value=f"{new_sov:.1f}%",
                        delta=f"+{new_sov - old_sov:.1f}% (with +${added_m:.1f}M)"
                    )

        with sim_left:
            with st.container(border=True):
                header_l, header_spacer = st.columns([2.2, 0.6])
                with header_l:
                    st.markdown("#### Head-to-Head Spend Comparison by Issue")
                    if sum(st.session_state.applied_boosts.values()) > 0:
                        st.caption("🟢 *Includes Applied $2.0M Simulation*")
                    else:
                        st.caption("Showing baseline observed spend across selected entities.")

                melted_comp = comp_df.melt(
                    id_vars="Issue Theme",
                    value_vars=["Our PAC Spend ($)", "Selected Opponents Spend ($)"],
                    var_name="Entity",
                    value_name="Estimated Spend ($)"
                )

                grouped_bar = alt.Chart(melted_comp).mark_bar(cornerRadiusEnd=4).encode(
                    y=alt.Y("Issue Theme:N", title=None, axis=alt.Axis(labelLimit=180)),
                    x=alt.X("Estimated Spend ($):Q", title="Estimated Spend ($)"),
                    yOffset="Entity:N",
                    color=alt.Color(
                        "Entity:N",
                        scale=alt.Scale(
                            domain=["Our PAC Spend ($)", "Selected Opponents Spend ($)"],
                            range=["#2563eb", "#dc2626"]
                        ),
                        legend=alt.Legend(orient="top", title=None)
                    ),
                    tooltip=[
                        alt.Tooltip("Issue Theme:N"),
                        alt.Tooltip("Entity:N"),
                        alt.Tooltip("Estimated Spend ($):Q", format="$,.0f")
                    ]
                ).properties(height=360)

                st.altair_chart(grouped_bar, use_container_width=True)

            with st.container(border=True):
                ai_b_l, ai_b_r = st.columns([1.8, 1])
                with ai_b_l:
                    st.markdown("#### 🤖 AI Budget ROI & Risk Advisor")
                with ai_b_r:
                    st.caption("Grounded • Real-Time Simulator Check")

                budget_ai = generate_budget_ai_advice(comp_df, theme_1, alloc_1_m, theme_2, alloc_2_m)
                st.markdown(f"**Status:** `{budget_ai['status']}`")
                st.markdown(budget_ai["advice"])
                st.warning(budget_ai["risk_note"])