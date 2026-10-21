import json
import math
import re
from datetime import datetime
from io import BytesIO
from urllib.parse import quote

import streamlit as st
from PIL import Image

# Optional Gemini SDK is listed in requirements.txt. The app remains usable in demo mode
# if the SDK is unavailable or no API key is configured.
try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None


# ---------------------------------------------------------------------------
# Page configuration and styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Multimodal Construction Intelligence Platform",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root { --ink:#172033; --muted:#667085; --line:#e5eaf1; --surface:#ffffff; }
.stApp { background: linear-gradient(180deg, #f5f8fc 0%, #eef3f9 100%); color:var(--ink); }
.block-container { padding-top: 1.4rem; padding-bottom: 2.8rem; max-width: 1440px; }
[data-testid="stSidebar"] { background: #111c2f; }
[data-testid="stSidebar"] * { color: #edf3fb; }
[data-testid="stSidebar"] input { color: #172033 !important; }
.hero {
  padding: 1.45rem 1.6rem; border-radius: 20px; color: white;
  background: linear-gradient(120deg, #172b4d 0%, #1d5674 58%, #248b83 100%);
  box-shadow: 0 10px 30px rgba(19,40,70,.13); margin-bottom: 1.2rem;
}
.hero h1 { margin: 0 0 .35rem 0; font-size: clamp(1.55rem, 3vw, 2.3rem); }
.hero p { margin: 0; opacity: .88; max-width: 900px; }
.eyebrow { text-transform: uppercase; letter-spacing: .14em; font-size: .72rem; font-weight: 800; opacity: .78; }
.panel {
  background: var(--surface); border: 1px solid var(--line); border-radius: 16px;
  padding: 1rem 1.1rem; box-shadow: 0 4px 16px rgba(20,40,70,.04); margin-bottom: .8rem;
}
.kpi {
  background: white; border: 1px solid var(--line); border-radius: 15px;
  padding: 1rem 1.1rem; min-height: 112px; box-shadow: 0 4px 14px rgba(20,40,70,.035);
}
.kpi-label { color: #667085; font-size: .82rem; margin-bottom: .4rem; }
.kpi-value { font-size: 1.65rem; font-weight: 800; color: #172033; line-height: 1.2; }
.kpi-note { color: #667085; font-size: .78rem; margin-top: .35rem; }
.pill { display:inline-block; border-radius:999px; padding:.28rem .7rem; font-size:.78rem; font-weight:800; }
.pill-green { background:#dcfce7; color:#166534; }
.pill-amber { background:#fef3c7; color:#92400e; }
.pill-red { background:#fee2e2; color:#991b1b; }
.pill-blue { background:#dbeafe; color:#1e40af; }
.section-title { font-size:1.18rem; font-weight:800; margin:.25rem 0 .75rem 0; color:#172033; }
.small-muted { color:#667085; font-size:.85rem; }
div[data-testid="stMetric"] { background:white; padding:.9rem 1rem; border:1px solid #e5eaf1; border-radius:14px; }
.stTabs [data-baseweb="tab-list"] { gap: .35rem; }
.stTabs [data-baseweb="tab"] { background:#e8eef6; border-radius:10px 10px 0 0; padding:.7rem 1rem; }
.stTabs [aria-selected="true"] { background:white !important; }
div.stButton > button, div.stDownloadButton > button { border-radius:10px; font-weight:700; }
hr { border-color:#e5eaf1; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Constants, helpers, and demo data
# ---------------------------------------------------------------------------
SEVERITIES = ["Low", "Medium", "High", "Critical"]
DEFECT_TYPES = ["Hairline", "Settlement", "Structural Shear", "Thermal Expansion", "Spalling"]
STRUCTURE_FACTORS = {
    "Residential house": {"base_rate": 1850, "steel": 3.8, "cement": 0.40, "market_factor": 1.08},
    "Apartment building": {"base_rate": 2200, "steel": 4.5, "cement": 0.45, "market_factor": 1.12},
    "Commercial building": {"base_rate": 2600, "steel": 5.2, "cement": 0.50, "market_factor": 1.18},
    "Warehouse / industrial": {"base_rate": 1650, "steel": 3.2, "cement": 0.34, "market_factor": 1.04},
    "Institutional building": {"base_rate": 2400, "steel": 4.8, "cement": 0.48, "market_factor": 1.14},
}
TIER_FACTORS = {
    "Tier 1 city": {"cost": 1.18, "market": 1.35},
    "Tier 2 city": {"cost": 1.00, "market": 1.00},
    "Tier 3 city": {"cost": 0.88, "market": 0.78},
}
CURRENCY = "₹"


def inr(value, decimals=2):
    value = float(value)
    if abs(value) >= 1e7:
        return f"₹{value / 1e7:,.{decimals}f} Cr"
    if abs(value) >= 1e5:
        return f"₹{value / 1e5:,.{decimals}f} L"
    return f"₹{value:,.0f}"


def pill(label, kind="blue"):
    return f'<span class="pill pill-{kind}">{label}</span>'


def severity_kind(severity):
    return {"Low": "green", "Medium": "amber", "High": "red", "Critical": "red"}.get(severity, "blue")


def safe_json_loads(text):
    """Parse plain JSON or JSON embedded in a Markdown code fence."""
    if isinstance(text, dict):
        return text
    raw = str(text).strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start:end + 1])
        raise


def get_gemini_client(api_key):
    if not api_key or not api_key.strip() or genai is None:
        return None
    try:
        return genai.Client(api_key=api_key.strip())
    except Exception:
        return None


def gemini_json(client, prompt, image=None):
    """Call Gemini and require a JSON-only response. Errors are handled by the caller."""
    if client is None:
        raise RuntimeError("Gemini is not configured.")
    contents = [prompt]
    if image is not None:
        image_bytes = image.getvalue()
        mime = image.type or "image/jpeg"
        if types is not None:
            contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime))
        else:
            contents.append(image_bytes)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ) if types is not None else None,
    )
    return safe_json_loads(response.text)


def demo_safety_report():
    return {
        "safety_score": 76,
        "overall_status": "Needs Attention",
        "ppe_detection": {
            "helmet": {"detected": True, "compliance": "Partially compliant", "notes": "Head protection appears present for visible workers; verify every person."},
            "high_visibility_vest": {"detected": True, "compliance": "Partially compliant", "notes": "High-visibility clothing is visible on some workers."},
            "steel_toe_boots": {"detected": None, "compliance": "Not verifiable", "notes": "Footwear cannot be reliably confirmed from this view."},
            "safety_harness": {"detected": False, "compliance": "Potential gap", "notes": "Harness use is not clearly visible where elevated work may be occurring."},
        },
        "hazards": [
            {"hazard": "Unprotected elevated work area", "severity": "High", "location": "Upper work zone", "remedy": "Stop work at the edge until compliant guardrails or a properly anchored fall-arrest system is verified."},
            {"hazard": "PPE coverage cannot be confirmed for every worker", "severity": "Medium", "location": "General work area", "remedy": "Conduct a worker-by-worker PPE check and document corrective actions before work continues."},
            {"hazard": "Potential housekeeping / trip exposure", "severity": "Low", "location": "Ground-level work zone", "remedy": "Keep access routes clear and secure loose materials and cables."},
        ],
        "limitations": "Demonstration data only; not an analysis of the uploaded image."
    }


def demo_crack_report():
    return {
        "defect_type": "Hairline",
        "estimated_crack_width_mm": 0.3,
        "confidence": "Low",
        "observations": ["A fine surface line is used as an illustrative example.", "Depth and movement cannot be inferred from a single photo."],
        "possible_root_causes": ["Plastic or drying shrinkage", "Surface finish / render movement"],
        "recommended_repairs": [
            {"action": "Map and monitor the crack", "details": "Record length and width with a crack gauge and recheck for progression."},
            {"action": "Repair after engineering review", "details": "Use a compatible flexible sealant or approved repair system if confirmed non-structural."}
        ],
        "urgency": "Routine inspection",
        "engineer_review_required": True,
        "limitations": "Demonstration data only; not an analysis of the uploaded image."
    }


def call_or_demo(client, prompt, image, demo_fn):
    if client is None:
        return demo_fn(), True
    try:
        return gemini_json(client, prompt, image), False
    except Exception as exc:
        st.warning(f"Gemini analysis could not be completed ({type(exc).__name__}). Showing demonstration data instead.")
        return demo_fn(), True


def render_kpi(label, value, note=""):
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def download_json(data, filename):
    return st.download_button(
        "Download JSON report",
        data=json.dumps(data, indent=2, ensure_ascii=False),
        file_name=filename,
        mime="application/json",
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏗️ MCIP")
    st.caption("Multimodal Construction Intelligence Platform")
    st.divider()
    st.markdown("### AI configuration")
    api_key = st.text_input(
        "Google Gemini API key",
        type="password",
        placeholder="Paste API key (optional)",
        help="Key is held in this Streamlit session only. For production deployments, prefer Streamlit secrets.",
    )
    client = get_gemini_client(api_key)
    if client:
        st.success("Gemini enabled for this session")
    elif genai is None:
        st.info("Gemini SDK unavailable — demonstration mode")
    else:
        st.info("Demonstration mode — add a key to enable image analysis")
    st.divider()
    st.markdown("### Architecture")
    st.markdown(
        "1. **Streamlit UI** — uploads, forms and reports\n"
        "2. **Gemini 2.5 Flash** — optional multimodal analysis\n"
        "3. **Rule-based estimator** — budget and materials\n"
        "4. **Pollinations prompt link** — conceptual visualisation\n"
        "5. **Session state only** — no external database"
    )
    st.divider()
    st.markdown("### Feature navigation")
    st.caption("Use the tabs across the main workspace.")
    st.markdown("- Site safety & PPE audit\n- Crack diagnostics\n- Planning & valuation\n- Daily site assistant")
    st.caption("Engineering outputs are decision support, not a substitute for a competent professional.")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Skill Wallet · Capstone Project 1</div>
      <h1>Multimodal Construction Intelligence Platform</h1>
      <p>AI-assisted site safety reviews, concrete defect triage, early-stage cost planning, and weather-aware daily guidance — in one workspace.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.warning(
    "Important: AI-generated visual assessments and illustrative cost estimates are not certified inspections, "
    "structural designs, valuations, or legal interpretations of OSHA / Indian standards. Verify on site with qualified professionals."
)

tab_safety, tab_cracks, tab_planning, tab_assistant = st.tabs([
    "1 · Site Safety & PPE Audit",
    "2 · Crack Diagnostics",
    "3 · Planning & Valuation",
    "4 · Daily Site Assistant",
])


# ---------------------------------------------------------------------------
# Tab 1: Site safety and PPE audit
# ---------------------------------------------------------------------------
with tab_safety:
    st.markdown('<div class="section-title">Site Safety & PPE Audit</div>', unsafe_allow_html=True)
    st.write("Upload a site photograph to review visible PPE and potential hazards. The image may not show all relevant context.")
    uploaded_safety = st.file_uploader(
        "Construction site photo",
        type=["jpg", "jpeg", "png"],
        key="safety_upload",
        help="Use JPG or PNG. Avoid uploading identifiable worker information unless you are authorized to do so.",
    )
    if uploaded_safety:
        try:
            img = Image.open(uploaded_safety)
            st.image(img, caption="Uploaded site photograph", use_container_width=True)
        except Exception:
            st.error("This image could not be opened. Please upload a valid JPG or PNG.")

    safety_prompt = """
You are an occupational construction safety assistant. Analyze only visible evidence in the provided construction-site image.
Return valid JSON only with this schema:
{
  "safety_score": integer 0-100,
  "overall_status": "Compliant" | "Needs Attention" | "Unsafe",
  "ppe_detection": {
    "helmet": {"detected": true|false|null, "compliance": string, "notes": string},
    "high_visibility_vest": {"detected": true|false|null, "compliance": string, "notes": string},
    "steel_toe_boots": {"detected": true|false|null, "compliance": string, "notes": string},
    "safety_harness": {"detected": true|false|null, "compliance": string, "notes": string}
  },
  "hazards": [{"hazard": string, "severity": "Low"|"Medium"|"High"|"Critical", "location": string, "remedy": string}],
  "limitations": string
}
Use null / "Not verifiable" when the image does not support a conclusion. Do not claim to certify compliance.
Flag visible fall, electrical, excavation, struck-by, housekeeping, access, or material-storage risks. Recommend immediate isolation for credible imminent danger.
Use OSHA principles and relevant Indian construction practice as context, but do not claim that IS 456 is a PPE standard (IS 456 concerns plain and reinforced concrete). Do not invent measurements or infer unseen PPE.
"""
    safety_col1, safety_col2 = st.columns([1, 2])
    with safety_col1:
        analyze_safety = st.button("Analyze site photo", type="primary", use_container_width=True, disabled=uploaded_safety is None)
    with safety_col2:
        st.caption("Gemini mode analyzes the uploaded image. Demo mode returns clearly labelled sample data.")
    if analyze_safety:
        with st.spinner("Reviewing visible PPE and site hazards..."):
            report, demo_mode = call_or_demo(client, safety_prompt, uploaded_safety, demo_safety_report)
        report["_metadata"] = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "mode": "demonstration" if demo_mode else "gemini",
            "image_filename": uploaded_safety.name if uploaded_safety else None,
        }
        st.session_state["safety_report"] = report
        st.session_state["safety_demo"] = demo_mode

    if "safety_report" in st.session_state:
        report = st.session_state["safety_report"]
        if st.session_state.get("safety_demo"):
            st.info("Demonstration report — values below are illustrative and are not based on your uploaded image.")
        score = report.get("safety_score", "N/A")
        status = str(report.get("overall_status", "Unknown"))
        status_kind = "green" if status.lower() == "compliant" else ("red" if status.lower() == "unsafe" else "amber")
        k1, k2, k3 = st.columns(3)
        with k1: render_kpi("Safety score", f"{score}/100", "Visual screening indicator only")
        with k2: render_kpi("Overall status", status, "Review findings before action")
        with k3: render_kpi("Hazards identified", str(len(report.get("hazards", []))), "Visible / potential hazards")
        st.markdown(pill(status, status_kind), unsafe_allow_html=True)
        st.markdown("#### PPE detection checklist")
        ppe = report.get("ppe_detection", {})
        ppe_names = {
            "helmet": "Helmet",
            "high_visibility_vest": "High-visibility vest",
            "steel_toe_boots": "Steel-toe boots",
            "safety_harness": "Safety harness",
        }
        rows = []
        for key, label in ppe_names.items():
            item = ppe.get(key, {}) if isinstance(ppe, dict) else {}
            detected = item.get("detected", None) if isinstance(item, dict) else None
            status_text = "Visible" if detected is True else ("Not visible" if detected is False else "Not verifiable")
            rows.append({"PPE item": label, "Visual finding": status_text, "Compliance": item.get("compliance", "Not assessed"), "Notes": item.get("notes", "")})
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.markdown("#### Identified hazards & corrective actions")
        hazards = report.get("hazards", [])
        if hazards:
            for hazard in hazards:
                sev = str(hazard.get("severity", "Medium"))
                st.markdown(
                    f'<div class="panel"><div style="display:flex;justify-content:space-between;gap:1rem;align-items:center">'
                    f'<strong>{hazard.get("hazard", "Unspecified hazard")}</strong>{pill(sev, severity_kind(sev))}</div>'
                    f'<div class="small-muted" style="margin:.4rem 0">Location: {hazard.get("location", "Not specified")}</div>'
                    f'<div>{hazard.get("remedy", "Review and correct the hazard with the site safety lead.")}</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No hazards were returned. This does not prove the site is safe.")
        if report.get("limitations"):
            st.caption(f"Limitations: {report['limitations']}")
        download_json(report, "site_safety_audit.json")


# ---------------------------------------------------------------------------
# Tab 2: Crack diagnostics
# ---------------------------------------------------------------------------
with tab_cracks:
    st.markdown('<div class="section-title">Crack Diagnostics & Structural Integrity</div>', unsafe_allow_html=True)
    st.write("Upload a concrete or wall surface photo for preliminary visual triage.")
    uploaded_crack = st.file_uploader(
        "Concrete / wall surface photo",
        type=["jpg", "jpeg", "png"],
        key="crack_upload",
    )
    if uploaded_crack:
        try:
            Image.open(uploaded_crack)
            st.image(uploaded_crack, caption="Uploaded surface photograph", use_container_width=True)
        except Exception:
            st.error("This image could not be opened. Please upload a valid JPG or PNG.")

    crack_prompt = """
Act as a cautious construction defect triage assistant. Analyze only visible image evidence and return valid JSON:
{
 "defect_type": "Hairline"|"Settlement"|"Structural Shear"|"Thermal Expansion"|"Spalling"|"Uncertain",
 "estimated_crack_width_mm": number|null,
 "confidence": "Low"|"Medium"|"High",
 "observations": [string],
 "possible_root_causes": [string],
 "recommended_repairs": [{"action": string, "details": string}],
 "urgency": "Routine inspection"|"Prompt engineer review"|"Urgent / restrict access",
 "engineer_review_required": boolean,
 "limitations": string
}
Do not estimate crack width as a reliable measurement without a scale in the image; use null if not measurable. Mention that lighting, perspective, surface coating, and lack of scale limit diagnosis. Consider shrinkage, settlement, shear, thermal movement, corrosion-related spalling, and water ingress as possibilities, not confirmed causes.
Only recommend epoxy injection where appropriate after an engineer establishes that the crack type and activity suit it. Structural re-grouting is not a universal repair. For widening cracks, displacement, exposed/corroded reinforcement, falling concrete, or suspected load-bearing damage, advise prompt qualified structural engineer review and restrict access if unsafe. Do not certify structural integrity.
"""
    if st.button("Diagnose surface", type="primary", disabled=uploaded_crack is None, key="crack_analyze"):
        with st.spinner("Reviewing the visible defect..."):
            report, demo_mode = call_or_demo(client, crack_prompt, uploaded_crack, demo_crack_report)
        report["_metadata"] = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "mode": "demonstration" if demo_mode else "gemini",
            "image_filename": uploaded_crack.name if uploaded_crack else None,
        }
        st.session_state["crack_report"] = report
        st.session_state["crack_demo"] = demo_mode

    if "crack_report" in st.session_state:
        report = st.session_state["crack_report"]
        if st.session_state.get("crack_demo"):
            st.info("Demonstration report — this is sample content, not an analysis of the uploaded image.")
        urgency = str(report.get("urgency", "Unknown"))
        urgent_kind = "red" if "urgent" in urgency.lower() else ("amber" if "prompt" in urgency.lower() else "green")
        c1, c2, c3 = st.columns(3)
        with c1: render_kpi("Defect classification", report.get("defect_type", "Uncertain"), "Preliminary visual triage")
        width = report.get("estimated_crack_width_mm")
        with c2: render_kpi("Estimated width", f"{width} mm" if width is not None else "Not measurable", "A photo without scale cannot establish width")
        with c3: render_kpi("Urgency", urgency, f"Confidence: {report.get('confidence', 'Not stated')}")
        st.markdown(pill(urgency, urgent_kind), unsafe_allow_html=True)
        st.markdown("#### Visual observations")
        for item in report.get("observations", []):
            st.markdown(f"- {item}")
        st.markdown("#### Possible root causes")
        causes = report.get("possible_root_causes", [])
        if causes:
            for item in causes: st.markdown(f"- {item}")
        else:
            st.write("No causes returned.")
        st.markdown("#### Recommended engineering review / repairs")
        for repair in report.get("recommended_repairs", []):
            st.markdown(f"**{repair.get('action', 'Review')}** — {repair.get('details', '')}")
        if report.get("engineer_review_required", True):
            st.warning("Qualified structural engineer review is required before selecting a repair method.")
        if report.get("limitations"):
            st.caption(f"Limitations: {report['limitations']}")
        download_json(report, "crack_diagnostics_report.json")


# ---------------------------------------------------------------------------
# Tab 3: Planning, materials and illustrative valuation
# ---------------------------------------------------------------------------
with tab_planning:
    st.markdown('<div class="section-title">Planning & Real Estate Valuation</div>', unsafe_allow_html=True)
    st.write("Create an early-stage cost and materials estimate. Replace these illustrative rates with local BOQ, drawings, soil conditions and current supplier quotations before budgeting.")
    with st.form("planning_form"):
        p1, p2 = st.columns(2)
        with p1:
            structure = st.selectbox("Structure type", list(STRUCTURE_FACTORS.keys()))
            built_up_sqft = st.number_input("Built-up area (sq ft)", min_value=100, max_value=2_000_000, value=2500, step=100)
        with p2:
            tier = st.selectbox("Location tier", list(TIER_FACTORS.keys()), index=1)
            completion = st.slider("Construction completion", min_value=0, max_value=100, value=35, step=5)
        submit_plan = st.form_submit_button("Calculate project estimate", type="primary", use_container_width=True)

    if submit_plan or "plan_result" not in st.session_state:
        factors = STRUCTURE_FACTORS[structure]
        tier_factor = TIER_FACTORS[tier]
        area = float(built_up_sqft)
        total_budget = area * factors["base_rate"] * tier_factor["cost"]
        incurred = total_budget * completion / 100
        # Illustrative asset value uses a separate location and building-type multiplier.
        asset_value = total_budget * factors["market_factor"] * tier_factor["market"]
        steel_kg = area * factors["steel"]
        cement_bags = area * factors["cement"]
        st.session_state["plan_result"] = {
            "structure_type": structure,
            "built_up_area_sqft": int(area),
            "location_tier": tier,
            "completion_percent": completion,
            "estimated_total_project_budget_inr": round(total_budget, 2),
            "estimated_capex_incurred_inr": round(incurred, 2),
            "illustrative_current_market_asset_valuation_inr": round(asset_value, 2),
            "estimated_steel_kg": round(steel_kg, 1),
            "estimated_cement_bags_50kg": round(cement_bags, 1),
            "assumptions": {
                "base_cost_rate_inr_per_sqft": factors["base_rate"],
                "location_cost_factor": tier_factor["cost"],
                "market_value_multiplier": factors["market_factor"] * tier_factor["market"],
                "steel_intensity_kg_per_sqft": factors["steel"],
                "cement_intensity_bags_per_sqft": factors["cement"],
            },
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "disclaimer": "Illustrative planning model only. Not a professional valuation, BOQ, quantity survey, or construction estimate."
        }

    plan = st.session_state["plan_result"]
    m1, m2, m3 = st.columns(3)
    with m1: render_kpi("Total project budget", inr(plan["estimated_total_project_budget_inr"]), "Illustrative construction cost")
    with m2: render_kpi("Capital expenditure incurred", inr(plan["estimated_capex_incurred_inr"]), f'{plan["completion_percent"]}% completion assumed')
    with m3: render_kpi("Asset valuation", inr(plan["illustrative_current_market_asset_valuation_inr"]), "Indicative multiplier-based estimate")
    st.markdown("#### Milestone progress")
    st.progress(plan["completion_percent"] / 100, text=f'{plan["completion_percent"]}% estimated complete')
    st.markdown("#### Material requirement indicators")
    q1, q2, q3 = st.columns(3)
    with q1: render_kpi("Steel", f'{plan["estimated_steel_kg"]:,.0f} kg', "Rule-of-thumb intensity")
    with q2: render_kpi("Cement", f'{plan["estimated_cement_bags_50kg"]:,.0f} bags', "50 kg bags; indicative")
    with q3: render_kpi("Area", f'{plan["built_up_area_sqft"]:,} sq ft', f'{plan["structure_type"]} · {plan["location_tier"]}')
    with st.expander("View assumptions and estimate details"):
        st.json(plan["assumptions"])
        st.caption(plan["disclaimer"])
    download_json(plan, "planning_and_valuation_estimate.json")

    st.markdown("#### Conceptual 3D architectural visualisation")
    st.write("Generate a prompt for a conceptual render. The link opens Pollinations AI in a new browser tab; availability, output, and service terms are controlled by that provider.")
    v1, v2 = st.columns(2)
    with v1:
        style = st.selectbox("Architectural style", ["Contemporary Indian", "Minimalist modern", "Sustainable tropical", "Industrial", "Traditional courtyard"])
    with v2:
        render_time = st.selectbox("Lighting", ["Golden hour", "Bright daylight", "Blue hour", "Overcast soft light"])
    prompt = (
        f"Photorealistic conceptual 3D architectural visualization of a {plan['structure_type'].lower()}, "
        f"{plan['built_up_area_sqft']} square feet, {style.lower()} architecture in an Indian {plan['location_tier'].lower()} urban context, "
        f"{render_time.lower()}, realistic materials, clear structural massing, landscaped site, professional architectural render, "
        "wide angle exterior view, no text, no watermark. Concept only, not a construction drawing."
    )
    st.text_area("Generated prompt", value=prompt, height=100, key="render_prompt")
    pollinations_url = "https://image.pollinations.ai/prompt/" + quote(prompt, safe="")
    st.link_button("Open conceptual render in Pollinations AI", pollinations_url, use_container_width=True)
    st.caption("Do not use generated imagery for permitting, dimensions, structural coordination, or construction execution.")


# ---------------------------------------------------------------------------
# Tab 4: Weather-aware guidance and conversational assistant
# ---------------------------------------------------------------------------
with tab_assistant:
    st.markdown('<div class="section-title">Daily Site Assistant & Weather-Aware Advisory</div>', unsafe_allow_html=True)
    weather = st.selectbox("Current / forecast site condition", ["Normal", "High Wind", "Heavy Rain", "Extreme Heat"])
    alerts = {
        "Normal": {
            "kind": "green", "title": "Routine controls",
            "items": [
                "Complete the daily toolbox talk and task-specific risk assessment.",
                "Inspect access routes, scaffolds, lifting gear and PPE before work.",
                "Follow the approved method statement, permit-to-work and concrete QA plan."
            ]
        },
        "High Wind": {
            "kind": "red", "title": "Wind-related controls",
            "items": [
                "Pause crane lifts, hoisting and work at height when wind exceeds equipment / method-statement limits; follow the manufacturer's limits and site lifting plan.",
                "Secure loose sheets, formwork, signage and materials; establish exclusion zones around unstable loads.",
                "Reassess scaffold, edge protection and suspended-platform conditions before restarting."
            ]
        },
        "Heavy Rain": {
            "kind": "red", "title": "Rain, flooding and electrical hazards",
            "items": [
                "Stop work in flooded excavations and assess collapse, access and dewatering risks before entry.",
                "Isolate damaged or water-exposed electrical equipment; use tested RCD/GFCI protection and qualified electricians.",
                "Protect fresh concrete from washout and standing water; use approved curing and protection methods after placement conditions are assessed.",
                "Inspect slopes, trenches, access paths and temporary works after rainfall before resuming."
            ]
        },
        "Extreme Heat": {
            "kind": "amber", "title": "Heat-stress controls",
            "items": [
                "Schedule strenuous work for cooler periods where possible; provide shaded rest, potable water and acclimatisation.",
                "Use a buddy system and train workers to recognise heat illness; escalate symptoms promptly.",
                "Control concrete placement temperature, evaporation and curing using the approved mix and project specification; do not add water without authorization."
            ]
        }
    }
    alert = alerts[weather]
    st.markdown(
        f'<div class="panel"><div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.5rem">'
        f'{pill(weather, alert["kind"])}<strong>{alert["title"]}</strong></div></div>',
        unsafe_allow_html=True,
    )
    for item in alert["items"]:
        st.markdown(f"- {item}")
    st.caption("Weather guidance is general. Follow site-specific limits, equipment manuals, local law and the appointed safety officer.")

    st.divider()
    st.markdown("#### Ask the construction assistant")
    st.write("Ask about sequencing, materials, quality checks, safety planning, or interpreting project documentation. Do not use this chatbot as an emergency response service.")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Ask a construction engineering question...")
    if user_question:
        st.session_state["chat_history"].append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)
        with st.chat_message("assistant"):
            if client:
                try:
                    history_text = "\n".join(
                        f'{m["role"].upper()}: {m["content"]}'
                        for m in st.session_state["chat_history"][-10:]
                    )
                    assistant_prompt = f"""
You are a cautious construction site assistant. Current site weather selection: {weather}.
Give practical, concise, context-aware guidance. Ask clarifying questions when project context is missing.
Do not claim to replace a licensed structural engineer, safety officer, local authority, or project specification.
For imminent danger, recommend stopping work, isolating the area, and contacting site emergency / safety leads.
Do not invent code clauses or precise engineering calculations. Distinguish general guidance from verified requirements.
Conversation:
{history_text}
ASSISTANT:
"""
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=assistant_prompt,
                    )
                    answer = response.text or "I couldn't produce a response. Please try again."
                except Exception as exc:
                    answer = (
                        f"I couldn't reach Gemini ({type(exc).__name__}). In demo mode, I can still provide general guidance: "
                        "identify the task and site conditions, check the approved drawings and method statement, complete a task-specific risk assessment, "
                        "and confirm any safety-critical decision with the responsible site professional."
                    )
            else:
                answer = (
                    "Demonstration mode: I can offer a general checklist, but this response is not AI-generated. "
                    "First define the activity, location, materials, equipment and current site conditions. Then check the approved drawings, "
                    "method statement, applicable standards and task-specific risk assessment. For safety-critical or structural decisions, "
                    "pause the activity if needed and consult the responsible engineer or safety officer."
                )
            st.markdown(answer)
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})

    chat_col1, chat_col2 = st.columns([1, 4])
    with chat_col1:
        if st.button("Clear chat history"):
            st.session_state["chat_history"] = []
            st.rerun()


st.divider()
st.caption(
    "Multimodal Construction Intelligence Platform · Skill Wallet Capstone Project 1 · "
    "Session-based prototype with no external database. Review all outputs before operational use."
)
