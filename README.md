# Multimodal Construction Intelligence Platform
**Skill Wallet Capstone Project 1**

A Streamlit prototype for AI-assisted construction-site safety review, concrete crack triage, early-stage planning/valuation, and weather-aware site guidance.

## Features

1. **Site Safety & PPE Audit** — upload JPG/PNG, review visible PPE and hazards, see a safety score/status, and export JSON.
2. **Crack Diagnostics & Structural Integrity** — upload a surface photo, classify a possible defect, show possible causes and engineering-review actions, and export JSON.
3. **Planning & Real Estate Valuation** — estimate project budget, incurred CAPEX, illustrative asset valuation, steel and cement indicators, milestone progress, and create a Pollinations AI conceptual-render prompt.
4. **Daily Site Assistant** — weather-aware safety reminders and a multi-turn Gemini assistant.

## Run locally

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit.

## Gemini API key

- You can paste a Google Gemini API key into the sidebar at runtime.
- The app does not write the key to a file or database; it is used in the current Streamlit session.
- Without a key, the app remains usable in **demonstration mode**. Safety/crack tabs show illustrative sample reports and explicitly label them as such.
- For a deployed project, use Streamlit Community Cloud **Secrets** rather than asking users to paste a key. This prototype intentionally supports session-only entry.

## Deploy to Streamlit Community Cloud

1. Push `app.py` and `requirements.txt` to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository and select `app.py`.
3. Deploy. No database, local disk persistence, or additional services are required.
4. For a private project-level Gemini key, add a secret such as:

```toml
GEMINI_API_KEY = "your-key-here"
```

To use a configured secret automatically, update the app to read `st.secrets.get("GEMINI_API_KEY", "")` and keep the secret out of source control. Never commit a real API key.

## Important limitations

- **Not a certified safety inspection or engineering assessment.** Image models can miss hazards, misclassify PPE, and cannot reliably measure crack width without a scale.
- **Standards:** IS 456 is a plain and reinforced concrete code, not a PPE standard. OSHA principles and Indian project/site requirements must be checked against the jurisdiction and actual work conditions.
- **Cost assumptions are illustrative:** per-square-foot rates, steel intensity, cement intensity, and market multipliers are simplified defaults. Replace them with a local BOQ, design, schedule, current rates, and professional valuation.
- **Pollinations AI:** the app builds a prompt and opens the provider URL. Availability and terms are controlled by the external provider.
- **Privacy:** uploads are processed in the app session and may be sent to Gemini if the user enables AI mode. Do not upload sensitive images unless authorized.
- **Production hardening:** before operational deployment, add access control, audit logging, rate limits, approved model/version management, structured output validation, monitoring, and formal engineering review workflows.

## Project structure

```text
multimodal_construction_intelligence/
├── app.py
├── requirements.txt
└── README.md
```
