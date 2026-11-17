# NC Midterm Message & $2M Allocation Intelligence Tracker

## Project Description
An interactive Streamlit analytics consultancy web application designed for a North Carolina independent expenditure PAC digital strategy director. The app aggregates live political advertising data from the Meta Ad Library and Google Public Ads BigQuery transparency datasets to track statewide midterm ad spending ($565M+ market), analyze competitor messaging across seven fixed issue themes, run a head-to-head Super PAC gap analysis, and simulate a real-time $2.0M war chest deployment with dynamic Share of Voice (SOV) projections and context-grounded AI tactical shift briefs.

## App Deployment URL

[https://share.streamlit.io/FoxSpikes/NC-Midterm-Message-Funds-Allocation-Tracker](https://share.streamlit.io/FoxSpikes/NC-Midterm-Message-Funds-Allocation-Tracker)

## Local Setup Instructions

Prerequisites: Ensure you have Python and `uv` installed on your machine.

```bash
git clone [https://github.com/FoxSpikes/NC-Midterm-Message-Funds-Allocation-Tracker.git](https://github.com/FoxSpikes/NC-Midterm-Message-Funds-Allocation-Tracker.git)
cd NC-Midterm-Message-Funds-Allocation-Tracker
uv sync
uv run streamlit run app.py
