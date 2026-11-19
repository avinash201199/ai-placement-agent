"""
app.py - The Streamlit UI.  Run it with:  streamlit run app.py
"""

import os

import streamlit as st

from agent import demo_extract_profile, extract_profile, run_agent, run_demo_agent
from tools import read_resume_text

st.set_page_config(page_title="AI Placement Agent", page_icon="🤖")
st.title("🤖 AI Placement Agent")
st.caption("Upload your resume. The agent finds matching jobs, shows your skill gaps "
           "and writes a referral message. Built with Let's Code: lets-code.co.in")

DEMO = not os.getenv("LLM_API_KEY")
if DEMO:
    st.info("Demo mode: no API key found, so the steps run in a fixed order with the real tools. "
            "Add your key in .env to let the AI agent plan and write the report.")

resume = st.file_uploader("Upload your resume (PDF)", type="pdf")

if resume and st.button("Find my jobs 🚀"):
    with st.status("Agent is working...", expanded=True) as status:
        st.write("Reading your resume...")
        text = read_resume_text(resume)
        profile = demo_extract_profile(text) if DEMO else extract_profile(text)
        st.json(profile)

        report = (run_demo_agent if DEMO else run_agent)(profile, on_step=st.write)
        status.update(label="Done ✅", state="complete", expanded=True)

    st.markdown(report)
