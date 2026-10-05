"""Placeholder; the full UI is built in Phase 7 (spec section 15)."""
import os

import streamlit as st

st.set_page_config(page_title="Car Rental Assistant")
st.title("Car Rental Assistant")
st.info(f"UI coming soon. Agent API: {os.getenv('AGENT_API_URL', 'http://localhost:8000')}")
