import streamlit as st
import os
import requests
import json
from groq import Groq

st.set_page_config(page_title="Live AI Scraper", layout="wide")
if "results" not in st.session_state: st.session_state.results = None

col1, col2 = st.columns([1, 2])
with col1:
    st.header("Control Panel")
    url = st.text_input("Website URL", value="https://example.com")
    prompt = st.text_area("What to extract?", value="Get all the main paragraphs.")
    if st.button("Scrape & Update", type="primary"):
        with st.spinner("Agent working..."):
            try:
                scrape_url = "https://r.jina.ai/" + url
                headers = {'Accept': 'application/json'}
                res = requests.get(scrape_url, headers=headers, timeout=15)
                text = res.json()['data']['content']
                client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
                ai_prompt = f"Extract from text. Request: {prompt}\n\nText:\n{text[:12000]}\n\nRespond ONLY with valid JSON."
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": ai_prompt}],
                    temperature=0.1
                )
                try:
                    clean = response.choices[0].message.content
                    if "`json" in clean: clean = clean.split("`json", 1)[1]
                    if "`" in clean: clean = clean.split("`", 1)[0]
                    st.session_state.results = json.loads(clean)
                except:
                    st.session_state.results = {"Data": response.choices[0].message.content}
            except Exception as e: st.error(e)

with col2:
    st.header("Live Dashboard View")
    if st.session_state.results: st.json(st.session_state.results)
    else: st.info("Click the button on the left to fetch data.")
