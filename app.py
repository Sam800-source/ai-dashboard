import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import os, requests, json
from groq import Groq

st.set_page_config(page_title="AI & ML Command Center", layout="wide")
st.title("🧠 AI & ML Command Center")
tab1, tab2, tab3 = st.tabs(["📊 Data Explorer", "🤖 ML Laboratory", "🕷️ AI Web Scraper"])

# --- TAB 1: DATA EXPLORER & CHARTS ---
with tab1:
    st.header("Upload Data & Visualize")
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.dataframe(df.head(500), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("X Axis", df.columns)
        with col2:
            y_axis = st.selectbox("Y Axis", df.columns)
            
        chart_type = st.radio("Chart Type", ["Line", "Bar", "Scatter", "Histogram"], horizontal=True)
        
        if st.button("Generate Chart"):
            if chart_type == "Line": fig = px.line(df, x=x_axis, y=y_axis)
            elif chart_type == "Bar": fig = px.bar(df, x=x_axis, y=y_axis)
            elif chart_type == "Scatter": fig = px.scatter(df, x=x_axis, y=y_axis)
            else: fig = px.histogram(df, x=x_axis)
            st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: ML LABORATORY ---
with tab2:
    st.header("Train Machine Learning Models (Free & Local)")
    if uploaded_file:
        target = st.selectbox("Select Target Column to Predict", df.columns)
        task = st.selectbox("Task Type", ["Classification (Categories)", "Regression (Numbers)"])
        model_choice = st.selectbox("Select Model", ["Random Forest", "Logistic Regression", "Linear Regression"])
        
        if st.button("🚀 Train Model"):
            # Clean data (drop non-numeric for simplicity)
            numeric_df = df.select_dtypes(include=['number']).dropna()
            if target not in numeric_df.columns:
                st.error("Target column must be numbers for this demo.")
            else:
                X = numeric_df.drop(columns=[target])
                y = numeric_df[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                
                if model_choice == "Random Forest": model = RandomForestClassifier()
                elif model_choice == "Logistic Regression": model = LogisticRegression(max_iter=1000)
                else: model = LinearRegression()
                
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                
                if task == "Classification (Categories)":
                    acc = accuracy_score(y_test, predictions)
                    st.success(f"Model Trained! Accuracy: {acc * 100:.2f}%")
                else:
                    mse = mean_squared_error(y_test, predictions)
                    st.success(f"Model Trained! Mean Squared Error: {mse:.2f}")
                
                st.subheader("Predictions vs Actual")
                result_df = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
                st.dataframe(result_df.head(), use_container_width=True)

# --- TAB 3: AI WEB SCRAPER ---
with tab3:
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
                    response = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": ai_prompt}], temperature=0.1)
                    try:
                        clean = response.choices[0].message.content
                        if "`json" in clean: clean = clean.split("`json", 1)[1]
                        if "`" in clean: clean = clean.split("`", 1)[0]
                        st.session_state.results = json.loads(clean)
                    except: st.session_state.results = {"Data": response.choices[0].message.content}
                except Exception as e: st.error(e)
    with col2:
        st.header("Live Dashboard View")
        if st.session_state.results: st.json(st.session_state.results)
        else: st.info("Click the button on the left to fetch data.")
