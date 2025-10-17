import streamlit as st
import requests

# The URL of your running LangServe endpoint
API_URL = "http://localhost:8000/research-assistant/invoke" 

st.title("🔬 AI Research Assistant")

question = st.text_input("Enter your research topic:", key="user_input")

if st.button("Generate Report") and question:
    with st.spinner("🚀 Generating detailed report..."):
        try:
            # 1. Prepare the JSON payload for the LangServe API
            payload = {"input": {"question": question}}

            # 2. Make the POST request to your running backend
            response = requests.post(API_URL, json=payload, timeout=300) 
            response.raise_for_status() # Raise an exception for bad status codes

            # 3. Extract the final report (assuming the output structure)
            # The response from LangServe /invoke is usually: {"output": "...", "metadata": {}}
            report_content = response.json().get("output", "Error: No output found.")

            st.markdown("---")
            st.subheader("✅ Research Report")
            st.markdown(report_content) # Render the report using Markdown
            st.markdown("---")

        except requests.exceptions.RequestException as e:
            st.error(f"An API error occurred. Is the LangServe server running? Error: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

st.caption("Powered by LangServe and OpenRouter.")