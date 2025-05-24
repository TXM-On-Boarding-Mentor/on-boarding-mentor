# 💬 Chatbot template

A simple Streamlit app that shows how to build a chatbot using OpenAI's GPT-3.5.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chatbot-template.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Apply for API key 
Apply API key from: 
- https://platform.openai.com/api-keys
- https://aistudio.google.com/app/apikey

Replace <YOUR API KEY> with your real API key in example.env and rename example.env to .env 

```bash
OPENAI_API_KEY = <YOUR API KEY>
GEMINI_API_KEY = <YOUR API KEY>
```


3. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```

### Live Demo
https://main-deploy.streamlit.app/


4. Setup virtual environment 
setup
```bash
python -m venv venv
```

activate
```bash
source venv/bin/activate
```
