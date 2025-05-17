import streamlit as st
from openai import OpenAI
import time
from dotenv import load_dotenv
import os
import autogen
from autogen import ConversableAgent, LLMConfig
from autogen import AssistantAgent, UserProxyAgent
from autogen.code_utils import content_str
from utils.ui_helper import UIHelper

class OrchestratorAgent:
    def __init__(self):
        self.user_name = "On-boarding Mentor"
        self.user_image = "https://www.w3schools.com/howto/img_avatar.png"
        self.placeholderstr = "Please input your command"
        self.seed = 42
        self._load_environment()
        UIHelper.config_page()
        self._setup_llm_configs()
        self._initialize_agents()

    def _load_environment(self):
        """Load environment variables from .env file."""
        load_dotenv(override=True)
        self.openai_api_key = st.secrets["OPENAI_API_KEY"]
        self.gemini_api_key = st.secrets["GEMINI_API_KEY"]

    def _setup_llm_configs(self):
        """Set up LLM configurations for Gemini and OpenAI."""
        self.llm_config_gemini = LLMConfig(
            api_type="google",
            model="gemini-2.0-flash-lite",
            api_key=self.gemini_api_key
        )
        self.llm_config_openai = LLMConfig(
            api_type="openai",
            model="gpt-4o-mini",
            api_key=self.openai_api_key
        )

    def _initialize_agents(self):
        """Initialize assistant and user proxy agents."""
        self.assistant = AssistantAgent(
            name="assistant",
            system_message=(
               "You are a helpful agent coordinator for an onboarding website. "
                "Your role is to guide new employees through the platform and help them understand how to use its features. "
                "When users ask questions like 'How should I start with this website?', respond with an overview of the site's purpose: "
                "'This website helps you get familiar with your company and onboarding process! The website allows you to: "
                "1. Visualize the enterprise culture, "
                "2. Understand organization stakeholders, and "
                "3. Grow with the company using your personal note.' "
                "Direct users to the appropriate subpages as follows: "
                "1. For visualizing the enterprise culture, direct them to the 'Create Word Cloud' page. "
                "2. For rthe information of their notes, direct them to the 'Chat with Notes' page. "
                "3. For growing with the company, direct them to the 'Upload Notes' page. "
                "Answer all user questions in a concise and helpful way based on this structure."
            ),
            llm_config=self.llm_config_openai,
            max_consecutive_auto_reply=1
        )
        self.user_proxy = UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            code_execution_config=False,
            is_termination_msg=lambda x: content_str(x.get("content")).find("ALL DONE") >= 0
        )

    def stream_data(self, stream_str):
        """Stream data with a delay for each word."""
        for word in stream_str.split(" "):
            yield word + " "
            time.sleep(0.05)

    def generate_response(self, prompt):
        """Generate a response from the assistant based on the prompt."""
        prompt_template = f"{prompt}"
        result = self.user_proxy.initiate_chat(
            recipient=self.assistant,
            message=prompt_template
        )
        return result.chat_history

    def show_chat_history(self, chat_history, container):
        """Display chat history entries (both user_proxy and assistant)."""
        for entry in chat_history:
            role = entry.get('role')
            content = entry.get('content', '').strip()
            if not content or 'ALL DONE' in content:
                continue
            # Persist
            st.session_state.messages.append({"role": role, "content": content})
            # Render
            if role == 'assistant':
                container.chat_message('assistant').markdown(content)
            else:
                container.chat_message('user', avatar=self.user_image).markdown(content)
                
                
                
                

    def run(self):
        """Main method to run the application with full history display."""
        st.title(f"💬 {self.user_name}")
        UIHelper.setup_sidebar()
        chat_container = st.container()

        # Initialize session_state for messages
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Render all past messages
        for msg in st.session_state.messages:
            role = msg["role"]
            content = msg["content"]
            if role == "assistant":
                chat_container.chat_message("assistant").markdown(content)
            else:
                chat_container.chat_message("user", avatar=self.user_image).markdown(content)
                
                

        # Handle new user input
        if prompt := st.chat_input(placeholder=self.placeholderstr, key="chat_bot"):
            # Append and display user message
            #st.session_state.messages.append({"role": "user", "content": prompt})
            #chat_container.chat_message("user", avatar=self.user_image).markdown(prompt)

            # Generate and display all new history entries
            history = self.generate_response(prompt)
            self.show_chat_history(history, chat_container)

if __name__ == "__main__":
    orchestrator = OrchestratorAgent()
    orchestrator.run()
