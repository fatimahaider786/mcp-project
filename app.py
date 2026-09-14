import streamlit as st
import os
import resend
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Load environment variables from .env file
load_dotenv()

# --- Page Configuration & Styling ---
st.set_page_config(
    page_title="AI Assistant & MCP Tools",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stChatInputContainer {
        padding-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Smart AI Assistant with Tools")
st.caption("Aapka personal AI agent jo math calculations, CRM management aur real emails bhej sakta hai!")

# --- Fetch Credentials from Environment Variables ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "onboarding@resend.dev")

# Set Resend API key
if RESEND_API_KEY:
    resend.api_key = RESEND_API_KEY

# Check if credentials are set
if not GROQ_API_KEY or not RESEND_API_KEY:
    st.error("⚠️ `.env` file mein `GROQ_API_KEY` ya `RESEND_API_KEY` missing hai!")
    st.stop()

# --- Define Tools ---
@tool
def add(a: int, b: int) -> int:
    """Adds two numbers together."""
    return a + b

@tool
def greet(name: str) -> str:
    """Greets a person by name."""
    return f"Hello, {name}! Welcome to our system."

@tool
def send_lead_to_crm(name: str, email: str, company: str) -> str:
    """Saves a new lead into the CRM system."""
    return f"Success: Lead {name} from {company} ({email}) has been saved to the CRM."

@tool
def send_email(to_email: str, subject: str, body: str) -> str:
    """Sends a real email to any target email address using Resend API."""
    if not RESEND_API_KEY:
        return "Error: Resend API Key missing in environment variables!"
    try:
        params = {
            "from": SENDER_EMAIL,
            "to": [to_email],
            "subject": subject,
            "text": body,
        }
        response = resend.Emails.send(params)
        return f"Success: Email successfully sent to {to_email}! (ID: {response.get('id')})"
    except Exception as e:
        return f"Error sending email: {str(e)}"

tools = [add, greet, send_lead_to_crm, send_email]
tool_map = {tool.name: tool for tool in tools}

# --- Initialize LLM with Tool Binding ---
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, groq_api_key=GROQ_API_KEY)
llm_with_tools = llm.bind_tools(tools)

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Chat Input
if user_input := st.chat_input("Yahan apna sawal likhein (jaise: 'Sum of 50 and 60' ya 'Send email to...')"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("AI soch raha hai aur tools run kar raha hai..."):
            try:
                # Build LangChain messages from history
                lc_messages = []
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        lc_messages.append(HumanMessage(content=msg["content"]))
                    else:
                        lc_messages.append(AIMessage(content=msg["content"]))

                # First LLM call
                response = llm_with_tools.invoke(lc_messages)
                
                # Check if model wants to call tools
                while response.tool_calls:
                    lc_messages.append(response)
                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        
                        if tool_name in tool_map:
                            tool_result = tool_map[tool_name].invoke(tool_args)
                        else:
                            tool_result = f"Error: Tool {tool_name} not found."
                        
                        lc_messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]))
                    
                    # Next LLM call after tool execution
                    response = llm_with_tools.invoke(lc_messages)

                bot_response = response.content
            except Exception as e:
                bot_response = f"Koi error aa gaya hai: {str(e)}"
            
            st.markdown(bot_response)
            st.session_state.messages.append({"role": "assistant", "content": bot_response})