import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

load_dotenv(".env")  # pastikan .env berisi setting OLLAMA jika diperlukan

st.set_page_config(page_title="Chatbot Wawancara Kerja - UAS")
st.title("MATA CORPORATION")
st.caption("UAS Big Data 2025")

st.markdown("""
    <style>
    .user-wrapper {
        display: flex;
        justify-content: flex-end;
    }
    .user-bubble {
        background-color: #5e5e5e;
        color: white;
        padding: 0.7em 1em;
        border-radius: 15px;
        margin: 0.5em 0;
        max-width: 70%;
        text-align: left;
        display: inline-block;
        word-wrap: break-word;
    }
    .assistant-bubble {
        background-color: #2b2b2b;
        color: white;
        padding: 0.7em 1em;
        border-radius: 15px;
        margin: 0.5em 0;
        max-width: 70%;
        text-align: left;
        display: inline-block;
        word-wrap: break-word;
    }
    </style>
""", unsafe_allow_html=True)

base_url = "http://localhost:11434"
model = "llama3.2:3b"
session_id = "default_session"

# Ambil history dari database SQLite
def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///chat_history.db")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("Mulai Wawancara"):
    st.session_state.chat_history = []
    history = get_session_history(session_id)
    history.clear()

# Tampilkan riwayat obrolan
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"""
            <div class="user-wrapper">
                <div class="user-bubble">{msg["content"]}</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="assistant-bubble">{msg["content"]}</div>
        """, unsafe_allow_html=True)

# Setup Langchain LLM
llm = ChatOllama(base_url=base_url, model=model)

system_prompt = """
You are an HR interviewer for MATA CORPORATION which is a software company.
Dont Add Expressions like (firmly, Please respond as if we were in a real interview) and dont use name.
Ask the user about their name first.
Ask the user 6 interview questions, one at a time.
Evaluate their answers, and after the 5th question, decide if the candidate is ACCEPT or REJECT. And only answer REJECT or ACCEPT with a short reason based on:
 - relevant skills
 - clarity of answers
Output the decision at the end only.
"""

system = SystemMessagePromptTemplate.from_template(system_prompt)
human = HumanMessagePromptTemplate.from_template("{input}")
messages = [system, MessagesPlaceholder(variable_name="history"), human]
prompt = ChatPromptTemplate(messages=messages)
chain = prompt | llm | StrOutputParser()

runnable_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

def chat_with_llm(session_id, input):
    for output in runnable_with_history.stream({"input": input}, config={"configurable": {"session_id": session_id}}):
        yield output

prompt_input = st.chat_input("Jawab pertanyaan di sini...")

if prompt_input:
    st.session_state.chat_history.append({"role": "user", "content": prompt_input})

    st.markdown(f"""
        <div class="user-wrapper">
            <div class="user-bubble">{prompt_input}</div>
        </div>
    """, unsafe_allow_html=True)

    
    response_container = st.empty()
    response_text = ""

    for chunk in chat_with_llm(session_id, prompt_input):
        response_text += chunk
        response_container.markdown(f"""<div class="assistant-bubble">{response_text}▌</div>""", unsafe_allow_html=True)

    response_container.markdown(f"""<div class="assistant-bubble">{response_text}</div>""", unsafe_allow_html=True)
    st.session_state.chat_history.append({"role": "assistant", "content": response_text})

