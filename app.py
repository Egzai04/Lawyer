import streamlit as st
from rag_pipeline import load_and_index_pdf, build_rag_chain, get_answer

st.set_page_config(
    page_title="Indian Constitution Chatbot",
    page_icon="⚖️",
    layout="centered"
)

st.title("⚖️ EGZAI'S PERSONAL LAWYER")
st.caption("Any legal problem? — don,t worry, Egzai's assistant will help you")

# Load and index PDF only once per session
@st.cache_resource(show_spinner="⚖️ Indian lawyer loading...")
def initialize_chain():
    vectorstore = load_and_index_pdf()
    chain_and_retriever = build_rag_chain(vectorstore)  # returns (chain, retriever)
    return chain_and_retriever

chain_and_retriever = initialize_chain()

# Initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if user_input := st.chat_input("Mention your problem..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Searching the Constitution..."):
            result = get_answer(chain_and_retriever, user_input)
            answer = result["answer"]
            sources = result["sources"]

        st.markdown(answer)

        # Show source chunks
        if sources:
            with st.expander("📜 Relevant Constitutional Provisions"):
                for i, src in enumerate(sources, 1):
                    st.markdown(f"**Source {i} — Page {src['page']}**")
                    st.caption(src["snippet"] + "...")
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})

## `.env` file
HUGGINGFACEHUB_API_TOKEN="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"