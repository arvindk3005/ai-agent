import os
import time
import tempfile
from dotenv import load_dotenv

import streamlit as st
from PIL import Image
import google.generativeai as genai

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------- LOAD ENV ----------------

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)


# ---------------- PAGE ----------------

st.set_page_config(
    page_title="AI Multi Agent",
    page_icon="🤖"
)

st.title("🤖 AI Multi Agent")
st.write("Chat + PDF + Image + Audio + Video + Notes + Calculator")


# ---------------- SESSION STATE ----------------

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None

if "uploaded_file_name" not in st.session_state:
    st.session_state["uploaded_file_name"] = None

if "media_file" not in st.session_state:
    st.session_state["media_file"] = None

if "file_type" not in st.session_state:
    st.session_state["file_type"] = None


# ---------------- MODELS ----------------

chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

media_model = genai.GenerativeModel(
    "gemini-2.5-flash"
)


# ---------------- TOOLS ----------------

@tool
def calculator(expression: str) -> str:
    """Use this tool for math calculations."""

    try:
        allowed = "0123456789+-*/(). "

        if not all(char in allowed for char in expression):
            return "Invalid expression."

        result = eval(expression)

        return f"Answer: {result}"

    except Exception as e:
        return f"Calculation error: {e}"


@tool
def save_note(note: str) -> str:
    """Save notes into notes.txt"""

    with open("notes.txt", "a") as file:
        file.write(note + "\n")

    return "Note saved successfully."


@tool
def read_notes(query: str = "") -> str:
    """Read saved notes."""

    if not os.path.exists("notes.txt"):
        return "No notes available."

    with open("notes.txt", "r") as file:
        notes = file.read()

    if notes.strip() == "":
        return "No notes available."

    return notes


tools = [
    calculator,
    save_note,
    read_notes
]


# ---------------- AGENT ----------------

agent = create_agent(
    model=chat_model,
    tools=tools,
    system_prompt="""
You are a smart AI assistant.

Rules:
- Use calculator tool for calculations.
- Use save_note tool if user asks to save notes.
- Use read_notes tool if user asks to read notes.
- Uploaded files are handled separately.
"""
)


# ---------------- PDF PROCESSING ----------------

def process_pdf(pdf_path):

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    # LIMIT CHUNKS FOR FREE TIER
    chunks = chunks[:40]

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore


# ---------------- WAIT FOR MEDIA ----------------

def wait_for_media(file):

    while file.state.name == "PROCESSING":
        time.sleep(2)
        file = genai.get_file(file.name)

    if file.state.name == "FAILED":
        raise ValueError("File processing failed.")

    return file


# ---------------- FILE UPLOADER ----------------

uploaded_file = st.file_uploader(
    "Upload PDF / Image / Audio / Video",
    type=[
        "pdf",
        "png",
        "jpg",
        "jpeg",
        "mp3",
        "wav",
        "m4a",
        "mp4",
        "mov",
        "avi"
    ]
)


# ---------------- HANDLE UPLOAD ----------------

if uploaded_file is not None:

    if st.session_state["uploaded_file_name"] != uploaded_file.name:

        suffix = "." + uploaded_file.name.split(".")[-1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(uploaded_file.read())

            temp_path = temp_file.name

        file_type = uploaded_file.type

        st.session_state["uploaded_file_name"] = uploaded_file.name
        st.session_state["file_type"] = file_type

        # PDF
        if file_type == "application/pdf":

            with st.spinner("Processing PDF..."):

                vectorstore = process_pdf(temp_path)

                st.session_state["vectorstore"] = vectorstore
                st.session_state["media_file"] = None

            st.success("PDF uploaded successfully!")

        # IMAGE
        elif file_type.startswith("image"):

            image = Image.open(temp_path)

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

            with st.spinner("Uploading image..."):

                media_file = genai.upload_file(temp_path)

                media_file = wait_for_media(media_file)

                st.session_state["media_file"] = media_file
                st.session_state["vectorstore"] = None

            st.success("Image uploaded successfully!")

        # AUDIO
        elif file_type.startswith("audio"):

            with st.spinner("Uploading audio..."):

                media_file = genai.upload_file(temp_path)

                media_file = wait_for_media(media_file)

                st.session_state["media_file"] = media_file
                st.session_state["vectorstore"] = None

            st.success("Audio uploaded successfully!")

        # VIDEO
        elif file_type.startswith("video"):

            with st.spinner("Uploading video..."):

                media_file = genai.upload_file(temp_path)

                media_file = wait_for_media(media_file)

                st.session_state["media_file"] = media_file
                st.session_state["vectorstore"] = None

            st.success("Video uploaded successfully!")


# ---------------- CHAT HISTORY ----------------

for message in st.session_state["messages"]:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# ---------------- CHAT INPUT ----------------

user_input = st.chat_input("Ask anything...")


# ---------------- MAIN LOGIC ----------------

if user_input:

    st.session_state["messages"].append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    file_type = st.session_state.get("file_type")
    vectorstore = st.session_state.get("vectorstore")
    media_file = st.session_state.get("media_file")

    ai_reply = ""

    # ---------- PDF RAG ----------

    if (
        file_type == "application/pdf"
        and vectorstore is not None
    ):

        docs = vectorstore.similarity_search(
            user_input,
            k=4
        )

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = f"""
You are a PDF assistant.

Answer ONLY from the PDF context below.

If answer is not found,
say:
"Answer not clearly found in PDF."

PDF Context:
{context}

Question:
{user_input}
"""

        try:
            response = chat_model.invoke(prompt)
            ai_reply = response.content

        except Exception as e:
            error_text = str(e)

            if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text or "quota" in error_text.lower():
                ai_reply = (
                    "Gemini free-tier quota is exhausted right now. "
                    "Wait for the retry time shown in terminal, use a smaller PDF, "
                    "or switch to another API key/billing-enabled project."
                )
            else:
                ai_reply = f"PDF answer error: {e}"

    # ---------- IMAGE / AUDIO / VIDEO ----------

    elif media_file is not None:

        try:
            response = media_model.generate_content([
                media_file,
                user_input
            ])

            ai_reply = response.text

        except Exception as e:
            error_text = str(e)

            if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text or "quota" in error_text.lower():
                ai_reply = (
                    "Gemini free-tier quota is exhausted right now. "
                    "Wait for the retry time shown in terminal, use fewer requests, upload a smaller file, "
                    "or switch to another API key/billing-enabled project."
                )
            else:
                ai_reply = f"Media analysis error: {e}"

    # ---------- NORMAL CHAT / TOOLS ----------

    else:

        try:
            response = agent.invoke({
                "messages": st.session_state["messages"]
            })

            last_message = response["messages"][-1]

            try:
                ai_reply = last_message.text

            except Exception:

                try:
                    ai_reply = last_message.content[0]["text"]

                except Exception:
                    ai_reply = str(last_message.content)

        except Exception as e:
            error_text = str(e)

            if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text or "quota" in error_text.lower():
                ai_reply = (
                    "Gemini free-tier quota is exhausted right now. "
                    "Wait for the retry time shown in terminal, reduce requests, "
                    "or switch to another API key/billing-enabled project."
                )
            else:
                ai_reply = f"Chat error: {e}"

    # ---------- SAVE CHAT ----------

    st.session_state["messages"].append({
        "role": "assistant",
        "content": ai_reply
    })

    with st.chat_message("assistant"):
        st.write(ai_reply)