import os
import tempfile
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


# ---------------------------------------------------------
# 1. Load environment variables
# ---------------------------------------------------------
#load_dotenv()
os.environ['OPENAI_API_KEY'] = "your api key"

CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)



# ---------------------------------------------------------
# 2. Streamlit page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Document RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Document RAG Assistant")
st.caption(
    "Upload documents, build a vector database, ask questions, "
    "and get answers grounded in the uploaded documents."
)


# ---------------------------------------------------------
# 3. Session state
# ---------------------------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = []


# ---------------------------------------------------------
# 4. Helper: load one uploaded file
# ---------------------------------------------------------
def load_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name

    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(temp_path)
            documents = loader.load()

        elif suffix == ".docx":
            loader = Docx2txtLoader(temp_path)
            documents = loader.load()

        elif suffix in [".txt", ".md"]:
            loader = TextLoader(
                temp_path,
                encoding="utf-8",
            )
            documents = loader.load()

        else:
            raise ValueError(
                f"Unsupported file type: {suffix}"
            )

        # Add original filename to metadata.
        for doc in documents:
            doc.metadata["source"] = uploaded_file.name

        return documents

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


# ---------------------------------------------------------
# 5. Helper: build RAG vector store
# ---------------------------------------------------------
def build_vectorstore(uploaded_files, chunk_size, chunk_overlap):
    all_documents = []

    for uploaded_file in uploaded_files:
        docs = load_uploaded_file(uploaded_file)
        all_documents.extend(docs)

    if not all_documents:
        raise ValueError("No readable documents were found.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(
        all_documents
    )

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    # A unique collection avoids collisions between uploads.
    collection_name = (
        "rag_upload_" + uuid.uuid4().hex
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
    )

    return vectorstore, len(all_documents), len(chunks)


# ---------------------------------------------------------
# 6. Helper: ask the RAG system
# ---------------------------------------------------------
def answer_with_rag(question, retriever):
    # A. RETRIEVAL
    retrieved_docs = retriever.invoke(question)

    # B. BUILD CONTEXT FROM RETRIEVED RESULTS
    context_parts = []

    for i, doc in enumerate(
        retrieved_docs,
        start=1,
    ):
        source = doc.metadata.get(
            "source",
            "Unknown",
        )
        page = doc.metadata.get(
            "page",
            None,
        )

        page_text = (
            f", page {page + 1}"
            if isinstance(page, int)
            else ""
        )

        context_parts.append(
            f"""[Retrieved Chunk {i}]
Source: {source}{page_text}

{doc.page_content}
"""
        )

    context = "\n\n".join(
        context_parts
    )

    # C. LANGCHAIN_OPENAI CHAT MODEL
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a document question-answering assistant.

Answer the user's question using ONLY the retrieved context.

Rules:
1. Do not invent facts.
2. If the answer is not present in the context, say:
   "The answer is not available in the uploaded documents."
3. Keep the answer clear and concise.
4. When useful, mention the source filename.

Retrieved context:
{context}
""",
            ),
            (
                "human",
                "{question}",
            ),
        ]
    )

    messages = prompt.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    # D. FINAL GENERATION
    response = llm.invoke(messages)

    return response.content, retrieved_docs, context


# ---------------------------------------------------------
# 7. Sidebar: upload and indexing
# ---------------------------------------------------------
with st.sidebar:
    st.header("1. Upload documents")

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT or MD files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    st.header("2. Chunk settings")

    chunk_size = st.slider(
        "Chunk size",
        min_value=200,
        max_value=2000,
        value=700,
        step=100,
    )

    chunk_overlap = st.slider(
        "Chunk overlap",
        min_value=0,
        max_value=500,
        value=100,
        step=25,
    )

    top_k = st.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=10,
        value=4,
        step=1,
    )

    if st.button(
        "Build RAG Knowledge Base",
        type="primary",
        use_container_width=True,
    ):
        if not uploaded_files:
            st.warning(
                "Please upload at least one document."
            )
        else:
            try:
                with st.spinner(
                    "Reading, chunking and embedding documents..."
                ):
                    vectorstore, doc_count, chunk_count = build_vectorstore(
                        uploaded_files,
                        chunk_size,
                        chunk_overlap,
                    )

                    st.session_state.vectorstore = vectorstore
                    st.session_state.retriever = (
                        vectorstore.as_retriever(
                            search_kwargs={
                                "k": top_k
                            }
                        )
                    )

                    st.session_state.uploaded_names = [
                        f.name for f in uploaded_files
                    ]

                    st.session_state.messages = []

                st.success(
                    f"Knowledge base created. "
                    f"{doc_count} document sections → "
                    f"{chunk_count} chunks."
                )

            except Exception as exc:
                st.exception(exc)

    if st.session_state.uploaded_names:
        st.subheader("Indexed files")
        for name in st.session_state.uploaded_names:
            st.write("•", name)


# ---------------------------------------------------------
# 8. Explain the RAG flow
# ---------------------------------------------------------
with st.expander("How this application works"):
    st.markdown(
        """
```text
Uploaded Documents
        ↓
Document Loaders
        ↓
Text Chunking
        ↓
OpenAIEmbeddings
(text-embedding-3-small)
        ↓
Chroma Vector Database
        ↓
User Question
        ↓
Retriever.invoke(question)
        ↓
Top Relevant Chunks
        ↓
Retrieved Chunks + Question
        ↓
ChatOpenAI
        ↓
Final Grounded Answer
```

The embedding model is used for **retrieval**.

`ChatOpenAI` is used for **final answer generation**.
"""
    )


# ---------------------------------------------------------
# 9. Show previous chat messages
# ---------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):
            with st.expander(
                "Retrieved source chunks"
            ):
                for i, source in enumerate(
                    message["sources"],
                    start=1,
                ):
                    st.markdown(
                        f"**Chunk {i}** — "
                        f"{source['source']}"
                    )

                    if source.get(
                        "page"
                    ) is not None:
                        st.caption(
                            f"Page: "
                            f"{source['page']}"
                        )

                    st.write(
                        source["content"]
                    )


# ---------------------------------------------------------
# 10. User question
# ---------------------------------------------------------
question = st.chat_input(
    "Ask a question about the uploaded documents..."
)

if question:
    if st.session_state.retriever is None:
        st.warning(
            "Upload documents and build the RAG knowledge base first."
        )
        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(
            "Retrieving relevant chunks and generating answer..."
        ):
            answer, retrieved_docs, context = answer_with_rag(
                question,
                st.session_state.retriever,
            )

        st.markdown(answer)

        source_rows = []

        with st.expander(
            "Retrieved source chunks"
        ):
            for i, doc in enumerate(
                retrieved_docs,
                start=1,
            ):
                source = doc.metadata.get(
                    "source",
                    "Unknown",
                )
                page = doc.metadata.get(
                    "page",
                    None,
                )

                display_page = (
                    page + 1
                    if isinstance(page, int)
                    else None
                )

                st.markdown(
                    f"**Chunk {i} — {source}**"
                )

                if display_page is not None:
                    st.caption(
                        f"Page: {display_page}"
                    )

                st.write(
                    doc.page_content
                )

                source_rows.append(
                    {
                        "source": source,
                        "page": display_page,
                        "content": doc.page_content,
                    }
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": source_rows,
        }
    )
