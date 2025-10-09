**📘 Smart Homework Helper**

An AI-powered homework assistant that helps students get step-by-step explanations for any question.
It can also read and understand your uploaded PDF notes to give context-based answers — just like a personalized tutor!

**🚀 Overview**

The Smart Homework Helper uses the power of Google Gemini (Generative AI) and LangChain to make studying easier.
You can ask any question — from math problems to science concepts — and even upload your class notes in PDF format.
The system will extract information from your notes, search Wikipedia if needed, and give you a clear, easy-to-understand explanation.

🎯 **Features**

📚 PDF Upload Support – Upload your handwritten or class notes in PDF format.

🧠 Context-Aware Answers – The model learns from your uploaded notes.

🌐 Wikipedia Integration – Automatically fetches factual information from Wikipedia.

💬 Step-by-Step Explanations – Generates simple and detailed answers for better understanding.

🧮 Python Tool Access – Allows the agent to run simple calculations or Python code internally.

⚡ Real-Time Response – Instant replies through the integrated ChatGoogleGenerativeAI model.

**🏗️ Tech Stack**

Frontend: Streamlit (or HTML & CSS if you later use Flask UI)

Backend: Python

Framework: LangChain

AI Model: Google Gemini (via langchain_google_genai)

Embeddings: HuggingFace and Google Generative AI Embeddings

Vector Store: FAISS for semantic search and retrieval

Document Loader: PyPDFLoader for reading PDF files

Text Splitter: RecursiveCharacterTextSplitter for breaking large documents into smaller chunks

**⚙️ How It Works**

Upload Notes

The user uploads a PDF file of their notes.

The file is processed using PyPDFLoader.

Text Chunking & Embedding

The text is split into smaller chunks.

Each chunk is converted into a numeric embedding using HuggingFace embeddings.

Vector Database (FAISS)

All chunks are stored in a FAISS vector database for fast retrieval.

Retriever-Augmented Generation (RAG)

When you ask a question, the retriever finds the most relevant chunks from your notes.

The Gemini model uses these chunks to generate an accurate, step-by-step answer.

Fallback to Wikipedia

If no notes are uploaded, the model fetches data directly from Wikipedia or runs simple Python code when needed.

**🔑 Environment Setup**

You must set your Google API Key to use the Gemini model.

In your terminal, run:
set GOOGLE_API_KEY="AIzaSyDJx6Qkp6xa7r1I-3Fb5w68RyqBadL74_s"      # Windows

Or directly inside the code (already included):

os.environ["GOOGLE_API_KEY"] = "AIzaSyDJx6Qkp6xa7r1I-3Fb5w68RyqBadL74_s"

**🧠 Learning Outcomes**

Learned how to integrate LLMs (Gemini) with LangChain.

Understood Retrieval-Augmented Generation (RAG) pipeline.

Implemented document processing using PyPDFLoader and FAISS.

Designed an interactive AI-powered assistant with real-time reasoning.

Explored embeddings, retrievers, and agent-based architectures in AI systems.

**🖼️ Output Preview**

1️⃣ Upload PDF Notes 
2️⃣ Ask Any Homework Question
3️⃣ Get Instant Step-by-Step Answers

**📜 License**

This project is open-source and free to use for educational purposes.
