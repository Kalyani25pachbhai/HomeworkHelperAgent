import os
from flask import Flask, render_template, request, session
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

app = Flask(__name__)

app.secret_key = os.environ.get("FLASK_SECRET_KEY", "a_strong_secret_key_for_demo") 

PDF_PATH = "/tmp/notes.pdf"
DB_PATH = "/tmp/faiss_db"

os.environ["GOOGLE_API_KEY"] = "AIzaSyDJx6Qkp6xa7r1I-3Fb5w68RyqBadL74_s" 

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

embeddings_model = GoogleGenerativeAIEmbeddings(model="text-embedding-004")

wiki = WikipediaAPIWrapper()
tools = [
    Tool(name="Wikipedia", func=wiki.run, description="Useful for factual homework queries."),
    PythonREPLTool()
]

agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=False,
    handle_parsing_errors=True 
)

def load_notes(pdf_file_path):
    """Loads PDF, chunks it, and creates/saves a FAISS vector store."""
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    global embeddings_model
    db = FAISS.from_documents(chunks, embeddings_model)
    db.save_local(DB_PATH)
    return db.as_retriever()

def get_retriever():
    """Checks if the FAISS DB exists and loads it, otherwise returns None."""
    if os.path.exists(DB_PATH):
        print("Loading existing FAISS database.")
        global embeddings_model
        db = FAISS.load_local(DB_PATH, embeddings_model, allow_dangerous_deserialization=True)
        return db.as_retriever()
    return None

def homework_helper(query, retriever=None):
    """Answers the question, using the retriever (notes) if provided, or the agent (tools) otherwise."""
    prompt = f"""
    You are a Homework Helper. Explain the answer to the following question in a systematic, easy-to-read format. 
    Use a clear markdown structure, including bolding, headings, and bullet points.
    
    Question: {query}
    """
    
    if retriever:
        try:
            qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
            return qa.run(query)
        except Exception as e:
            print(f"Retrieval QA failed: {e}. Falling back to general agent.")
            return agent.run(prompt)
    else:
        return agent.run(prompt)

@app.route("/", methods=["GET"])
def index():
    uploaded_filename = session.get('uploaded_filename')
    return render_template("index.html", uploaded_filename=uploaded_filename)

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question")
    pdf_file = request.files.get("notes")
    uploaded_filename = session.get('uploaded_filename') 
    retriever = None

    if pdf_file and pdf_file.filename != '':
        try:
            pdf_file.save(PDF_PATH)
            retriever = load_notes(PDF_PATH)
            
            uploaded_filename = pdf_file.filename
            session['uploaded_filename'] = uploaded_filename
            
            
        except Exception as e:
            print(f"Error processing PDF: {e}")
            session.pop('uploaded_filename', None)
            
            answer = homework_helper(question)
            return render_template("index.html", 
                                   question=question, 
                                   answer=f"⚠️ Error processing your notes. The system encountered an error: '{e}'. Answering using public knowledge instead.",
                                   uploaded_filename=None)
    
    else:
        retriever = get_retriever()
        
    answer = homework_helper(question, retriever)
    
    return render_template("index.html", 
                           question=question, 
                           answer=answer,
                           uploaded_filename=uploaded_filename)

if __name__ == "__main__":
    app.run(debug=True)