import os
import shutil
from flask import Flask, render_template, request, session
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import Document

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
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100) 
    chunks = splitter.split_documents(docs)
    
    global embeddings_model
    db = FAISS.from_documents(chunks, embeddings_model)
    
    if not os.path.exists(DB_PATH):
        os.makedirs(DB_PATH, exist_ok=True)
        
    db.save_local(DB_PATH)
    
    return db.as_retriever(search_kwargs={"k": 6})

def get_retriever():
    if os.path.exists(DB_PATH) and os.path.isdir(DB_PATH):
        print(f"Attempting to load existing FAISS database from {DB_PATH}.")
        global embeddings_model
        try:
            db = FAISS.load_local(
                DB_PATH, 
                embeddings_model, 
                allow_dangerous_deserialization=True
            )
            print("FAISS database loaded successfully.")
            
            return db.as_retriever(search_kwargs={"k": 6})
        
        except Exception as e:
            print(f"CRITICAL ERROR loading FAISS DB from {DB_PATH}: {e}. Deleting corrupted data.")
            try:
                shutil.rmtree(DB_PATH) 
            except Exception as clean_e:
                print(f"Error during cleanup: {clean_e}")
            return None 
    return None

def homework_helper(query, retriever=None):
    general_agent_prompt = f"""
    You are a Homework Helper. Explain the answer to the following question in a systematic, easy-to-read format. 
    Use a clear markdown structure, including bolding, headings, and bullet points.
    
    Question: {query}
    """

    if retriever:
        try:
            qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
            
            strict_rag_query = (
                f"Answer the following question based ONLY on the provided context. "
                f"Be hyper-specific: if the answer is a name, number, or date, provide the literal text found in the notes. "
                f"If the answer is a list, provide the complete list. If not found, say 'Not found in notes'. "
                f"Question: {query}"
            )
            
            return qa.run(strict_rag_query) 
            
        except Exception as e:
            print(f"Retrieval QA failed: {e}. Falling back to general agent.")
            return agent.run(general_agent_prompt)
            
    else:
        print("No retriever available. Using general agent for public knowledge search.")
        return agent.run(general_agent_prompt)

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
            os.makedirs(os.path.dirname(PDF_PATH), exist_ok=True) 

            pdf_file.save(PDF_PATH)
            retriever = load_notes(PDF_PATH)
            
            uploaded_filename = pdf_file.filename
            session['uploaded_filename'] = uploaded_filename
            
        except Exception as e:
            print(f"Error processing PDF: {e}")
            session.pop('uploaded_filename', None)
            
            # Removed uploaded_filename argument here
            answer = homework_helper(question)
            return render_template("index.html", 
                                   question=question, 
                                   answer=f"⚠️ Error processing your notes. The system encountered an error: '{e}'. Answering using public knowledge instead.",
                                   uploaded_filename=None)
    
    else:
        retriever = get_retriever()
        
    # Removed uploaded_filename argument here
    answer = homework_helper(question, retriever)
    
    return render_template("index.html", 
                           question=question, 
                           answer=answer,
                           uploaded_filename=uploaded_filename)

if __name__ == "__main__":
    app.run(debug=True)