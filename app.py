from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
from langchain_groq import ChatGroq

app = Flask(__name__)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
HUGGINGFACE_API_KEY = os.environ.get('HUGGINGFACE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

# Set environment variables if needed
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["HUGGINGFACE_API_KEY"] = HUGGINGFACE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize embeddings
embeddings = download_hugging_face_embeddings()

index_name = "medbot"

# Connect to existing Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize HuggingFace LLM endpoint
llm = ChatGroq(
    api_key=os.environ["GROQ_API_KEY"],
    model_name="gemma2-9b-it"  # Or another Groq-supported model
)

# Prepare prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create document combining chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Create retrieval augmented generation chain
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg", "")
    if not msg:
        return "No message received", 400

    print(f"User input: {msg}")

    response = rag_chain.invoke({"input": msg})
    answer = response.get("answer", "Sorry, I could not generate a response.")

    print(f"Response: {answer}")

    return answer


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
