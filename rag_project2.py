from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

llm = Ollama(model="mistral", temperature=0.3)


path="/Users/burakmemis/Downloads/ggvsee-englisch.pdf"
def pdf_loader(path):
    return PyMuPDFLoader(path).load()


def build_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "Section ", "(1)", "(2)"]
    )
    return splitter.split_documents(docs)

def build_vectordb(chunks, embedding, persist_dir):
    vectordb = Chroma.from_documents(documents=chunks,embedding=embeddings,persist_directory=persist_dir)
    return vectordb

system_message ="""
You are a legal and compliance assistant.
- The user may ask questions in Turkish.
- Source documents are in English.
- ALWAYS answer in Turkish.
- Use ONLY the provided context.
- Cite section numbers exactly.
- If the answer is not explicitly stated in the context, say:
"Bu soru verilen dokümanda açıkça tanımlanmamıştır."
""".strip()
# 1. Load
docs = pdf_loader(path)

# 2. Split (ileride section-aware ile değiştir)
chunks = build_chunks(docs)

# 3. Embedding
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Vector DB
vectordb = build_vectordb(
    chunks=chunks,
    embedding=embeddings,
    persist_dir="./chroma_ggvsee"
)

retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# 5. Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("human", "{question}")
])

# 6. RAG Chain
chain = (
    {
        "context": lambda x: retriever.invoke(x["question"]),
        "question": lambda x: x["question"]
    }
    | prompt
    | llm
)

while True:
    user_input = input("\n🧑 Soru: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    response = chain.invoke({"question": user_input})
    print("\n🤖 Cevap:\n", response)


    


