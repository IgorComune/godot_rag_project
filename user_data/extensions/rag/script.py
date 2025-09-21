from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Caminho relativo do seu Chroma DB
PERSIST_DIRECTORY = "godot_chroma_db"

# Inicializa embeddings
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Inicializa Chroma DB
vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_function)

# ----------------------------
# Função que modifica o input
# ----------------------------
def input_modifier(prompt, state=None, is_chat=False):
    try:
        docs = vectordb.similarity_search(prompt, k=3)
        context_text = "\n".join([doc.page_content for doc in docs])
        print("[RAG DEBUG] Contexto adicionado ao prompt:")
        print(context_text)
        return f"{context_text}\nPergunta: {prompt}\nResposta:"
    except Exception as e:
        print(f"[RAG Extension] Erro ao consultar o Chroma DB: {e}")
        return prompt  # fallback: prompt original

# ----------------------------
# Configura parâmetros da extensão
# ----------------------------
params = {
    "display_name": "RAG Extension",
    "is_tab": False,  # fica no final da aba Text generation
}

# ----------------------------
# Hooks do WebUI
# ----------------------------
def setup():
    """
    Retorna os hooks que o WebUI vai chamar.
    """
    return {
        "input_modifier": input_modifier,
    }
