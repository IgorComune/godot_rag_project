# Godot RAG Project

![Open Source](https://img.shields.io/badge/Open%20Source-Yes-brightgreen)

## Background
A few months ago, I got into the world of Reinforcement Learning and somehow ended up experimenting with **Godot**.  
Then I had a wild idea:  

**Create an AI Agent that supports game development in Godot.**  

Since Godot is open source, I decided the entire project should be open source too.  

---

## Setup

### Requirements
- Python 3.13  
- [Dolphin3.0-Llama3.1–8B-Q3_K.gguf](#)  
- [text-generation-webui](https://github.com/oobabooga/text-generation-webui)  
- [LangChain](https://www.langchain.com/)  
- PyTorch  
- Chroma DB  

### Installation
```bash
# Clone the repository
git clone https://github.com/IgorComune/godot_rag_project.git

# Create virtual environment (Python 3.13 recommended)
# Example using Anaconda:
conda create -n godot_rag python=3.13
conda activate godot_rag

# Install main dependencies
pip install -r requirements.txt

# Install portable dependencies for WebUI
pip install -r requirements/portable/requirements.txt --upgrade
How It Works
Scrape all links from the Godot Documentation.

Download all content from the scraped links.

Vectorize content into a ChromaDB.

Test ChromaDB queries.

Download WebUI, load the LLM model.

Integrate everything together.

Optional: If you don’t want to scrape the docs, you can just unzip the pre-scraped file godot_docs/pages.zip.

Running the Project
bash
Copy code
# Step 1: Scrape links
python godot_scraper_links.py

# Step 2: Scrape content
python godot_scraper_content.py

# Step 3: Vectorize content
python txt_to_vector.py

# Step 4: Run WebUI server with RAG extension
python server.py --extensions rag
RAG Extension
This extension allows your input text to query the ChromaDB and enrich the LLM prompt with context.

python
Copy code
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

PERSIST_DIRECTORY = "godot_chroma_db"
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_function)

def input_modifier(prompt, state=None, is_chat=False):
    try:
        docs = vectordb.similarity_search(prompt, k=3)
        context_text = "\n".join([doc.page_content for doc in docs])
        print("[RAG DEBUG] Contexto adicionado ao prompt:")
        print(context_text)
        return f"{context_text}\nPergunta: {prompt}\nResposta:"
    except Exception as e:
        print(f"[RAG Extension] Erro ao consultar o Chroma DB: {e}")
        return prompt

params = {
    "display_name": "RAG Extension",
    "is_tab": False,
}

def setup():
    return {
        "input_modifier": input_modifier,
    }
Flow:
Your input text → query ChromaDB → ChromaDB answer → send to LLM → LLM answers you

Notes
I’m moving to Unity to explore Reinforcement Learning, so this project may be abandoned if no one takes over.

The project has a lot of problems: mixture of Portuguese and English, useless files, lack of coding standards…

Feel free to fork, clean it up, or continue developing it ❤️.

License
This project is fully open-source. Do whatever you want with it.

Repository
https://github.com/IgorComune/godot_rag_project

css
Copy code
