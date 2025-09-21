import os
import sys
from typing import List

import torch
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --- Dependency Check ---
try:
    import torch
    import langchain_chroma
except ImportError:
    print("Error: A required dependency (PyTorch or langchain-chroma) is not installed.")
    print("Please ensure both are installed. For PyTorch, follow instructions at https://pytorch.org/get-started/locally/")
    print("For the other, run: pip install langchain-huggingface langchain-chroma")
    sys.exit(1)


class VectorDBQuery:
    """
    A class to load an existing Chroma DB and perform similarity searches.
    """

    def __init__(
        self,
        persist_directory: str = "godot_chroma_db",
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ):
        """
        Initializes the VectorDBQuery class.

        Args:
            persist_directory (str): The directory where the Chroma DB is stored.
            model_name (str): The name of the Sentence Transformer model to be used.
        """
        self.persist_directory = persist_directory
        self.model_name = model_name
        self.embedding_function = self._initialize_embedding_function()
        self.chroma_settings = Settings(anonymized_telemetry=False)
        self.vector_store = self._load_vector_store()

    def _initialize_embedding_function(self):
        """
        Initializes and returns the embedding function using SentenceTransformerEmbeddings.
        Uses the GPU (CUDA) if available.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing embedding model on device: '{device}'")

        if not torch.cuda.is_available():
            print("Warning: CUDA not available. Using CPU. "
                  "Processing will be significantly slower.")

        try:
            return HuggingFaceEmbeddings(
                model_name=self.model_name, model_kwargs={"device": device}
            )
        except Exception as exc:
            print(f"Error initializing embedding model '{self.model_name}': {exc}")
            raise

    def _load_vector_store(self) -> Chroma:
        """
        Loads an existing Chroma DB from the persistent directory.

        Returns:
            Chroma: The loaded vector store instance.
        """
        if not os.path.isdir(self.persist_directory):
            print(f"Error: Chroma DB directory '{self.persist_directory}' not found.")
            print("You need to run the 'txt_to_vector.py' script first to create the database.")
            sys.exit(1)

        print(f"Loading Chroma DB from '{self.persist_directory}'...")
        try:
            store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function,
                client_settings=self.chroma_settings,
            )
            print("Chroma DB loaded successfully.")
            return store
        except Exception as exc:
            print(f"Could not load Chroma DB from '{self.persist_directory}': {exc}")
            sys.exit(1)

    def query(self, query_text: str, k: int = 5) -> List[Document]:
        """
        Performs a similarity search in the vector store.

        Args:
            query_text (str): The query text.
            k (int): The number of results to return.

        Returns:
            List[Document]: A list of matching documents.
        """
        if not self.vector_store:
            print("Error: Vector store is not loaded.")
            return []

        print(f"\nPerforming similarity search for: '{query_text}'")
        results = self.vector_store.similarity_search(query_text, k=k)
        return results


def main_interactive_query():
    """
    Sets up the query engine and runs an interactive loop
    to accept user queries from the terminal.
    """
    # --- Configuration ---
    PERSIST_DIRECTORY = "godot_chroma_db"
    MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

    print("--- Query Vector Database ---")
    db_query = VectorDBQuery(
        persist_directory=PERSIST_DIRECTORY,
        model_name=MODEL_NAME,
    )

    while True:
        try:
            query = input("\nEnter your query (or 'exit' to quit): ")
            if query.lower() in ["exit", "quit"]:
                break
            if not query:
                continue

            search_results = db_query.query(query, k=3)

            if search_results:
                print("\n--- Top 3 Search Results ---")
                for i, doc in enumerate(search_results):
                    print(f"Result {i+1}:")
                    source = (
                        doc.metadata.get("source", "N/A") if doc.metadata else "N/A"
                    )
                    print(f"  Source: {source}")
                    content_preview = doc.page_content.replace("\n", " ").strip()
                    print(f"  Content: {content_preview}\n")
            else:
                print("No results found for your query.")

        except KeyboardInterrupt:
            print("\nExiting...")
            break

    print("--- Process Finished ---")


if __name__ == "__main__":
    main_interactive_query()