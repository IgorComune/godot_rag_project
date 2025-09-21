import os
import shutil  # Added for cleanup in example usage
import sys  # To exit gracefully if dependencies are missing

from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --- Dependency Check ---
try:
    import torch
    from chromadb.config import Settings
    import langchain_chroma
    # Disable ChromaDB telemetry to avoid errors
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
except ImportError:
    print("Error: A required dependency (PyTorch, ChromaDB, or langchain-chroma) is not installed.")
    print("Please ensure all are installed. For PyTorch, follow instructions at https://pytorch.org/get-started/locally/")
    print("For others, run: pip install chromadb-client langchain-huggingface langchain-chroma")
    sys.exit(1)


class TextToVector:
    """
    A class to load text documents from a directory, split them into chunks,
    and vectorize them using Chroma DB with a multilingual Sentence Transformer model.
    """

    def __init__(
        self,
        docs_path: str = "godot_docs/pages",
        persist_directory: str = "godot_chroma_db",
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ):
        """
        Initializes the TextToVector class.

        Args:
            docs_path (str): The path to the directory containing the .txt documents.
            persist_directory (str): The directory where the Chroma DB will be persisted.
            model_name (str): The name of the Sentence Transformer model to use for
                              embeddings. 'paraphrase-multilingual-MiniLM-L12-v2' is a
                              good balance of performance and size for multilingual tasks.
        """
        self.docs_path = docs_path
        self.persist_directory = persist_directory
        self.model_name = model_name
        self.embedding_function = self._initialize_embedding_function()
        
        # Enhanced telemetry disabling
        self.chroma_settings = Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=True
        )
        self.vector_store = None

    def _initialize_embedding_function(self):
        """
        Initializes and returns the embedding function using SentenceTransformerEmbeddings.
        Note: The first time this runs, the model will be downloaded automatically
        from the Hugging Face Hub and cached locally.
        """
        # Determine the device to use (GPU if available, otherwise CPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing embedding model on device: '{device}'")

        # Check if the user has a less powerful GPU and might want to force CPU
        if not torch.cuda.is_available():
            print("Warning: CUDA not available. Falling back to CPU. "
                  "Processing will be significantly slower.")

        try:
            return HuggingFaceEmbeddings(
                model_name=self.model_name, 
                model_kwargs={"device": device},
                show_progress=True
            )
        except Exception as exc:
            print(f"Error initializing embedding model '{self.model_name}': {exc}")
            raise

    def _load_documents(self) -> List[Document]:
        """
        Loads all .txt files from the specified directory using DirectoryLoader.

        Returns:
            List[Document]: A list of loaded documents.
        """
        if not os.path.isdir(self.docs_path):
            print(
                f"Error: Document path '{self.docs_path}' does not exist or "
                "is not a directory."
            )
            return []

        print(f"Loading documents from '{self.docs_path}'...")
        # Use DirectoryLoader to recursively find and load all .txt files.
        # It will use TextLoader for each file, which is what was done manually before.
        loader = DirectoryLoader(
            self.docs_path,
            glob="**/*.txt",  # Recursively search for .txt files
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
            use_multithreading=True,
        )

        try:
            documents = loader.load()
            if not documents:
                print(f"No .txt files found or loaded from '{self.docs_path}'.")
                return []
            print(f"Found and loaded {len(documents)} documents.")
            return documents
        except Exception as e:
            print(f"An error occurred while loading documents: {e}")
            return []

    def process_and_vectorize(self):
        """
        Orchestrates the document loading, splitting, and vectorization process.
        Persists the vectorized data to Chroma DB.
        """
        all_documents = self._load_documents()
        if not all_documents:
            print("No documents were loaded. Aborting vectorization.")
            return

        # DirectoryLoader automatically adds 'source' metadata with the file path.
        print(f"Total documents loaded: {len(all_documents)}")

        # Split documents into smaller chunks for better embedding and retrieval.
        # Adjust chunk_size and chunk_overlap based on your specific document
        # characteristics.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, 
            chunk_overlap=200, 
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        chunks = text_splitter.split_documents(all_documents)
        print(f"Split documents into {len(chunks)} chunks.")

        if not chunks:
            print("No chunks were generated from the loaded documents. "
                  "Vectorization aborted.")
            return

        print(f"Starting vectorization and persisting to '{self.persist_directory}'...")
        
        # Suppress telemetry warnings during vectorization
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*telemetry.*")
            try:
                self.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embedding_function,
                    persist_directory=self.persist_directory,
                    client_settings=self.chroma_settings,
                )
                print("\nVectorization complete and Chroma DB persisted successfully!")
                print(f"Created vector store with {len(chunks)} chunks.")
            except Exception as exc:
                print(f"Error during vectorization or persisting Chroma DB: {exc}")
                # Print more detailed error information
                import traceback
                traceback.print_exc()

    def get_vector_store(self):
        """
        Returns the initialized Chroma vector store.

        If the vector store hasn't been created, it will attempt to load it from the
        persist directory or return None if it doesn't exist.
        """
        if self.vector_store:
            return self.vector_store
        else:
            try:
                # Attempt to load an existing Chroma DB if it hasn't been created
                # in this session.
                print(f"Attempting to load Chroma DB from '{self.persist_directory}'...")
                
                # Suppress telemetry warnings during loading
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*telemetry.*")
                    self.vector_store = Chroma(
                        persist_directory=self.persist_directory,
                        embedding_function=self.embedding_function,
                        client_settings=self.chroma_settings,
                    )
                
                print("Chroma DB loaded successfully.")
                return self.vector_store
            except Exception as exc:
                print(f"Could not load Chroma DB from '{self.persist_directory}': {exc}")
                print("You might need to run process_and_vectorize() first.")
                return None


def run_vectorization_pipeline():
    """
    Main function to orchestrate the text vectorization process.
    It checks for source files, configures paths, and runs the vectorizer.
    """
    # Disable telemetry at the environment level
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    os.environ["CHROMA_TELEMETRY"] = "False"
    
    # A quick check to ensure the main documents directory exists.
    if not os.path.isdir("godot_docs/pages"):
        print("Error: The source directory 'godot_docs/pages' was not found.")
        print(
            "Please run the 'godot_scraper_content.py' or 'godot_scraper_links.py' "
            "script first to generate the text files."
        )
        sys.exit(1)

    # A quick check to ensure the main documents directory is not empty.
    if not os.listdir("godot_docs/pages"):
        print("Warning: The source directory 'godot_docs/pages' is empty.")
        print("There are no documents to process.")

    # --- Configuration ---
    DOCS_PATH = "godot_docs/pages"
    PERSIST_DIRECTORY = "godot_chroma_db"
    MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

    # --- Cleanup ---
    # To ensure a clean build, we remove the old database directory.
    if os.path.isdir(PERSIST_DIRECTORY):
        print(f"Removing existing vector database at '{PERSIST_DIRECTORY}'...")
        shutil.rmtree(PERSIST_DIRECTORY)
        print("Existing database removed.")

    print("--- Starting Text to Vector Conversion ---")
    print(f"Source documents: '{DOCS_PATH}'")
    print(f"Vector DB path:   '{PERSIST_DIRECTORY}'")
    print(f"Embedding model:  '{MODEL_NAME}'")
    print("-" * 40)

    vectorizer = TextToVector(
        docs_path=DOCS_PATH,
        persist_directory=PERSIST_DIRECTORY,
        model_name=MODEL_NAME,
    )

    vectorizer.process_and_vectorize()

    # --- Example Query (Optional) ---
    # print("\n--- Testing the Vector Database with a Query ---")
    # vector_store = vectorizer.get_vector_store()
    # if vector_store:
    #     query = "How do I create a script in Godot?"
    #     print(f"Query: '{query}'")

    #     try:
    #         search_results = vector_store.similarity_search(query, k=3)

    #         if search_results:
    #             print("\n--- Top 3 Search Results ---")
    #             for i, doc in enumerate(search_results):
    #                 print(f"Result {i+1}:")
    #                 print(f"  Source: {doc.metadata.get('source', 'N/A')}")
    #                 print(f"  Content: {doc.page_content[:250].replace(chr(10), ' ')}...\n")
    #         else:
    #             print("No results found for the query.")
    #     except Exception as e:
    #         print(f"Error during similarity search: {e}")
    # else:
    #     print("Could not load the vector store to perform a query.")

    print("\n--- Process Finished ---")


if __name__ == "__main__":
    run_vectorization_pipeline()