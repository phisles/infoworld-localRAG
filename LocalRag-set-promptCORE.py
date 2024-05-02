# Import necessary libraries
import os
import nltk
import spacy
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import SQLiteVSS
from langchain_community.document_loaders import TextLoader
from nltk.tokenize import sent_tokenize
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain import hub

# Download NLTK data if needed
# nltk.download('punkt')

# Load the SpaCy model for NLP
nlp = spacy.load("en_core_web_sm")

# Define the directory containing the text files
directory = "/Users/pisles/InfoWorldRAG/data"
all_documents = []

# Load documents
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory, filename)
        loader = TextLoader(file_path)
        documents = loader.load()
        all_documents.extend(documents)
        print(f"Loaded file: {filename}")

# Tokenize documents into sentences
sentences = [sent_tokenize(doc.page_content) for doc in all_documents]
flat_sentences = [sentence for sublist in sentences for sentence in sublist]

# Embed sentences
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = [embedding_function.encode(sentence) for sentence in flat_sentences]



# Setup SQLite database for embeddings
db_file_path = "/tmp/vss.db"
if os.path.exists(db_file_path):
    os.remove(db_file_path)
db = SQLiteVSS.from_texts(texts=flat_sentences, embedding=embedding_function, table="state_union", db_file=db_file_path)

# Initialize the language model with callbacks
llm = Ollama(model="llama3", verbose=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

# Get user query and process it
question = input("Please enter your question: ")
doc = nlp(question)
entities = [ent.text for ent in doc.ents]
keywords = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

# Use the LangChain hub to get the QA prompt
QA_CHAIN_PROMPT = hub.pull("pisles/rag-prompt-llama-ps")

# Define the QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=db.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)

# Generate a response
result = qa_chain.invoke({"query": " ".join(entities + keywords)})
cleaned_result = result.get('result', '').replace('[INST]', '').replace('[/INST]', '').strip()

# Print the results
print(cleaned_result)
