"""
Fully local retrieval-augmented generation, step by step
https://www.infoworld.com/article/3715181/fully-local-retrieval-augmented-generation-step-by-step.html
Note: install sqslitvss extension with 'pip install sqlite-vss'
"""

# LocalRAG.py
# LangChain is a framework and toolkit for interacting with LLMs programmatically

# LangChain is a framework and toolkit for interacting with LLMs programmatically

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import SQLiteVSS
from langchain_community.document_loaders import TextLoader
from nltk.tokenize import sent_tokenize
import os
import nltk

# LLM
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Download NLTK data if needed (only have to do once)
#nltk.download('punkt')

# Load the document using a LangChain text loader
loader = TextLoader("/Users/pisles/InfoWorldRAG/stateoftheunion2023.txt")
documents = loader.load()

# Split the document into sentences using NLTK
sentences = []
for doc in documents:
    sentences.extend(sent_tokenize(doc.page_content))

# Use the sentence transformer package with the all-MiniLM-L6-v2 embedding model
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Delete the existing database file to start fresh
db_file_path = "/tmp/vss.db"
if os.path.exists(db_file_path):
    os.remove(db_file_path)

# Now initialize SQLiteVSS
db = SQLiteVSS.from_texts(
    texts=sentences,
    embedding=embedding_function,
    table="state_union",
    db_file=db_file_path
)

# First, we will do a simple retrieval using similarity search
# Query
#question = "What was said about Yale"
#data = db.similarity_search(question)

# print results
#print(data[0].page_content)


llm = Ollama(
    model = "llama3",
    verbose = True,
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]),
)

# QA chain
from langchain.chains import RetrievalQA
from langchain import hub

# LangChain Hub is a repository of LangChain prompts shared by the community
QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")

# Print the prompt
#print("Prompt:", QA_CHAIN_PROMPT)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    # we create a retriever to interact with the db using an augmented context
    retriever=db.as_retriever(), 
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)

# Define the query
question = "What was said about Nancy Pelosi?"

# Print the query
print("Query:", question)

# Generate response
result = qa_chain.invoke({"query": question})

# Print the available attributes of the retriever object
#print("Context provided to the language model:")
#print(qa_chain.retriever.search_kwargs.get('context'))


# Print the results of the LLM query
print("Results:")
print(result)



