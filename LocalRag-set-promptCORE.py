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
# nltk.download('punkt')

# Define the directory containing the text files
directory = "/Users/pisles/InfoWorldRAG/data"

# Initialize an empty list to store all documents
all_documents = []

# Loop through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".txt"):  # Check if the file is a text file
        file_path = os.path.join(directory, filename)
        
        # Load the document using a LangChain text loader
        loader = TextLoader(file_path)
        documents = loader.load()
        
        # Extend the list of all documents with the documents loaded from the current file
        all_documents.extend(documents)

        # Print the name of the file that has been loaded
        print(f"Loaded file: {filename}")


# Split the documents into sentences using NLTK
sentences = []
for doc in all_documents:
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

# Initialize the LLM
llm = Ollama(
    model="llama3",
    verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

# Define the query
question = input("Please enter your question: ")

# Print the query
#print("Query:", question)

# QA chain
from langchain.chains import RetrievalQA
from langchain import hub

# LangChain Hub is a repository of LangChain prompts shared by the community
QA_CHAIN_PROMPT = hub.pull("pisles/rag-prompt-llama-ps")


# Print the prompt
# print("Prompt:", QA_CHAIN_PROMPT)

# Create the QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    # We create a retriever to interact with the db using an augmented context
    retriever=db.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)

# Generate response
# Generate response
result = qa_chain.invoke({"query": question})

# Extract the actual result content
# Initialize the result string
cleaned_result = result.get('result', '')

# Remove multiple types of tags
tags_to_remove = ['[INST]', '[/INST]', '<<SYS>>', '<< /SYS >>']
for tag in tags_to_remove:
    cleaned_result = cleaned_result.replace(tag, '')

# Strip any leading or trailing whitespace
cleaned_result = cleaned_result.strip()

# Print the results of the LLM query
#print("Results:")
#print(cleaned_result)