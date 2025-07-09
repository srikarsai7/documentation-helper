from dotenv import load_dotenv
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

loader = ReadTheDocsLoader(
    "langchain-docs/api.python.langchain.com/en/latest/"
)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=50
)
docs = splitter.split_documents(docs)

for d in docs:
    d.metadata["source"] = d.metadata["source"].replace(
        "langchain-docs", "https:/"
    )

vstore = PineconeVectorStore(
    index_name="langchain-doc-index",
    embedding=embeddings
)

BATCH = 500

for i in range(0, len(docs), BATCH):
    batch_docs = docs[i : i + BATCH]
    texts      = [d.page_content for d in batch_docs]
    metadatas  = [d.metadata      for d in batch_docs]

    vstore.add_texts(
        texts,
        metadatas=metadatas,
        batch_size=BATCH,
        embedding_chunk_size=100
    )

print("✅ Ingestion complete!")
