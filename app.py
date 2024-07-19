import os
import requests
import dataset
import gradio as gr
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urldefrag
from joblib import Parallel, delayed
from datetime import datetime
from llama_index.core import PromptTemplate
from llama_index.core import VectorStoreIndex
from llama_index.llms.gemini import Gemini
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from llama_index.embeddings.gemini import GeminiEmbedding
from pinecone import Pinecone, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.indices import VectorStoreIndex
from llama_index.core import StorageContext



os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')
os.environ["PINECONE_API_KEY"] = os.getenv('PINECONE_API_KEY')


documents = []
    
def get_headings_and_text(html_content, url, page_title):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all the h2 and h3 tags
    headings = soup.find_all(['h2', 'h3'])
    
    content_dict = {}
    
    for heading in headings:
        # Get the text of the heading
        heading_text = heading.get_text(strip=True)
        
        # Find the next sibling that contains text content
        next_sibling = heading.find_next_sibling()
        
        while next_sibling and next_sibling.name not in ['p', 'div']:
            next_sibling = next_sibling.find_next_sibling()
        
        if next_sibling:
            # Get the text of the next sibling
            content_text = next_sibling.get_text(separator=' ', strip=True)
            content_text = content_text.replace(u'\xa0', u' ')
            documents.append(Document(text='## '+ heading_text + " # " + content_text, metadata= {"source": url, "date":datetime.now().strftime("%Y-%m-%d"),"title": page_title}))
    
def store_page(url, title):
    print('Visited page:', url)
    print('       title:', title)
    db['pages'].upsert({'url': url, 'title': title}, ['url'])

def store_links(from_url, links):
    db.begin()
    for to_url in links:
        db['links'].upsert({'from_url': from_url, 'to_url': to_url}, ['from_url', 'to_url'])
    db.commit()

def get_random_unvisited_pages(amount=10):
    result = db.query('''SELECT * FROM links
        WHERE to_url NOT IN (SELECT url FROM pages)
        ORDER BY RANDOM() LIMIT {}'''.format(amount))
    return [r['to_url'] for r in result]

def should_visit(base_url, url):
    if url is None:
        return None
    full_url = urljoin(base_url, url)
    full_url = urldefrag(full_url)[0]
    if not full_url.startswith(base_url):
        # This is an external URL
        return None
    ignore = []
    if any([i in full_url for i in ignore]):
        # This is a page to be ignored
        return None
    return full_url

def get_title_and_links_and_body(base_url, url):
    html = requests.get(url).text
    html_soup = BeautifulSoup(html, 'html.parser')
    page_title = html_soup.find(id='page-title')
    page_title = page_title.text if page_title else ''
    page_body = html
    
    links = []
    for link in html_soup.find_all("a"):
        link_url = should_visit(base_url, link.get('href'))
        if link_url:
            links.append(link_url)
    return url, page_title, links, page_body

def call_webscraper(base_url):
    urls_to_visit = [base_url]
    i = 0
    while urls_to_visit and i < 25:
            scraped_results = Parallel(n_jobs=5, backend="threading")(
                delayed(get_title_and_links_and_body)(base_url, url) for url in urls_to_visit
            )
            for url, page_title, links, page_body in scraped_results:
                store_page(url, page_title)
                store_links(url, links)
                get_headings_and_text(page_body, url, page_title)
            urls_to_visit = get_random_unvisited_pages()
            i = i + 1

    node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=32)
    return node_parser.get_nodes_from_documents(documents)


db = dataset.connect('sqlite:///wikipedia.db')
base_url = 'https://www.csuchico.edu/'

nodes = call_webscraper(base_url)


# print("Number of Documents:", len(documents))
# print("Number of Nodes:", len(nodes))
# print(nodes[0])

## Embeddings

embed_model = GeminiEmbedding(
    model_name="models/text-embedding-004",
    embed_batch_size=16
)

## Pinecone Vector Store

pinecone = Pinecone()
pinecone

INDEX_NAME = "rag"

if INDEX_NAME in pinecone.list_indexes().names():
  pinecone.delete_index(INDEX_NAME)

pinecone.create_index(
    name=INDEX_NAME,
    dimension=768,
    metric="dotproduct",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

pinecone_index = pinecone.Index(INDEX_NAME)

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index_loaded = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model
)

index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
    embed_model=embed_model
)


## Query Engine

#BLOCK_ONLY_HIGH
safety_settings={
  HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
  HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
  HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
  HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

llm = Gemini(
    model_name="models/gemini-pro",
    temperature=0,
    max_tokens=256,
    safety_settings=safety_settings
)

query_engine = index_loaded.as_chat_engine(
    llm=llm,
    similarity_top_k=6,
)

def query_bot(query):
    global conversation_summary
    response = query_engine.chat(query)
    return response

response = query_bot("What is the University name?")
response.response

response = query_bot("Why choose chico?")
response.response

## Gradio Demo

# Generates response using the question answering chain defined earlier
def generate(query):

  response_str = ""
  response = query_engine.query(query)
  return response.response
  # for token in response.response:
  #   response_str += token
  #   yield response_str

with gr.Blocks() as demo:
  gr.Markdown("""
  # Retrieval Augmented Generation with Gemini Pro and Pinecone: WildCatBot
  ### This demo uses the Gemini Pro LLM and Pinecone Vector Search for fast and performant Retrieval Augmented Generation (RAG).
  ### This bot is for answering questions related to California State University, Chico.
  """)

  gr.Markdown("## Enter your question")
  with gr.Row():
    with gr.Column():
      ques = gr.Textbox(label="Question", placeholder="Enter text here", lines=2)
    with gr.Column():
      ans = gr.Textbox(label="Answer", lines=4, interactive=False)
  with gr.Row():
    with gr.Column():
      btn = gr.Button("Submit")
    with gr.Column():
      clear = gr.ClearButton([ques, ans])

  btn.click(fn=generate, inputs=[ques], outputs=[ans])
  examples = gr.Examples(
        examples=[
            "Why choose Chico?",
        ],
        inputs=[ques],
    )

demo.queue().launch(debug=True,share=True)
