import os
import json
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import requests
import datetime
from langchain_chroma.vectorstores import Chroma
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
embedding = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
splitter = RecursiveCharacterTextSplitter(chunk_overlap=100, chunk_size=1000)

import glob
def get_most_recent_file(base_path):
    files = [f for f in glob.glob(os.path.join(base_path, '*')) if os.path.isfile(f)]
    if not files:
        print(f"No files found in {base_path}")
        return None
    most_recent_file = max(files, key=os.path.getmtime)
    print(f"Most recent file: {most_recent_file}")
    return most_recent_file


def convert_db(urls ,persist_directory =None):
    print("urls=  == = =",urls)
    texts = []
    with open(get_most_recent_file("text")) as f:
        history = json.load(f)
    for article in history["Articles"]:
        if article["id"] in urls:
            print(article["id"])
            texts.append(Document(page_content=article['brief'] + article["content"] , metadata={"source": article["urls"] , "heading": article["title"] }))
    
    # for url in urls:
    #     print("done")
    #     response = requests.get(url)
    #     soup = BeautifulSoup(response.content , 'html.parser')
    #     haeding = soup.find("h1").get_text()
    #     image = soup.find("img").get("src")
    #     author = soup.find("a" , class_ = "author")
    #     if not author:
    #         author = "Unknown"
    #     else:
    #         author = author.get_text()
    #     main_content = soup.find('main') or soup.find('article') or soup.find('div' , class_ = "content")
    #     chunks = splitter.split_text(main_content.get_text(strip=True))
    #     for chunk in chunks:
    #         chunk = Document(page_content=chunk, metadata={"source": url, "heading": haeding , "image": image , "author": author})
    #         texts.append(chunk)
    os.makedirs("db", exist_ok=True)
    if not persist_directory:
        persist_directory = f"db/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    print(texts , persist_directory)
    vector_db = Chroma.from_documents(texts, embedding, persist_directory=persist_directory)
    return persist_directory
