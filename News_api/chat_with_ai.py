import os
from langchain_google_genai import  ChatGoogleGenerativeAI
from . import convert_db
import json
import datetime
from langchain_chroma.vectorstores import Chroma
from dotenv import load_dotenv

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings


# def chat_with_ai():
#     conversation_history = []
    
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == 'quit':
#             break
        
#         conversation_history.append(f"Human: {user_input}")
        
#         question_answer_chain = create_stuff_documents_chain(llm, prompt)
#         rag_chain = create_retrieval_chain(retriever, question_answer_chain)
#         response = rag_chain.invoke({"input": user_input})
        
#         ai_response = response["answer"]
#         print(f"AI: {ai_response}")
        
#         conversation_history.append(f"AI: {ai_response}")
#     return conversation_history


class Chat:
    instances = {}
    instances_urls = {}
    def __init__(self, urls = None):
        if urls is not None:
            load_dotenv()
            self.conversation_history = []
            os.makedirs("chats", exist_ok=True)
            if not os.path.exists("chats/history.json"):
                with open("chats/history.json", "w") as f:
                    json.dump({"history": []}, f)
            with open("chats/history.json", "r") as f:
                history = json.load(f)
            found = False
            for i in history["history"]:
                if i["urls"] == urls:
                    persistent_dir = i["storage"]
                    found = True
                    break
            if not found:
                persistent_dir = convert_db.convert_db(urls)
            
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            self.vdb = Chroma(persist_directory=persistent_dir, embedding_function=self.embeddings)
            self.retriever = self.vdb.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

            system_prompt = (
                "You are an friendly assistant for friendly chat tasks. "
                "Use the following pieces of retrieved context to answer"
                "gave me the detailed discription of the context withing 10 -13 lines"
                "you have given a question undersatnd the context and answer the question briefly."
                "remember that you can use your own knowledge to be creative "
                "if the context recived wont match then use your own knowledge to answer it but make it truthfull"
                "\n\n"
            )

            self.prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input} , context {context}"),
                ]
            )

            self.question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
            self.rag_chain = create_retrieval_chain(self.retriever, self.question_answer_chain)

    @classmethod
    def get_chat_instance(self, chat_id , urls=None):
        print("urls" , urls)
        print("chat_id" , chat_id)
        print("instances = ",self.instances)
        if chat_id in self.instances:
            return self.instances[chat_id]
        if urls is None:
            raise ValueError("URLs must be provided for new chat instances")
        urls_s = ','.join(str(url) for url in urls)
        if urls_s in self.instances_urls:
            return  self.instances_urls[urls_s]
        instance = Chat(urls=urls)
        print(instance.conversation_history)
        self.instances[chat_id] = instance
        self.instances_urls[urls_s] = chat_id
        return  chat_id

    def save_chat_instance(self, chat_id):
        if not os.path.exists("chats"):
            os.makedirs("chats")
        with open(f"chats/{chat_id}.json", "w") as f:
            json.dump(self.conversation_history, f)

    def chat_with_ai(self, user_input):
        if user_input.lower() == 'quit':
            return self.conversation_history
        retrieved_docs = self.retriever.get_relevant_documents(user_input)
        print()
        self.conversation_history.append(f"Human: {user_input}, Retrieved Docs: {retrieved_docs}")
        print("self.conversation_history = ",self.conversation_history)
        response = self.rag_chain.invoke({"input": user_input})

        ai_response = response["answer"]
        print(f"AI: {ai_response}")

        self.conversation_history.append(f"AI: {ai_response}")
        return ai_response

# Start the conversation just gave the messae you want and it returns the thing
# Print the entire conversation history
## usage chat = Chat()
### chat.chat_with_ai("hello ther")
#### chat.conversation)history --> to get all the history