from flask import Flask
import openai
from flask import Flask,jsonify,request
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
persist_directory = 'docs/chroma/'
from langchain.vectorstores import FAISS

app = Flask(__name__)

OPENAI_API_KEY  = "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"
openai.api_key = OPENAI_API_KEY



def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.0, 
    )
    content = response.choices[0].message["content"]
    return content


chat4 = ChatOpenAI(openai_api_key = OPENAI_API_KEY, temperature = 0.9, model="gpt-4")


@app.route("/", methods=["GET"])
def index():
    return jsonify({"RESPONSE": "WORKING"})

@app.route("/chat",methods=["POST"])
def chat():
    if request.json:
        data = request.json['data']
        prompt = request.json['question']
        history = ChatMessageHistory()
        memory = ConversationBufferMemory()
        profiles = []
        for i in range(len(data)):
            profiles.append(data[i]['val'])

        print("success 1")

        def getSplits(user_data):
            r_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=10,
            separators=["\n\n", "\n", "(?<=\. )", " ", ""])
            return(r_splitter.split_text(user_data))
        splits = []

        for i in range(len(profiles)):
            split = getSplits(profiles[i])
            splits.append(split)

        class Document:
            def __init__(self, page_content, metadata):
                self.page_content = page_content
                self.metadata = metadata


        print("success 2")

        docs = []
        for j in range (len(splits)):
            for i in range(len(splits[j])):
                doc = Document(splits[j][i], {"userID": data[j]['name'],"index": i})
                docs.append(doc)
        pd = 'docs/chroma/'
        embedding = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)

        print("success 3")

        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            persist_directory=pd
        )

        def get_completion(prompt, model="gpt-3.5-turbo"):
            messages = [{"role": "user", "content": prompt}]
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0.9, 
            )
            
            content = response.choices[0].message["content"]
            return content
            print(vectordb._collection.count())
            
        if prompt:
            history = ChatMessageHistory()
            question = prompt
            history.add_user_message(question)
            res = vectordb.max_marginal_relevance_search(question,k=10, fetch_k=20)
            mem = memory.load_memory_variables({})

            print("success 4")


            prompt = f"""
            Generate a thorough response using the data in {res} and link with the history in {mem}. 
            Use userIDs as references. Do not add opinions or suggestions from your own.
        
            text : ```{res}```
            history: ```{mem}```
            """
            print("success 5")
            response1 = get_completion(prompt)
            history.add_ai_message(response1)
            memory.save_context({"input": question}, {"output": response1})
            return jsonify({"response":response1})