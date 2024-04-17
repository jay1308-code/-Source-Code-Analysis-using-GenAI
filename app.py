from langchain.vectorstores import Chroma
from src.helper import load_embeddings
from dotenv import load_dotenv
import os
from src.helper import repo_ingestion
from flask import Flask, render_template, jsonify, request
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain


app = Flask(__name__)


load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


embeddings = load_embeddings()
# persist_directory = "db"
# # Now we can load the persisted database from disk, and use it as normal.
# new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)



# llm = ChatOpenAI()
# memory = ConversationSummaryMemory(llm=llm, memory_key = "chat_history", return_messages=True)
# qa = ConversationalRetrievalChain.from_llm(llm, retriever=new_db.as_retriever(search_type="mmr", search_kwargs={"k":8}), memory=memory)




@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')



@app.route('/chatbot', methods=["GET", "POST"])
def gitRepo():

    if request.method == 'POST':
        user_input = request.form['question']
        repo_ingestion(user_input)
        os.system("python src\store_index.py")

    return jsonify({"response": str(user_input) })



@app.route("/get", methods=["GET", "POST"])
def chat():
    new_db = FAISS.load_local("faiss_index", embeddings)
    llm = ChatOpenAI()
    memory = ConversationSummaryMemory(llm=llm, memory_key = "chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=new_db.as_retriever(search_type="mmr", search_kwargs={"k":8}), memory=memory)

    msg = request.form["msg"]
    input = msg
    print(input)

    if input == "clear":
        os.system("rm -rf repo")

    result = qa(input)
    print(result['answer'])
    return str(result["answer"])





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)