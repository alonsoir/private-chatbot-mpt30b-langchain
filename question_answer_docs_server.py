from flask import Flask, jsonify, request, send_file
from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.vectorstores import Chroma
import os
import time
import threading

from constants import CHROMA_SETTINGS

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get("PERSIST_DIRECTORY")
model_path = os.environ.get("MODEL_PATH")
target_source_chunks = int(os.environ.get("TARGET_SOURCE_CHUNKS", 4))

# Initialize the model outside of load_model()
print("embeddings_model_name is ",embeddings_model_name)
print("persist_directory is ",persist_directory)
print("model_path is ",model_path)
print("target_source_chunks is ",target_source_chunks)
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
    client_settings=CHROMA_SETTINGS,
)
retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
qa = None  # Define it here to be accessible from handle_question()

app = Flask(__name__)

llm = None  # Inicializar llm fuera de la función load_model()

def load_model():
    try:
        global llm  # Indicar que se usará la variable global llm
        start = time.time()

        # check if the model is already downloaded
        if os.path.exists(model_path):
            print("Loading model...", model_path)

            # initialize llm
            llm = CTransformers(
                model=os.path.abspath(model_path),
                model_type="mpt",
                callbacks=[StreamingStdOutCallbackHandler()],
                config={"temperature": 0.1, "stop": ["", "|<"]},
            )
            end = time.time()

            print(f"\n> llm initialized! (took {round(end - start, 2)} s.):")

            return llm  # Devolver el objeto llm en lugar de True
        else:
            raise ValueError(
                "Model not found. Please run `poetry run python download_model.py` to download the model."
            )
    except Exception as e:
        print(str(e))
        raise

@app.route('/')
def index():
    return send_file('answer.html')

@app.route('/question', methods=['POST'])
def handle_question():
    global qa

    try:
        # Get the question from the POST request
        question = request.json['question']
        print("(Using POST) question is: ",question)
        # Return an immediate response to the client with a "processing" message
        response = {'status': 'processing'}
        threading.Thread(target=process_question, args=(question,)).start()

        return jsonify(response)
    except Exception as e:
        print(str(e))
        return jsonify({'error': 'An error occurred while processing the question.'})

def init():
    global qa
    print("Initializing model for the first time...")
    start = time.time()

    if not qa:
        qa = RetrievalQA.from_chain_type(
            llm=load_model(),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
            )
    end = time.time()
    print(f"\n> Init model (took {round(end - start, 2)} s.):")


def process_question(question):
    global qa

    try:
        # Get the answer from the chain
        print("Thinking... Please note that this can take a few minutes.")
        start = time.time()
        res = qa(question)
        answer, docs = res["result"], res["source_documents"]
        end = time.time()

        # Print the result
        print("\n\n> Question:")
        print(question)
        print(f"\n> Answer (took {round(end - start, 2)} s.):")
        print(answer)

        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)
    except Exception as e:
        print(str(e))
        raise

if __name__ == "__main__":
    init()
    app.run(host='127.0.0.1', port=8080, debug=True)
