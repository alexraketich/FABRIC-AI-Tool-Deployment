import getpass
import os
import configparser
import requests
import re
import logging
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from flask import (
    Flask, request, render_template, session,
    redirect, url_for, jsonify, flash, send_file, Response
)

from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.chat_models import init_chat_model

from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

load_dotenv()

# ----------------------------------------------------------------------------------------------------- #
# -------------------------------------------- Configuration ------------------------------------------ #
# ----------------------------------------------------------------------------------------------------- #
FLASK_SECRET_KEY = os.getenv('FLASK_SECRET_KEY')
OPEN_AI_SECRET = os.getenv('OPEN_AI_SECRET')

LOG_DIR = os.getenv('LOG_DIR')

QA_DB_FILE = os.getenv('QA_DB_FILE')
CG_DB_FILE = os.getenv('CG_DB_FILE')

QA_PROMPT = os.getenv('QA_PROMPT')
CG_PROMPT = os.getenv('CG_PROMPT')

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY

# ----------------------------------------------------------------------------------------------------- #
# -------------------------------------------- Logging Setup ------------------------------------------ #
# ----------------------------------------------------------------------------------------------------- #
os.makedirs(LOG_DIR, exist_ok=True) # ensure directory exists

date_str = datetime.now().strftime("%Y%m%d%H%M")
main_log_loc = Path(LOG_DIR, date_str+'_main.log')
query_log_loc = Path(LOG_DIR, date_str+'_query.log')
MAIN_LOG_FILE = str(main_log_loc)
QUERY_LOG_FILE = str(query_log_loc)

app.logger.setLevel(logging.INFO)

main_file_handler = logging.FileHandler(MAIN_LOG_FILE)
main_file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [%(pathname)s:%(lineno)d]'
))
main_file_handler.setLevel(logging.INFO)

CUSTOM = 25
logging.addLevelName(CUSTOM, "Custom")
def custom(self, message, *args, **kwargs):
    if self.isEnabledFor(CUSTOM):
        self._log(CUSTOM, message, args, **kwargs)
logging.Logger.custom = custom

custom_file_handler = logging.FileHandler(QUERY_LOG_FILE)
custom_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
custom_file_handler.setLevel(CUSTOM)
custom_file_handler.addFilter(lambda record:record.levelno==CUSTOM)

app.logger.addHandler(main_file_handler)
app.logger.addHandler(custom_file_handler)

# ----------------------------------------------------------------------------------------------------- #
# -------------------------------------- Defining Global Entities ------------------------------------- #
# ----------------------------------------------------------------------------------------------------- #
qa_model = "gpt-4o-mini"
cg_model = "gpt-4o-mini"
tester = 'RENCI'

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
tokenizer_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3").to(device)
tokenizer_model.eval()
app.logger.info(f"Loaded reranker model: {tokenizer_model.name_or_path}")

filter_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
cant_help_emb = filter_model.encode('I cannot help you with that', convert_to_tensor=True)

# ----------------------------------------------------------------------------------------------------- #
# --------------------------------------------- Home Route -------------------------------------------- #
# ----------------------------------------------------------------------------------------------------- #
@app.route('/', methods=['POST'])
def generate_response() -> str:
    api_key = request.headers.get("X-API-KEY")
    if api_key != FLASK_SECRET_KEY:
        return Response("Invalid API key\n", status=403)

    data = request.get_json()
    query = data.get('query')
    tool_type = data.get('tool_type')
    if not query:
        app.logger.info(f"Bad request with no query")
        return Response('No query specified\n', status=400)
    if not tool_type:
        tool_type = 'QA'

    app.logger.info(f"Query: {query}\nTool type: {tool_type}")

    t_start = time.perf_counter()

    # Choose models, DB, num_docs and template based on the tool type
    if tool_type.lower() == "code generation":
        model = cg_model
        db_loc = CG_DB_FILE
        template = CG_PROMPT
        num_docs = 2
        temp = 0
    else:
        model = qa_model
        db_loc = QA_DB_FILE
        template = QA_PROMPT
        num_docs = 6
        temp = 0.2

    embedding_model = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')
    vectorstore = Chroma(persist_directory=db_loc,
          embedding_function=embedding_model)
    app.logger.info(f"Retriever successfully created")

    prompt = PromptTemplate.from_template(template)

    if model == "gpt-4o-mini":

        openai_secret = OPEN_AI_SECRET
        llm = init_chat_model("gpt-4o-mini", model_provider="openai",
                                openai_api_key=openai_secret, temperature=temp)
    else:
        default_window = 16384
        llm = OllamaLLM(model=model, num_ctx=default_window, temperature=temp)
    app.logger.info("Model successfully initialized")

    # ----------------------------------------------------------------------------------------------------- #
    # -------------------------------------- Define Application Steps ------------------------------------- #
    # ----------------------------------------------------------------------------------------------------- #
    def retrieve(state: State, k=200):
        start = time.perf_counter()
        retrieved_docs = vectorstore.similarity_search(state["question"], k=k)
        end = time.perf_counter()
        app.logger.info(f"[TIMING] Retrieval only = {end - start:.3f} sec")
        return {"context": retrieved_docs}


    def rerank(state: State) -> dict:
        start = time.perf_counter()
        query = state["question"]
        docs = state["context"]
        pairs = [(query, doc.page_content) for doc in docs]

        with torch.no_grad():
            inputs = tokenizer.batch_encode_plus(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = tokenizer_model(**inputs)
            scores = outputs.logits.squeeze().tolist()

        if isinstance(scores, float):
            scores = [scores]

        for doc, score in zip(docs, scores):
            doc.metadata["rerank_score"] = score

        reranked_docs = sorted(docs, key=lambda x: x.metadata["rerank_score"], reverse=True)
        end = time.perf_counter()
        app.logger.info(f"[TIMING] Reranking only = {end - start:.3f} sec")
        return {"context": reranked_docs[:num_docs]}

    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        start = time.perf_counter()
        response = llm.invoke(messages)
        end = time.perf_counter()
        app.logger.info(f"[TIMING] LLM generation = {end - start:.3f} sec")
        return {"answer": response}

    graph_builder = StateGraph(State)
    graph_builder.add_node("retrieve", lambda state: retrieve(state, k=30))
    graph_builder.add_node("rerank", rerank)
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "rerank")
    graph = graph_builder.compile()

    # --- Measure retrieval + reranking combined ---
    t0 = time.perf_counter()
    rag_state = graph.invoke({"question": query})
    t1 = time.perf_counter()
    app.logger.info(f"[TIMING] Retrieval + Reranking total = {t1 - t0:.3f} sec")

    # --- Measure LLM generation ---
    t2 = time.perf_counter()
    try:
        response = generate(rag_state)
    except Exception as e:
        response = {'answer': "(No response provided)"}
    t3 = time.perf_counter()
    app.logger.info(f"[TIMING] LLM Response (wrapper) = {t3 - t2:.3f} sec")

    # --- Total ---
    t_end = time.perf_counter()
    app.logger.info(f"[TIMING] Total end-to-end latency = {t_end - t_start:.3f} sec")

    # Clean the response
    res = ""
    if (response["answer"].content)[3:11] == "markdown":
        res += remove_last_backtick_block(clean_markdown_ticks(response["answer"].content))
    else:
        res += response["answer"].content

    res_emb = filter_model.encode(res, convert_to_tensor=True)
    sim = cos_sim(res_emb, cant_help_emb).item()
    if sim < 0.70:
        res += print_context_list(rag_state["context"], tool_type=tool_type)
    else:
        app.logger.info(f"Similarity score is {sim}, so LLM cannot help the user with this query.")

    app.logger.custom(f"QUERY: {query}\nRESPONSE: {res}\nMODEL: {model}\nTOOL: {tool_type}\nTESTER: {tester}")

    return jsonify({
        'query': query,
        'response': res,
        'model_used': model
    }), 200

# ----------------------------------------------------------------------------------------------------- #
# ---------------------------------- Helper functions for print output -------------------------------- #
# ----------------------------------------------------------------------------------------------------- #
def clean_markdown_ticks(text):
    lines = text.splitlines(True)
    if lines:
        lines.pop(0)
        lines.pop(-1)
    return "".join(lines)

def print_context_list(contexts, tool_type):
    sources_with_urls = []
    for document in contexts:
        key = ''
        if tool_type.lower() == "code generation":
            source = os.path.basename(document.metadata['source']).replace("py", "ipynb")
            key = 'url'
        else:


            source = document.metadata['title']
            key = 'source'

        url = document.metadata[key]
        sources_with_urls.append(f"[{source}]({url})")
    final = "\n".join(f"- {link}" for link in sources_with_urls)
    return "\n\n" + "## Sources\n" + final + "\n\n --- \n\n"

def remove_last_backtick_block(content: str) -> str:
    pattern = r"```[\t ]*\n```[\t ]*\n?"
    cleaned_text = re.sub(pattern, "```\n", content)
    return cleaned_text

if __name__ == '__main__':
    app.run(debug=True, host='gh3-internal.ccs.uky.edu', port=6008)
