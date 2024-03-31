import streamlit as st
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from llama_cpp import Llama
from huggingface_hub import hf_hub_download 
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from huggingface_hub import hf_hub_download
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader
import streamlit as st
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
import os
import re
from langchain_community.llms import Ollama

template = f"""Use the following pieces of information to answer the user's question.
If you don't know the answer or if the data might be outdated, just say that you don't know or acknowledge the potential time lapse, don't try to make up an answer.

Context: {{context}}
Question: {{question}}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
def set_custom_prompt():     
    prompt = PromptTemplate(template=template, input_variables=['context', 'question'])
    return prompt

def load_data(url):
  data1 = YoutubeLoader.from_youtube_url(url , add_video_info=True)
  data = data1.load()
  return data

def split_data(data):
  splitter = RecursiveCharacterTextSplitter(chunk_size = 2000 , chunk_overlap = 200)
  splits = splitter.split_documents(data)
  return splits

def init_llm():
#   model_name = "google/gemma-2b-it"
#   model_file = "gemma-2b-it.gguf"
#   HF_TOKEN = st.secrets["HF_TOKEN"]
#   model_pth = hf_hub_download(model_name,
#                                   filename=model_file,
#                                   local_dir='/content',
#                                   token= HF_TOKEN)
#   callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])  
#   llm = LlamaCpp(model_path = model_pth ,  max_tokens = 2000  , n_gpu_layers = -1 ,callback_manager= callback_manager, verbose=True,)
  llm = Ollama(model="mistral:latest")
  return llm 
  
def init_db(splits):
  embedding_func = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
  Chroma.from_documents(splits , embedding_func ,  persist_directory="./chroma_db5")

def init_chain(llm , db_chroma , prompt): 
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                            chain_type='stuff',
                                            retriever= db_chroma.as_retriever(search_kwargs={ "k": 4}),
                                            chain_type_kwargs={'prompt': prompt},
                                            )
    return qa_chain

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db_chroma = Chroma(persist_directory="./chroma_db5", embedding_function= embeddings)
    llm = init_llm()
    qa_prompt = set_custom_prompt()
    qa = init_chain(llm,  db_chroma  , qa_prompt)
    return qa
  
st.set_page_config(page_title="InsightBOT : Your YouTube Companion" ,page_icon = "🤖")  
st.title("InsightBOT 🤖")

st.sidebar.subheader("Youtube URL 🔗")

url = st.sidebar.text_input('Enter Youtube Video URL:')

if st.sidebar.button("Analyze Video"):
    st.video(url, format='video/mp4')
    
    with st.spinner("Extracting insights... 🧠💡"):
        data = load_data(url)
        splits = split_data(data)
        db = init_db(splits)
   
if st.sidebar.button('Summarise Video'):
    with st.spinner('Writing video synopsis... 🖊️'):
        data = load_data(url)
        splits = split_data(data)
        llm = init_llm()
        sum_chain = load_summarize_chain(llm  = llm , chain_type = "map_reduce")
        summary = sum_chain.run(splits)
        st.write(summary)

st.markdown("Summarise and Engage with Your Video Content! 💡")

st.sidebar.markdown("---")
st.sidebar.caption("Created By: Sidhant Manale ")
               
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": f"Hey! How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

st.sidebar.markdown("#")
st.sidebar.markdown("#")
st.sidebar.subheader("Clear Chat")
if st.sidebar.button("Reset"):
    st.session_state.messages = []

if prompt1 := st.chat_input():
    
        st.session_state.messages.append({"role": "user", "content": prompt1})
        st.chat_message("user").write(prompt1)
        with st.spinner("Thinking..."):
            qa_result = qa_bot()
            response = qa_result({'query': prompt1})
            helpful_answer = response['result']
              
            if 'source_documents' in response and response['source_documentsz ']:
                document = response['source_documents'][0]
                metadata = document.metadata
                file = metadata['source'].split("\\")[-1]
                source = os.path.splitext(file)[0]
                assistant_answer = f"{helpful_answer} \n\n Source : {source} Video"
            else:
                source = "Llama"
                assistant_answer = f"{helpful_answer} \n\n Source : {source} Model"
                
            st.session_state.messages.append({"role": "assistant", "content": helpful_answer})
            st.chat_message("assistant").write(helpful_answer)
            
        
        

        



 