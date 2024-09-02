import streamlit as st
import os
from llama_index.embeddings.openai import OpenAIEmbedding 
from llama_index.llms.openai import OpenAI
from src.chat_engine import get_chat_engine
from src.query_engine import EnhancedQueryEngine
from src.pinecone_retriever import PineconeRetriever
from src.data_indexer import DataIndexer

def get_app_model():
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-large",
        dimensions=5,
        api_key=os.environ['OPENAI_API_KEY']
    )
    indexer = DataIndexer(
        dataset_name="movies",
        embedding_dimension=5,
        embed_model=embed_model,
        pinecone_api_key=os.environ['PINECONE_API_KEY']
    )
    vector_store = indexer.get_vector_store()
    
    retriever = PineconeRetriever(
        vector_store=vector_store,
        embed_model=embed_model,
        query_mode="default",
        similarity_top_k=10
    )
    llm = OpenAI(
        model="gpt-4o",
        api_key=os.environ['OPENAI_API_KEY']
    )
    
    query_engine = EnhancedQueryEngine(
        retriever=retriever,
        llm=llm,
        streaming=True
    )
        
    chat_engine = get_chat_engine(
        chat_mode="condense_question",
        query_engine=query_engine,
        llm=llm,
        streaming=True
    )
    
    return chat_engine
    

class ChatView:
    def __init__(self):
        st.set_page_config(
            page_title="MovieMate AI",
            page_icon="🎬",
            layout="wide",
            initial_sidebar_state="auto",
            menu_items=None
        )
        st.title("MovieMateAI 🎬🤖")
        st.info("Ask me anything about movies", icon="🎥")

    def get_user_input(self):
        return st.chat_input("Message MovieMateAI")

    def display_response(self, response):
        st.write(response)


class ChatController:
    def __init__(self):
        self.view = ChatView()
        self.chat_engine = get_app_model()
        self._initialize_session_state()

    def _initialize_session_state(self):
        if "messages" not in st.session_state.keys():
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Ask me anything about movies",
                }
            ]

        if "chat_engine" not in st.session_state.keys():
            st.session_state.chat_engine = self.chat_engine

    def run(self):
        user_input = self.view.get_user_input()
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                response_stream = st.session_state.chat_engine.stream_chat(user_input)
                st.write_stream(response_stream.response_gen)
                message = {"role": "assistant", "content": response_stream.response}
                st.session_state.messages.append(message)



if __name__ == "__main__":
    controller = ChatController()
    controller.run()
