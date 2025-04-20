import os
import faiss
import threading
from typing import List

from langchain.memory import (
    ConversationBufferMemory,
    CombinedMemory,
    VectorStoreRetrieverMemory,
)
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

from BRAIN_llm_init import llama_slow as llm
from BRAIN_llm_init import llama_fast as classifier_llm

# Module-level placeholders
embeddings = None
vector_store = None
buffer_memory = None
retriever_memory = None
combined_memory = None
memory_chain = None


def initialize():
    global embeddings, vector_store, buffer_memory, retriever_memory, combined_memory, memory_chain

    # ———— Build empty FAISS index ————
    embeddings = OpenAIEmbeddings()
    dim = len(embeddings.embed_query("test"))
    index = faiss.IndexFlatL2(dim)
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={},
    )

    # ———— Async helper to classify/store chunks ————
    def async_store_chunks(texts: List[str], forced_category: str = None):
        def worker(chunks: List[str], forced: str):
            metadatas = []
            for chunk in chunks:
                if forced in {"me", "user"}:
                    category = forced
                else:
                    prompt = (
                        "Classify the following memory chunk. "
                        "Output exactly one word, either me or user. "
                        "Do NOT output anything else:\n\n"
                        f"{chunk}"
                    )
                    raw = classifier_llm.predict(prompt).strip().lower()
                    category = "me" if raw.startswith("me") else "user"
                metadatas.append({"category": category})
            vector_store.add_texts(chunks, metadatas=metadatas)

        threading.Thread(target=worker, args=(texts, forced_category), daemon=True).start()

    # ———— Preload story if available ————
    try:
        story_path = os.path.join(os.path.dirname(__file__), "story.txt")
        with open(story_path, "r", encoding="utf-8") as f:
            story = f.read()
        story_chunks = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        ).split_text(story)
        async_store_chunks(story_chunks, forced_category="me")
    except FileNotFoundError:
        pass

    # ———— Sliding-Window Buffer Memory ————
    class SlidingWindowBufferMemory(ConversationBufferMemory):
        # allow setting attributes not declared as fields
        model_config = {"extra": "allow"}  # for pydantic v2; if you’re on v1, replace with a Config subclass
        
        def __init__(
            self,
            memory_key: str,
            input_key: str,
            buffer_size: int,
            vector_store: FAISS,
        ):
            super().__init__(memory_key=memory_key, input_key=input_key)
            # now safe to assign
            self.buffer_size = buffer_size
            self.vector_store = vector_store

        def save_context(self, inputs: dict, outputs: dict) -> None:
            super().save_context(inputs, outputs)
            msgs = self.chat_memory.messages
            max_msgs = self.buffer_size * 2
            if len(msgs) > max_msgs:
                user_msg = msgs.pop(0)
                ai_msg   = msgs.pop(0)
                async_store_chunks([user_msg.content], forced_category="user")
                async_store_chunks([ai_msg.content],    forced_category="me")

    buffer_memory = SlidingWindowBufferMemory(
        memory_key="history",
        input_key="input",
        buffer_size=5,
        vector_store=vector_store,
    )

    # ———— Read-Only Retriever Memory ————
    class ReadOnlyRetrieverMemory(VectorStoreRetrieverMemory):
        def save_context(self, inputs: dict, outputs: dict) -> None:
            # no-op
            return

    retriever_memory = ReadOnlyRetrieverMemory(
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        memory_key="long_term",
        input_key="input",
    )

    # ———— Combine memories & build chain ————
    combined_memory = CombinedMemory(memories=[buffer_memory, retriever_memory])

    memory_chain_prompt = PromptTemplate.from_template(
        """Your output is one sentence max.
    You are a funny live streamer who is a cute Japanese anime character girl called Sama.
    Be creative and engage based on past chat history.
    Ask a question, say something about yourself, or bring up something to do with your audience.

    Relevant long‑term memories:
    {long_term}

    Recent chat (last 5 turns):
    {history}

    User: {input}
    AI:"""
    )
    
    memory_chain = LLMChain(llm=llm, prompt=memory_chain_prompt, verbose=False)

# Ensure initialization on import
initialize()