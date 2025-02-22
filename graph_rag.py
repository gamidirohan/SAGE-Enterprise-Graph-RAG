import os
import logging
import dotenv
import streamlit as st

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO)
st.title("Graph RAG: Query Your Knowledge Graph")
st.write("Enter your question to query the Neo4j knowledge graph.")

# Instantiate the Neo4j connector
from langchain_community.graphs import Neo4jGraph
graph = Neo4jGraph()

# Instantiate the LLM (using Groq DeepSeek API in JSON mode)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model_name="deepseek-r1-distill-llama-70b",
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.0,
    model_kwargs={"response_format": {"type": "json_object"}}
)

# Create the Graph RAG chain
from langchain.chains import GraphCypherQAChain
chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, verbose=True)
st.write("Knowledge Graph RAG is ready!")

# Streamlit query interface
question = st.text_input("Ask your question:")
if question:
    with st.spinner("Generating answer..."):
        result = chain.invoke({"query": question})
    if result.get('result'):
        st.subheader("Answer")
        st.write(result['result'])
    else:
        st.write("No result:", result)
