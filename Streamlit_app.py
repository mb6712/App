#!pip install streamlit
#!pip install elasticsearch
#!pip install sentence_transformers
# streamlit UI
# pip install streamlit elasticsearch sentence_transformers transformers

import streamlit as st
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch

indexName = "PMID"

try:
    es = Elasticsearch(
        "https://localhost:9200",
        basic_auth=("elastic", "your_password"),  # Update with your actual password
        ca_certs="path/to/your/http_ca.crt"  # Update with the actual path to your certificate
    )
    if es.ping():
        st.success("Successfully connected to Elasticsearch!")
    else:
        st.error("Oops!! Cannot connect to Elasticsearch.")
except Exception as e:
    st.error(f"Connection Error: {e}")

def search(input_query):
    # Initialize PubMedBERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    
    # Encode the query using PubMedBERT
    inputs = tokenizer(input_query, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    query_vector = outputs.last_hidden_state.mean(dim=1).detach().numpy()

    query = {
        "field": "AbstractVector",  # Ensure this matches the field in your Elasticsearch documents
        "query_vector": query_vector.tolist(),
        "k": 10,
        "num_candidates": 500
    }
    res = es.knn_search(index=indexName, knn=query, source=["Title", "Abstract"])
    results = res["hits"]["hits"]

    return results

def main():
    st.title("PubMed Article Search")

    # Input: User enters search query
    search_query = st.text_input("Enter your search query")

    # Button: User triggers the search
    if st.button("Search"):
        if search_query:
            # Perform the search and get results
            results = search(search_query)

            # Display search results
            st.subheader("Search Results")
            for result in results:
                with st.container():
                    if '_source' in result:
                        try:
                            st.header(f"{result['_source']['Title']}")
                        except Exception as e:
                            st.error(e)
                        
                        try:
                            st.write(f"Abstract: {result['_source']['Abstract']}")
                        except Exception as e:
                            st.error(e)
                        st.divider()

if __name__ == "__main__":
    main()
