import streamlit as st
from pinecone import Pinecone
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

# Initialize Pinecone
pineconev=st.secrets["pinecone_api"]
pc = Pinecone(pineconev)

# Create or connect to an index
index_name = "auggie"
index =pc.Index("auggie")

# Streamlit app interface
st.title("Auggie Data Entry")
st.write("Allows data to be uploaded into Auggie Vector Database")


with st.form(key="form", clear_on_submit=True):
    st.subheader("DATA")
    Id= st.text_area("Enter ID ")
    symptom= st.text_area("Enter Symptoms")
    associated_conditions= st.text_area("Enter Associated conditions")
    Relationships = st.text_area("Enter Relationship")

    submit_button = st.form_submit_button(label="submit")

if submit_button:
    # Combine all inputs into a single string
    combined_input = f"{Id} {symptom} {associated_conditions} {Relationships}"
    embeddings_response = pc.inference.embed(
        model='multilingual-e5-large',
        inputs=[combined_input],  # Pass as a single input
        parameters={"input_type": "passage", "truncate": "END"}
    )

    # Extract embedding values
    embedding_values = embeddings_response.data[0]['values']

    # Prepare the document
    document = {
        "id": Id,
        "values": embedding_values,
        "metadata": {
            "symptom": symptom,
            "condition": associated_conditions,
            "relationships": Relationships
        }
    }

    # Upsert the document to Pinecone
    index.upsert([(document["id"], document["values"], document["metadata"])], namespace="case-study")
    
    st.write(f"Document {Id} added to index {index_name} with embeddings.")

st.title("Force-Directed Graph Visualization from Pinecone Data")
st.write("This app retrieves data from Pinecone and visualizes it as a force-directed graph.")
 # Query Pinecone to get all documents

# Create a form for query input
with st.form(key="query"):
    query_text = st.text_input("Enter your query")
    query_button = st.form_submit_button(label="Submit")

# Check if the form is submitted
if query_button:
    # Perform embedding for the query
    query_embedding_response = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query_text],
        parameters={"input_type": "query", "truncate": "END"}
    )
    
 # Print the embeddings response to check its structure 
    st.write("Query embedding response:", query_embedding_response)

    # Extract the query embedding
    query_embedding = query_embedding_response[0].values
    
    # Query Pinecone
    documents = index.query(
        namespace='case-study',
        vector=query_embedding,
        top_k=4,
        include_values=False,
        include_metadata=True,
    )

    st.write("Retrieved documents:", documents)
    
    # Extract nodes and edges
    nodes = [] 
    edges = []
    highlighted_nodes= set()
    for document in documents['matches']: 
        data = document['metadata']
        nodes.append(document['id'])
        highlighted_nodes.add(document['id'])
        for related in data.get('relationships', []): 
            edges.append((document['id'], related))

    # Create a graph
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for edge in edges:
        G.add_edge(edge[0], edge[1], title=edge[1])

   # Create a Pyvis network 
    net = Network(height="750px", width="100%", notebook=True) 
  # Add nodes with different sizes based on whether they are highlighted
    for node in G.nodes(): 
        size = 30 if node in highlighted_nodes else 10 
        net.add_node(node, size=size, title=node) 
    # Add edges 
    for edge in G.edges(data=True): 
        net.add_edge(edge[0], edge[1], title=edge[2]['title'])

# Generate the interactive network graph 
    net.show("graph.html") 
    # Display the graph in Streamlit 
    HtmlFile = open("graph.html", 'r', encoding='utf-8') 
    source_code = HtmlFile.read() 
    st.components.v1.html(source_code, height=750)
