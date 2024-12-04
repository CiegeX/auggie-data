import streamlit as st
from pinecone import Pinecone
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from transformers import pipeline
import uuid 


# Initialize Pinecone and Openai
pineconev=st.secrets["pinecone_api"]
pc = Pinecone(pineconev)

# Create or connect to an index
index_name = "auggie"
index =pc.Index("auggie")

# Create Pipeline for preprocessing text
pipe = pipeline("token-classification", model="blaze999/Medical-NER")

# Ensure session state initialization
if 'entities' not in st.session_state:
    st.session_state.entities = []
if 'selected_entities' not in st.session_state:
    st.session_state.selected_entities = []

# Streamlit app interface
st.title("Auggie Data Entry")
st.write("Allows data to be uploaded into Auggie Vector Database")

with st.form(key="form", clear_on_submit=True):
    st.subheader("Author Notes")
    author_notes = st.text_area("Enter Notes here")
    submit_button = st.form_submit_button(label="Submit")

def merge_entities(entities):
    merged_entities = []
    current_entity = None
    current_text = []

    for entity in entities:
        entity_type = entity['entity']
        text = entity['word'].replace("▁", "").strip()

        if entity_type.startswith('B-'):
            if current_entity and current_text:
                merged_entities.append({"entity": current_entity, "text": " ".join(current_text)})
            current_entity = entity_type[2:]
            current_text = [text]
        elif entity_type.startswith('I-') and current_entity:
            current_text.append(text)
        else:
            if current_entity and current_text:
                merged_entities.append({"entity": current_entity, "text": " ".join(current_text)})
            current_entity = None
            current_text = []

    if current_entity and current_text:
        merged_entities.append({"entity": current_entity, "text": " ".join(current_text)})

    return merged_entities

# Check if the form is submitted
if submit_button and author_notes:
    # Use the transformer pipeline to analyze the input
    results = pipe(author_notes)

    # Extract and merge entities from the results
    entities = merge_entities(results)

    # Store the entities in session state
    st.session_state.entities = entities

# Display the results in a scrollable expander
if st.session_state.entities:
    with st.expander("Extracted Entities"):
        for i, entity in enumerate(st.session_state.entities):
            checkbox_label = f"Relationship: {entity['entity']}, Node: {entity['text']}"
            checkbox_key = f"entity_{i}"
            checkbox_value = entity in st.session_state.selected_entities

            if st.checkbox(checkbox_label, key=checkbox_key, value=checkbox_value):
                if entity not in st.session_state.selected_entities:
                    st.session_state.selected_entities.append(entity)
            else:
                if entity in st.session_state.selected_entities:
                    st.session_state.selected_entities.remove(entity)

# Button to upload selected entities
if st.button("Upload to Pinecone") and st.session_state.selected_entities:
    combined_text = ', '.join([d["text"] for d in st.session_state.selected_entities])

    if isinstance(combined_text, str):
        # Check for a disease_disorder entity to use as the ID
        disease_entity = next((d["text"] for d in st.session_state.selected_entities if d["entity"] == "DISEASE_DISORDER"), None)

        # Use the disease entity as the ID if available, otherwise generate a unique ID
        unique_id = disease_entity if disease_entity else str(uuid.uuid4())

        # Convert combined text to a single vector using an embedding model
        embeddings_response = pc.inference.embed(
            model='multilingual-e5-large',
            inputs=[combined_text],
            parameters={"input_type": "passage", "truncate": "END"}
        )
        combined_embedding = embeddings_response[0]['values']

        try:
            # Retrieve existing metadata if it exists
            fetch_response = index.fetch(ids=[unique_id], namespace="case-study")
            existing_metadata = fetch_response['vectors'].get(unique_id, {}).get('metadata', {})
        except KeyError:
            existing_metadata = {}
         # Ensure relationships and conditions are correctly paired and ordered 
        existing_conditions = existing_metadata.get("associated conditions", "").split(", ") if existing_metadata.get("associated conditions", "") else [] 
        existing_relationships = existing_metadata.get("relationships", "").split(", ") if existing_metadata.get("relationships", "") else [] 
        new_relationships_conditions = [(d["entity"], d["text"]) for d in st.session_state.selected_entities if d["text"] not in existing_conditions] 
        updated_relationships_conditions = list(zip(existing_relationships, existing_conditions)) + new_relationships_conditions
        # Convert to single strings maintaining order 
        relationships = ', '.join([pair[0] for pair in updated_relationships_conditions]) 
        conditions = ', '.join([pair[1] for pair in updated_relationships_conditions])
        
        updated_metadata = { 
            "relationships": relationships, 
            "associated conditions": conditions 
        }
        
        # Upsert the combined vector with updated metadata to Pinecone
        index.upsert([
            (unique_id, combined_embedding, updated_metadata)
        ], namespace="case-study")

        st.write("Combined vector uploaded to Pinecone successfully!")
        st.session_state.selected_entities = []
    else:
        st.write("Error: Combined text is not in the correct format")

# Display the currently selected entities
st.write("Selected entities:")
for entity in st.session_state.selected_entities:
    st.write(f"Entity: {entity['entity']}, Text: {entity['text']}")

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
    #st.write("Query embedding response:", query_embedding_response)

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

    #st.write("Retrieved documents:", documents)
    
# Initialize sets and lists
    nodes = set()  # Using a set to avoid duplicate nodes
    edges = []
    associated_map = {}
    highlighted_nodes = set()

# Process documents
    for document in documents['matches']: 
        data = document['metadata']
        doc_id = document['id']
        nodes.add(doc_id)
        highlighted_nodes.add(doc_id)

    # Correctly split associated conditions
        associated_conditions = data.get('associated_conditions', '').split(', ')
        for condition in associated_conditions:
            if condition not in associated_map:
                associated_map[condition] = []
            associated_map[condition].append(doc_id)
            nodes.add(condition)

    # Ensure 'relationships' are correctly processed as a list of whole items
        relationships = data.get('relationships', [])
        if isinstance(relationships, str):
            relationships = relationships.split(', ')

        for related in relationships: 
            edges.append((doc_id, related))
            


# Connect nodes based on similar associate conditions 
    for condition, related_nodes in associated_map.items(): 
        for i in range(len(related_nodes)): 
            for j in range(i + 1, len(related_nodes)): 
                edges.append((related_nodes[i], related_nodes[j]))
                
# Create a graph
    G = nx.Graph()
    G.add_nodes_from(nodes)

    for edge in edges:
        G.add_edge(edge[0], edge[1])
        G.edges[edge[0], edge[1]]['title'] = edge[1]
          
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
