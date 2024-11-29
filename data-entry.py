import streamlit as st
from pinecone import Pinecone
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from transformers import pipeline
import uuid 

# Initialize Pinecone
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

