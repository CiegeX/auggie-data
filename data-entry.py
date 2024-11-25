import streamlit as st
from pinecone import Pinecone

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


