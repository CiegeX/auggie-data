import json
import streamlit as st
from pinecone import Pinecone
from openai import OpenAI
import uuid

# Initialize Pinecone and OpenAI
pineconev = st.secrets["pinecone_api"]
pc = Pinecone(pineconev)
client = OpenAI(api_key=st.secrets["openai_api"])

# Create or connect to an index
index_name = "auggie-transcripts"
index = pc.Index(index_name)

# Ensure session state for storing response_data
if 'response_data' not in st.session_state:
    st.session_state['response_data'] = []
if "selected_entities" not in st.session_state: 
    st.session_state["selected_entities"] = []
st.title('Auggie Data Entry')

def get_response(message):
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": """1. given medical notes, we want to break them up into a directed graph
2.1 if the note itself has an acronym for an entity then use it to name the entity but do not invent your own
2.2 the longer name or names can be added as `synonyms` property to the entity
2.3 when deciding the name for an Entity don't break multiple words down to the point that they lose their meaning for example don't shorten calcium gluconate to just calcium on your own
2.4 anything that looks like a definition or meaning or purpose or explanation should become part of the `details` property for an Entity. Don't add extra context if not from text!
2.6.1 prioritize conforming to the following relationship names whenever possible: "IS_TREATED_WITH", "IS_ASSOCIATED_WITH", "HAS_SIGN", "IS_SIGN_OF", "HAS_SYMPTOM", "IS_SYMPTOM_OF", "IS_SIGN_OF", "HAS_COMPLICATION", "IS_TRIGGERED_BY", "TRIGGERS", "HAS_MNEMONIC", "HAS_RISK_FACTOR", "IS_RISK_FACTOR_FOR", "CAN_BE_SCORED_USING"
2.6.4 if the theme or idea or meaning or gist of the relationship in the note doesn't match any of the above suggested ones then use whatever new relationship name you best see fit
2.7 we want to show an entity and its synonyms that will be created as a property of that entity
2.8 If entities are implied but not explicitly described (e.g., 'signs and symptoms' without details) then don't try to process it because its missing data! 
3. we also want to display pinecone upsert that goes with each of the pieces mentioned in #2
4. escape apostrophes in field values
7. The output MUST be in JSON where it can look like { markup: [ {markup: human readable, pineConeUpserts: [actual upsert]} ] } but please format it nicely with indentations
8. when defining a relationship the id must match source name
                {
                    "markup": [
                        {
                            "id": "Behcet",
                            "values": {
                                "name": "Behcet",
                                "synonyms": ["Behçet"],
                                "details": "A condition associated with painful genital ulcers, vasculitis, aphthous ulcers, and uveitis. It does not include fever or erythema nodosum."
                                "relationship": ["HAS_SYMPTOM", "HAS_SYMPTOM"],
                                "target": ["Genital Ulcers", "Uveitis"]
                            }
                        },
                        {
                            "id": "Genital Ulcers",
                            "values": {
                                "name": "Genital Ulcers",
                                "synonyms": ["Painful genital ulcers"],
                                "details": "Painful sores in the genital region, characteristic of Behcet disease."
                                "relationship": "IS_SYMPTOM_OF",
                                "source": "Behcet"
                            }
                        },
                        {
                            "id": "Uveitis",
                            "values": {
                                "name": "Uveitis",
                                "synonyms": [],
                                "details": "Inflammation of the uvea, often associated with Behcet disease."
                                "relationship": "IS_SYMPTOM_OF",
                                "source": "Behcet"
                            }
                        }
                    ]
                }
                """
            },
            {
                "role": "user",
                "content": message
            }
        ]
    )
    response_content = response.choices[0].message.content.strip()

    # Log response content for debugging
    st.write(f"Raw Response Content: {response_content}")  # Debugging line to print raw response content

    if not response_content:
        st.error("Error: The response content is empty.")
        return ""

    # Remove the backticks and "json" label if present
    if response_content.startswith("```json") and response_content.endswith("```"):
        response_content = response_content[7:-3].strip()

    # Properly escape single quotes
    response_content = response_content.replace("\\'", "'")

    # Log cleaned response content for debugging
    st.write(f"Cleaned Response Content: {response_content}")  # Debugging line to print cleaned response content

    try:
        response_json = json.loads(response_content)  # Convert string to JSON
        st.write(f"Parsed JSON: {response_json}")  # Debugging line to print parsed JSON

        # Extract data from 'markup' and flatten the structure
        pine_cone_upserts = []
        if 'markup' in response_json:
            for entry in response_json['markup']:
                if 'id' in entry and 'values' in entry:
                    pine_cone_upserts.append(entry)
                else:
                    st.error("Error: 'id' or 'values' key not found in entry.")
        else:
            st.error("Error: 'markup' key not found in JSON response.")
            return "Error: 'markup' key not found."

        st.session_state['response_data'].extend(pine_cone_upserts)
    except json.JSONDecodeError as e:
        st.error(f"JSONDecodeError: {e}")
        st.error(f"Response Content for Debugging: {response_content}")
        return f"Error: Invalid JSON response. Details: {str(e)}"

    return response_content

# Input for user message
input_placeholder = st.empty()
message = input_placeholder.text_input("You: ", key="chat_input")

if st.button("Send", key="send_button"):
    if message:
        response_content = get_response(message)
        if response_content and "Error:" not in response_content:  # Only append if the response is not empty and valid
            st.rerun()  # Re-run the script to update the checkboxes

# Display the response data with checkboxes
#st.write(f"Session State Response Data: {st.session_state['response_data']}")  # Debugging line to print response data
for i, item in enumerate(st.session_state['response_data']):
    if isinstance(item, dict):  # Ensure the item is a dictionary
        entity = item.get('values', {}).get('name', 'Unknown Entity')
        relationships = ", ".join([f"{relation['relationship']} with {relation['target']}" for relation in item.get('values', {}).get('relationships', []) if 'relationship' in relation and 'target' in relation])
        checkbox_label = f":green[Node: {item['id']}]  \n :blue[Edge: {item['values'].get('relationship')}]  \n :green[Source:  {item['values'].get('source')}]  \n :green[Target: {item['values'].get('target')}]  \n Details: {item['values'].get('details')}"
        if st.checkbox(checkbox_label, key=f"item_{i}"):
            if item not in st.session_state.selected_entities:
                st.session_state.selected_entities.append(item)    
        else:
            if item in st.session_state.selected_entities:
                st.session_state.selected_entities.remove(item)

# Display the selected entities
st.write("Selected entities:")
for entity in st.session_state.selected_entities:
    if isinstance(entity, dict):
        st.write(f"ID: {entity.get('id', 'Unknown ID')}")

# Button to upload selected entities
if st.button("Upload to Pinecone") and st.session_state.selected_entities:
    for entity in st.session_state.selected_entities:
        if isinstance(entity, dict):
            entity_id = entity.get('id', str(uuid.uuid4()))  # Use entity ID or generate a unique one
            entity_text = entity.get("values", {}).get("name", "Unknown Entity")
            
            # Get embedding
            embeddings_response = pc.inference.embed(
                model='multilingual-e5-large',
                inputs=[entity_text],
                parameters={"input_type": "passage", "truncate": "END"}
            )
            embedding = embeddings_response[0]['values']

            # Prepare metadata
            metadata = {
                "source": entity.get("values", {}).get("source", ""),
                "target": entity.get("values", {}).get("target", ""),
                "relationship": entity.get("values", {}).get("relationship", ""),
                "details": entity.get("values", {}).get("details", ""),
                "synonyms": entity.get("values", {}).get("synonyms", [])
            }

            # Upsert into Pinecone
            index.upsert([(entity_id, embedding, metadata)], namespace="test")
            
    st.write("Selected entities uploaded to Pinecone successfully!")
    st.session_state.selected_entities = []

st.title("Query")


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
        namespace='test',
        vector=query_embedding,
        top_k=4,
        include_values=False,
        include_metadata=False,
    )

    #st.write("Retrieved documents:", documents)

    for doc in documents["matches"]:
        st.write(f"{doc['id']} with a score of : {doc['score']}")
