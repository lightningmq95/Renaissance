# %%
import dspy
import json
import os
import spacy
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
from datetime import datetime
from pymongo import MongoClient
from fastapi import FastAPI, HTTPException

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MONGODB_URI = os.getenv('MONGODB_URI')

# %%
# MongoDB Atlas connection
client = MongoClient(MONGODB_URI)
db = client['renai']
collection = db['events']

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# %%
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# %%
lm = dspy.LM('gemini/gemini-2.0-flash-exp', api_key=GEMINI_API_KEY)
dspy.configure(lm=lm)

# %%
class Entity(BaseModel):
    entity: str
    type: str

class EventEntity(BaseModel):
    entity: str
    type: str
    role: str

class Event(BaseModel):
    action: str
    type: str
    date: str
    location: str
    entities: List[EventEntity]

# %%
# Define request and response models
class TranscriptRequest(BaseModel):
    text: str
    speaker: str

class QueryRequest(BaseModel):
    question: str

class EventResponse(BaseModel):
    action: str
    type: str
    date: str
    location: str
    entities: List[EventEntity]

class QueryResponse(BaseModel):
    responses: List[str]

# %%
class ExtractEvents(dspy.Signature):
    """Extract a list of relevant events, each containing Event type, date, location and participating entities (if any, along with their role in the specific event) information from text, current date and given entities."""

    text: str = dspy.InputField()
    speaker: str = dspy.InputField(desc="the speaker of the text")
    entities: List[Entity] = dspy.InputField(desc="a list of entities and their metadata")
    current_date: str = dspy.InputField(desc="the current date to convert relative dates like 'today', 'yesterday', 'tomorrow' to actual dates")
    
    events: List[Event] = dspy.OutputField(desc="a list of events being talked about, either happening during the meeting or being referenced to, should NOT include events to happen in the future, and their metadata with fields: action(What Happened), type, date (convert relative dates like 'today', 'yesterday', 'tomorrow' to actual dates), location, entities (fetched from input)")


# %%
class KnowledgeExtraction(dspy.Module):
    def __init__(self):
        self.cot2 = dspy.ChainOfThought(ExtractEvents)

    def normalize_text(self, text):
        # Normalize text to title case
        return text.title()

    def extract_entities(self, text):
        doc = nlp(text)
        entities = []
        for ent in doc.ents:
            normalized_entity = self.normalize_text(ent.text)
            entities.append(Entity(entity=normalized_entity, type=ent.label_))
        return entities

    def forward(self, text, speaker):
        entities = self.extract_entities(text)
        current_date = datetime.now().strftime('%Y-%m-%d')
        events = self.cot2(text=text, speaker=speaker, entities=entities, current_date=current_date)
        return events

class RAG(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought('context, question -> response')

    def forward(self, question):
        # Fetch records from MongoDB
        records = collection.find()

        corpus = []
        for record in records:
            context = ""
            for key, value in record.items():
                if key == "_id":
                    continue
                # Convert key to proper case
                proper_case_key = key.replace('_', ' ').title().replace(' ', '')
                context += f"{proper_case_key}: {value}\n"
            corpus.append(context)
            
        embedder = dspy.Embedder('gemini/text-embedding-004')
        search = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=5)
        context = search(question).passages
        return self.respond(context=context, question=question)


# %%
app = FastAPI()

# %%
# Initialize modules
knowledge_extraction = KnowledgeExtraction()
rag = RAG()

# %%
@app.post("/extract-events", response_model=List[Event])
def extract_events(request: List[TranscriptRequest]):
    try:
        event_list = []
        for req in request:
            response = knowledge_extraction(text=req.text, speaker=req.speaker)
            event_list.extend(response.events)
        return event_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query-rag", response_model=str)
def query_rag(request: QueryRequest):
    try:
        responses = rag(question=request.question)
        return responses.response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



