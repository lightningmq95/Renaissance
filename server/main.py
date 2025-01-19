# %%
import time
from flask import jsonify, request
import markdown
import dspy
import json
import os
import spacy
import warnings
import assemblyai as aai
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
from datetime import datetime
from pymongo import MongoClient
from fastapi import FastAPI, HTTPException
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
import logging
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
warnings.filterwarnings("ignore")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
llm = GoogleGenerativeAI(
    model="gemini-pro", google_api_key=GEMINI_API_KEY, temperature=0
)

# LLM Chain configurations for refining, todos, and summary
refining_template = """Refine the following text from a company meeting transcription. 
Include speaker labels where available. 
Ensure the language is clear, concise, and professional, while preserving the original meaning.

Text: {text}"""

todo_template = """Transform the following meeting transcript into a JSON formatted to-do list. 
Extract key tasks mentioned during the discussion and format them as an array of JSON objects.
Each task object should include these fields:
- "description": the task description
- "assignee": the person responsible (if mentioned, otherwise "Unassigned")
- "deadline": the deadline (if mentioned, otherwise null)
- "status": "pending"
- "mentioned_by": the speaker who mentioned the task (if available)

Return only valid JSON with this structure:
{{"todos": [
    {{"description": "task", "assignee": "person", "deadline": "date", "status": "pending", "mentioned_by": "speaker"}},
    ...
]}}

Transcript: {text}"""

summary_template = """Summarize the following text from a company meeting transcription. 
Your summary must strictly adhere to the following guidelines to ensure accuracy:
1. **Do not infer or assume** any information not explicitly stated in the transcript.
2. Highlight only the key points that are **clearly mentioned** in the text.
3. Clearly differentiate between **discussions**, **decisions made**, and **action items**.
4. Use only the information provided in the transcript and maintain a professional tone.
5. If certain information is unclear or incomplete in the transcript, state that explicitly instead of making assumptions.

Ensure the summary is concise, structured, and professional. 

Transcript: {text}"""

chains = {
    "refining": LLMChain(
        llm=llm,
        prompt=PromptTemplate(template=refining_template, input_variables=["text"]),
    ),
    "todo": LLMChain(
        llm=llm, prompt=PromptTemplate(template=todo_template, input_variables=["text"])
    ),
    "summary": LLMChain(
        llm=llm,
        prompt=PromptTemplate(template=summary_template, input_variables=["text"]),
    ),
}
# %%
# MongoDB Atlas connection
client = MongoClient(MONGODB_URI)
db = client["renai"]
collection = db["events"]

db_two = client["realtime"]
todos_collection = db_two["todos"]
summary_collection = db_two["summary"]
transcript_collection = db_two["transcript"]

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# %%
# Send a ping to confirm a successful connection
try:
    client.admin.command("ping")
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# %%
lm = dspy.LM("gemini/gemini-2.0-flash-exp", api_key=GEMINI_API_KEY)
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
    entities: List[Entity] = dspy.InputField(
        desc="a list of entities and their metadata"
    )
    current_date: str = dspy.InputField(
        desc="the current date to convert relative dates like 'today', 'yesterday', 'tomorrow' to actual dates"
    )

    events: List[Event] = dspy.OutputField(
        desc="a list of events being talked about, either happening during the meeting or being referenced to, should NOT include events to happen in the future, and their metadata with fields: action(What Happened), type, date (convert relative dates like 'today', 'yesterday', 'tomorrow' to actual dates), location, entities (fetched from input)"
    )


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
        current_date = datetime.now().strftime("%Y-%m-%d")
        events = self.cot2(
            text=text, speaker=speaker, entities=entities, current_date=current_date
        )
        return events


class RAG(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought("context, question -> response")

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
                proper_case_key = key.replace("_", " ").title().replace(" ", "")
                context += f"{proper_case_key}: {value}\n"
            corpus.append(context)

        embedder = dspy.Embedder("gemini/text-embedding-004")
        search = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=5)
        context = search(question).passages
        return self.respond(context=context, question=question)


# %%
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
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


# @app.post("/upload_audio")
# async def upload_audio(audio: UploadFile = File(...)):
#     try:
#         # Create recordings directory if it doesn't exist
#         os.makedirs("recordings", exist_ok=True)

#         # Generate filename with timestamp
#         timestamp = time.strftime("%Y%m%d-%H%M%S")
#         filename = f"meeting_{timestamp}.wav"
#         filepath = os.path.join("recordings", filename)

#         # Save the uploaded file
#         with open(filepath, "wb") as buffer:
#             contents = await audio.read()
#             buffer.write(contents)

#         # Process with AssemblyAI
#         config = aai.TranscriptionConfig(speaker_labels=True)
#         transcription = aai.Transcriber().transcribe(filepath, config=config)

#         diarized_text = ""
#         segments = []
#         for utterance in transcription.utterances:
#             speaker = f"Speaker {utterance.speaker}"
#             text = utterance.text
#             diarized_text += f"{speaker}: {text}\n"
#             segments.append(
#                 {
#                     "speaker": speaker,
#                     "text": text,
#                     "start": utterance.start,
#                     "end": utterance.end,
#                 }
#             )

#         if not diarized_text.strip():
#             raise HTTPException(status_code=500, detail="Empty transcript generated")

#         # Generate summary and todos
#         summary = chains["summary"].invoke({"text": diarized_text}).get("text", "")
#         todos_output = chains["todo"].invoke({"text": diarized_text})

#         try:
#             todos = json.loads(todos_output.get("text", ""))
#         except json.JSONDecodeError as e:
#             raise HTTPException(status_code=500, detail="Invalid JSON returned by LLM")

#         summary_html = markdown.markdown(summary)

#         return {
#             "success": True,
#             "transcript": segments,
#             "summary": summary_html,
#             "todos": todos,
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_audio")
async def upload_audio(audio: UploadFile = File(...)):
    try:
        logger.info(f"Starting audio upload process for file: {audio.filename}")

        # Create recordings directory if it doesn't exist
        os.makedirs("recordings", exist_ok=True)
        logger.info("Created recordings directory")

        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"meeting_{timestamp}.wav"
        filepath = os.path.join("recordings", filename)
        logger.info(f"Generated filepath: {filepath}")

        try:
            # Save the uploaded file
            with open(filepath, "wb") as buffer:
                contents = await audio.read()
                buffer.write(contents)
            logger.info("Successfully saved audio file")
        except Exception as e:
            logger.error(f"Failed to save audio file: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Failed to save audio file: {str(e)}"
            )

        try:
            # Process with AssemblyAI
            logger.info("Starting AssemblyAI transcription")
            config = aai.TranscriptionConfig(speaker_labels=True)
            transcription = aai.Transcriber().transcribe(filepath, config=config)
            logger.info("Completed AssemblyAI transcription")
        except Exception as e:
            logger.error(f"AssemblyAI transcription failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Transcription failed: {str(e)}"
            )

        # Process transcription results
        diarized_text = ""
        segments = []
        try:
            for utterance in transcription.utterances:
                speaker = f"Speaker {utterance.speaker}"
                text = utterance.text
                diarized_text += f"{speaker}: {text}\n"
                segments.append(
                    {
                        "speaker": speaker,
                        "text": text,
                        "start": utterance.start,
                        "end": utterance.end,
                    }
                )
            logger.info("Successfully processed transcription segments")
        except Exception as e:
            logger.error(f"Failed to process transcription segments: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Failed to process transcription: {str(e)}"
            )

        if not diarized_text.strip():
            logger.error("Empty transcript generated")
            raise HTTPException(status_code=500, detail="Empty transcript generated")

        try:
            # Generate summary and todos
            logger.info("Generating summary")
            summary = chains["summary"].invoke({"text": diarized_text}).get("text", "")
            logger.info("Generating todos")
            todos_output = chains["todo"].invoke({"text": diarized_text})
        except Exception as e:
            logger.error(f"Failed to generate summary or todos: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Failed to generate summary or todos: {str(e)}"
            )

        try:
            todos_data = json.loads(todos_output.get("text", ""))

            # Add metadata to todos
            meeting_metadata = {
                "meeting_id": timestamp,
                "created_at": datetime.now(),
                "audio_file": filename,
            }

            # Store todos
            todos_for_db = []
            for todo in todos_data.get("todos", []):
                todo_with_metadata = {
                    **todo,
                    **meeting_metadata,
                    "updated_at": datetime.now(),
                }
                todos_for_db.append(todo_with_metadata)

            # Store summary
            summary_document = {
                **meeting_metadata,
                "content": summary,
                "updated_at": datetime.now(),
            }

            # Store transcript
            transcript_document = {
                **meeting_metadata,
                "full_text": diarized_text,
                "segments": segments,
                "updated_at": datetime.now(),
            }

            # Insert all documents into their respective collections
            if todos_for_db:
                logger.info(f"Inserting {len(todos_for_db)} todos into MongoDB")
                todos_collection.insert_many(todos_for_db)
                logger.info("Successfully inserted todos into MongoDB")

            logger.info("Inserting summary into MongoDB")
            summary_collection.insert_one(summary_document)
            logger.info("Successfully inserted summary into MongoDB")

            logger.info("Inserting transcript into MongoDB")
            transcript_collection.insert_one(transcript_document)
            logger.info("Successfully inserted transcript into MongoDB")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON returned by LLM: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail="Invalid JSON returned by LLM")
        except Exception as e:
            logger.error(f"Failed to store data in MongoDB: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Failed to store data: {str(e)}"
            )

        summary_html = markdown.markdown(summary)

        logger.info("Successfully completed audio upload and processing")
        return {
            "success": True,
            "transcript": segments,
            "summary": summary_html,
            "todos": todos_data,
            "meeting_id": timestamp,
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in upload_audio: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/todos", response_model=List[dict])
def get_todos():
    todos = list(todos_collection.find({}, {"_id": 0}))
    return todos


@app.get("/transcript", response_model=List[dict])
def get_transcripts():
    transcripts = list(transcript_collection.find({}, {"_id": 0}))
    return transcripts


@app.get("/summary", response_model=List[dict])
def get_summary():
    summaries = list(summary_collection.find({}, {"_id": 0}))
    return summaries


@app.post("/query-rag", response_model=str)
def query_rag(request: QueryRequest):
    try:
        responses = rag(question=request.question)
        return responses.response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
