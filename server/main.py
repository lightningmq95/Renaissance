# %%
import time
import markdown
import dspy
import json
import os
import spacy
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
from fastapi import UploadFile, File, Form
import traceback
import dateparser
import logging
from typing import Optional



from fastapi.middleware.cors import CORSMiddleware
from fastapi import File, UploadFile

import google.generativeai as genai
import assemblyai as aai
import time

load_dotenv()

import os
import numpy as np
import cv2
import easyocr
import assemblyai as aai
from moviepy.video.io.VideoFileClip import VideoFileClip
from collections import Counter
from dotenv import load_dotenv


load_dotenv()
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
reader = easyocr.Reader(['en'])
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
llm = GoogleGenerativeAI(
    model="gemini-pro", google_api_key=GEMINI_API_KEY, temperature=0
)
aai.settings.api_key = ASSEMBLYAI_API_KEY


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
# Define Data models to be used by DSPy
class Entity(BaseModel):
    entity: str
    type: str


class EventEntity(BaseModel):
    entity: str
    type: str
    role: str

class Event(BaseModel):
    action: str
    type: Optional[str] = None
    date: Optional[str] = None
    location: Optional[str] = None
    entities: Optional[List[EventEntity]] = []


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


class TodoResponse(BaseModel):
    task: str
    priority: str
    deadline: str


# %%
# Programming the ExtractEvents LLM Model
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
# Programming the KnowledgeExtraction module
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
        print(events)
        return events

# Programming the RAG Module
class RAG(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought("context, question -> response")

    def forward(self, question):
        # Fetch records from MongoDB
        collections = ["events", "todos"]
        corpus = []

        for collection_name in collections:
            collection = db[collection_name]
            records = collection.find()

            for record in records:
                context = ""
                for key, value in record.items():
                    if key == "_id":
                        continue
                    # Convert key to proper case
                    proper_case_key = key.replace("_", " ").title().replace(" ", "")
                    context += f"{proper_case_key}: {value}\n"
                corpus.append(context)

        records = collection.find()

        embedder = dspy.Embedder("gemini/text-embedding-004")
        search = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=5)
        context = search(question).passages
        return self.respond(context=context, question=question)

# TaskExtractor class
class TaskExtractor:
    def __init__(self, gemini_api_key):
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel("gemini-pro")
        self.request_count = 0
        self.last_request_time = 0
        self.RATE_LIMIT_REQUESTS = 60
        self.MIN_REQUEST_INTERVAL = 1

    def _rate_limit(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - time_since_last)
        self.last_request_time = time.time()
        self.request_count += 1
        if current_time - self.last_request_time >= 60:
            self.request_count = 0

    def batch_utterances(self, utterances: List, batch_size: int = 5):
        return [
            utterances[i : i + batch_size]
            for i in range(0, len(utterances), batch_size)
        ]

    def process_transcript(self, utterances_info):
        all_tasks = []
        utterance_batches = self.batch_utterances(utterances_info)
        for batch in utterance_batches:
            batch_text = "\n".join(
                [
                    f"Speaker {u['speaker']} ({u['start']:.2f} - {u['end']:.2f}): {u['text']}"
                    for u in batch
                ]
            )
            tasks = self.extract_tasks(batch_text)
            for task in tasks:
                matching_utterance = None
                max_overlap = 0
                for utterance in batch:
                    task_words = set(task["task"].lower().split())
                    utterance_words = set(utterance["text"].lower().split())
                    overlap = len(task_words.intersection(utterance_words))
                    if overlap > max_overlap:
                        max_overlap = overlap
                        matching_utterance = utterance
                if matching_utterance:
                    task.update(
                        {
                            "speaker": matching_utterance["speaker"],
                            "timestamp_start": matching_utterance["start"],
                            "timestamp_end": matching_utterance["end"],
                        }
                    )
                    all_tasks.append(task)
            time.sleep(4)
        return all_tasks

    def extract_tasks(self, text_batch):
        self._rate_limit()
        # prompt = f"""
        # Extract action items and tasks from the following conversation text...
        # {text_batch}
        # """
        prompt = """
        Extract action items and tasks from the following conversation text. 
        Look for any statements that imply something needs to be done, assignments given, or commitments made.
        Include both explicit tasks ("I'll do X") and implicit tasks ("We need to X", "X should be done").
        
        Provide results in JSON with these keys:
        - task: The task description (required)
        - deadline: The deadline mentioned, or null if not available
        - priority: Either 'high', 'medium', or 'low' based on urgency words and context
        
        Text: {text}
        
        Return an empty array [] if no tasks are found.
        Ensure the response is valid JSON.
        """.format(text=text_batch)
        try:
            response = self.model.generate_content(prompt)
            json_str = response.text.strip("```json").strip("```")
            return json.loads(json_str)
        except Exception as e:
            print(f"Error: {e}")
            return []

def parse_relative_date(relative_date_str):
    """Parse relative date strings like 'today', 'tomorrow', 'next week' into actual dates"""
    parsed_date = dateparser.parse(relative_date_str)
    if parsed_date:
        return parsed_date.strftime("%Y-%m-%d")
    return relative_date_str

def transcribe_audio(audio_file_path):
    config = aai.TranscriptionConfig(speaker_labels=True)
    transcript = aai.Transcriber().transcribe(audio_file_path, config)
    results = []
    for utterance in transcript.utterances:
        start = utterance.start / 1000
        end = utterance.end / 1000
        result = (start, end, utterance.speaker, utterance.text)
        results.append(result)
    return results

def process_contour_region(frame, x, y, w, h):
    roi = frame[y:y+h, x:x+w]
    results = reader.readtext(roi)
    text = ' '.join([result[1] for result in results])
    return text.strip() if text.strip() else "Unknown Speaker"

def get_transcript_text(video_file_path, mode):
    """
    Process video and return transcript with speaker identification
    Args:
        video_path (str): Path to video file
        meeting_type (str): '1' for Google Meet, '2' for Zoom
    Returns:
        list: List of dicts with speaker and transcript
    """
    cap = cv2.VideoCapture(video_file_path)
    transcription_results = transcribe_audio(video_file_path)
    transcript_list = []

    for start_time, end_time, speaker, text in transcription_results:
        text_counter = Counter()
        duration = end_time - start_time
        analysis_end = start_time + min(duration, 10)

        for t in np.arange(start_time, analysis_end, 0.5):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret:
                continue

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            if mode == '1':  # Google Meet
                mask = cv2.inRange(hsv, np.array([100,50,50]), np.array([130,255,255]))
            else:  # Zoom
                mask = cv2.inRange(hsv, np.array([40,50,50]), np.array([80,255,255]))

            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

            if len(cnts) > 0:
                area = max(cnts, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(area)
                speaker_name = process_contour_region(frame, x, y, w, h)
                text_counter[speaker_name] += 1

        most_common_speaker = text_counter.most_common(1)[0][0] if text_counter else "Unknown Speaker"
        
        transcript_dict = {"start_minutes": start_time, "end_minutes": end_time, "speaker": most_common_speaker, "text": text}
        transcript_list.append(transcript_dict)

    cap.release()
    cv2.destroyAllWindows()
    return transcript_list



# %%
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:5173",
    "http://localhost:8080",
]

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
# All Queries

@app.post("/extract-events", response_model=List[Event])
def extract_events(request: List[TranscriptRequest]):
    try:
        event_list = []
        for req in request:
            response = knowledge_extraction(text=req.text, speaker=req.speaker)
            event_list.extend(response.events)
        events_dicts = [event.dict() for event in event_list]
        collection = db["events"]
        collection.insert_many(events_dicts)
        return event_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    """
    Endpoint to query the RAG model with a question.

    Args:
        request (QueryRequest): The request containing the question.

    Returns:
        str: The response from the RAG model.
    """
    try:
        responses = rag(question=request.question)
        return responses.response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get-todos", response_model=List[TodoResponse])
def get_todos():
    """
    Endpoint to upload a video file, process it to extract transcript, events, and tasks,
    and save the results to the MongoDB database.

    Args:
        mode (int): The mode of the meeting (e.g., 1 for Google Meet, 2 for Zoom).
        file (UploadFile): The uploaded video file.

    Returns:
        JSON response with the status of the operation.
    """
    try:
        collection = db["todos"]
        todos = collection.find({"deadline": {"$exists": True, "$ne": None}})
        todo_responses = []

        for record in todos:
            todo_response = {
                "task": str(record.get("task", "")),
                "deadline": str(record.get("deadline", "")),
                "priority": str(record.get("priority", "")),
            }
            todo_responses.append(todo_response)

        print(todo_responses)
        return todo_responses
        # print(corpus)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/upload-video")
async def upload_video(mode: int = Form(...), file: UploadFile = File(...)):
    """
    Endpoint to upload a video file, process it to extract transcript, events, and tasks,
    and save the results to the MongoDB database.

    Args:
        mode (int): The mode of the meeting (1 for Google Meet, 2 for Zoom).
        file (UploadFile): The uploaded video file.

    Process:
        1. Save the uploaded video file to the server.
        2. Extract the transcript from the video file based on the meeting mode.
        3. Fetch events from the transcript and save them to the 'events' collection in MongoDB.
        4. Fetch tasks from the transcript and save them to the 'todos' collection in MongoDB.

    Returns:
        JSON response with the status of the operation.
    """
    try:
        # Save the uploaded file
        file_location = f"videos/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(file.file.read())
        
        # Process the video file as needed
        transcript = get_transcript_text(file_location, mode)
        # print("hello")
        collection = db['transcripts']
        meeting_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        collection.insert_one({"segments": transcript, "meeting_id": meeting_id})
        # print("wagwan")
        utterances_info = [
            {
                "speaker": u['speaker'],
                "text": u['text'],
                "start": u['start_minutes'],
                "end": u['end_minutes']
            } for u in transcript
        ]

        # Extract events from transcript 
        event_list = []
        for req in transcript:
            response = knowledge_extraction(text=req['text'], speaker=req['speaker'])
            event_list.extend(response.events)
        events_dicts = [event.dict() for event in event_list]
        print(event_list)
        collection = db['events']
        if events_dicts:
            collection.insert_many(events_dicts)

        # Extract tasks from transcript
        extractor = TaskExtractor(GEMINI_API_KEY)
        tasks_list = extractor.process_transcript(utterances_info)
        formatted_tasks = []
        for task in tasks_list:
            deadline = task.get("deadline", None)
            if deadline:
                deadline = parse_relative_date(deadline)

            formatted_task = {
                "task": task.get("task", ""),
                "deadline": deadline,
                "priority": task.get("priority", "low"),
                "timestamp": {
                    "start": round(task.get("timestamp_start", 0), 2),
                    "end": round(task.get("timestamp_end", 0), 2),
                },
            }
            formatted_tasks.append(formatted_task)
        collection = db['todos']
        if formatted_tasks:
            collection.insert_many(formatted_tasks)
 
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
