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
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import assemblyai as aai
import time
import itertools

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MONGODB_URI = os.getenv('MONGODB_URI')
AIA_API_KEY = os.getenv('AIA_API_KEY')

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

class TodoResponse(BaseModel):
    task: str
    priority: str
    deadline: str

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
        collections = ['events', 'todos']
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
                    proper_case_key = key.replace('_', ' ').title().replace(' ', '')
                    context += f"{proper_case_key}: {value}\n"
                corpus.append(context)

        records = collection.find()
            
        embedder = dspy.Embedder('gemini/text-embedding-004')
        search = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=5)
        context = search(question).passages
        return self.respond(context=context, question=question)

def milliseconds_to_minutes(milliseconds):
    return milliseconds / (1000 * 60)

def batch_list(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

aai.settings.api_key = AIA_API_KEY

# TaskExtractor class (unchanged from provided logic)
class TaskExtractor:
    def __init__(self, gemini_api_key):
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-pro')
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
        return [utterances[i:i + batch_size] for i in range(0, len(utterances), batch_size)]

    def process_transcript(self, utterances_info):
        all_tasks = []
        utterance_batches = self.batch_utterances(utterances_info)
        for batch in utterance_batches:
            batch_text = "\n".join([
                f"Speaker {u['speaker']} ({u['start']:.2f} - {u['end']:.2f}): {u['text']}" 
                for u in batch
            ])
            tasks = self.extract_tasks(batch_text)
            for task in tasks:
                matching_utterance = None
                max_overlap = 0
                for utterance in batch:
                    task_words = set(task['task'].lower().split())
                    utterance_words = set(utterance['text'].lower().split())
                    overlap = len(task_words.intersection(utterance_words))
                    if overlap > max_overlap:
                        max_overlap = overlap
                        matching_utterance = utterance
                if matching_utterance:
                    task.update({
                        "speaker": matching_utterance['speaker'],
                        "timestamp_start": matching_utterance['start'],
                        "timestamp_end": matching_utterance['end']
                    })
                    all_tasks.append(task)
            time.sleep(4)
        return all_tasks

    def extract_tasks(self, text_batch):
        self._rate_limit()
        prompt = f"""
        Extract action items and tasks from the following conversation text...
        {text_batch}
        """
        try:
            response = self.model.generate_content(prompt)
            json_str = response.text.strip('```json').strip('```')
            return json.loads(json_str)
        except Exception as e:
            print(f"Error: {e}")
            return []

def format_todo_list(tasks):
    """Format tasks into a readable todo list with optional timestamp handling"""
    if not tasks:
        return "No tasks found in the transcript."
        
    formatted_list = []
    
    # Sort tasks by priority (high -> medium -> low)
    priority_order = {"high": 0, "medium": 1, "low": 2}
    sorted_tasks = sorted(tasks, key=lambda x: priority_order.get(x.get('priority', 'low'), 3))
    
    for task in sorted_tasks:
        # Build the task string components
        deadline_str = f" (Deadline: {task['deadline']})" if task.get('deadline') else ""
        priority_str = f"[{task['priority'].upper()}]" if task.get('priority') else ""
        
        # Only add timestamp and speaker if they exist
        speaker_str = f"Speaker {task['speaker']}: " if task.get('speaker') else ""
        
        if task.get('timestamp_start') is not None and task.get('timestamp_end') is not None:
            timestamp_str = f"[{task['timestamp_start']:.2f} - {task['timestamp_end']:.2f}]"
        else:
            timestamp_str = ""
        
        todo_item = f"- {speaker_str}{task['task']} {priority_str}{deadline_str} {timestamp_str}".strip()
        formatted_list.append(todo_item)
    
    return "\n".join(formatted_list)

def process_audio_to_tasks(audio_file_path, aai_api_key, gemini_api_key):
    """Process audio file to extract tasks with improved error handling"""
    try:
        # Set up AssemblyAI
        aai.settings.api_key = aai_api_key
        
        # Configure transcription
        config = aai.TranscriptionConfig(speaker_labels=True)
        
        print("\n=== Starting Audio Transcription ===")
        transcript = aai.Transcriber().transcribe(audio_file_path, config)

        print("\n=== Full Transcript Text ===")
        print(transcript.text)
        
        print("\n=== Processing Utterances ===")
        utterances_info = []
        
        for utterance in transcript.utterances:
            # Convert timestamps to minutes
            start_minutes = utterance.start / (1000 * 60)
            end_minutes = utterance.end / (1000 * 60)
            
            utterance_data = {
                "speaker": utterance.speaker,
                "text": utterance.text,
                "start": start_minutes,
                "end": end_minutes
            }
            utterances_info.append(utterance_data)
            
            print(f"\nUtterance Details:")
            print(f"Speaker: {utterance.speaker}")
            print(f"Text: {utterance.text}")
            print(f"Time: {start_minutes:.2f} - {end_minutes:.2f}")

        if not utterances_info:
            print("No utterances found in transcript")
            return []

        # Create formatted transcript
        formatted_transcript = "\n".join([
            f"{u['start']:.2f}\n"
            f"Speaker {u['speaker']}: {u['text']}\n"
            f"{u['end']:.2f}\n"
            for u in utterances_info
        ])
        
        print("\n=== Extracting Tasks ===")
        extractor = TaskExtractor(gemini_api_key)
        
        # Try batch processing first
        print("\nTrying batched processing...")
        tasks = extractor.process_transcript(utterances_info)
        
        # If batch processing finds no tasks, try direct processing
        if not tasks:
            print("\nBatch processing found no tasks. Trying direct processing...")
            direct_tasks = extractor.extract_tasks(formatted_transcript)
            
            # If direct processing found tasks, add timing information
            if direct_tasks:
                print("\nDirect processing found tasks. Adding timing information...")
                tasks = []
                for task in direct_tasks:
                    # Find the most relevant utterance for this task
                    best_match = None
                    max_overlap = 0
                    
                    for utterance in utterances_info:
                        task_words = set(task['task'].lower().split())
                        utterance_words = set(utterance['text'].lower().split())
                        overlap = len(task_words.intersection(utterance_words))
                        
                        if overlap > max_overlap:
                            max_overlap = overlap
                            best_match = utterance
                    
                    if best_match:
                        task.update({
                            "speaker": best_match['speaker'],
                            "timestamp_start": best_match['start'],
                            "timestamp_end": best_match['end']
                        })
                        tasks.append(task)
                    else:
                        # If no matching utterance found, add task without timing info
                        tasks.append(task)

        return tasks
            
    except Exception as e:
        print(f"\n=== Error in Processing ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nTraceback:")
        import traceback
        traceback.print_exc()
        return []

def parse_relative_date(relative_date_str):
    """Parse relative date strings like 'today', 'tomorrow', 'next week' into actual dates"""
    parsed_date = dateparser.parse(relative_date_str)
    if parsed_date:
        return parsed_date.strftime('%Y-%m-%d')
    return relative_date_str

def save_tasks_to_json(tasks, output_file='todo_list.json'):
    """
    Save tasks to a JSON file with standardized format
    Args:
        tasks: List of task dictionaries
        output_file: Path to output JSON file
    """
    if not tasks:
        return False
        
    # Standardize the task format for JSON output
    formatted_tasks = []
    for task in tasks:
        deadline = task.get('deadline', None)
        if deadline:
            deadline = parse_relative_date(deadline)
        
        formatted_task = {
            "task": task.get('task', ''),
            "deadline": deadline,
            "priority": task.get('priority', 'low'),
            "timestamp": {
            "start": round(task.get('timestamp_start', 0), 2),
            "end": round(task.get('timestamp_end', 0), 2)
            }
        }
        formatted_tasks.append(formatted_task)
    
    # Save to JSON file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_tasks, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        return False
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
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        events_dicts = [event.dict() for event in event_list]
        collection = db['events']
        collection.insert_many(events_dicts)
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

@app.get("/get-todos", response_model=List[TodoResponse])
def get_todos():
    try:
        collection = db['todos']
        todos = collection.find({"deadline": {"$exists": True, "$ne": None}})
        todo_responses = []

        for record in todos:
            todo_response = {
                'task':str(record.get('task', '')),
                'deadline':str(record.get('deadline', '')),
                'priority':str(record.get('priority', '')),
            }
            todo_responses.append(todo_response)

        print(todo_responses)
        return todo_responses
        # print(corpus)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-audio-tasks")
def process_audio_tasks(audio_file_path: str):
    try:
        extractor = TaskExtractor(GEMINI_API_KEY)
        aai.settings.api_key = AIA_API_KEY
        config = aai.TranscriptionConfig(speaker_labels=True)
        transcript = aai.Transcriber().transcribe(audio_file_path, config)
        utterances_info = [
            {
                "speaker": u.speaker,
                "text": u.text,
                "start": u.start,
                "end": u.end
            } for u in transcript.utterances
        ]
        tasks = extractor.process_transcript(utterances_info)
        if save_tasks_to_json(tasks):
            return {"tasks": tasks, "message": "Tasks successfully saved to todo_list.json"}
        else:
            return {"tasks": tasks, "message": "Failed to save tasks to JSON file"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
