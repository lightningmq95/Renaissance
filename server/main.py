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
import dateparser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from fastapi.middleware.cors import CORSMiddleware
from fastapi import File, UploadFile

import google.generativeai as genai
import assemblyai as aai
import time
import itertools

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


def milliseconds_to_minutes(milliseconds):
    return milliseconds / (1000 * 60)


def batch_list(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


aai.settings.api_key = ASSEMBLYAI_API_KEY


# TaskExtractor class (unchanged from provided logic)
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


def format_todo_list(tasks):
    """Format tasks into a readable todo list with optional timestamp handling"""
    if not tasks:
        return "No tasks found in the transcript."

    formatted_list = []

    # Sort tasks by priority (high -> medium -> low)
    priority_order = {"high": 0, "medium": 1, "low": 2}
    sorted_tasks = sorted(
        tasks, key=lambda x: priority_order.get(x.get("priority", "low"), 3)
    )

    for task in sorted_tasks:
        # Build the task string components
        deadline_str = (
            f" (Deadline: {task['deadline']})" if task.get("deadline") else ""
        )
        priority_str = f"[{task['priority'].upper()}]" if task.get("priority") else ""

        # Only add timestamp and speaker if they exist
        speaker_str = f"Speaker {task['speaker']}: " if task.get("speaker") else ""

        if (
            task.get("timestamp_start") is not None
            and task.get("timestamp_end") is not None
        ):
            timestamp_str = (
                f"[{task['timestamp_start']:.2f} - {task['timestamp_end']:.2f}]"
            )
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
                "end": end_minutes,
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
        formatted_transcript = "\n".join(
            [
                f"{u['start']:.2f}\n"
                f"Speaker {u['speaker']}: {u['text']}\n"
                f"{u['end']:.2f}\n"
                for u in utterances_info
            ]
        )

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
                        task_words = set(task["task"].lower().split())
                        utterance_words = set(utterance["text"].lower().split())
                        overlap = len(task_words.intersection(utterance_words))

                        if overlap > max_overlap:
                            max_overlap = overlap
                            best_match = utterance

                    if best_match:
                        task.update(
                            {
                                "speaker": best_match["speaker"],
                                "timestamp_start": best_match["start"],
                                "timestamp_end": best_match["end"],
                            }
                        )
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
        return parsed_date.strftime("%Y-%m-%d")
    return relative_date_str



def get_transcript_text(video_file_path, mode):
    # aai.settings.api_key = AIA_API_KEY
    # config = aai.TranscriptionConfig(speaker_labels=True)
    # transcript = aai.Transcriber().transcribe(video_file_path, config)
    transcript = [
  {
    "start_minutes": 0.01,
    "end_minutes": 0.28,
    "speaker": "A",
    "text": "Hello, everyone. Thank you guys for coming to our weekly student success meeting. And let's just get started. So I have our list of chronically absent students here. And I've been noticing a troubling trend. A lot of students are skipping on Fridays. Does anyone have any idea what's going on?"
  },
  {
    "start_minutes": 0.29,
    "end_minutes": 0.43,
    "speaker": "C",
    "text": "I've heard some of my mentees talking about how it's really hard to get out of bed on Fridays. It might be good if we did something like a pancake breakfast to encourage them to come."
  },
  {
    "start_minutes": 0.44,
    "end_minutes": 0.49,
    "speaker": "A",
    "text": "I think that's a great idea. Let's try that next week."
  },
  {
    "start_minutes": 0.5,
    "end_minutes": 0.74,
    "speaker": "D",
    "text": "It might also be because a lot of students have been getting sick now that it's getting colder outside. I've had a number of students come by my office with symptoms like sniffling and coughing. We should put up posters with tips for not getting sick since it's almost flu season. Like, you know, wash your hands after the bathroom, stuff like that."
  },
  {
    "start_minutes": 0.75,
    "end_minutes": 1.0,
    "speaker": "A",
    "text": "I think that's a good idea and it'll be a good reminder for the teachers as well. So one other thing I wanted to talk about. There's a student I've noticed here, John Smith. He's missed seven days already and it's only November. Does anyone have an idea what's going on with him?"
  },
  {
    "start_minutes": 1.0,
    "end_minutes": 1.22,
    "speaker": "C",
    "text": "I might be able to fill in the gaps there. I talked to John today and he's really stressed out. He's been dealing with helping his parents take care of his younger siblings during the day. It might actually be a good idea if he spoke to the guidance counselor a little bit."
  },
  {
    "start_minutes": 1.23,
    "end_minutes": 1.52,
    "speaker": "B",
    "text": "I can talk to John today if you want to send him to my office after you meet with him. It's a lot to deal with for a middle schooler. Great, thanks. And I can help out with the family's childcare needs. I'll look for some free or low cost resources in the community to share with John and he can share them with his family."
  },
  {
    "start_minutes": 1.52,
    "end_minutes": 1.62,
    "speaker": "A",
    "text": "Great. Well, some really good ideas here today. Thanks for coming. And if no one has anything else, I think we can wrap up."
  }
]
    return transcript

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

    # Save to JSON file
    try:
        with open(output_file, "w", encoding="utf-8") as f:
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
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
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
        collection = db["events"]
        collection.insert_many(events_dicts)
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


@app.get("/get-todos", response_model=List[TodoResponse])
def get_todos():
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


@app.post("/process-audio-tasks")
def process_audio_tasks(audio_file_path: str):
    try:
        extractor = TaskExtractor(GEMINI_API_KEY)
        aai.settings.api_key = ASSEMBLYAI_API_KEY
        config = aai.TranscriptionConfig(speaker_labels=True)
        transcript = aai.Transcriber().transcribe(audio_file_path, config)
        utterances_info = [
            {"speaker": u.speaker, "text": u.text, "start": u.start, "end": u.end}
            for u in transcript.utterances
        ]
        tasks = extractor.process_transcript(utterances_info)
        if save_tasks_to_json(tasks):
            return {
                "tasks": tasks,
                "message": "Tasks successfully saved to todo_list.json",
            }
        else:
            return {"tasks": tasks, "message": "Failed to save tasks to JSON file"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/upload-video")
async def upload_video(mode: int, file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        file_location = f"videos/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(file.file.read())
        
        # Process the video file as needed
        # For now, just print a message
        transcript = get_transcript_text(file_location, mode)
        collection = db['transcripts']
        meeting_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        collection.insert_one({"segments": transcript, "meeting_id": meeting_id})
        utterances_info = [
            {
                "speaker": u['speaker'],
                "text": u['text'],
                "start": u['start_minutes'],
                "end": u['end_minutes']
            } for u in transcript
        ]
        # print(utterances_info)  
        event_list = []
        for req in transcript:
            response = knowledge_extraction(text=req['text'], speaker=req['speaker'])
            event_list.extend(response.events)
        events_dicts = [event.dict() for event in event_list]
        collection = db['events']
        collection.insert_many(events_dicts)

        extractor = TaskExtractor(GEMINI_API_KEY)
        tasks_list = extractor.process_transcript(utterances_info)
        collection = db['todos']
        print("Tasks: ", tasks_list)
        collection.insert_many(tasks_list)


 
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
