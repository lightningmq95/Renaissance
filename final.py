# %%
import os
import numpy as np
import cv2
import easyocr
import assemblyai as aai
from moviepy.video.io.VideoFileClip import VideoFileClip
from collections import Counter
from dotenv import load_dotenv

# %%
load_dotenv()
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
reader = easyocr.Reader(['en'])

# %%
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

# %%
def process_contour_region(frame, x, y, w, h):
    roi = frame[y:y+h, x:x+w]
    results = reader.readtext(roi)
    text = ' '.join([result[1] for result in results])
    return text.strip() if text.strip() else "Unknown Speaker"

# %%
def process_video_transcript(video_path, meeting_type):
    """
    Process video and return transcript with speaker identification
    Args:
        video_path (str): Path to video file
        meeting_type (str): '1' for Google Meet, '2' for Zoom
    Returns:
        list: List of dicts with speaker and transcript
    """
    cap = cv2.VideoCapture(video_path)
    transcription_results = transcribe_audio(video_path)
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
            
            if meeting_type == '1':  # Google Meet
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
        
        transcript_dict = {
            most_common_speaker: {
                "text": text,
                "start_time": start_time,
                "end_time": end_time
            }
        }
        transcript_list.append(transcript_dict)

    cap.release()
    cv2.destroyAllWindows()
    return transcript_list

# %%
process_video_transcript("samples/green.mp4", 2)


