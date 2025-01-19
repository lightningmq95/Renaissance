import React from "react";
import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import RecordRTC from "recordrtc";

const RealTime = () => {
  const [recorder, setRecorder] = useState(null);
  const [timer, setTimer] = useState(null);
  const [seconds, setSeconds] = useState(0);
  const [isRecording, setIsRecording] = useState(false);
  const [status, setStatus] = useState("");
  const [transcript, setTranscript] = useState([]);
  const [summary, setSummary] = useState("");
  const [todos, setTodos] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    if (isRecording && !timer) {
      const interval = setInterval(() => {
        setSeconds((prev) => prev + 1);
      }, 1000);
      setTimer(interval);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [isRecording, timer]);

  useEffect(() => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    document.getElementById("timer").textContent = `${minutes
      .toString()
      .padStart(2, "0")}:${remainingSeconds.toString().padStart(2, "0")}`;
  }, [seconds]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const newRecorder = new RecordRTC(stream, {
        type: "audio",
        mimeType: "audio/wav",
        recorderType: RecordRTC.StereoAudioRecorder,
      });

      newRecorder.startRecording();
      setRecorder(newRecorder);
      setIsRecording(true);
      setStatus("Recording...");
    } catch (err) {
      console.error("Error starting recording:", err);
      setStatus("Error: Could not start recording");
    }
  };

  const stopRecording = async () => {
    setStatus("Processing...");
    setIsRecording(false);
    recorder.stopRecording(async () => {
      const blob = recorder.getBlob();
      const formData = new FormData();
      formData.append("audio", blob, "recording.wav");

      try {
        const response = await fetch("http://127.0.0.1:8000/upload_audio", {
          method: "POST",
          body: formData,
        });

        const data = await response.json();
        if (data.success) {
          displayResults(data);
          setStatus("Processing complete");
        } else {
          setStatus("Error: " + data.error);
        }
      } catch (err) {
        console.error("Error uploading recording:", err);
        setStatus("Error uploading recording");
      }
    });
  };

  const displayResults = (data) => {
    setTranscript(data.transcript);
    setSummary(data.summary);
    setTodos(data.todos.todos);
  };

  return (
    <div className="flex justify-center h-screen w-screen">
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold mb-8 text-center">
          Meeting Recorder
        </h1>

        <div className="flex justify-center space-x-4 mb-8">
          <button
            onClick={startRecording}
            disabled={isRecording}
            className="bg-green-500 text-white px-6 py-2 rounded-lg hover:bg-green-600"
          >
            Start Recording
          </button>
          <button
            onClick={stopRecording}
            disabled={!isRecording}
            className="bg-red-500 text-white px-6 py-2 rounded-lg hover:bg-red-600"
          >
            Stop Recording
          </button>
        </div>

        <div id="status" className="text-center mb-8 text-gray-600">
          {status}
        </div>

        <div id="timer" className="text-center mb-8 text-2xl font-mono"></div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white p-6 rounded-lg shadow overflow-y-auto h-[28rem]">
            <h2 className="text-md font-bold mb-4">Transcript</h2>
            <div id="transcript" className="space-y-2">
              {transcript.map((segment, index) => (
                <p key={index} className="mb-2">
                  <span className="font-bold">{segment.speaker}</span>:{" "}
                  {segment.text}
                </p>
              ))}
            </div>
          </div>

          <div className="space-y-2 w-[46rem]">
            <div className="bg-white p-6 rounded-lg shadow overflow-y-auto h-[13.75rem]">
              <h2 className="text-md font-bold mb-4">Summary</h2>
              <div id="summary" dangerouslySetInnerHTML={{ __html: summary }} />
            </div>
            <div className="bg-white p-6 rounded-lg shadow overflow-y-auto h-[13.75rem]">
              <h2 className="text-md font-bold mb-4">Todo List</h2>
              <div id="todos">
                {todos.map((todo, index) => (
                  <div key={index} className="mb-4 p-4 bg-gray-50 rounded">
                    <p className="font-bold">{todo.description}</p>
                    <p className="text-sm">Assignee: {todo.assignee}</p>
                    <p className="text-sm">
                      Deadline: {todo.deadline || "Not specified"}
                    </p>
                    <p className="text-sm">Mentioned by: {todo.mentioned_by}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RealTime;
