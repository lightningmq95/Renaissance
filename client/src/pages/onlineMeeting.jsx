import React, { useState } from "react";

const OnlineMeeting = () => {
  const [videoFile, setVideoFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState("");

  // Handle file upload
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith("video/")) {
      setVideoFile(file); // Set the selected video file
    } else {
      alert("Please upload a valid video file.");
    }
  };

  // Handle video upload to FastAPI
  const handleUpload = async () => {
    if (!videoFile) {
      alert("Please select a video file.");
      return;
    }

    setUploading(true);
    const formData = new FormData();
    formData.append("video", videoFile);

    try {
      const response = await fetch("http://localhost:8000/upload-video", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setMessage("Video uploaded successfully.");
      } else {
        setMessage("Failed to upload video.");
      }
    } catch (error) {
      setMessage("An error occurred during upload.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div>
      <div className="pt-10">
        <h1 className="text-center text-3xl font-bold">
          Upload Video For Transcription
        </h1>
      </div>
      <div className="flex flex-col items-center justify-center h-[calc(100vh-300px)]">
        <div className="text-center flex flex-col">
          {/* File upload button */}
          <label
            htmlFor="video-upload"
            className="bg-blue-600 text-white font-bold py-3 px-6 rounded-lg shadow-lg cursor-pointer hover:bg-blue-700 transition"
          >
            Upload Video
          </label>
          <input
            id="video-upload"
            type="file"
            accept="video/*"
            onChange={handleFileChange}
            className="hidden"
          />

          {/* Upload button */}
          <button
            onClick={handleUpload}
            className="mt-4 bg-green-600 text-white py-2 px-6 rounded-lg"
            disabled={uploading}
          >
            {uploading ? "Uploading..." : "Submit Video"}
          </button>

          {/* Message */}
          {message && <p className="mt-4 text-lg">{message}</p>}
        </div>
      </div>
    </div>
  );
};

export default OnlineMeeting;
