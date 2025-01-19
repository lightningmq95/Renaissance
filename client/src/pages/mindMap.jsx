import React, { useContext, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { TranscriptContext } from "../../context/transcriptContext";
import Markmap from "./Markup";

const MindMap = () => {
  const [searchParams] = useSearchParams();
  const created_at = searchParams.get("created_at");
  const { transcript } = useContext(TranscriptContext);
  const [mindmapMarkdown, setMindmapMarkdown] = useState(null);
  const [generationStatus, setGenerationStatus] = useState("idle"); // "idle", "generating", "generated"

  const filteredTranscript = transcript
    ? transcript.filter((t) => t.created_at === created_at)
    : [];

  if (filteredTranscript.length === 0) {
    return <div>No transcript found for this meeting</div>;
  }

  const transcriptRecord = filteredTranscript[0];

  const generateMindmap = async () => {
    setGenerationStatus("generating");

    try {
      const response = await fetch("http://localhost:8000/generate_mindmap", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ full_text: transcriptRecord.full_text }),
      });

      if (!response.ok) {
        throw new Error("Failed to generate mindmap");
      }

      const data = await response.json();
      setMindmapMarkdown(data.markdown);
      setGenerationStatus("generated");
    } catch (error) {
      console.error("Error generating mindmap:", error);
      setGenerationStatus("idle"); // Reset status on error
    }
  };

  const getButtonText = () => {
    if (generationStatus === "generating") return "Generating...";
    if (generationStatus === "generated") return "Generated";
    return "Generate";
  };

  return (
    <div className="p-5">
      <div className="text-xl font-bold">
        Meeting Date: {new Date(transcriptRecord.created_at).toLocaleString()}
      </div>
      <h1 className="text-lg font-bold">Transcript: </h1>
      <div>{transcriptRecord.full_text}</div>
      <button
        className="bg-slate-500 rounded-full px-3 py-2 text-white mt-5"
        onClick={generateMindmap}
        disabled={generationStatus === "generating"} // Disable button while generating
      >
        {getButtonText()}
      </button>
      <br />
      <br />
      {mindmapMarkdown && <Markmap markdown={mindmapMarkdown} />}
    </div>
  );
};

export default MindMap;
