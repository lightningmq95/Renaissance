import React, { createContext, useState, useEffect } from "react";

export const TranscriptContext = createContext();

export const TranscriptContextProvider = ({ children }) => {
  const [transcript, setTranscript] = useState(() => {
    const storedData = localStorage.getItem("transcript");
    return storedData ? JSON.parse(storedData) : [];
  });

  useEffect(() => {
    localStorage.setItem("transcript", JSON.stringify(transcript));
  }, [transcript]);

  return (
    <TranscriptContext.Provider value={{ transcript, setTranscript }}>
      {children}
    </TranscriptContext.Provider>
  );
};
