import React, { createContext, useState } from "react";

export const SummaryContext = createContext();

export const SummaryContextProvider = ({ children }) => {
  const [summary, setSummary] = useState("");

  return (
    <SummaryContext.Provider value={{ summary, setSummary }}>
      {children}
    </SummaryContext.Provider>
  );
};
