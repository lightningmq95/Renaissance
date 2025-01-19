import React, { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm"; // For GitHub-flavored Markdown
import { useContext } from "react";
import { TranscriptContext } from "../../context/transcriptContext";
import { useNavigate } from "react-router-dom";

const Meetings = () => {
  const [cards, setCards] = useState([]);
  const [expandedCard, setExpandedCard] = useState(null);
  const { transcript, setTranscript } = useContext(TranscriptContext);
  const navigate = useNavigate();

  // All of the data from all of the collections are fetched to show the recorded meetings
  useEffect(() => {
    const fetchAndOrganizeData = async () => {
      try {
        // Fetch data from `todos` and `summary` endpoints
        const todosResponse = await fetch("http://localhost:8000/todos");
        const summaryResponse = await fetch("http://localhost:8000/summary");
        const transcriptResponse = await fetch(
          "http://localhost:8000/transcript"
        );

        if (!todosResponse.ok || !summaryResponse.ok) {
          throw new Error("Failed to fetch data");
        }

        const todosData = await todosResponse.json();
        const summaryData = await summaryResponse.json();
        const transcriptData = await transcriptResponse.json();

        setTranscript(transcriptData);

        // console.log(todosData);
        // console.log(summaryData);

        // Organize data by matching `created_at` fields
        const organizedData = summaryData.map((summary) => {
          // Filter all todos that match the current summary's `created_at`
          const matchingTodos = todosData.filter(
            (todo) => todo.created_at === summary.created_at
          );

          // Return a single object with grouped todos and the corresponding summary
          return {
            created_at: summary.created_at,
            summary: summary.content, // Add summary content
            todos: matchingTodos.map((todo) => ({
              description: todo.description,
              assignee: todo.assignee,
              deadline: todo.deadline,
              status: todo.status,
            })), // Include all relevant fields from todos
          };
        });

        // Sort data by `created_at` (newest first)
        const sortedData = organizedData.sort(
          (a, b) => new Date(b.created_at) - new Date(a.created_at)
        );

        console.log(sortedData);

        setCards(sortedData);
      } catch (error) {
        console.error("Error fetching or organizing data:", error);
      }
    };

    fetchAndOrganizeData();
  }, []);

  const toggleExpand = (index) => {
    setExpandedCard(expandedCard === index ? null : index); // Toggle expanded state
  };

  const handleGenerateMindMap = (card) => {
    navigate(`/mindmap?created_at=${encodeURIComponent(card.created_at)}`);
  };

  return (
    <div className="bg-gray-100 min-h-screen p-8">
      <h1 className="text-2xl font-bold text-center mb-6">Meetings</h1>

      <div className="space-y-4">
        {cards.map((card, index) => (
          <div
            key={index}
            className="bg-white p-6 rounded-lg shadow-lg border border-gray-200 cursor-pointer"
            onClick={() => toggleExpand(index)}
          >
            {/* Card Header */}
            <div className="flex justify-between items-center">
              <h3 className="text-lg font-semibold text-gray-800">
                {new Date(card.created_at).toLocaleString()}
              </h3>
              <button
                className="bg-slate-500 rounded-full px-3 py-2 text-white"
                onClick={() => handleGenerateMindMap(card)}
              >
                Generate MindMap
              </button>
            </div>

            {/* Expanded Content */}
            {expandedCard === index && (
              <div className="mt-4">
                {/* Summary */}
                <div className="mb-4">
                  <h4 className="text-md font-bold mb-2">Summary:</h4>
                  <ReactMarkdown
                    className="text-gray-600"
                    remarkPlugins={[remarkGfm]}
                  >
                    {card.summary}
                  </ReactMarkdown>
                </div>

                {/* Todos */}
                <div>
                  <h4 className="text-md font-bold mb-2">Todos:</h4>
                  {card.todos.length > 0 ? (
                    <ul className="list-disc list-inside space-y-2">
                      {card.todos.map((todo, index) => (
                        <div key={index} className="text-gray-600">
                          <p>
                            <strong>Description:</strong> {todo.description}
                          </p>
                          <p>
                            <strong>Assignee:</strong> {todo.assignee}
                          </p>
                          <p>
                            <strong>Deadline:</strong>{" "}
                            {todo.deadline === null
                              ? "Not Mentioned"
                              : todo.deadline}
                          </p>
                          <p>
                            <strong>Status:</strong> {todo.status}
                          </p>
                        </div>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-gray-600">No todos available.</p>
                  )}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default Meetings;
