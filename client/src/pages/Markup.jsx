import React, { useEffect, useRef } from "react";

const Markmap = ({ markdown }) => {
  const svgRef = useRef(null);
  const markmapInstanceRef = useRef(null); // Add ref to store markmap instance

  useEffect(() => {
    // Clear any existing content first
    if (markmapInstanceRef.current) {
      if (svgRef.current) {
        svgRef.current.innerHTML = "";
      }
      markmapInstanceRef.current = null;
    }

    const loadLibraries = async () => {
      try {
        await loadScript("https://cdn.jsdelivr.net/npm/d3@6");
        await loadScript(
          "https://cdn.jsdelivr.net/npm/markmap-lib@0.14.3/dist/browser/index.min.js"
        );
        await loadScript(
          "https://cdn.jsdelivr.net/npm/markmap-view@0.14.3/dist/index.min.js"
        );

        await new Promise((resolve) => setTimeout(resolve, 100));

        if (!window.markmap) {
          throw new Error("Markmap library is not loaded correctly.");
        }

        // Add custom CSS for text alignment
        const existingStyle = document.querySelector("#markmap-style");
        if (!existingStyle) {
          const style = document.createElement("style");
          style.id = "markmap-style";
          style.textContent = `
            .markmap-node .markmap-label {
              transform: translateY(20px) !important;
              dominant-baseline: middle !important;
              alignment-baseline: middle !important;
            }
            .markmap-link {
              stroke-width: 1.5;
            }
          `;
          document.head.appendChild(style);
        }

        const { Transformer, Markmap } = window.markmap;
        const transformer = new Transformer();
        const { root } = transformer.transform(markdown);

        if (svgRef.current && !markmapInstanceRef.current) {
          markmapInstanceRef.current = new Markmap(svgRef.current, {
            maxWidth: 300,
            color: (node) => {
              const level = node.depth;
              return ["#2196f3", "#4caf50", "#ff9800", "#f44336"][level % 4];
            },
            paddingX: 16,
            autoFit: true,
            initialExpandLevel: 2,
            duration: 500,
            nodeMinWidth: 150,
            nodeMinHeight: 50,
            spacingVertical: 60,
            spacingHorizontal: 120,
          });

          markmapInstanceRef.current.setData(root);
          markmapInstanceRef.current.fit();
        }
      } catch (error) {
        console.error("Error loading Markmap libraries:", error);
      }
    };

    if (markdown) {
      loadLibraries();
    }

    return () => {
      if (svgRef.current) {
        svgRef.current.innerHTML = "";
      }
      markmapInstanceRef.current = null;
    };
  }, [markdown]);

  return (
    <div
      style={{
        width: "100%",
        height: "100vh",
        backgroundColor: "#f5f5f5",
        overflow: "hidden",
      }}
    >
      <svg
        ref={svgRef}
        style={{
          width: "100%",
          height: "100%",
        }}
      />
    </div>
  );
};

const loadScript = (src) => {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) {
      resolve();
      return;
    }

    const script = document.createElement("script");
    script.src = src;
    script.onload = resolve;
    script.onerror = reject;
    document.head.appendChild(script);
  });
};

export default Markmap;
