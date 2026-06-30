import { useEffect, useState } from "react";

// Cycling status lines tied to the cooking metaphor, swapped every
// INTERVAL_MS so a slow backend response doesn't feel like a stalled UI.
const STATUS_MESSAGES = [
  "Whisking up your recipe...",
  "Preheating the oven...",
  "Sifting through the recipe book...",
  "Folding in the details...",
  "Plating it up...",
];

const INTERVAL_MS = 1800;

function LoadingBubble() {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setIndex((prev) => (prev + 1) % STATUS_MESSAGES.length);
    }, INTERVAL_MS);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="message assistant loading" aria-live="polite">
      <svg
        className="whisk-spinner"
        viewBox="0 0 24 24"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        aria-hidden="true"
      >
        <path
          d="M12 2v7M9 5.5c0 1.5 1.3 2.5 3 2.5s3-1 3-2.5"
          stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"
        />
        <path
          d="M12 9c-3 0-5.5 2.8-5.5 6.5S9 22 12 22s5.5-2.8 5.5-6.5"
          stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"
        />
        <path
          d="M12 9c3 0 5.5 2.8 5.5 6.5"
          stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" opacity="0.5"
        />
      </svg>
      <span className="loading-text">{STATUS_MESSAGES[index]}</span>
      <span className="loading-dots">
        <span>.</span><span>.</span><span>.</span>
      </span>
    </div>
  );
}

export default LoadingBubble;
