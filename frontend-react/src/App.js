import React, { useState } from 'react';
import './App.css';

function App() {
  const [prompt, setPrompt] = useState('');
  const [generatedScript, setGeneratedScript] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const API_URL = 'http://localhost:8000/generate-full-short-script/';

  const handleGenerateScript = async () => {
    if (!prompt.trim()) {
      setError("Please enter a prompt, logline, or story idea.");
      return;
    }

    setIsLoading(true);
    setError('');
    setGeneratedScript('');

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt }),
      });

      if (!response.ok) {
        // Try to get more detailed error from backend
        let errorMsg = `Error: ${response.status} ${response.statusText}`;
        try {
          const errorData = await response.json();
          errorMsg += `. ${errorData.error || errorData.detail || ''}`;
        } catch (e) { /* Ignore if body isn't JSON */ }
        throw new Error(errorMsg);
      }

      const data = await response.json();

      if (data.script) {
        setGeneratedScript(data.script);
      } else if (data.error) {
        throw new Error(data.error);
      } else {
        throw new Error("Received an unexpected response from the server.");
      }

    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };
  
  // A few creative prompts to get the user started
  const examplePrompts = [
    "A lonely lighthouse keeper discovers that the foghorn's sound can calm a mythical sea monster.",
    "In a city where it constantly rains, a detective must solve the mystery of a man who seemingly evaporated.",
    "An old watchmaker realizes one of his clocks can turn back time, but only by ten seconds, forcing him to perfectly time a rescue.",
    "Two rival street magicians in New Orleans discover their 'illusions' are actually real magic, and they've attracted unwanted attention from an ancient magical society."
  ];

  return (
    <div className="container">
      <h1>AI Script Generator</h1>
      <p className="subtitle">Just like ChatGPT, give a prompt and get a complete short script!</p>
      
      {error && <div className="error-message">{error}</div>}

      <div className="form-group">
        <label htmlFor="prompt">Enter Your Story Idea:</label>
        <textarea
          id="prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="e.g., A robot chef in a high-end restaurant begins to question its existence after creating a dish that tastes like a forgotten human emotion."
          rows={5}
        />
      </div>

      <div className="example-prompts">
        <p>Or try an example:</p>
        {examplePrompts.map((p, i) => (
          <button key={i} className="example-btn" onClick={() => setPrompt(p)}>
            {p.substring(0, 40)}...
          </button>
        ))}
      </div>

      <button className="generate-btn" onClick={handleGenerateScript} disabled={isLoading}>
        {isLoading ? 'Writing...' : 'Generate Full Script'}
      </button>

      {isLoading && <div className="loader"></div>}

      {generatedScript && (
        <div className="script-container">
            <h2>Generated Script</h2>
            <pre className="script-output">{generatedScript}</pre>
        </div>
      )}
    </div>
  );
}

export default App;