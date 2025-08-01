// frontend-react/src/index.js
import React from 'react';
import ReactDOM from 'react-dom/client'; // Updated for React 18+
import './index.css'; // Standard global CSS file
import App from './App'; // Your main App component
import reportWebVitals from './reportWebVitals';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();