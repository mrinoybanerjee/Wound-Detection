// Import necessary modules
import React from "react"; // React library
import ReactDOM from "react-dom/client"; // ReactDOM library for DOM manipulations
import "./index.css"; // Import global CSS styles
import App from "./App"; // Import main App component

// Get the root DOM node where the React application will be mounted
const root = ReactDOM.createRoot(document.getElementById("root"));

// Render the App component into the root DOM node
// The App component is wrapped in React.StrictMode, a tool for highlighting potential problems in an application
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
