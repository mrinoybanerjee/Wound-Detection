import React, { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import Button from "@mui/material/Button";
import { CssBaseline, Box, Snackbar, CircularProgress } from "@mui/material";
import * as tf from "@tensorflow/tfjs";

// Local imports
import { CLASSES } from "./constants";

/**
 * Main application component.
 * It uses TensorFlow.js to load a model and make predictions based on the webcam feed.
 */
function App() {
  // References and states
  const webcamRef = useRef(null); // Reference to the webcam component
  const [model, setModel] = useState(null); // State for the loaded model
  const [isLoading, setIsLoading] = useState(false); // State for loading status
  const [isSnackbarOpen, setIsSnackbarOpen] = useState(false); // State for Snackbar visibility
  const [predictedClass, setPredictedClass] = useState(null); // State for the predicted class

  // Constraints for the webcam video feed
  const videoConstraints = {
    facingMode: "environment", // Switch to "environment" for rear camera on mobile devices
  };

  /**
   * Load the TensorFlow.js model.
   */
  async function loadModel() {
    const modelURL = `${process.env.PUBLIC_URL}/model_files/model.json`;
    const model = await tf.loadGraphModel(modelURL);
    setModel(model);
    console.log("Model loaded");
  }

  /**
   * Reset the fields after a prediction.
   */
  const resetFields = () => {
    setIsLoading(false);
    setPredictedClass(null);
    setIsSnackbarOpen(false);
  };

  /**
   * Capture a frame from the webcam feed, preprocess it, and make a prediction.
   */
  const onCapture = async () => {
    console.log("Captured");
    if (!model || !webcamRef.current) return;

    setIsLoading(true);
    const imageSrc = webcamRef.current.getScreenshot();
    // Convert base64 string to Image
    const image = new Image();
    image.src = imageSrc;

    // Ensure the image is loaded before proceeding
    await new Promise((resolve, reject) => {
      image.onload = () => resolve();
      image.onerror = reject;
    });

    // Preprocess the image
    const tensor = tf.browser.fromPixels(image).toFloat().div(tf.scalar(255.0));
    const resized = tf.image.resizeBilinear(tensor, [224, 224]);
    const normalized = resized.sub(tf.scalar(0.5)).mul(tf.scalar(2));

    // Prepare the tensor for prediction
    const batched = normalized.expandDims(0); // Now shape is [1, height, width, channels]
    const transposed = batched.transpose([0, 3, 1, 2]); // Correctly transposes to [1, channels, height, width]

    // Make a prediction
    const predictions = await model.executeAsync(transposed);
    const predictionsArray = await predictions.array();
    console.log(predictions);
    const predictedClass = predictionsArray[0].indexOf(
      Math.max(...predictionsArray[0])
    );

    // Update the states
    setIsLoading(false);
    setPredictedClass(CLASSES[predictedClass]);
    setIsSnackbarOpen(true);

    // Dispose of tensors to free up GPU memory
    tensor.dispose();
    resized.dispose();
    batched.dispose();
    transposed.dispose();
  };

  // Load the model when the component mounts
  useEffect(() => {
    loadModel();
  }, []);

  // Render the component
  return (
    <>
      <CssBaseline /> {/* Ensures consistent baseline styling */}
      <Box
        sx={{
          position: "fixed", // Full screen, fixed positioning
          top: 0,
          left: 0,
          width: "100vw",
          height: "100vh",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center", // Center content vertically for smaller devices
          alignItems: "center", // Center content horizontally
          overflow: "hidden", // Prevent scrolling
        }}
      >
        <Webcam
          ref={webcamRef}
          audio={false}
          height="100%"
          width="100%"
          videoConstraints={videoConstraints}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            minWidth: "100%",
            minHeight: "100%",
            objectFit: "cover",
          }}
        />
        <Button
          variant="contained"
          color="primary"
          onClick={onCapture}
          sx={{
            position: "fixed",
            bottom: 20, // Adjust as needed for visual comfort
            zIndex: 10, // Ensure it's above the webcam feed
          }}
        >
          {isLoading ? (
            <CircularProgress style={{ color: "white" }} size={24} />
          ) : (
            "Capture"
          )}
        </Button>

        <Snackbar
          open={isSnackbarOpen}
          autoHideDuration={5000}
          onClose={resetFields}
          message={`Predicted class: ${predictedClass}`}
        />
      </Box>
    </>
  );
}

export default App;
