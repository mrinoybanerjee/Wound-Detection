import React from "react";
import Webcam from "react-webcam";
import Button from "@mui/material/Button";
import { CssBaseline, Box } from "@mui/material";

function App() {
  const videoConstraints = {
    facingMode: "environment", // Switch to "environment" for rear camera on mobile devices
  };

  const capture = () => {
    // Logic for capturing the image
    console.log("Captured");
  };

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
          onClick={capture}
          sx={{
            position: "fixed",
            bottom: 20, // Adjust as needed for visual comfort
            zIndex: 10, // Ensure it's above the webcam feed
          }}
        >
          Click Picture
        </Button>
      </Box>
    </>
  );
}

export default App;
