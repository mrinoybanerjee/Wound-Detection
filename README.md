# Wound Detection System
This project combines the power of PyTorch, TensorFlow and React to create a comprehensive wound detection tool that leverages deep learning and a user-friendly web interface.
Check it out [here](https://main--eclectic-chebakia-d6873a.netlify.app/) 🚀

## Overview
This project consists of two main parts:
1. **PyTorch Modeling**: Jupyter notebooks that contain the modeling work using PyTorch, including the creation, training, and validation of a CNN model for wound detection.
2. **React Web UI**: A Progressive Web App (PWA) built with React that uses TensorFlow.js to load and run the trained model directly in the browser, allowing for real-time wound detection using a device's camera.

## Important directories
- `model/`: Contains the machine learning model files, Jupyter notebooks, visualizations, and requirements.
- `model/Notebooks`: Jupyter notebooks for model development and validation.
- `model/Visualizations`: Visual assets used in the model's exploratory data analysis.
- `model/model.pth`: The saved PyTorch model weights.
- `web-ui/`: The source code for the React-based PWA.
- `web-ui/public/model_files`: TensorFlow.js model files split into shards for optimized loading.

## Getting Started
To get started with the project, clone the repository and navigate to the respective directories for instructions on setting up the PyTorch environment or the React application.

### PyTorch Model
1. Install dependencies: `pip install -r model/requirements.txt`
2. Explore the Jupyter notebooks in `model/Notebooks` to understand the model development process.

### React Web UI
1. Navigate to the `web-ui` directory.
2. Install dependencies: `npm install`
3. Start the development server: `npm start`
4. The application should now be running and accessible via a web browser.

## Directory structure
```
.
├── Demo Slides.pdf
├── README.md
├── model
│   ├── Notebooks
│   │   ├── modeling.ipynb
│   │   ├── non_deep_learning_model.ipynb
│   │   └── validation.ipynb
│   ├── Visualizations
│   │   └── Visualizations.png
│   ├── model.pth
│   └── requirements.txt
└── web-ui
    ├── README.md
    ├── package-lock.json
    ├── package.json
    ├── public
    │   ├── favicon.ico
    │   ├── index.html
    │   ├── logo192.png
    │   ├── logo512.png
    │   ├── manifest.json
    │   ├── model_files
    │   │   ├── group1-shard10of23.bin
    │   │   ├── group1-shard11of23.bin
    │   │   ├── group1-shard12of23.bin
    │   │   ├── group1-shard13of23.bin
    │   │   ├── group1-shard14of23.bin
    │   │   ├── group1-shard15of23.bin
    │   │   ├── group1-shard16of23.bin
    │   │   ├── group1-shard17of23.bin
    │   │   ├── group1-shard18of23.bin
    │   │   ├── group1-shard19of23.bin
    │   │   ├── group1-shard1of23.bin
    │   │   ├── group1-shard20of23.bin
    │   │   ├── group1-shard21of23.bin
    │   │   ├── group1-shard22of23.bin
    │   │   ├── group1-shard23of23.bin
    │   │   ├── group1-shard2of23.bin
    │   │   ├── group1-shard3of23.bin
    │   │   ├── group1-shard4of23.bin
    │   │   ├── group1-shard5of23.bin
    │   │   ├── group1-shard6of23.bin
    │   │   ├── group1-shard7of23.bin
    │   │   ├── group1-shard8of23.bin
    │   │   ├── group1-shard9of23.bin
    │   │   └── model.json
    │   └── robots.txt
    └── src
        ├── App.css
        ├── App.js
        ├── index.css
        └── index.js
8 directories, 45 files
```

## Contributors
- Abhishek Murthy ([@rootsec1](https://github.com/rootsec1))
- Mrinoy Bannerjee ([@mrinoybanerjee](https://github.com/mrinoybanerjee))
- Sai Samyukta Palle ([@pallesaisamyukta](https://github.com/pallesaisamyukta))








