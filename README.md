# Wound Detection System

This project combines the power of PyTorch, TensorFlow and React to create a comprehensive wound detection tool that leverages deep learning and a user-friendly web interface.
Check it out [here](https://main--eclectic-chebakia-d6873a.netlify.app/) 🚀

## Overview

This project consists of two main parts:

1. **PyTorch Modeling**: Jupyter notebooks that contain the modeling work using PyTorch, including the creation, training, and validation of a CNN model for wound detection.
2. **React Web UI**: A Progressive Web App (PWA) built with React that uses TensorFlow.js to load and run the trained model directly in the browser, allowing for real-time wound detection using a device's camera.

## Important directories

- `models`: Store for all the trained models.
- `scripts`: Store for all the scripts. The `model.py` script is used to trigger the training pipeline.
- `web-ui/`: The source code for the React-based PWA.
- `web-ui/public/model_files`: TensorFlow.js model files split into shards for optimized loading.

## Getting Started

To get started with the project, clone the repository and navigate to the respective directories for instructions on setting up the PyTorch environment or the React application.

### PyTorch Model

1. Install dependencies: `pip install -r requirements.txt`
2. Explore the Jupyter notebooks in the `notebooks` directory to understand the model development process.
3. To trigger the training pipeline, run `python scripts/model.py`.

### React Web UI

1. Navigate to the `web-ui` directory.
2. Install dependencies: `npm install`
3. Start the development server: `npm start`
4. The application should now be running and accessible via a web browser.


# Google drive link for trained models

Access trained models for this project here: https://drive.google.com/drive/folders/1W22YZyyfLSrQ_cgc6cGwR64Gmvct08P8?usp=drive_link

## Directory structure

```
.
├── 540 Module Project 1.pdf
├── README.md
├── data
│   ├── outputs
│   └── raw
│       └── WoundDataset.zip
├── models
│   ├── EfficientNet_model_fold_4.pth
│   ├── Efficientnet_model_fold_4_tfjs
│   │   ├── group1-shard1of4.bin
│   │   ├── group1-shard2of4.bin
│   │   ├── group1-shard3of4.bin
│   │   ├── group1-shard4of4.bin
│   │   └── model.json
│   └── custom_resnet.pth
├── notebooks
│   ├── InceptionV3_Model.ipynb
│   ├── Model_pth_to_tfjs.ipynb
│   ├── ResNet50_Model.ipynb
│   ├── efficient_net.ipynb
│   ├── non_deep_learning_model.ipynb
│   └── resNet_34.ipynb
├── requirements.txt
├── scripts
│   ├── constants.py
│   ├── make_dataset.py
│   ├── model.py
│   └── setup.py
└── web-ui
    ├── package-lock.json
    ├── package.json
    ├── public
    │   ├── favicon.ico
    │   ├── index.html
    │   ├── logo192.png
    │   ├── logo512.png
    │   ├── manifest.json
    │   ├── model_files
    │   │   ├── group1-shard1of4.bin
    │   │   ├── group1-shard2of4.bin
    │   │   ├── group1-shard3of4.bin
    │   │   ├── group1-shard4of4.bin
    │   │   └── model.json
    │   └── robots.txt
    └── src
        ├── App.css
        ├── App.js
        ├── constants.js
        ├── index.css
        └── index.js

12 directories, 39 files
```

## Converting Pytorch model to TensorflowJS

This guide covers converting a PyTorch model file (.pth) to a TensorFlow.js compatible format. Follow the steps below to perform the conversion.

1. Converting model to ONNX format
First, convert your PyTorch model to ONNX format using the following code:

import torch
from efficientnet_pytorch import EfficientNet
import torch.onnx
import onnx

### Load the model with the correct number of output classes

### Converting PyTorch Model to TensorFlow.js

This guide covers converting a PyTorch model file (.pth) to a TensorFlow.js compatible format. Follow the steps below to perform the conversion.

#### 1. Converting model to ONNX format
First, convert your PyTorch model to ONNX format using the following code:

```python
import torch
from efficientnet_pytorch import EfficientNet
import torch.onnx
import onnx

# Load the model with the correct number of output classes
model_name = 'efficientnet-b0'  # Adjust based on the specific EfficientNet variant you used
num_classes = 10  # The number of classes in your dataset
model = EfficientNet.from_name(model_name, num_classes=num_classes)

# Load the state dictionary
checkpoint_path = '/path/to/your/model.pth'  # Update this path
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)

model.eval()  # Set the model to evaluation mode

# Input to the model
dummy_input = torch.randn(1, 3, 224, 224)  # Adjust the size if necessary for EfficientNet

# Export the model to ONNX
onnx_path = "/path/to/save/model.onnx"  # Update this path
torch.onnx.export(model, dummy_input, onnx_path,
                  export_params=True,
                  opset_version=10,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'},
                                'output': {0: 'batch_size'}})

print('Model exported to ONNX format.')
```

#### 2. Installing necessary libraries
Install the required libraries to convert ONNX to TensorFlow and then to TensorFlow.js by running:

```shell
pip install onnx tensorflow onnx-tf
pip install tensorflow_probability
pip install tf2onnx
```

#### 3. Convert Model from ONNX to TensorFlow
Convert the ONNX model to TensorFlow format using the following code:

```python
import onnx
import tensorflow as tf
from onnx_tf.backend import prepare

# Load the ONNX model
onnx_model = onnx.load("/path/to/your/model.onnx")  # Update this path

# Prepare the ONNX model for inference
tf_rep = prepare(onnx_model)

# Export the model to TensorFlow format
tf_rep.export_graph("/path/to/save/model.pb")  # Update this path

print('Model exported to TensorFlow format.')
```

#### 4. Install libraries to convert TensorFlow to TensorFlow.js
Install the required package for the conversion:

```shell
npm install -g @tensorflow/tfjs-node
```

#### 5. Convert TensorFlow Model to TensorFlow.js
Finally, convert the TensorFlow model to TensorFlow.js format with the following command:

```shell
tensorflowjs_converter --input_format=tf_saved_model --output_node_names='output_node_name' --control_flow_v2=true /path/to/your/model.pb /path/to/save/tfjs_model
```

Replace `/path/to/your/model.pb` with the path to your TensorFlow model file and `/path/to/save/tfjs_model` with the desired output directory for the TensorFlow.js model.

Please ensure to replace placeholder paths with actual paths relevant to your model and environment. This guide assumes you have EfficientNet or a compatible model; adjust as necessary for your specific model architecture.



## Contributors

- Abhishek Murthy ([@rootsec1](https://github.com/rootsec1))
- Mrinoy Bannerjee ([@mrinoybanerjee](https://github.com/mrinoybanerjee))
- Sai Samyukta Palle ([@pallesaisamyukta](https://github.com/pallesaisamyukta))
