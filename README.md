# AI Movie Script Generator using Phi-2

This project is a full-stack application that leverages Microsoft's Phi-2, a powerful small language model (SLM), to generate original movie scripts from a user-provided prompt. The application features a Python backend using FastAPI to serve the AI model and a modern, interactive frontend built with React.

The project also includes a complete fine-tuning pipeline, allowing you to train the base Phi-2 model on your own dataset of scripts to create a specialized, expert screenwriter AI.

 <!-- It's a good idea to take a screenshot of your app and upload it to a service like imgur.com, then replace this URL with your own screenshot URL. -->

## Features

-   **AI-Powered Script Generation:** Uses the Phi-2 language model to generate creative and coherent short film scripts.
-   **Interactive Frontend:** A clean user interface built with React allows for easy prompt input and displays the generated script in a formatted way.
-   **High-Performance Backend:** Built with FastAPI, the Python backend is asynchronous and efficient, capable of handling AI model inference.
-   **GPU Accelerated:** Automatically utilizes an NVIDIA GPU with CUDA for fast model loading and generation if available.
-   **Model Fine-Tuning Pipeline:** Includes scripts and instructions to fine-tune the base Phi-2 model on a custom dataset, creating a specialized model that understands screenplay format perfectly.
-   **LoRA Integration:** Uses Parameter-Efficient Fine-Tuning (PEFT) with LoRA, making it possible to fine-tune the model on consumer-grade GPUs.

## Tech Stack

-   **Backend:**
    -   Python
    -   FastAPI (for the web server)
    -   PyTorch
    -   Hugging Face Transformers (for model loading and generation)
    -   Hugging Face PEFT & TRL (for fine-tuning)
    -   BitsAndBytes (for 8-bit/4-bit quantization)
-   **Frontend:**
    -   React.js
    -   JavaScript (ES6+)
    -   HTML5 & CSS3
    -   npm

## Prerequisites

Before you begin, ensure you have the following installed on your system:

-   [Python](https://www.python.org/downloads/) (v3.10 or v3.11 recommended)
-   [Node.js and npm](https://nodejs.org/en/) (LTS version recommended)
-   [Git](https://git-scm.com/)
-   **For GPU Support:** An NVIDIA GPU with the appropriate [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) installed.

## Getting Started: Running the Project

Follow these steps to get the application running locally on your machine.

### Clone the Repository

First, clone the project from GitHub to your local machine:
```bash
git clone https://github.com/Danush6123/Movie-Script-Generation-using-LLM.git
cd Movie-Script-Generation-using-LLM
```
### DEVELOPERS
**Danush G** - https://github.com/Danush6123

**Rachana  P** - https://github.com/Rachana904

**Rishith P** - https://github.com/rishith15

**Harshini P Raiker** - https://github.com/harshinipraiker
