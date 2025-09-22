# Offline AI Tutor

This project is a multimodal AI tutor that supports English and Urdu, works completely offline, and is optimized for both x86_64 and ARM (Raspberry Pi) architectures. It features text and voice chat, OCR from images, and specialized models for different subjects.

## Features

- **Multilingual:** Supports both English and Urdu for the user interface and interactions.
- **Multimodal:** Accepts text, image, and audio inputs.
- **Offline-first:** All models run locally, requiring no internet connection.
- **Cross-Platform:** Optimized for both standard computers (x86_64) and single-board computers like Raspberry Pi (ARM).
- **Session Management:** Saves and manages chat sessions in a local database.
- **Dynamic Model Switching:** Automatically switches to a specialized language model for subjects like Mathematics to provide more accurate responses.

## Setup

1.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Install System Dependencies (for ARM/Raspberry Pi):**
    If you are running this on an ARM-based system like a Raspberry Pi, you need to install the Tesseract OCR engine.
    ```bash
    sudo apt-get update
    sudo apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-urd
    ```

3.  **Download Models:**
    This project requires several models to be downloaded and placed in the `models/` directory. The required model paths are configured in `config.json`. Please ensure you have the models corresponding to the paths in the configuration file.

## Running the Application

To start the AI Tutor, run the main GUI file:

```bash
python main_gui.py
```

This will launch a Gradio web interface in your browser.

## How to Use

-   **Version Selection:** Choose between `x86_64` (for standard PCs) and `arm` (for Raspberry Pi) from the "Version" dropdown in the sidebar. The application will load the appropriate models.
-   **Personalize Tutoring:** Select your desired language, grade, and subject from the sidebar to tailor the tutor's responses.
-   **Interact:** You can type your questions, upload an image containing text, or record/upload an audio message. The tutor will process the input and respond with optional speech output