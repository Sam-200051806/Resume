# Resume Analyzer

This project is a Streamlit application that allows users to upload their resumes in PDF format and receive responses based on the content of the uploaded document. The application utilizes various libraries for document processing, embeddings, and querying to provide insightful feedback to users.

## Project Structure

```
resume-analyzer
├── src
│   ├── app.py               # Main entry point of the Streamlit application
│   ├── document_loader.py    # Functions to load and process PDF documents
│   ├── embeddings.py         # Setup for Hugging Face embeddings model
│   ├── query_engine.py       # Logic for querying embeddings and retrieving information
│   └── utils.py             # Utility functions for the application
├── .env                      # Environment variables for configuration
├── requirements.txt          # List of dependencies for the project
└── README.md                 # Documentation for the project
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd resume-analyzer
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the root directory and add the necessary environment variables, such as the Pinecone index name.

5. **Run the application:**
   ```bash
   streamlit run src/app.py
   ```

## Usage Guidelines

- Open the application in your web browser.
- Upload your resume in PDF format using the provided interface.
- Interact with the application to receive responses based on the content of your resume.

## Overview of Functionality

- **Document Loading:** The application uses `document_loader.py` to extract text from uploaded PDF resumes.
- **Embeddings Setup:** The `embeddings.py` file initializes the Hugging Face embeddings model for processing the extracted text.
- **Querying:** The `query_engine.py` file contains the logic for querying the embeddings and retrieving relevant information based on user input.
- **Utilities:** Common utility functions are managed in `utils.py` to streamline the application.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.