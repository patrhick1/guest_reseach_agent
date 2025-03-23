# Podcast Guest Research Web Interface

A modern web interface for automating podcast guest research using the Podcast Guest Research Automation System.

## Overview

This web application provides a user-friendly interface to the Podcast Guest Research Automation API. It allows podcast hosts to easily research upcoming guests by retrieving social media profiles, historical posts, and generating comprehensive research reports.

## Features

- Simple form interface to input podcast episode details
- Integration with the research API backend
- Displays guest information including LinkedIn profiles
- Links to generated research reports
- Responsive design that works on desktop and mobile devices
- Support for both podcast RSS feed and direct social media research methods

## Project Structure

The application consists of the following components:

- `main.py`: The backend that handles the research workflow
- `server.py`: A Flask server that serves the web interface and communicates with main.py
- `research_state.py`: A shared module for state management
- `run.py`: A utility script to run both servers together (primarily for local development)
- `templates/`: HTML templates for the web interface
- `static/css/`: CSS stylesheets for the web interface
- `static/js/`: JavaScript files for the web interface

## Prerequisites

- Python 3.8+ with the dependencies installed
- Dependencies listed in requirements.txt
- Modern web browser (Chrome, Firefox, Safari, or Edge)

## Setup Instructions

1. Clone this repository:
   ```
   git clone <repository-url>
   cd podcast-guest-research
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file with your API keys and configuration
   - Required variables are listed in the next section

4. Run the application:
   ```
   # Run both frontend and backend together (recommended for local development)
   python run.py
   
   # Run only the frontend (if backend is already running elsewhere)
   python run.py --frontend-only
   
   # Specify custom ports
   python run.py --frontend-port 5000 --backend-port 8080
   ```

## Environment Variables

The following environment variables can be set in a `.env` file:

```
# API Keys
OPENAI_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
LISTENNOTES_API_KEY=your_listennotes_api_key
TAVILY_API_KEY=your_tavily_api_key
PODSCANAPI=your_podscan_api_key
APIFY_CLIENT_TOKEN=your_apify_token

# Google API
GOOGLE_REPORT_FOLDER_ID=your_google_drive_folder_id

# Server Configuration
PORT=8080  # Backend port
FRONTEND_PORT=5000  # Frontend port
```

## Usage

1. Start the application using `python run.py`
2. Open your browser and navigate to `http://localhost:5000`
3. Fill in the research form with the following information:
   - **Research Method**: Select either "Podcast" or "Social Media" research method
   - **Episode Title** (required for podcast method): The exact title of the podcast episode
   - **Airtable Record ID** (required): The Airtable record ID where results will be saved
   - **RSS Feed URL** (optional for podcast method): The RSS feed URL of the podcast
   - **Guest Name** (optional for podcast method): The name of the guest for RSS feed search
   - **Host Podcast Name** (optional): The name of the host podcast
   - **LinkedIn URL** (optional for social media method): The LinkedIn profile URL of the guest
   - **Twitter URL** (optional for social media method): The Twitter profile URL of the guest

4. Click "Start Research" to begin the research process. The system will:
   - Extract guest information from podcast metadata or social media profiles
   - Find social media profiles
   - Retrieve historical posts
   - Generate a research report
   - Save results to Google Drive and Airtable

5. View the research results, including:
   - Guest name
   - LinkedIn profile link
   - Link to the generated research report

## Deploying on Replit

For deploying on Replit, follow these steps:

1. Create a new Replit project and import this repository
2. Set up your environment variables in Replit's Secrets tab (equivalent to .env file)
3. Create a `.replit` file in the root directory with the following configuration:

```
language = "python3"
entrypoint = "server.py"
run = "python server.py"
```

4. Create a `pyproject.toml` file for dependency management:

```toml
[tool.poetry]
name = "podcast-guest-research"
version = "0.1.0"
description = "A web interface for automating podcast guest research"

[tool.poetry.dependencies]
python = "^3.8"
flask = "^2.3.0"
python-dotenv = "^1.0.0"
pydantic = "^2.0.0"
typing-extensions = "^4.5.0"
requests = "^2.31.0"
langgraph = "^0.0.2"
langchain-core = "^0.0.10"
langchain-openai = "^0.0.2"
langchain-community = "^0.0.2"
langchain-google-genai = "^0.0.1"
tiktoken = "^0.4.0"
aiohttp = "^3.8.5"
apify-client = "^1.0.0"
google-api-python-client = "^2.100.0"
google-auth = "^2.23.0"
google-auth-oauthlib = "^1.0.0"
google-auth-httplib2 = "^0.1.0"
tavily-python = "^0.2.8"
```

5. Modify `server.py` to import and initialize the research functionality directly (no need for separate processes):

Add the following near the end of the file, before the `if __name__ == '__main__'` block:

```python
# Import research functionality
from main import research_podcast_guest

# Initialize any necessary components
# ... 
```

6. Update `if __name__ == '__main__'` block in `server.py` to use Replit's environment:

```python
if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
```

7. Click the Run button in Replit to start your application

Note: For Replit deployment, the `run.py` script is not necessary as Replit manages the process differently.

## Running in Production

For production deployments (non-Replit), consider the following:

1. Set `debug=False` in server.py for production environments
2. Use a production WSGI server like Gunicorn:
   ```
   # For the frontend
   gunicorn -w 4 server:app
   ```
3. Set up proper logging and monitoring
4. Use environment variables for all sensitive information

## Troubleshooting

- If the form submission fails, check that the backend API is running
- Ensure all required API keys are set in the backend's `.env` file
- Check browser console for JavaScript errors
- Verify network requests in your browser's developer tools
- If the application doesn't start, check terminal output for errors

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- FontAwesome for the icons
- The FastAPI and Flask frameworks
- LangChain and LangGraph for AI-powered analysis 