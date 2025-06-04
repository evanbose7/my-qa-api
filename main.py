import os
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import json
from bs4 import BeautifulSoup
from dotenv import load_dotenv # For loading environment variables from .env file

# Load environment variables from .env file
load_dotenv()

# Initialize the FastAPI application
app = FastAPI(
    title="Website Q&A API",
    description="API to fetch website content and answer questions using Gemini AI.",
    version="1.0.0",
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Define the API endpoint for answering questions from a website
@app.get("/answer-from-website", summary="Answer a question based on website content")
async def answer_from_website(
    url: str = Query(..., description="The URL of the website to fetch content from."),
    question: str = Query(..., description="The question to answer based on the website content.")
):
    """
    Fetches content from a given website URL, extracts readable text,
    and then uses the Gemini AI to answer a user's question based on that content.
    """
    # Basic validation for inputs is handled by FastAPI's Query(...)
    # If url or question are missing, FastAPI will automatically return a 422 Unprocessable Entity.

    try:
        # Step 1: Fetch the content of the website
        print(f"Fetching content from: {url}")
        # Use a User-Agent header to mimic a browser and avoid some blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, timeout=15, headers=headers) # Increased timeout for potentially slower websites
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        html_content = response.text
        print("Website content fetched successfully.")

        # Step 2: Parse the HTML and extract relevant text
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script, style, and other non-content tags to clean up the text
        for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'form']):
            script_or_style.extract()

        # Get all text from the body, then clean up whitespace
        text_content = soup.get_text(separator=' ', strip=True)

        # Limit the text content to avoid exceeding token limits for the LLM
        # A typical limit for context is around 30,000 tokens for Gemini 2.0 Flash,
        # but it's safer to use a conservative character limit.
        max_text_length = 15000 # Approximately 15,000 characters
        if len(text_content) > max_text_length:
            text_content = text_content[:max_text_length] + "..."
            print(f"Text content truncated to {max_text_length} characters.")
        else:
            print(f"Extracted text content length: {len(text_content)} characters.")

        if not text_content.strip():
            raise HTTPException(status_code=400, detail='No readable text content found on the website.')

        # Step 3: Construct the prompt for the Gemini AI
        prompt = (
            f"Based on the following website content, answer the question:\n\n"
            f"Website Content:\n---\n{text_content}\n---\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        print("Prompt constructed for Gemini AI.")

        # Prepare the chat history for the Gemini API request
        chat_history = []
        chat_history.append({"role": "user", "parts": [{"text": prompt}]})

        # Define the payload for the Gemini API request
        payload = {
            "contents": chat_history
        }

        # Get API key from environment variable
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set in environment variables.")

        # Define the API URL for the Gemini 2.0 Flash model
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

        # Make the POST request to the Gemini API
        print("Calling Gemini API...")
        llm_response = requests.post(api_url, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        llm_response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

        # Parse the JSON response from the LLM
        llm_result = llm_response.json()
        print("Gemini API call successful.")

        # Check if the response contains valid content
        if llm_result.get('candidates') and len(llm_result['candidates']) > 0 and \
           llm_result['candidates'][0].get('content') and \
           llm_result['candidates'][0]['content'].get('parts') and \
           len(llm_result['candidates'][0]['content']['parts']) > 0:
            # Extract the generated answer
            answer_text = llm_result['candidates'][0]['content']['parts'][0].get('text')
            # Send the answer as a JSON response
            return JSONResponse(content={'answer': answer_text})
        else:
            # Handle cases where the LLM response structure is unexpected or content is missing
            print(f'Unexpected LLM response structure: {llm_result}')
            raise HTTPException(status_code=500, detail='Failed to get an answer from the AI.')

    except requests.exceptions.RequestException as e:
        # Catch and display any errors during the website fetching or API call
        print(f'Network/HTTP error: {e}')
        raise HTTPException(status_code=500, detail=f'Could not access the website or connect to AI: {e}')
    except Exception as e:
        # Catch any other unexpected errors
        print(f'An unexpected error occurred: {e}')
        raise HTTPException(status_code=500, detail='An internal server error occurred.')

# To run this API locally, you would use Uvicorn:
# uvicorn your_file_name:app --reload --port 5000
# (e.g., uvicorn website_qa_api:app --reload --port 5000 if your file is website_qa_api.py)