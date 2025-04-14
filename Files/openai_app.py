from fastapi import FastAPI, HTTPException  # Import FastAPI for creating the API and HTTPException for error handling
from pydantic import BaseModel  # Import BaseModel from Pydantic to define request/response models
from openai import OpenAI  # Import OpenAI API client to interact with OpenAI models
import os  # To interact with the operating system, such as reading environment variables
from dotenv import load_dotenv  # To load environment variables from a .env file

# Load environment variables from .env
load_dotenv()  # This will read the .env file and set the environment variables

# Get the OpenAI API key from the environment variables
api_key = os.getenv("OPENAI_KEY")  # Retrieve the OpenAI API key using the environment variable name

# Initialize FastAPI app
app = FastAPI()  # This initializes the FastAPI application instance

# Define request model using Pydantic's BaseModel
class PromptRequest(BaseModel):
    prompt: str = "provide summary"  # Default prompt value if the user doesn't provide one
    model: str = "gpt-3.5-turbo"  # Default model set to "gpt-3.5-turbo", but can be changed to "gpt-4" or other models

# Define response model using Pydantic's BaseModel
class PromptResponse(BaseModel):
    response: str  # This will store the OpenAI model's response to the request

# Define endpoint for the POST request to interact with OpenAI
@app.post("/chat/")
async def chat_with_openai(request: PromptRequest):
    try:
        # Initialize OpenAI client with the provided API key
        client = OpenAI(api_key=api_key)

        # Call the OpenAI API to generate a response based on the provided prompt and model
        response = client.chat.completions.create(
            model=request.model,  # Model to be used, either GPT-3.5 or GPT-4
            messages=[  # This is the conversation history (just one message in this case)
                {"role": "user", "content": request.prompt}  # The user-provided prompt
            ],
            temperature=0.7  # Set the randomness of the response (0.7 is a good middle ground)
        )

        # Return the OpenAI model's response wrapped in a Pydantic model for structured response
        return PromptResponse(response=response.choices[0].message.content)  # Extracting the model's response content

    except Exception as e:
        # If there was an error, raise an HTTPException with a status code 500 and error details
        raise HTTPException(status_code=500, detail=str(e))  # This will return a server error with the exception message
