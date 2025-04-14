# importing libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_experimental.agents import create_csv_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType
import os

# Initialize OpenAI API key
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_KEY")

# Initialize the language model
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini") # Temperature means Controls randomness in the model's output 
                                                     # and 0 means deterministic (you'll get the same response every time for the same input).

# Create the agent
agent = create_csv_agent(
    llm=llm,
    path="/home/anudeep/Desktop/gen ai/document/mtsamples.csv",  # Update with the correct path
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    allow_dangerous_code=True
)

# Initialize FastAPI app
app = FastAPI()

# Define request and response models
class AgentRequest(BaseModel):  # AgentRequest: Expects an object with a field called input (a question/query as a string).
    input: str

class AgentResponse(BaseModel):  # AgentResponse: Returns an object with a field called output (the agent’s answer).
    output: str

# Define the API endpoint
@app.post("/run-agent/", response_model=AgentResponse)
async def run_agent(request: AgentRequest):
    try:
        # Run the agent with the provided input
        response = agent.run(request.input)
        return AgentResponse(output=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
