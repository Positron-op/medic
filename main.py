import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDdK_Q6dLIg1vevncfSekPsYuJ65C4xKtI")

# Initialize Gemini
model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-pro")
        print("Gemini API configured successfully")
    except Exception as e:
        print(f"Error initializing Gemini: {e}")
        model = None
else:
    print("Warning: GEMINI_API_KEY environment variable not set")

class PromptRequest(BaseModel):
    prompt: str
    context: Optional[dict] = None

@app.post("/api/ai")
async def ai_endpoint(request: PromptRequest):
    if not model:
        raise HTTPException(
            status_code=500, 
            detail="Gemini API not configured. Please set GEMINI_API_KEY environment variable."
        )
    
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    context = f"""
    You are MediAid – a Local Drug Finder & Health Advice Chatbot for communities in Nigeria and West Africa.

    Instructions:
    - You help users with medical and health questions
    - Provide drug recommendations for common conditions
    - Give preventive health advice
    - Be cordial, professional, and supportive
    - Focus on locally available medications when possible
    - Include appropriate medical disclaimers
    - If the question seems serious, advise consulting a healthcare professional
    - Stay on medical/health topics only
    - Always format responses in markdown for better readability

    User question: "{prompt}"

    Remember to:
    1. First address the immediate question
    2. Provide relevant medical information
    3. Include locally available treatment options when applicable
    4. Add preventive measures if relevant
    5. Always end with the medical disclaimer
    """

    try:
        response = model.generate_content(context)
        response_text = response.text if hasattr(response, 'text') else str(response)
        if not response_text.endswith("⚠️"):
            response_text += "\n\n⚠️ This is for informational purposes only. Always consult a healthcare professional for proper diagnosis and treatment."
        return {"text": response_text}
    except Exception as e:
        print(f"Gemini API error: {e}")
        return {"error": "Sorry, I'm having trouble connecting to my AI service. Please try again later."}

@app.get("/")
async def root():
    return {"message": "MediAid API is running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "gemini_configured": model is not None,
        "api_key_set": GEMINI_API_KEY is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
