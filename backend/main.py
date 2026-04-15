from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware   #Allow frontend to talk to backend
from fastapi.responses import JSONResponse   #Sends proper JSON response
import uvicorn   #Runs server
from parser import parse_resume   #extract resume data
from scorer import score_resume
from matcher import match_job_description
import tempfile, os

app = FastAPI(title="Resume Analyzer API", version="1.0.0")   

app.add_middleware(
    CORSMiddleware,   #allow any frontend to connect
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Resume Analyzer API is running!"}

@app.post("/parse")
async def parse_endpoint(file: UploadFile = File(...)):   #accepts a resume file
    """Extract structured info from resume PDF/text."""
    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:   #Saved temporarily
            tmp.write(content)
            tmp_path = tmp.name
        parsed = parse_resume(tmp_path)
        os.unlink(tmp_path) #delete temp file
        return JSONResponse(content={"success": True, "data": parsed})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/score")
async def score_endpoint(file: UploadFile = File(...)):
    """Parse resume and return AI-generated score + suggestions."""
    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        parsed = parse_resume(tmp_path)
        os.unlink(tmp_path)
        result = score_resume(parsed)
        return JSONResponse(content={"success": True, "data": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/match")
async def match_endpoint(
    file: UploadFile = File(...),
    job_description: str = Form(...)
):
    """Match resume against a job description using embeddings."""
    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        parsed = parse_resume(tmp_path)
        os.unlink(tmp_path)
        score_data = score_resume(parsed)
        match_data = match_job_description(parsed, job_description)
        return JSONResponse(content={
            "success": True,
            "data": {
                **score_data,
                **match_data
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
