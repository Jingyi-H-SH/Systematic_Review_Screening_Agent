import os
import io
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from agent_system import run_screening_process

# Load environment variables from .env file for local development
load_dotenv()

app = FastAPI(
    title="Multi-Agent Systematic Review Screening API",
    description="Upload a CSV of articles to be screened by a team of AI agents.",
    version="1.0.0"
)

@app.get("/", tags=["Health Check"])
async def read_root():
    """A simple endpoint to check if the API is running."""
    return {"status": "API is running."}

@app.post("/screen/", tags=["Screening"])
async def screen_articles(
    criteria: str = Form(...),
    csv_file: UploadFile = File(...)
):
    """
    Asynchronously screens articles from an uploaded CSV file.

    - **criteria**: The inclusion/exclusion criteria for the screening.
    - **csv_file**: The CSV file containing 'title' and 'abstract' columns.
    """
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY environment variable not set.")

    # Ensure the uploaded file is a CSV
    if not csv_file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")

    try:
        # Read the uploaded file into a pandas DataFrame
        content = await csv_file.read()
        try:
            df_articles = pd.read_csv(io.BytesIO(content))
        except UnicodeDecodeError:
            print("UTF-8 decoding failed, trying with latin-1 encoding.")
            df_articles = pd.read_csv(io.BytesIO(content), encoding='latin-1')

        # --- Run the multi-agent screening process ---
        final_df = await run_screening_process(df_articles, criteria)

        # Convert the resulting DataFrame to a CSV in memory
        output_stream = io.StringIO()
        final_df.to_csv(output_stream, index=False)
        output_stream.seek(0)

        # Return the CSV as a downloadable file
        return StreamingResponse(
            iter([output_stream.read()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=screened_results.csv"}
        )

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")