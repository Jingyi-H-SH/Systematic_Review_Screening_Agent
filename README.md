# Multi-Agent Systematic Review Screening App

This web application uses a team of AI agents to screen academic articles for systematic reviews based on user-defined criteria. It is built with FastAPI and can be run locally or deployed to a cloud platform like OnRender.

## Project Structure

- `app.py`: The main FastAPI web server. It handles HTTP requests, file uploads, and serves the results.
- `agent_system.py`: Contains all the core logic for the AI agents, including prompts, data schemas, and the orchestration flow.
- `requirements.txt`: A list of all Python packages required to run the application.
- `.env`: A file for storing your local environment variables, specifically your OpenAI API key. **This file should not be committed to Git.**
- `README.md`: This file.

## How to Run Locally

1.  **Set up a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables:**
    - Create a new file named `.env`.
    - Open the `.env` file and add the following line, replacing `"your_api_key_here"` with your actual OpenAI API key:
      `OPENAI_API_KEY="your_api_key_here"`

4.  **Run the Application:**
    ```bash
    uvicorn app:app --reload
    ```
    The `--reload` flag automatically restarts the server when you make changes to the code.

5.  **Access the API:**
    - Open your web browser and go to `http://127.0.0.1:8000`.
    - For an interactive API documentation (where you can upload files and test the endpoint), go to `http://127.0.0.1:8000/docs`.

## How to Deploy to OnRender

1.  **Push to GitHub:**
    - Create a new repository on GitHub and push your project files (`app.py`, `agent_system.py`, `requirements.txt`, `README.md`). **Do not push the `.env` file.**

2.  **Create a New Web Service on OnRender:**
    - Log in to your OnRender dashboard.
    - Click "New +" -> "Web Service".
    - Connect your GitHub repository.

3.  **Configure the Service:**
    - **Environment:** Python
    - **Build Command:** `pip install -r requirements.txt`
    - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`

4.  **Add Environment Variables:**
    - In the "Environment" section of your service settings on OnRender, click "Add Environment Variable".
    - **Key:** `OPENAI_API_KEY`
    - **Value:** Paste your actual OpenAI API key here.

5.  **Deploy:**
    - Click "Create Web Service". OnRender will automatically build and deploy your application. Your API will be live at the URL provided by OnRender.