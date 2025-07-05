# BigQuery ADK Agent

This project implements an intelligent agent using the Google Agent Development Kit (ADK). The agent is capable of understanding questions in natural language, converting them into SQL queries, executing the query on a BigQuery database, and returning the results.

The key feature of this agent is its use of a "prompt engineering" script that inspects the BigQuery database schema to create a rich and detailed context, allowing the AI model to generate much more accurate and complex SQL queries.

## Project Structure
```
/adk_bigquery_agent/                  # Root project folder
|
├── .venv/                            # Virtual environment directory
|
├── bigquery_agent/                   # Python package containing the agent's source code
│   ├── __init__.py                   # Makes the directory a Python package
│   ├── .env                          # File to store credentials (not versioned)
│   ├── agent.py                      # Defines the main agent (root agent)
│   ├── config.yaml                   # Agent deployment settings
│   ├── generate_bigquery_prompt.py   # Script to analyze the DB and generate the context
│   ├── bigquery_context.txt          # Generated enhanced prompt file
│   ├── prompt.py                     # Stores the prompt template and joins with the context
│   └── tools.py                      # Contains the tool that executes SQL queries
|
├── deploy_agent_engine.ipynb         # Python notebook to step-by-step deploy on Vertex Agent Engine
├── requirements.txt                  # File listing Python dependencies
└── README.md                         # This file
```

## Prerequisites

* Python 3.12 or higher
* Access to a Google Cloud Project with BigQuery enabled.
* Appropriate IAM permissions for BigQuery data access.

## Installation and Execution Guide

Follow the steps below to set up and run the project.

### 1. Clone the Repository (Optional)

If you are starting on a new machine, clone the repository.

```
git clone git@github.com:speca-google/adk_bigquery_agent.git
cd adk_bigquery_agent
```

### 2. Create and Activate the Virtual Environment (venv)

It is a good practice to isolate the project's dependencies in a virtual environment.

# Create the virtual environment

```
python -m venv .venv
```

# Activate the virtual environment
```
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a file named `.env` inside the `mysql_agent/` directory and fill it with your credentials.

**`mysql_agent/.env` Example:**

```env
# --- Vertex AI Settings ---
# You can leave GOOGLE_GENAI_USE_VERTEXAI as "True" if deploying on GCP
GOOGLE_GENAI_USE_VERTEXAI="True"
GOOGLE_CLOUD_PROJECT="your-gcp-project-id"   # Project ID from GCP (where Agent is going to run)
GOOGLE_CLOUD_LOCATION="us-central1"          # Location where the agent will be deployed
GOOGLE_CLOUD_BUCKET="gs://your-agent-bucket" # Bucket for deployment on Agent Engine

# --- Agent Models ---
# The model used by the main agent to understand user intent and orchestrate tools.
ROOT_AGENT_MODEL="gemini-2.5-flash"
# The model used by the prompt generator script.
LLM_MODEL="gemini-2.5-flash"

# --- BigQuery Database Settings ---
BIGQUERY_PROJECT_ID="your-gcp-project-id"     # Your BigQuery Project ID
BIGQUERY_DATASET_ID="your-bigquery-dataset-id" # The Dataset ID to connect to (e.g., "my_analytics_data")
```

### 5. Generate the Database Context

Run the `generate_bigquery_prompt.py` script from the project's root directory. It will connect to your MySQL database, collect metadata and samples, and then use Gemini to generate a complete and optimized prompt file.

# Make sure your current directory is the project root
```
python bigquery_agent/generate_bigquery_prompt.py
````

After execution, a new file named `bigquery_context.txt` will be created in the `adk_bq_agent/` directory. This file contains the detailed context about your database schema that the agent will use.

### 6. Run the Agent Locally for Testing

Now that everything is configured, you can start the agent using ADK web.
```
adk web
```
The `adk web` command will open a web UI to test your agent. If you get a permission error, don't forget to authenticate your gcloud application-default credentials.
```
gcloud auth application-default login
```

### 7. Deploy this agent on Agent Engine

To deploy this agent, use the Python Notebook `deploy_agent_engine.ipynb`. This file contains a step-by-step guide for deployment.