# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# tools.py
import os
import logging
import datetime
from dotenv import load_dotenv

# --- BigQuery Imports ---
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError

# --- Vertex AI Search (Discovery Engine) Imports ---
from google.cloud import discoveryengine_v1 as discoveryengine

# --- ADK Imports ---
from google.adk.tools.tool_context import ToolContext
from google.oauth2.credentials import Credentials

# Load environment variables
load_dotenv()

# =======================================================================
# CONFIGURATION
# =======================================================================

# BigQuery Config
BIGQUERY_PROJECT_ID = os.environ.get("BIGQUERY_PROJECT_ID")
BIGQUERY_DATASET_ID = os.environ.get("BIGQUERY_DATASET_ID")

# Vertex AI Search (Datastore) Config
DATASTORE_PROJECT_ID = os.environ.get("DATASTORE_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
DATASTORE_LOCATION = os.environ.get("DATASTORE_LOCATION", "global") 
DATASTORE_ID = os.environ.get("DATASTORE_ID") 

# Auth ID for user-specific context
AUTH_ID = os.environ.get("AUTH_ID")

logging.basicConfig(level=logging.INFO)

# =======================================================================
# HELPER FUNCTIONS
# =======================================================================

def _serialize_rows(rows: list) -> list:
    """Internal helper to serialize BigQuery rows (handling dates/datetimes)."""
    serialized_rows = []
    for row in rows:
        serialized_row = {}
        # BigQuery Row objects usually act like dicts, but we force conversion 
        # to ensure compatibility with serialization logic.
        row_as_dict = dict(row.items()) if hasattr(row, 'items') else row

        for key, value in row_as_dict.items():
            if isinstance(value, (datetime.datetime, datetime.date)):
                serialized_row[key] = value.isoformat()
            else:
                serialized_row[key] = value
        serialized_rows.append(serialized_row)
    return serialized_rows


def _json_to_markdown_table(data_list: list) -> str:
    """Internal helper to convert list of dicts to Markdown table."""
    if not data_list:
        return "No results found."

    headers = list(data_list[0].keys())
    header_row = "| " + " | ".join(map(str, headers)) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"

    data_rows = []
    for row_dict in data_list:
        row_values = [str(row_dict.get(header, '')) for header in headers]
        data_rows.append("| " + " | ".join(row_values) + " |")

    return "\n".join([header_row, separator_row] + data_rows)

# =======================================================================
# TOOLS DEFINITION
# =======================================================================

# 1. VERTEX AI SEARCH TOOL (Retrieval)
# Implemented as a custom python function using the Discovery Engine API
# to avoid conflicts with mixing built-in tools and function tools.

def search_data_context(query: str, tool_context: ToolContext = None) -> str:
    """
    Searches the Vertex AI Datastore (Knowledge Base) for details about table schemas,
    column definitions, and relationships.
    
    Use this tool BEFORE writing SQL queries to ensure you have the correct table names
    and column schemas.

    Args:
        query (str): The search term. E.g., "schema for table users", "details of moodle dataset".

    Returns:
        str: Relevant text snippets and summary from the documentation files.
    """
    if not all([DATASTORE_PROJECT_ID, DATASTORE_LOCATION, DATASTORE_ID]):
        return "Error: Vertex AI Search configuration (DATASTORE_ID) is missing in .env."

    client = None

    try:
        # Attempt to get user-specific token from ADK context
        if tool_context and tool_context.state:
            access_token = tool_context.state.get(AUTH_ID)
            if access_token:
                user_creds = Credentials(token=access_token)
                client = discoveryengine.SearchServiceClient(credentials=user_creds)
            else:
                logging.warning(f"No token found for AUTH_ID {AUTH_ID} in search tool. Using default credentials.")
                client = discoveryengine.SearchServiceClient()
        else:
            client = discoveryengine.SearchServiceClient()

    except Exception as e:
        logging.warning(f"Error initializing Discovery Engine client with user token: {e}. Fallback to default.")
        client = discoveryengine.SearchServiceClient()

    try:
        # The full resource name of the search engine serving config
        serving_config = client.serving_config_path(
            project=DATASTORE_PROJECT_ID,
            location=DATASTORE_LOCATION,
            data_store=DATASTORE_ID,
            serving_config="default_search",
        )

        # Define search specs
        content_search_spec = discoveryengine.SearchRequest.ContentSearchSpec(
            snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
                return_snippet=True
            ),
            summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
                summary_result_count=3,
                include_citations=True,
                ignore_adversarial_query=True,
                ignore_non_summary_seeking_query=True,
            ),
        )

        request = discoveryengine.SearchRequest(
            serving_config=serving_config,
            query=query,
            page_size=3, # Top 3 results usually provide enough context
            content_search_spec=content_search_spec,
            query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(
                condition=discoveryengine.SearchRequest.QueryExpansionSpec.Condition.AUTO,
            ),
            spell_correction_spec=discoveryengine.SearchRequest.SpellCorrectionSpec(
                mode=discoveryengine.SearchRequest.SpellCorrectionSpec.Mode.AUTO
            ),
        )

        response = client.search(request)

        formatted_results = []
        
        # Check if we have a summary (generative answer provided by Vertex AI Search)
        if hasattr(response, 'summary') and response.summary.summary_text:
             formatted_results.append(f"### AI Summary:\n{response.summary.summary_text}\n")

        # Iterate over search results to extract snippets
        for result in response.results:
            doc_data = result.document.derived_struct_data
            title = doc_data.get("title", "Untitled Document")
            link = doc_data.get("link", "")
            
            # Extract snippets
            snippets = []
            if hasattr(result.document, "derived_struct_data") and "snippets" in result.document.derived_struct_data:
                for snippet in result.document.derived_struct_data["snippets"]:
                    snippets.append(snippet.get("snippet", ""))
            
            # If no snippets found in struct data, try to see if result object has direct access or fallback
            content_preview = "\n".join(snippets) if snippets else "No snippet available."
            
            formatted_results.append(
                f"---\n**Source:** {title}\n**Link:** {link}\n**Relevant Content:**\n...{content_preview}..."
            )

        if not formatted_results:
            return f"No detailed context found for query: '{query}'. Try different keywords."

        return "\n\n".join(formatted_results)

    except Exception as e:
        logging.error(f"Error searching Datastore: {e}")
        return f"Error retrieving context: {e}"


# 2. BIGQUERY TOOL (Execution)
# This is a custom function tool, which ADK supports natively.

def query_bigquery(sql_query: str, tool_context: ToolContext) -> dict:
    """
    Executes a raw SQL query against Google BigQuery and formats the entire
    result set into a single Markdown table.

    Args:
        sql_query (str): The complete and valid SQL query string to execute.

    Returns:
        dict: A dictionary containing a 'results_markdown' key with the data
              as a Markdown table string on success, or an 'error' key on failure.
    """
    # Validation
    if not BIGQUERY_PROJECT_ID:
        return {"error": "BIGQUERY_PROJECT_ID is not configured in .env."}
    
    client = None

    try: 
        # Attempt to get user-specific token from ADK context
        if tool_context and tool_context.state:
            access_token = tool_context.state.get(AUTH_ID)
            if access_token:
                user_creds = Credentials(token=access_token)
                client = bigquery.Client(project=BIGQUERY_PROJECT_ID, credentials=user_creds)
            else:
                logging.warning(f"No token found for AUTH_ID {AUTH_ID}. Using default credentials.")
                client = bigquery.Client(project=BIGQUERY_PROJECT_ID)
        else:
            client = bigquery.Client(project=BIGQUERY_PROJECT_ID)

    except Exception as e:        
        logging.warning(f"Error initializing BigQuery client with user token: {e}. Fallback to default.")
        client = bigquery.Client(project=BIGQUERY_PROJECT_ID)
        
    try:
        # Run the query
        query_job = client.query(sql_query)
        result_iterator = query_job.result()

        results_as_list = [dict(row.items()) for row in result_iterator]
        serialized_result = _serialize_rows(results_as_list)
        markdown_output = _json_to_markdown_table(serialized_result)

        return {"results_markdown": markdown_output}

    except GoogleAPIError as e:
        return {
            "error": "Failed to execute SQL query in BigQuery.",
            "details": f"BigQuery API Error: {e}",
            "sql_sent": sql_query
        }
    except Exception as e:
        return {
            "error": "An unexpected error occurred during execution.",
            "details": f"Error: {e}",
            "sql_sent": sql_query
        }