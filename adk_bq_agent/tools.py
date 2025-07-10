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
# Import BigQuery client from google-cloud-bigquery library
from google.cloud import bigquery
# Import Google API error for specific exception handling
from google.api_core.exceptions import GoogleAPIError
from dotenv import load_dotenv
import datetime # Import required for handling date/time objects

# Load environment variables from the .env file
load_dotenv()

# --- BigQuery Connection Details (from .env) ---
# For BigQuery, we primarily need the Project ID and Dataset ID.
# Authentication is typically handled via Application Default Credentials (ADC)
# or by setting the GOOGLE_APPLICATION_CREDENTIALS environment variable
# to point to a service account key file.
BIGQUERY_PROJECT_ID = os.environ.get("BIGQUERY_PROJECT_ID")
BIGQUERY_DATASET_ID = os.environ.get("BIGQUERY_DATASET_ID")


def _serialize_rows(rows: list) -> list:
    """
    Internal helper to iterate through rows and convert non-serializable
    types (like date/datetime) to strings in ISO format.
    BigQuery results are typically google.cloud.bigquery.Row objects,
    which behave like dictionaries but need explicit conversion for full compatibility.
    """
    serialized_rows = []
    for row in rows:
        serialized_row = {}
        # BigQuery Row objects have a 'keys()' method and are iterable by items.
        # We'll explicitly convert them to a dictionary first to ensure consistent access.
        if hasattr(row, 'items') and callable(getattr(row, 'items')):
            # Convert BigQuery Row object to a standard dictionary
            row_as_dict = dict(row.items())
        else:
            # Assume it's already a dictionary or can be treated as one
            row_as_dict = row

        for key, value in row_as_dict.items():
            if isinstance(value, (datetime.datetime, datetime.date)):
                serialized_row[key] = value.isoformat()
            else:
                serialized_row[key] = value
        serialized_rows.append(serialized_row)
    return serialized_rows


def _json_to_markdown_table(data_list: list) -> str:
    """
    Internal helper to convert a list of dictionaries into a Markdown formatted table.
    """
    if not data_list:
        return "No results found."

    # Use the keys from the first dictionary as headers
    # Convert to list to maintain order and allow indexing if needed
    headers = list(data_list[0].keys())
    header_row = "| " + " | ".join(map(str, headers)) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"

    # Create data rows
    data_rows = []
    for row_dict in data_list:
        # Ensure values are retrieved in the same order as headers
        row_values = [str(row_dict.get(header, '')) for header in headers]
        data_rows.append("| " + " | ".join(row_values) + " |")

    return "\n".join([header_row, separator_row] + data_rows)


def query_bigquery(sql_query: str) -> dict:
    """
    Executes a raw SQL query against Google BigQuery and formats the entire
    result set into a single Markdown table.

    Args:
        sql_query (str): The complete and valid SQL query string to execute.

    Returns:
        dict: A dictionary containing a 'results_markdown' key with the data
              as a Markdown table string on success, or an 'error' key on failure.
    """
    # Ensure both PROJECT_ID and DATASET_ID are configured
    if not all([BIGQUERY_PROJECT_ID, BIGQUERY_DATASET_ID]):
        missing_vars = []
        if not BIGQUERY_PROJECT_ID:
            missing_vars.append("BIGQUERY_PROJECT_ID")
        if not BIGQUERY_DATASET_ID:
            missing_vars.append("BIGQUERY_DATASET_ID")
        return {"error": f"BigQuery connection details are not fully configured in the environment. "
                         f"Please set {', '.join(missing_vars)} in your .env file."}

    client = None
    try:
        # Initialize BigQuery client with the project ID.
        # The dataset ID will typically be part of the SQL query itself (e.g., `project.dataset.table`).
        client = bigquery.Client(project=BIGQUERY_PROJECT_ID)

        # Run the query. The .result() method blocks until the query completes.
        query_job = client.query(sql_query)
        result_iterator = query_job.result()

        # Iterate through the results. Each row is a google.cloud.bigquery.Row object.
        results_as_list_of_dicts = []
        for row in result_iterator:
            # Convert each Row object into a standard Python dictionary for consistency
            # with _serialize_rows and _json_to_markdown_table.
            results_as_list_of_dicts.append(dict(row.items()))

        # First, serialize the data to handle date/datetime objects correctly.
        serialized_result = _serialize_rows(results_as_list_of_dicts)

        # Then, convert the entire result set into a single Markdown table string.
        markdown_output = _json_to_markdown_table(serialized_result)

        # Return the final formatted string in the response dictionary.
        return {"results_markdown": markdown_output}

    except GoogleAPIError as e:
        # Catch BigQuery specific API errors
        return {
            "error": "Failed to execute SQL query in BigQuery.",
            "details": f"BigQuery API Error: {e}",
            "sql_sent": sql_query
        }
    except Exception as e:
        # Catch any other unexpected errors
        return {
            "error": "An unexpected error occurred during BigQuery query execution.",
            "details": f"Error: {e}",
            "sql_sent": sql_query
        }
    finally:
        # The BigQuery client object does not typically require an explicit close
        pass
