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

# generate_bigquery_prompt.py
import os
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError
from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import GenerativeModel

# Load environment variables from your .env file
load_dotenv()

# --- Configuration ---
# BigQuery Connection Details (from .env)
# These are used to connect to BigQuery and specify the target dataset
BIGQUERY_PROJECT_ID = os.environ.get("BIGQUERY_PROJECT_ID")
BIGQUERY_DATASET_ID = os.environ.get("BIGQUERY_DATASET_ID")

# Google Cloud AI / Gemini Configuration (from .env)
GCP_PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
GCP_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION")
LLM_MODEL = os.environ.get("LLM_MODEL")

# The output filename for the generated prompt context.
OUTPUT_FILENAME = "bigquery_context.txt"

# =======================================================================
# HELPER FUNCTIONS TO FETCH BIGQUERY METADATA
# =======================================================================

def get_accessible_tables(client: bigquery.Client, project_id: str, dataset_id: str) -> list:
    """
    Fetches a list of accessible tables within a specific BigQuery dataset,
    along with their descriptions and partitioning/clustering info.
    Uses INFORMATION_SCHEMA.TABLES.
    """
    print(f"  Fetching tables for dataset: {project_id}.{dataset_id}...")
    query = f"""
        SELECT
            table_name,
            option_value AS table_description,
            -- Check for partitioning columns
            ARRAY_AGG(
                CASE WHEN option_name = 'partitioning_column' THEN option_value ELSE NULL END
                IGNORE NULLS
            ) AS partitioning_columns,
            -- Check for clustering columns (assuming they are stored as options or similar)
            ARRAY_AGG(
                CASE WHEN option_name = 'clustering_column' THEN option_value ELSE NULL END
                IGNORE NULLS
            ) AS clustering_columns
        FROM
            `{project_id}.{dataset_id}.INFORMATION_SCHEMA.TABLES` AS tables
        LEFT JOIN
            `{project_id}.{dataset_id}.INFORMATION_SCHEMA.TABLE_OPTIONS` AS options
        ON
            tables.table_name = options.table_name
            AND options.option_name = 'description' -- For table description
        WHERE
            tables.table_type = 'BASE TABLE'
        GROUP BY
            tables.table_name, options.option_value
        ORDER BY
            table_name;
    """
    try:
        query_job = client.query(query)
        tables_info = []
        for row in query_job.result():
            tables_info.append({
                'table_name': row.table_name,
                'description': row.table_description,
                'partitioning_columns': row.partitioning_columns,
                'clustering_columns': row.clustering_columns
            })
        return tables_info
    except GoogleAPIError as e:
        print(f"    ❌ Error fetching tables for dataset {dataset_id}: {e}")
        return []

def get_table_schema(client: bigquery.Client, project_id: str, dataset_id: str, table_name: str) -> list:
    """
    Retrieves the schema (columns, data types, and descriptions) for a specific BigQuery table.
    Uses INFORMATION_SCHEMA.COLUMNS.
    """
    try:
        info_schema_table = f"`{project_id}.{dataset_id}.INFORMATION_SCHEMA.COLUMNS`"
        query = f"""
            SELECT column_name, data_type, description
            FROM {info_schema_table}
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position;
        """
        query_job = client.query(query)
        schema_info = []
        for col in query_job.result():
            col_desc = f" ({col.description})" if col.description else ""
            schema_info.append(f"`{col.column_name}`: **{col.data_type.upper()}**{col_desc}")
        return schema_info
    except GoogleAPIError as e:
        print(f"    ❌ Error getting schema for table `{table_name}`: {e}")
        return [f"Error retrieving schema: {e}"]

def get_sample_rows(client: bigquery.Client, project_id: str, dataset_id: str, table_name: str, limit: int = 3) -> str:
    """
    Gets a few sample rows from the BigQuery table and formats them as a Markdown table.
    Uses a fully qualified table name.
    """
    try:
        # Fully qualified table name for data tables
        full_table_path = f"`{project_id}.{dataset_id}.{table_name}`"
        query = f"SELECT * FROM {full_table_path} LIMIT {limit};"
        query_job = client.query(query)
        rows = list(query_job.result())

        if not rows:
            return f"No sample rows found for table `{table_name}`."

        # Get column names from the schema of the first row (if available)
        colnames = [field.name for field in query_job.result().schema]

        # Format as a Markdown table
        header = f"| {' | '.join(colnames)} |"
        separator = f"|{'|'.join(['---'] * len(colnames))}|"
        
        # Convert row objects to list of values, handling non-string types
        body_rows = []
        for row in rows:
            # BigQuery Row objects behave like tuples/dicts; iterate over values directly.
            # Convert each value to string to ensure markdown compatibility.
            body_rows.append(f"| {' | '.join(map(str, row))} |")
        
        return f"{header}\n{separator}\n" + "\n".join(body_rows)
    except GoogleAPIError as e:
        print(f"    ❌ Warning: Could not retrieve samples for table `{table_name}`. Error: {e}")
        return f"Could not retrieve samples for table `{table_name}`. Details: {e}"
    except Exception as e:
        print(f"    ❌ An unexpected error occurred while getting samples for `{table_name}`: {e}")
        return f"Could not retrieve samples for table `{table_name}`. Details: {e}"


def get_column_data_analysis(client: bigquery.Client, project_id: str, dataset_id: str, table_name: str) -> list:
    """
    Performs basic data analysis on BigQuery table columns.
    Queries INFORMATION_SCHEMA.COLUMNS for column details and then performs aggregate queries.
    """
    analysis_lines = []
    full_table_path = f"`{project_id}.{dataset_id}.{table_name}`"

    try:
        # Get column details from INFORMATION_SCHEMA
        info_schema_query = f"""
            SELECT column_name, data_type
            FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position;
        """
        query_job = client.query(info_schema_query)
        columns = list(query_job.result())
    except GoogleAPIError as e:
        print(f"    ❌ Could not describe table `{table_name}` for analysis. Error: {e}")
        return [f"Could not analyze table. Error: {e}"]

    for col in columns:
        col_name = col.column_name
        data_type = col.data_type.lower() # Convert to lowercase for easier comparison
        
        safe_col_name = f"`{col_name}`" # Use backticks for column names if they contain special characters or are reserved words

        # Analysis for numeric types
        numeric_types = ['int64', 'float64', 'numeric', 'bignumeric']
        if any(ntype in data_type for ntype in numeric_types):
            try:
                # Use standard SQL aggregate functions
                query = f"SELECT MIN({safe_col_name}), MAX({safe_col_name}), AVG({safe_col_name}), COUNT(DISTINCT {safe_col_name}) FROM {full_table_path};"
                query_job = client.query(query)
                min_val, max_val, avg_val, distinct_count = next(query_job.result())
                
                # Format average to two decimal places if it's a float
                avg_formatted = f"{avg_val:.2f}" if isinstance(avg_val, float) else str(avg_val)

                if all(v is not None for v in [min_val, max_val, avg_val, distinct_count]):
                     analysis_lines.append(f"- **{col_name}**: Numeric. MIN=`{min_val}`, MAX=`{max_val}`, AVG=`{avg_formatted}`, Distinct Values=`{distinct_count}`")
            except Exception as e:
                # print(f"    ⚠️ Warning: Could not perform numeric analysis for column {col_name}. Error: {e}")
                pass  # Ignore columns that can't be aggregated or other errors

        # Analysis for string/text types
        text_types = ['string']
        if any(ttype in data_type for ttype in text_types):
            try:
                # Count distinct values
                distinct_query = f"SELECT COUNT(DISTINCT {safe_col_name}) FROM {full_table_path} WHERE {safe_col_name} IS NOT NULL;"
                query_job_distinct = client.query(distinct_query)
                distinct_count = next(query_job_distinct.result())[0]

                # Get top 5 most frequent values
                top_values_query = f"""
                    SELECT {safe_col_name}
                    FROM {full_table_path}
                    WHERE {safe_col_name} IS NOT NULL
                    GROUP BY {safe_col_name}
                    ORDER BY COUNT(*) DESC
                    LIMIT 5;
                """
                query_job_top = client.query(top_values_query)
                top_values = ', '.join([f'`{row[0]}`' for row in query_job_top.result()])
                
                if distinct_count > 0:
                    analysis_lines.append(f"- **{col_name}**: Text. Distinct Values=`{distinct_count}`. Top values: {top_values}")
            except Exception as e:
                # print(f"    ⚠️ Warning: Could not perform text analysis for column {col_name}. Error: {e}")
                pass # Ignore errors on complex text fields or empty results

    return analysis_lines if analysis_lines else ["No specific column analysis was possible."]


# =======================================================================
# FUNCTION TO GENERATE PROMPT WITH GEMINI
# =======================================================================
def generate_enhanced_prompt_with_gemini(database_context: str, project_id: str, dataset_id: str):
    """
    Uses Gemini to construct the full, enhanced prompt based on BigQuery context.
    Includes instructions for BigQuery specific SQL generation.
    """
    print("\n--- Generating Full Prompt with Gemini ---")
    if not all([GCP_PROJECT_ID, GCP_LOCATION, LLM_MODEL]):
        print("❌ GCP_PROJECT_ID, GCP_LOCATION, and LLM_MODEL must be set in your .env file for Gemini.")
        return None

    try:
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        model = GenerativeModel(LLM_MODEL)
    except Exception as e:
        print(f"❌ Error initializing Vertex AI: {e}")
        return None

    instruction_for_gemini = f"""
You are an expert BigQuery SQL developer and a master prompt engineer. Your goal is to construct a complete and highly effective prompt for converting natural language questions into BigQuery Standard SQL queries.
You have been provided with a detailed breakdown of a database below under the "DATABASE INFORMATION" section.
The BigQuery project ID is '{project_id}' and the dataset ID is '{dataset_id}'. All table references in queries MUST use the fully qualified path: `project_id.dataset_id.table_name`. For example, a table named 'users' in this dataset would be referred to as `{project_id}.{dataset_id}.users`.

Your task is to generate a complete prompt that includes the following, in this exact order:
1.  An "OVERVIEW" section that you will write.
2.  The full "DATABASE INFORMATION" (Schema, Examples, and Analysis) provided to you.
3.  A section of "IMPORTANT BIGQUERY NOTES" that you will write.
4.  A section with 7 new, complex, and insightful examples of questions and their corresponding BigQuery SQL queries. These examples should demonstrate how to join the provided tables.

CRITICAL INSTRUCTIONS:
- Your entire response will be the final content for the prompt. Start your response *directly* with `## OVERVIEW:`. Do not include any preamble or other text.
- **OVERVIEW:** Write a concise, natural language summary describing what this database appears to be used for, based on the table names (proceso, tramite, etapa, tarea) and their schemas. It looks like a workflow or process management system.
- **IMPORTANT BIGQUERY NOTES:** Create a bulleted list of key BigQuery Standard SQL rules. Include notes on using fully qualified table names (`project.dataset.table`), the importance of `JOIN` clauses between the tables, using `LIKE` for partial text matches, BigQuery's data types (e.g., `STRING`, `INT64`, `TIMESTAMP`), and common date/time functions (e.g., `FORMAT_TIMESTAMP`, `DATE_TRUNC`). **Crucially, also include notes on leveraging table descriptions, column descriptions, and how to utilize partitioning and clustering columns for query optimization (e.g., filtering by partition column to reduce scanned data).** Emphasize that all queries must be valid BigQuery Standard SQL.
- **EXAMPLES:** The examples must follow the exact format: `**Question:** "..."` followed on a new line by `**SQL Query:** "..."`. The SQL query must be a single line. These examples MUST be complex, using `JOIN`s, `GROUP BY`, `WHERE` clauses, and aggregate functions (`COUNT`, `AVG`, `SUM`, `MAX`, `MIN`, etc.) to answer realistic business questions about processes, tasks, and stages. Always use the fully qualified table names in your example queries.

---
{database_context}
"""

    print("Sending request to Gemini... (This may take a moment)")
    try:
        response = model.generate_content(instruction_for_gemini)
        return response.text
    except Exception as e:
        print(f"❌ An error occurred during the Gemini API call: {e}")
        return None

# =======================================================================
# MAIN ORCHESTRATION FUNCTION
# =======================================================================
def main():
    """
    Connects to BigQuery, analyzes its metadata, uses Gemini to build a final prompt,
    and writes it to a text file.
    """
    if not all([BIGQUERY_PROJECT_ID, BIGQUERY_DATASET_ID]):
        print("❌ Exiting. Please configure BIGQUERY_PROJECT_ID and BIGQUERY_DATASET_ID "
              "environment variables in your .env file.")
        return

    print("--- Starting BigQuery Database Analysis ---")
    db_context = {"schema": {}, "examples": {}, "analysis": {}}
    client = None
    try:
        client = bigquery.Client(project=BIGQUERY_PROJECT_ID)
        print(f"✅ Successfully initialized BigQuery client for project '{BIGQUERY_PROJECT_ID}'.")
        
        # Get table information including descriptions, partitioning, and clustering
        tables_info = get_accessible_tables(client, BIGQUERY_PROJECT_ID, BIGQUERY_DATASET_ID)
        if not tables_info:
            print(f"❌ No tables found in dataset '{BIGQUERY_DATASET_ID}'. Exiting.")
            return

        for i, table_data in enumerate(tables_info):
            table_name = table_data['table_name']
            table_description = table_data['description']
            partitioning_columns = table_data['partitioning_columns']
            clustering_columns = table_data['clustering_columns']

            print(f"  ({i+1}/{len(tables_info)}) Processing table: `{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}.{table_name}`")
            
            # Add table description to schema context
            table_schema_lines = []
            if table_description:
                table_schema_lines.append(f"Table Description: {table_description}")
            if partitioning_columns:
                table_schema_lines.append(f"Partitioned by: {', '.join(f'`{col}`' for col in partitioning_columns)}")
            if clustering_columns:
                table_schema_lines.append(f"Clustered by: {', '.join(f'`{col}`' for col in clustering_columns)}")

            # Append column schema (with column descriptions)
            column_schema_list = get_table_schema(client, BIGQUERY_PROJECT_ID, BIGQUERY_DATASET_ID, table_name)
            table_schema_lines.extend(column_schema_list)

            db_context["schema"][table_name] = table_schema_lines
            db_context["examples"][table_name] = get_sample_rows(client, BIGQUERY_PROJECT_ID, BIGQUERY_DATASET_ID, table_name)
            db_context["analysis"][table_name] = get_column_data_analysis(client, BIGQUERY_PROJECT_ID, BIGQUERY_DATASET_ID, table_name)

    except GoogleAPIError as e:
        print(f"\n❌ BigQuery API Error: {e}")
        return
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        return
    finally:
        # BigQuery client does not require explicit close
        print("\n✅ Database analysis complete.")

    # Assemble all gathered information into a single string for Gemini
    info_lines = ["\n# DATABASE INFORMATION\n"]
    info_lines.append("## Database Schema:")
    for table, schema_info in db_context["schema"].items():
        info_lines.append(f"\n### Table: `{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}.{table}`")
        info_lines.extend([f"- {line}" for line in schema_info]) # Use extend for list of lines
    
    info_lines.append("\n---\n## Table Data Samples:")
    for table, example in db_context["examples"].items():
        info_lines.append(f"\n### Samples for table `{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}.{table}`:\n{example}")

    info_lines.append("\n---\n## Column Data Analysis:")
    for table, analysis in db_context["analysis"].items():
        if analysis:
            info_lines.append(f"\n### Analysis of Table `{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}.{table}`:")
            info_lines.extend(analysis)
    
    database_context_for_gemini = "\n".join(info_lines)

    # Use Gemini to generate the final prompt content
    final_prompt_content = generate_enhanced_prompt_with_gemini(
        database_context_for_gemini,
        BIGQUERY_PROJECT_ID,
        BIGQUERY_DATASET_ID
    )

    if not final_prompt_content:
        print("\n❌ Prompt generation failed. No file will be written.")
        return

    # Write the result to the output file
    try:
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            f.write(final_prompt_content)
        print(f"\n✅ Success! Prompt saved to: **{OUTPUT_FILENAME}**")
    except IOError as e:
        print(f"\n❌ Error saving file: {e}")

if __name__ == "__main__":
    main()
