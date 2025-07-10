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
from dotenv import load_dotenv
import vertexai
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError
import json # Import for JSON serialization if needed for complex objects
from vertexai.preview.generative_models import GenerativeModel # Changed import path

# Load environment variables from your .env file
load_dotenv()

# --- Configuration ---
# BigQuery Connection Details (from .env)
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

def get_accessible_tables(client, project_id: str, dataset_id: str):
    """
    Fetches a list of accessible tables for the current dataset, including
    their descriptions, partitioning, and clustering columns.
    """
    print(f"  Fetching tables for dataset: {project_id}.{dataset_id}...")
    query = f"""
        SELECT
            t.table_name,
            t.table_type,
            COALESCE(td.option_value, 'No description available.') AS description,
            ARRAY_AGG(
                CASE
                    WHEN col.is_partitioning_column = 'YES' THEN col.column_name
                    ELSE NULL
                END IGNORE NULLS
            ) AS partitioning_columns,
            ARRAY_AGG(
                CASE
                    WHEN col.clustering_ordinal_position IS NOT NULL THEN col.column_name
                    ELSE NULL
                END IGNORE NULLS
            ) AS clustering_columns
        FROM
            `{project_id}`.`{dataset_id}`.INFORMATION_SCHEMA.TABLES AS t
        LEFT JOIN
            `{project_id}`.`{dataset_id}`.INFORMATION_SCHEMA.TABLE_OPTIONS AS td
            ON t.table_name = td.table_name AND td.option_name = 'description'
        LEFT JOIN
            `{project_id}`.`{dataset_id}`.INFORMATION_SCHEMA.COLUMNS AS col
            ON t.table_name = col.table_name
        WHERE
            t.table_type = 'BASE TABLE'
        GROUP BY
            t.table_name, t.table_type, td.option_value
        ORDER BY
            t.table_name;
    """
    try:
        query_job = client.query(query)
        # Convert Row objects to dictionaries
        tables_info = []
        for row in query_job.result():
            tables_info.append(dict(row))
        return tables_info
    except GoogleAPIError as e:
        print(f"    ❌ Error fetching tables for dataset {dataset_id}: {e}")
        return []

def get_table_schema(client, project_id: str, dataset_id: str, table_name: str):
    """Retrieves the schema (columns, data types, and descriptions) for a specific table."""
    try:
        # Use client.get_table to get full table metadata including column descriptions
        table_ref = client.dataset(dataset_id, project=project_id).table(table_name)
        table = client.get_table(table_ref) # API call
        
        schema_info = []
        for field in table.schema:
            col_desc = ""
            if field.description:
                col_desc = f" ({field.description})"
            schema_info.append(f"`{field.name}`: **{field.field_type.upper()}**{col_desc}")
        return schema_info
    except GoogleAPIError as e:
        print(f"    ❌ Error getting schema for table `{table_name}`: {e}")
        return [f"Error retrieving schema: {e}"]

def get_sample_rows(client, project_id: str, dataset_id: str, table_name: str, limit: int = 3):
    """Gets a few sample rows from the table and formats them as a Markdown table."""
    fully_qualified_table = f"`{project_id}.{dataset_id}.{table_name}`"
    try:
        query = f"SELECT * FROM {fully_qualified_table} LIMIT {limit};"
        query_job = client.query(query)
        
        rows = []
        for row in query_job.result():
            rows.append(dict(row)) # Convert to dictionary for easier processing

        if not rows:
            return f"No sample rows found for table `{table_name}`."

        # Get column names from the first row's keys or job schema if rows are empty
        colnames = list(rows[0].keys()) if rows else [field.name for field in query_job.schema]
        
        header = f"| {' | '.join(colnames)} |"
        separator = f"|{'|'.join(['---'] * len(colnames))}|"
        body = "\n".join([f"| {' | '.join(map(str, row.values()))} |" for row in rows]) # Use .values() to ensure order
        
        return f"{header}\n{separator}\n{body}"
    except GoogleAPIError as e:
        print(f"    ❌ Warning: Could not retrieve samples for table `{table_name}`. Error: {e}")
        return f"Could not retrieve samples for table `{table_name}`. Details: {e}"

def get_column_data_analysis(client, project_id: str, dataset_id: str, table_name: str):
    """Performs basic data analysis on table columns for BigQuery."""
    analysis_lines = []
    fully_qualified_table = f"`{project_id}.{dataset_id}.{table_name}`"

    try:
        # Fetch columns from INFORMATION_SCHEMA.COLUMNS to determine types
        columns_query = f"""
            SELECT column_name, data_type
            FROM `{project_id}`.`{dataset_id}`.INFORMATION_SCHEMA.COLUMNS
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position;
        """
        query_job = client.query(columns_query)
        columns_info = [dict(row) for row in query_job.result()]
    except GoogleAPIError as e:
        print(f"    ❌ Could not describe table `{table_name}` for analysis. Error: {e}")
        return [f"Could not analyze table. Error: {e}"]

    for col_info in columns_info:
        col_name = col_info['column_name']
        data_type = col_info['data_type']
        
        safe_col_name = f"`{col_name}`"

        # Analysis for numeric types
        numeric_types = ['INT64', 'BIGNUMERIC', 'FLOAT64', 'NUMERIC']
        if data_type.upper() in numeric_types:
            try:
                query = f"""
                    SELECT
                        MIN({safe_col_name}),
                        MAX({safe_col_name}),
                        AVG({safe_col_name}),
                        COUNT(DISTINCT {safe_col_name})
                    FROM {fully_qualified_table}
                    WHERE {safe_col_name} IS NOT NULL;
                """
                query_job = client.query(query)
                row = next(query_job.result())
                min_val, max_val, avg_val, distinct_count = row[0], row[1], row[2], row[3]
                if all(v is not None for v in [min_val, max_val, avg_val]):
                     analysis_lines.append(f"- **{col_name}**: Numeric. MIN=`{min_val}`, MAX=`{max_val}`, AVG=`{avg_val:.2f}`, Distinct Values=`{distinct_count}`")
            except GoogleAPIError as e:
                print(f"      Warning: Could not analyze numeric column {col_name}. Error: {e}")
                pass  # Ignore columns that can't be aggregated

        # Analysis for string/text types
        text_types = ['STRING']
        if data_type.upper() in text_types:
            try:
                # Get distinct count
                query_distinct = f"""
                    SELECT COUNT(DISTINCT {safe_col_name})
                    FROM {fully_qualified_table}
                    WHERE {safe_col_name} IS NOT NULL;
                """
                query_job_distinct = client.query(query_distinct)
                distinct_count = next(query_job_distinct.result())[0]

                # Get top values
                query_top_values = f"""
                    SELECT {safe_col_name}, COUNT(*) as cnt
                    FROM {fully_qualified_table}
                    WHERE {safe_col_name} IS NOT NULL
                    GROUP BY {safe_col_name}
                    ORDER BY cnt DESC
                    LIMIT 5;
                """
                query_job_top_values = client.query(query_top_values)
                top_values = ', '.join([f'`{row[0]}` ({row[1]})' for row in query_job_top_values.result()])
                
                if distinct_count > 0:
                    analysis_lines.append(f"- **{col_name}**: Text. Distinct Values=`{distinct_count}`. Top values: {top_values}")
            except GoogleAPIError as e:
                print(f"      Warning: Could not analyze text column {col_name}. Error: {e}")
                pass # Ignore errors on complex text fields
        
        # Analysis for timestamp/date types
        datetime_types = ['DATE', 'DATETIME', 'TIMESTAMP']
        if data_type.upper() in datetime_types:
            try:
                query = f"""
                    SELECT
                        MIN({safe_col_name}),
                        MAX({safe_col_name}),
                        COUNT(DISTINCT {safe_col_name})
                    FROM {fully_qualified_table}
                    WHERE {safe_col_name} IS NOT NULL;
                """
                query_job = client.query(query)
                row = next(query_job.result())
                min_val, max_val, distinct_count = row[0], row[1], row[2]
                if all(v is not None for v in [min_val, max_val]):
                    analysis_lines.append(f"- **{col_name}**: Date/Time. MIN=`{min_val}`, MAX=`{max_val}`, Distinct Values=`{distinct_count}`")
            except GoogleAPIError as e:
                print(f"      Warning: Could not analyze date/time column {col_name}. Error: {e}")
                pass


    return analysis_lines if analysis_lines else ["No specific column analysis was possible."]


# =======================================================================
# FUNCTION TO GENERATE PROMPT WITH GEMINI
# =======================================================================
def generate_enhanced_prompt_with_gemini(database_context: str):
    """Uses Gemini to construct the full, enhanced prompt based on BigQuery context."""
    print("\n--- Generating Full Prompt with Gemini ---")
    if not all([GCP_PROJECT_ID, GCP_LOCATION]):
        print("❌ GCP_PROJECT_ID and GCP_LOCATION must be set in your .env file for Gemini.")
        return None

    try:
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        # Model initialization now uses GenerativeModel imported from preview
        model = GenerativeModel(LLM_MODEL) 
    except Exception as e:
        print(f"❌ Error initializing Vertex AI: {e}")
        return None

    instruction_for_gemini = f"""
You are an expert BigQuery SQL developer and a master prompt engineer. Your goal is to construct a complete and highly effective prompt for converting natural language questions into BigQuery Standard SQL queries.
You have been provided with a detailed breakdown of a database below under the "DATABASE INFORMATION" section.

Your task is to generate a complete prompt that includes the following, in this exact order:
1.  An "OVERVIEW" section that you will write.
2.  The full "DATABASE INFORMATION" (Schema, Examples, and Analysis) provided to you.
3.  A section of "IMPORTANT BIGQUERY NOTES" that you will write.
4.  A section with 7 new, complex, and insightful examples of questions and their corresponding BigQuery Standard SQL queries. These examples should demonstrate how to join the provided tables.

CRITICAL INSTRUCTIONS:
- Your entire response will be the final content for the prompt. Start your response *directly* with `## OVERVIEW:`. Do not include any preamble or other text.
- **OVERVIEW:** Write a concise, natural language summary describing what this database appears to be used for, based on the table names (e.g., proceso, tramite, etapa, tarea) and their schemas. If table or column descriptions are available, use them to enrich this summary. It looks like a workflow or process management system.
- **IMPORTANT BIGQUERY NOTES:** Create a bulleted list of key BigQuery Standard SQL rules. Include notes on using backticks `` for table/column names, using single quotes for strings, the importance of `JOIN` clauses between the tables, using `LIKE` for partial text matches, date/time functions (e.g., `FORMAT_TIMESTAMP`, `DATE_TRUNC`), and how to leverage table descriptions, column descriptions, partitioning columns, and clustering columns for query optimization (e.g., filtering by partition column to reduce scanned data). 
    - Emphasize using fully qualified table names (project_id.dataset_id.table_name).
- **EXAMPLES:** The examples must follow the exact format: `**Question:** "..."` followed on a new line by `**SQL Query:** "..."`. The SQL query must be a single line. These examples MUST be complex, using `JOIN`s, `GROUP BY`, `WHERE` clauses, aggregate functions (`COUNT`, `AVG`, etc.), and demonstrate an understanding of BigQuery-specific features like fully qualified table names and potentially partitioning/clustering if the context provides such columns. Answer realistic business questions about processes, tasks, and stages.

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
        print("❌ Exiting. Please configure BIGQUERY_PROJECT_ID and BIGQUERY_DATASET_ID environment variables in your .env file.")
        return

    print("--- Starting BigQuery Database Analysis (It will take some minutes) ---")
    db_context = {"schema": {}, "examples": {}, "analysis": {}}
    
    try:
        client = bigquery.Client(project=BIGQUERY_PROJECT_ID)
        print(f"✅ Successfully connected to BigQuery project '{BIGQUERY_PROJECT_ID}'.")
        
        tables_info = get_accessible_tables(client, BIGQUERY_PROJECT_ID, BIGQUERY_DATASET_ID)
        if not tables_info:
            print(f"❌ No tables found in dataset '{BIGQUERY_DATASET_ID}'. Exiting.")
            return

        for i, table_details in enumerate(tables_info):
            table_name = table_details['table_name']
            table_description = table_details['description']
            partitioning_cols = table_details['partitioning_columns']
            clustering_cols = table_details['clustering_columns']

            print(f"  ({i+1}/{len(tables_info)}) Processing table: `{table_name}`")
            
            # Add table description to schema context
            db_context["schema"][table_name] = [f"Table Description: {table_description}"]
            if partitioning_cols:
                db_context["schema"][table_name].append(f"Partitioned by: {', '.join([f'`{col}`' for col in partitioning_cols])}")
            if clustering_cols:
                db_context["schema"][table_name].append(f"Clustered by: {', '.join([f'`{col}`' for col in clustering_cols])}")

            # Append column schema to the existing list
            column_schema = get_table_schema(client, BIGQUERY_PROJECT_ID, BIGQUERY_DATASET_ID, table_name)
            db_context["schema"][table_name].extend(column_schema)

            db_context["examples"][table_name] = get_sample_rows(client, BIGQUERY_PROJECT_ID, BIGQUERY_DATASET_ID, table_name)
            db_context["analysis"][table_name] = get_column_data_analysis(client, BIGQUERY_PROJECT_ID, BIGQUERY_DATASET_ID, table_name)

    except GoogleAPIError as e:
        print(f"\n❌ BigQuery Client Error: {e}")
        return
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        return
    finally:
        print("\n✅ Database analysis complete.")

    # Assemble all gathered information into a single string for Gemini
    info_lines = ["\n# DATABASE INFORMATION\n"]
    info_lines.append("## Database Schema:")
    for table, schema_details in db_context["schema"].items():
        info_lines.append(f"\n### Table: `{table}`")
        info_lines.extend([f"- {item}" for item in schema_details])
    
    info_lines.append("\n---\n## Table Data Samples:")
    for table, example in db_context["examples"].items():
        info_lines.append(f"\n### Samples for table `{table}`:\n{example}")

    info_lines.append("\n---\n## Column Data Analysis:")
    for table, analysis in db_context["analysis"].items():
        if analysis:
            info_lines.append(f"\n### Analysis of Table `{table}`:")
            info_lines.extend(analysis)
    
    database_context_for_gemini = "\n".join(info_lines)

    # Use Gemini to generate the final prompt content
    final_prompt_content = generate_enhanced_prompt_with_gemini(database_context_for_gemini)

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
