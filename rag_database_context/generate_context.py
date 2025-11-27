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

import os
import json
from dotenv import load_dotenv
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError
import vertexai
from vertexai.generative_models import GenerativeModel

# =======================================================================
# PATH CONFIGURATION
# =======================================================================

# Get the directory where this script is located (rag_database_context/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the target directory (adk_bq_agent/) relative to this script
# Assuming structure:
# root/
#   adk_bq_agent/  <-- Target for outputs and .env
#   rag_database_context/ <-- Where this script lives
TARGET_AGENT_DIR = os.path.join(CURRENT_DIR, '../adk_bq_agent')

# Normalize the path
TARGET_AGENT_DIR = os.path.normpath(TARGET_AGENT_DIR)

# Load environment variables
# 1. Try standard load (current dir or parents)
load_dotenv()
# 2. Explicitly try loading from the sibling agent directory if vars missing
if not os.environ.get("BIGQUERY_PROJECT_ID"):
    env_path = os.path.join(TARGET_AGENT_DIR, '.env')
    print(f"⚠️  Env vars missing. Attempting to load .env from: {env_path}")
    load_dotenv(env_path)

# =======================================================================
# CONFIGURATION
# =======================================================================

# Configuration file path (Local to this script)
DATABASE_CONFIG_FILE = os.path.join(CURRENT_DIR, "database_config.json")

# Google Cloud AI / Gemini Configuration
GCP_PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
GCP_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION")
LLM_MODEL = os.environ.get("LLM_MODEL")

# Directory to save individual table files
OUTPUT_DIRECTORY = os.path.join(CURRENT_DIR, "database_context_details")

# The output filename for the consolidated summary context (Inside adk_bq_agent)
SUMMARY_FILENAME = os.path.join(CURRENT_DIR, "database_context.txt")

# =======================================================================
# HELPER FUNCTIONS - CONFIG & METADATA
# =======================================================================

def load_database_config():
    """Loads the BigQuery configuration from a JSON file."""
    if not os.path.exists(DATABASE_CONFIG_FILE):
        print(f"❌ Configuration file '{DATABASE_CONFIG_FILE}' not found.")
        return None
    
    try:
        with open(DATABASE_CONFIG_FILE, 'r') as f:
            config = json.load(f)
            return config
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing '{DATABASE_CONFIG_FILE}': {e}")
        return None

def get_accessible_tables(client, project_id: str, dataset_id: str):
    """Fetches a list of accessible tables for the current dataset."""
    print(f"  Fetching tables for dataset: {project_id}.{dataset_id}...")
    query = f"""
        SELECT
            t.table_name,
            t.table_type
        FROM
            `{project_id}`.`{dataset_id}`.INFORMATION_SCHEMA.TABLES AS t
        WHERE
            t.table_type = 'BASE TABLE'
        ORDER BY
            t.table_name;
    """
    try:
        query_job = client.query(query)
        tables_info = [dict(row) for row in query_job.result()]
        return tables_info
    except GoogleAPIError as e:
        print(f"    ❌ Error fetching tables for dataset {dataset_id}: {e}")
        return []

def get_table_schema(client, project_id: str, dataset_id: str, table_name: str):
    """Retrieves the schema (columns, data types) for a specific table."""
    try:
        table_ref = client.dataset(dataset_id, project=project_id).table(table_name)
        table = client.get_table(table_ref)
        
        schema_info = []
        for field in table.schema:
            schema_info.append(f"- `{field.name}`: {field.field_type}")
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
        
        rows = [dict(row) for row in query_job.result()]

        if not rows:
            return "No sample rows found."

        # Get column names
        colnames = list(rows[0].keys()) if rows else [field.name for field in query_job.schema]
        
        # Build simple text table
        header = f"| {' | '.join(colnames)} |"
        separator = f"|{'|'.join(['---'] * len(colnames))}|"
        body = "\n".join([f"| {' | '.join(map(str, row.values()))} |" for row in rows])
        
        return f"{header}\n{separator}\n{body}"
    except GoogleAPIError as e:
        return f"Could not retrieve samples. Details: {e}"

def get_column_data_analysis(client, project_id: str, dataset_id: str, table_name: str):
    """Performs basic data analysis on table columns for BigQuery."""
    analysis_lines = []
    fully_qualified_table = f"`{project_id}.{dataset_id}.{table_name}`"

    try:
        # Fetch columns to determine types
        columns_query = f"""
            SELECT column_name, data_type
            FROM `{project_id}`.`{dataset_id}`.INFORMATION_SCHEMA.COLUMNS
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position;
        """
        query_job = client.query(columns_query)
        columns_info = [dict(row) for row in query_job.result()]
    except GoogleAPIError as e:
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
            except GoogleAPIError:
                pass 

        # Analysis for string/text types
        text_types = ['STRING']
        if data_type.upper() in text_types:
            try:
                query_distinct = f"""
                    SELECT COUNT(DISTINCT {safe_col_name})
                    FROM {fully_qualified_table}
                    WHERE {safe_col_name} IS NOT NULL;
                """
                query_job_distinct = client.query(query_distinct)
                distinct_count = next(query_job_distinct.result())[0]

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
            except GoogleAPIError:
                pass 
        
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
            except GoogleAPIError:
                pass

    return analysis_lines if analysis_lines else ["No specific column analysis was possible."]

# =======================================================================
# GEMINI GENERATION FUNCTIONS
# =======================================================================

def generate_table_description(model, table_name, schema, samples):
    """
    Uses Gemini to generate a concise description of the table.
    """
    # print(f"    🧠 Generating description for `{table_name}`...")
    prompt = f"""
You are a Data Engineer analyzing a BigQuery database.
Your task is to write a **single, concise paragraph** describing the purpose and content of the table `{table_name}`.

Use the provided Schema and Sample Data to infer what this table represents.
Focus on the "What" and "Why".

Table Name: {table_name}
Schema:
{chr(10).join(schema)}
Sample Data:
{samples}

Description:
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"    ❌ Error generating description: {e}")
        return "Description not available."

def generate_dataset_overview(model, dataset_id, tables_info_list):
    """
    Generates an overview for the entire dataset, including relationships and example queries.
    """
    print(f"  🧠 Generating OVERVIEW for dataset `{dataset_id}`...")
    
    # Prepare context for the prompt
    tables_context = ""
    for info in tables_info_list:
        tables_context += f"Table: {info['name']}\nDescription: {info['description']}\nSchema Summary: {', '.join(info['schema'][:5])}...\n\n"

    prompt = f"""
You are a Senior Data Architect. Analyze the following BigQuery dataset: `{dataset_id}`.

Using the provided table descriptions and schemas, generate a comprehensive overview file content.

The output must contain:
1. **Dataset Description**: A high-level summary of what this dataset represents (e.g., LMS, E-commerce, HR).
2. **Relationships**: Explain how the tables likely relate to each other (e.g., "users table joins with orders on user_id").
3. **Example Queries**: Provide exactly 3 complex SQL queries (Standard SQL) that demonstrate how to join these tables to answer meaningful business questions.

Input Data:
{tables_context}

Output Format:
[Dataset Description Paragraph]

[Relationships Paragraph]

[Example Queries Section with 3 SQL blocks and brief explanations]
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"    ❌ Error generating dataset overview: {e}")
        return "Overview could not be generated."

# =======================================================================
# FILE SAVING FUNCTIONS
# =======================================================================

def save_table_file(directory, dataset_id, table_name, description, schema, samples, analysis):
    filename = f"table_{table_name}_details.txt"
    filepath = os.path.join(directory, filename)
    
    content = [
        f"TABLE: {dataset_id}.{table_name}",
        f"DESCRIPTION: {description}",
        "\n=== SCHEMA ===",
        *schema,
        "\n=== SAMPLES ===",
        samples,
        "\n=== ANALYSIS ===",
        *analysis
    ]
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(content))
    except IOError as e:
        print(f"    ❌ Error saving file {filename}: {e}")

def save_dataset_overview(directory, dataset_id, content):
    filename = f"dataset_{dataset_id}_overview.txt"
    filepath = os.path.join(directory, filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"    ✅ Saved overview: {filename}")
    except IOError as e:
        print(f"    ❌ Error saving overview {filename}: {e}")

def save_global_context(global_context_sections):
    """Writes the consolidated summary file organized by dataset."""
    content = "BIGQUERY DATASET CONTEXT SUMMARY\n"
    content += "=" * 50 + "\n\n"
    
    for section in global_context_sections:
        content += section + "\n"
        content += "-" * 50 + "\n\n"
    
    try:
        with open(SUMMARY_FILENAME, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\n✅ Summary context saved to: **{SUMMARY_FILENAME}**")
    except IOError as e:
        print(f"\n❌ Error saving summary file: {e}")

# =======================================================================
# MAIN
# =======================================================================

def main():
    if not all([GCP_PROJECT_ID, GCP_LOCATION, LLM_MODEL]):
        print("❌ Missing GCP/Vertex AI configuration in .env file.")
        print(f"   (Checked .env in: {os.path.join(TARGET_AGENT_DIR, '.env')})")
        return

    # Load Database Configuration
    db_config = load_database_config()
    if not db_config:
        print("❌ Could not load database configuration. Exiting.")
        return

    bq_project_id = db_config.get("project_id")
    datasets_config = db_config.get("datasets", [])

    if not bq_project_id:
        print("❌ 'project_id' missing in database_config.json.")
        return
    
    if not datasets_config:
        print("❌ 'datasets' list missing or empty in database_config.json.")
        return

    # Ensure root output dir exists in the AGENT folder
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
        print(f"Created output directory: {OUTPUT_DIRECTORY}")

    # Initialize Clients
    try:
        bq_client = bigquery.Client(project=bq_project_id)
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        model = GenerativeModel(LLM_MODEL)
    except Exception as e:
        print(f"❌ Error initializing clients: {e}")
        return

    # To store final sections for database_context.txt
    global_context_sections = []

    print(f"--- Starting Processing for Project: {bq_project_id} ---")
    print(f"--- Outputs will be saved to: {TARGET_AGENT_DIR} ---")

    for ds_config in datasets_config:
        dataset_id = ds_config.get("name")
        target_tables = ds_config.get("tables", []) # If empty, process all
        
        if not dataset_id:
            print("⚠️ Skipping entry with missing dataset name.")
            continue

        print(f"\n📂 Processing Dataset: {dataset_id}")
        
        # 1. Create dataset subdirectory
        dataset_dir = os.path.join(OUTPUT_DIRECTORY, dataset_id)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        # 2. Fetch Accessible Tables (All base tables)
        tables_info = get_accessible_tables(bq_client, bq_project_id, dataset_id)
        if not tables_info:
            print(f"   ⚠️ No accessible tables found in {dataset_id}.")
            continue
        
        # 3. Filter Tables based on config
        tables_to_process = []
        if not target_tables:
            # If list is empty, select all
            tables_to_process = tables_info
            print(f"   ℹ️  No specific tables selected. Processing all {len(tables_to_process)} tables.")
        else:
            # Filter matches
            target_set = set(target_tables)
            tables_to_process = [t for t in tables_info if t['table_name'] in target_set]
            print(f"   ℹ️  Selected {len(tables_to_process)} tables from config out of {len(tables_info)} available.")
            
            # Warn about missing tables
            found_names = {t['table_name'] for t in tables_to_process}
            missing = target_set - found_names
            if missing:
                print(f"   ⚠️ Warning: The following configured tables were not found in BigQuery: {missing}")

        if not tables_to_process:
            print("   ⚠️ No tables to process for this dataset after filtering.")
            continue

        dataset_tables_data = [] 
        dataset_summary_lines = [] 

        # 4. Process Filtered Tables
        for i, table_details in enumerate(tables_to_process):
            table_name = table_details['table_name']
            print(f"   Processing ({i+1}/{len(tables_to_process)}): {table_name}")

            # Fetch Data
            schema = get_table_schema(bq_client, bq_project_id, dataset_id, table_name)
            samples = get_sample_rows(bq_client, bq_project_id, dataset_id, table_name)
            analysis = get_column_data_analysis(bq_client, bq_project_id, dataset_id, table_name)

            # Generate Description
            description = generate_table_description(model, table_name, schema, samples)

            # Save Individual File (in subfolder)
            save_table_file(dataset_dir, dataset_id, table_name, description, schema, samples, analysis)

            # Collect info for Overview and Context
            dataset_tables_data.append({
                'name': table_name,
                'schema': schema,
                'description': description
            })
            dataset_summary_lines.append(f"  - {table_name}: {description}")

        # 5. Generate Dataset Overview
        overview_content = generate_dataset_overview(model, dataset_id, dataset_tables_data)
        save_dataset_overview(dataset_dir, dataset_id, overview_content)

        # 6. Build Section for Global Context
        overview_summary = overview_content.split('\n\n')[0] if overview_content else "No description generated."
        
        section_text = f"DATASET: {dataset_id}\n"
        section_text += f"OVERVIEW: {overview_summary}\n"
        section_text += "TABLES:\n"
        section_text += "\n".join(dataset_summary_lines)
        
        global_context_sections.append(section_text)

    # 7. Save Global Context File
    if global_context_sections:
        save_global_context(global_context_sections)
    else:
        print("\n❌ No datasets processed successfully.")

if __name__ == "__main__":
    main()