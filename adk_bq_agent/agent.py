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
from google.adk.agents import Agent
from dotenv import load_dotenv

# Import the refactored tool and the enhanced prompt for BigQuery
from . import tools
from .prompt import BIGQUERY_PROMPT 

# Load environment variables from the .env file
load_dotenv()

# --- Agent Definition ---

# The name of the model to be used by the root agent.
# This should be a powerful model capable of reasoning and function calling.
ROOT_AGENT_MODEL = os.environ.get("ROOT_AGENT_MODEL", "gemini-2.5-flash")

# This is the main agent for interacting with the BigQuery database.
# Its instruction is the comprehensive prompt we've built, which contains all the
# database context and reasoning logic. The agent's tool is the SQL query executor.
root_agent = Agent(
    name="bigquery_agent",
    model=ROOT_AGENT_MODEL,
    description="An agent that understands questions about a BigQuery database, generates SQL, executes it, and provides answers.", # Updated description
    instruction=BIGQUERY_PROMPT, 
    tools=[
        tools.query_bigquery,
    ],
)
