## Overview 
This repository houses the REST API service that backs FABRIC testbed's Q&A tool. This is for internal team members only.

## Project Structure: 
- **app.py**: code for the Flask app  
- **.env**: to store secrets 

### Secrets management 
The project depends on creating a .env file to store any secrets necessary. This is the format that .env is expected to be structured in: 
```
FLASK_SECRET_KEY=<flask-app-secret>

OPEN_AI_SECRET=<openai-secret>

QA_DB_FILE=<path-to-QA-tool-vectorstore>
CG_DB_FILE=<path-to-code-generation-tool-vectorstore>

QA_PROMPT=<QA-tool-system-prompt>
CG_PROMPT=<code-generation-tool-system-prompt>

LOG_DIR=<path-to-directory-for-app-logs>
```                                                            

*README will be updated upon restructuring next week*
