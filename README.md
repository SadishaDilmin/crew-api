# CrewAI Dev & QA Automation POC

AI-powered agent workflows for development and QA automation using CrewAI.

## Features

### 1. Project Analysis
Analyzes project ideas for feasibility and creates development plans.
- **Research Agent** - Evaluates technical feasibility
- **Planning Agent** - Creates development roadmap
- **Review Agent** - Improves and finalizes recommendations

### 2. Code Review
Comprehensive code review with multiple specialized agents.
- **Quality Analyst** - Best practices, readability, maintainability
- **Security Analyst** - Vulnerabilities and security risks
- **Performance Engineer** - Optimization opportunities
- **Summarizer** - Prioritized action items

### 3. QA Test Generation
Generates test cases for features and code.
- **Test Strategist** - Overall test strategy
- **Unit Test Specialist** - Detailed unit tests
- **Integration Test Specialist** - API and data flow tests
- **E2E Test Specialist** - User journey tests

### 4. Bug Analysis
Analyzes bugs and suggests fixes.
- **Bug Analyst** - Root cause analysis
- **Code Investigator** - Pinpoints bug location
- **Solution Architect** - Fix implementation and prevention

## Setup

### Backend

```bash
cd crew-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set Gemini API key
export GEMINI_API_KEY=your_api_key_here

# Run the server
uvicorn main:app --reload
```

### Frontend

```bash
cd crewai-poc

# Install dependencies
npm install

# Run development server
npm run dev
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze-project` | POST | Analyze project idea |
| `/code-review` | POST | Review code |
| `/generate-tests` | POST | Generate test cases |
| `/analyze-bug` | POST | Analyze bug reports |
| `/health` | GET | Health check |

## Environment Variables

- `GEMINI_API_KEY` - Required for CrewAI agents (Gemini Pro)
- `GOOGLE_API_KEY` - Optional alias if you prefer Google naming

## Tech Stack

- **Backend**: FastAPI + CrewAI
- **Frontend**: Next.js + Tailwind CSS
