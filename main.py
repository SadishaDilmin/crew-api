from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from crewai import Agent, Task, Crew, LLM
from typing import Optional, List
from dotenv import load_dotenv
import os
import requests
import secrets
import httpx
from urllib.parse import urlencode

load_dotenv()

app = FastAPI(title="CrewAI Dev & QA Automation POC")

# Add CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== GitHub OAuth Configuration ====================

GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET", "")
GITHUB_REDIRECT_URI = os.getenv("GITHUB_REDIRECT_URI", "http://localhost:8000/auth/github/callback")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# In-memory token storage (use Redis/DB in production)
github_tokens = {}

# ==================== Health Check ====================

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "CrewAI Dev & QA Automation POC"}

# ==================== LLMConfiguration ====================

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "minimax")  # minimax, openai, gemini

def get_llm(force_provider: str = None) -> LLM:
    """Get LLM instance. Uses MiniMax by default, with OpenAI/Gemini fallback."""
    provider = force_provider or LLM_PROVIDER
    
    if provider == "minimax":
        api_key = os.getenv("MINIMAX_API_KEY")
        if not api_key:
            raise ValueError("MINIMAX_API_KEY not set.")
        model = os.getenv("MINIMAX_MODEL", "minimax/MiniMax-M2.5")
        # LiteLLM handles MiniMax routing automatically
        return LLM(
            model=model,
            api_key=api_key
        )
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set.")
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        return LLM(model=f"openai/{model}", api_key=api_key)
    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set.")
        model = os.getenv("GEMINI_MODEL", "gemini/gemini-2.5-flash")
        return LLM(model=model, api_key=api_key)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


# ==================== API Key Validation ====================

@app.get("/validate-api-key")
def validate_api_key():
    """Validate the configured LLM API key."""
    provider = LLM_PROVIDER
    
    try:
        if provider == "minimax":
            api_key = os.getenv("MINIMAX_API_KEY")
            if not api_key:
                raise HTTPException(status_code=400, detail="MINIMAX_API_KEY not set.")
            model_name = os.getenv("MINIMAX_MODEL", "minimax/MiniMax-M2.5")
            # Extract model name without prefix for API call
            api_model = model_name.replace("minimax/", "")
            # Test MiniMax API with a simple request
            response = requests.post(
                "https://api.minimaxi.chat/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": api_model,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 5
                },
                timeout=30
            )
            if response.status_code == 200:
                return {
                    "valid": True,
                    "provider": "MiniMax",
                    "model": model_name,
                    "message": "MiniMax API key is valid",
                    "benefits": [
                        "80% cheaper than GPT-4 for similar quality",
                        "Optimized for multi-agent workflows",
                        "Fast response times",
                        "Great for code review, test generation, bug analysis"
                    ]
                }
            else:
                raise HTTPException(status_code=401, detail=f"MiniMax API error: {response.text}")
        else:
            # Fallback validation for other providers
            llm = get_llm()
            return {"valid": True, "provider": provider, "message": "API key configured"}
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Connection error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid API key: {str(e)}")


@app.get("/validate-gemini-key")
def validate_gemini_key():
    """Legacy endpoint - redirects to new validation."""
    return validate_api_key()

# ==================== Input Models ====================

class CodeReviewInput(BaseModel):
    code: Optional[str] = None
    file_path: Optional[str] = None
    github_url: Optional[str] = None
    gitlab_url: Optional[str] = None
    language: Optional[str] = "auto-detect"
    context: Optional[str] = ""

class QATestInput(BaseModel):
    feature_description: str
    code: Optional[str] = None
    file_path: Optional[str] = None
    github_url: Optional[str] = None
    test_type: Optional[str] = "all"

class BugAnalysisInput(BaseModel):
    bug_description: str
    error_logs: Optional[str] = ""
    code: Optional[str] = None
    file_path: Optional[str] = None
    github_url: Optional[str] = None

# ==================== Utility Functions ====================

def fetch_code_from_source(code: Optional[str], file_path: Optional[str], github_url: Optional[str], gitlab_url: Optional[str]) -> str:
    """Fetch code from various sources: direct code, local file, GitHub, or GitLab."""
    if code:
        return code
    
    if file_path:
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail=f"File not found: {file_path}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
    
    if github_url:
        try:
            # Convert GitHub web URL to raw URL
            if "github.com" in github_url and "/blob/" in github_url:
                raw_url = github_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
            else:
                raw_url = github_url
            response = requests.get(raw_url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Error fetching GitHub file: {str(e)}")
    
    if gitlab_url:
        try:
            # Convert GitLab web URL to raw URL
            if "/-/blob/" in gitlab_url:
                raw_url = gitlab_url.replace("/-/blob/", "/-/raw/")
            else:
                raw_url = gitlab_url
            response = requests.get(raw_url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Error fetching GitLab file: {str(e)}")
    
    raise HTTPException(status_code=400, detail="Must provide one of: code, file_path, github_url, or gitlab_url")

# ==================== GitHub OAuth Endpoints ====================

@app.get("/auth/github/login")
def github_login():
    """Redirect to GitHub OAuth login."""
    if not GITHUB_CLIENT_ID:
        raise HTTPException(status_code=500, detail="GitHub OAuth not configured. Set GITHUB_CLIENT_ID.")
    
    state = secrets.token_urlsafe(32)
    params = {
        "client_id": GITHUB_CLIENT_ID,
        "redirect_uri": GITHUB_REDIRECT_URI,
        "scope": "repo read:user",
        "state": state,
    }
    github_auth_url = f"https://github.com/login/oauth/authorize?{urlencode(params)}"
    return {"auth_url": github_auth_url, "state": state}


@app.get("/auth/github/callback")
async def github_callback(code: str = Query(...), state: str = Query(...)):
    """Handle GitHub OAuth callback and exchange code for access token."""
    if not GITHUB_CLIENT_ID or not GITHUB_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="GitHub OAuth not configured.")
    
    # Exchange code for access token
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://github.com/login/oauth/access_token",
            data={
                "client_id": GITHUB_CLIENT_ID,
                "client_secret": GITHUB_CLIENT_SECRET,
                "code": code,
                "redirect_uri": GITHUB_REDIRECT_URI,
            },
            headers={"Accept": "application/json"},
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to exchange code for token")
        
        token_data = response.json()
        
        if "error" in token_data:
            raise HTTPException(status_code=400, detail=token_data.get("error_description", "OAuth error"))
        
        access_token = token_data.get("access_token")
        
        # Get user info
        user_response = await client.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
        )
        user_data = user_response.json()
        user_id = str(user_data.get("id"))
        
        # Store token
        github_tokens[user_id] = {
            "access_token": access_token,
            "user": user_data,
        }
        
        # Redirect to frontend with user_id
        return RedirectResponse(url=f"{FRONTEND_URL}?github_user_id={user_id}&github_login={user_data.get('login')}")


@app.get("/auth/github/user/{user_id}")
def get_github_user(user_id: str):
    """Get stored GitHub user info."""
    if user_id not in github_tokens:
        raise HTTPException(status_code=404, detail="User not found. Please login again.")
    return {"user": github_tokens[user_id]["user"]}


@app.get("/auth/github/repos/{user_id}")
async def get_github_repos(user_id: str, page: int = 1, per_page: int = 30):
    """Get user's GitHub repositories."""
    if user_id not in github_tokens:
        raise HTTPException(status_code=401, detail="Not authenticated. Please login with GitHub.")
    
    access_token = github_tokens[user_id]["access_token"]
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://api.github.com/user/repos?page={page}&per_page={per_page}&sort=updated",
            headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch repos")
        
        repos = response.json()
        return {
            "repos": [
                {
                    "id": r["id"],
                    "name": r["name"],
                    "full_name": r["full_name"],
                    "description": r["description"],
                    "private": r["private"],
                    "html_url": r["html_url"],
                    "language": r["language"],
                    "updated_at": r["updated_at"],
                    "default_branch": r["default_branch"],
                }
                for r in repos
            ]
        }


@app.get("/auth/github/repos/{user_id}/{owner}/{repo}/contents")
async def get_repo_contents(user_id: str, owner: str, repo: str, path: str = ""):
    """Get contents of a repository directory."""
    if user_id not in github_tokens:
        raise HTTPException(status_code=401, detail="Not authenticated. Please login with GitHub.")
    
    access_token = github_tokens[user_id]["access_token"]
    
    async with httpx.AsyncClient() as client:
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        response = await client.get(
            url,
            headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch contents")
        
        contents = response.json()
        
        # Handle single file vs directory
        if isinstance(contents, dict):
            return {"type": "file", "content": contents}
        
        return {
            "type": "directory",
            "contents": [
                {
                    "name": item["name"],
                    "path": item["path"],
                    "type": item["type"],
                    "size": item.get("size", 0),
                    "sha": item["sha"],
                }
                for item in contents
            ]
        }


@app.get("/auth/github/repos/{user_id}/{owner}/{repo}/file")
async def get_file_content(user_id: str, owner: str, repo: str, path: str):
    """Get content of a specific file from a repository."""
    if user_id not in github_tokens:
        raise HTTPException(status_code=401, detail="Not authenticated. Please login with GitHub.")
    
    access_token = github_tokens[user_id]["access_token"]
    
    async with httpx.AsyncClient() as client:
        # Get file content using raw endpoint
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{path}"
        response = await client.get(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch file")
        
        return {"content": response.text, "path": path}


class GitHubFileInput(BaseModel):
    user_id: str
    owner: str
    repo: str
    path: str


def fetch_code_with_github_auth(github_input: GitHubFileInput) -> str:
    """Fetch code from GitHub using authenticated user's token."""
    if github_input.user_id not in github_tokens:
        raise HTTPException(status_code=401, detail="Not authenticated. Please login with GitHub.")
    
    access_token = github_tokens[github_input.user_id]["access_token"]
    url = f"https://raw.githubusercontent.com/{github_input.owner}/{github_input.repo}/HEAD/{github_input.path}"
    
    response = requests.get(url, headers={"Authorization": f"Bearer {access_token}"}, timeout=10)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to fetch file from GitHub")
    
    return response.text


# ==================== Updated Input Models with GitHub Auth ====================

class CodeReviewInputV2(BaseModel):
    # Option 1: Direct code
    code: Optional[str] = None
    # Option 2: GitHub authenticated access
    github_auth: Optional[GitHubFileInput] = None
    # Option 3: Public URL
    github_url: Optional[str] = None
    gitlab_url: Optional[str] = None
    file_path: Optional[str] = None
    # Common fields
    language: Optional[str] = "auto-detect"
    context: Optional[str] = ""


class QATestInputV2(BaseModel):
    feature_description: str
    code: Optional[str] = None
    github_auth: Optional[GitHubFileInput] = None
    github_url: Optional[str] = None
    file_path: Optional[str] = None
    test_type: Optional[str] = "all"


class BugAnalysisInputV2(BaseModel):
    bug_description: str
    error_logs: Optional[str] = ""
    code: Optional[str] = None
    github_auth: Optional[GitHubFileInput] = None
    github_url: Optional[str] = None
    file_path: Optional[str] = None


def fetch_code_v2(
    code: Optional[str],
    github_auth: Optional[GitHubFileInput],
    github_url: Optional[str],
    gitlab_url: Optional[str],
    file_path: Optional[str]
) -> str:
    """Fetch code with GitHub auth support."""
    if code:
        return code
    
    if github_auth:
        return fetch_code_with_github_auth(github_auth)
    
    return fetch_code_from_source(None, file_path, github_url, gitlab_url)


# ==================== V2 Endpoints with GitHub Auth ====================

@app.post("/v2/code-review")
def code_review_v2(data: CodeReviewInputV2):
    """Code review with GitHub auth support."""
    code = fetch_code_v2(data.code, data.github_auth, data.github_url, data.gitlab_url, data.file_path)
    llm = get_llm()
    
    quality_agent = Agent(
        role="Code Quality Analyst",
        goal="Analyze code for quality, readability, maintainability, and adherence to best practices",
        backstory="Senior developer with expertise in clean code principles, SOLID design patterns, and code quality metrics",
        llm=llm
    )

    security_agent = Agent(
        role="Security Analyst",
        goal="Identify security vulnerabilities, potential exploits, and suggest security improvements",
        backstory="Cybersecurity expert specializing in application security, OWASP guidelines, and secure coding practices",
        llm=llm
    )

    performance_agent = Agent(
        role="Performance Engineer",
        goal="Analyze code for performance issues, optimization opportunities, and scalability concerns",
        backstory="Performance optimization specialist with experience in high-scale systems and efficient algorithms",
        llm=llm
    )

    summary_agent = Agent(
        role="Review Summarizer",
        goal="Consolidate all reviews into actionable summary with prioritized recommendations",
        backstory="Tech lead experienced in code reviews and providing constructive feedback to development teams",
        llm=llm
    )

    context_info = f"Additional context: {data.context}" if data.context else ""
    
    quality_task = Task(
        description=f"Review this {data.language} code for quality and best practices:\n\n```\n{code}\n```\n{context_info}\n\nProvide specific suggestions for improving readability, maintainability, and adherence to coding standards.",
        agent=quality_agent,
        expected_output="Detailed code quality analysis with specific improvement suggestions"
    )

    security_task = Task(
        description=f"Analyze this code for security vulnerabilities:\n\n```\n{code}\n```\n\nIdentify potential security issues, injection risks, authentication/authorization concerns, and suggest fixes.",
        agent=security_agent,
        expected_output="Security vulnerability report with risk levels and remediation steps"
    )

    performance_task = Task(
        description=f"Analyze this code for performance:\n\n```\n{code}\n```\n\nIdentify performance bottlenecks, memory issues, inefficient algorithms, and optimization opportunities.",
        agent=performance_agent,
        expected_output="Performance analysis with optimization recommendations"
    )

    summary_task = Task(
        description="Consolidate all code review findings into a prioritized action list. Categorize issues by severity (Critical, High, Medium, Low) and provide a clear summary for the development team.",
        agent=summary_agent,
        expected_output="Consolidated review summary with prioritized action items"
    )

    crew = Crew(
        agents=[quality_agent, security_agent, performance_agent, summary_agent],
        tasks=[quality_task, security_task, performance_task, summary_task],
        verbose=True
    )

    result = crew.kickoff()
    return {"result": str(result)}


@app.post("/v2/generate-tests")
def generate_tests_v2(data: QATestInputV2):
    """Generate tests with GitHub auth support."""
    code = None
    if data.code or data.github_auth or data.github_url or data.file_path:
        code = fetch_code_v2(data.code, data.github_auth, data.github_url, None, data.file_path)
    
    llm = get_llm()
    
    test_strategist = Agent(
        role="QA Test Strategist",
        goal="Design comprehensive test strategy covering all testing levels and edge cases",
        backstory="Senior QA architect with 12 years of experience in test planning and quality assurance strategies",
        llm=llm
    )

    unit_test_agent = Agent(
        role="Unit Test Specialist",
        goal="Create detailed unit test cases with mocks, assertions, and edge cases",
        backstory="Test automation engineer specializing in unit testing, TDD, and testing frameworks like Jest, PyTest, JUnit",
        llm=llm
    )

    code_context = f"\n\nRelated code:\n```\n{code}\n```" if code else ""
    
    strategy_task = Task(
        description=f"Create a test strategy for this feature:\n\n{data.feature_description}{code_context}\n\nIdentify all test levels needed, key scenarios, edge cases, and acceptance criteria.",
        agent=test_strategist,
        expected_output="Comprehensive test strategy document with test levels and key scenarios"
    )

    unit_task = Task(
        description=f"Generate unit test cases for:\n\n{data.feature_description}{code_context}\n\nInclude test setup, assertions, mocks/stubs needed, and edge cases. Provide actual test code examples.",
        agent=unit_test_agent,
        expected_output="Detailed unit test cases with code examples"
    )

    crew = Crew(
        agents=[test_strategist, unit_test_agent],
        tasks=[strategy_task, unit_task],
        verbose=True
    )

    result = crew.kickoff()
    return {"result": str(result)}


@app.post("/v2/analyze-bug")
def analyze_bug_v2(data: BugAnalysisInputV2):
    """Analyze bugs with GitHub auth support."""
    code = None
    if data.code or data.github_auth or data.github_url or data.file_path:
        code = fetch_code_v2(data.code, data.github_auth, data.github_url, None, data.file_path)
    
    llm = get_llm()
    
    bug_analyst = Agent(
        role="Bug Analyst",
        goal="Thoroughly analyze bug description and error logs to identify root cause",
        backstory="Debugging expert with deep experience in troubleshooting complex software issues across multiple tech stacks",
        llm=llm
    )

    solution_architect = Agent(
        role="Solution Architect",
        goal="Propose comprehensive fix with implementation details and prevention strategies",
        backstory="Principal engineer experienced in bug fixing, patch development, and implementing robust solutions",
        llm=llm
    )

    error_context = f"\n\nError logs:\n```\n{data.error_logs}\n```" if data.error_logs else ""
    code_context = f"\n\nRelated code:\n```\n{code}\n```" if code else ""

    analysis_task = Task(
        description=f"Analyze this bug report:\n\n{data.bug_description}{error_context}{code_context}\n\nIdentify the symptoms, potential causes, and impact of the bug.",
        agent=bug_analyst,
        expected_output="Detailed bug analysis with root cause identification"
    )

    solution_task = Task(
        description=f"Based on the bug analysis, propose a comprehensive fix:\n\nBug: {data.bug_description}{code_context}\n\nProvide implementation details, code changes, test suggestions, and prevention strategies.",
        agent=solution_architect,
        expected_output="Solution proposal with fix implementation and prevention strategies"
    )

    crew = Crew(
        agents=[bug_analyst, solution_architect],
        tasks=[analysis_task, solution_task],
        verbose=True
    )

    result = crew.kickoff()
    return {"result": str(result)}


# ==================== Previous Input Models ====================

class ProjectInput(BaseModel):
    idea: str


# Note: get_llm() is defined at the top of the file after health check


# ==================== Project Analysis Workflow ====================

@app.post("/analyze-project")
def analyze_project(data: ProjectInput):
    """Analyze a project idea for feasibility and create development plan."""
    llm = get_llm()
    
    research_agent = Agent(
        role="Project Researcher",
        goal="Analyze project idea and explain feasibility, technical requirements, and potential challenges",
        backstory="Expert software researcher with 15 years of experience evaluating project ideas and technical feasibility",
        llm=llm
    )

    planning_agent = Agent(
        role="Project Planner",
        goal="Create detailed development plan with milestones, tech stack recommendations, and timeline",
        backstory="Senior project manager experienced in agile methodologies and software development lifecycle",
        llm=llm
    )

    review_agent = Agent(
        role="Project Reviewer",
        goal="Review, improve, and provide actionable recommendations for the project plan",
        backstory="Principal architect with expertise in system design and project optimization",
        llm=llm
    )

    research_task = Task(
        description=f"Analyze this project idea thoroughly: {data.idea}. Assess technical feasibility, required technologies, potential challenges, and market viability.",
        agent=research_agent,
        expected_output="Detailed feasibility analysis with technical requirements and challenges"
    )

    planning_task = Task(
        description="Based on the research, create a comprehensive step-by-step development plan including tech stack, milestones, resource requirements, and estimated timeline.",
        agent=planning_agent,
        expected_output="Complete development plan with phases, milestones, and tech stack"
    )

    review_task = Task(
        description="Review the development plan, identify gaps, suggest improvements, and provide final recommendations for successful implementation.",
        agent=review_agent,
        expected_output="Final reviewed plan with improvements and actionable recommendations"
    )

    crew = Crew(
        agents=[research_agent, planning_agent, review_agent],
        tasks=[research_task, planning_task, review_task],
        verbose=True
    )

    result = crew.kickoff()
    return {"result": str(result)}


# ==================== Code Review Workflow ====================

@app.post("/code-review")
def code_review(data: CodeReviewInput):
    """Perform comprehensive code review with multiple specialized agents."""
    # Fetch code from source
    code = fetch_code_from_source(data.code, data.file_path, data.github_url, data.gitlab_url)
    
    llm = get_llm()
    
    quality_agent = Agent(
        role="Code Quality Analyst",
        goal="Analyze code for quality, readability, maintainability, and adherence to best practices",
        backstory="Senior developer with expertise in clean code principles, SOLID design patterns, and code quality metrics",
        llm=llm
    )

    security_agent = Agent(
        role="Security Analyst",
        goal="Identify security vulnerabilities, potential exploits, and suggest security improvements",
        backstory="Cybersecurity expert specializing in application security, OWASP guidelines, and secure coding practices",
        llm=llm
    )

    performance_agent = Agent(
        role="Performance Engineer",
        goal="Analyze code for performance issues, optimization opportunities, and scalability concerns",
        backstory="Performance optimization specialist with experience in high-scale systems and efficient algorithms",
        llm=llm
    )

    summary_agent = Agent(
        role="Review Summarizer",
        goal="Consolidate all reviews into actionable summary with prioritized recommendations",
        backstory="Tech lead experienced in code reviews and providing constructive feedback to development teams",
        llm=llm
    )

    context_info = f"Additional context: {data.context}" if data.context else ""
    
    quality_task = Task(
        description=f"Review this {data.language} code for quality and best practices:\n\n```\n{code}\n```\n{context_info}\n\nProvide specific suggestions for improving readability, maintainability, and adherence to coding standards.",
        agent=quality_agent,
        expected_output="Detailed code quality analysis with specific improvement suggestions"
    )

    security_task = Task(
        description=f"Analyze this code for security vulnerabilities:\n\n```\n{code}\n```\n\nIdentify potential security issues, injection risks, authentication/authorization concerns, and suggest fixes.",
        agent=security_agent,
        expected_output="Security vulnerability report with risk levels and remediation steps"
    )

    performance_task = Task(
        description=f"Analyze this code for performance:\n\n```\n{code}\n```\n\nIdentify performance bottlenecks, memory issues, inefficient algorithms, and optimization opportunities.",
        agent=performance_agent,
        expected_output="Performance analysis with optimization recommendations"
    )

    summary_task = Task(
        description="Consolidate all code review findings into a prioritized action list. Categorize issues by severity (Critical, High, Medium, Low) and provide a clear summary for the development team.",
        agent=summary_agent,
        expected_output="Consolidated review summary with prioritized action items"
    )

    crew = Crew(
        agents=[quality_agent, security_agent, performance_agent, summary_agent],
        tasks=[quality_task, security_task, performance_task, summary_task],
        verbose=True
    )

    result = crew.kickoff()
    return {"result": str(result)}


# ==================== QA Test Generation Workflow ====================

@app.post("/generate-tests")
def generate_tests(data: QATestInput):
    """Generate comprehensive test cases for a feature or code."""
    # Fetch code from source if provided
    code = None
    if data.code or data.file_path or data.github_url:
        code = fetch_code_from_source(data.code, data.file_path, data.github_url, None)
    
    llm = get_llm()
    
    test_strategist = Agent(
        role="QA Test Strategist",
        goal="Design comprehensive test strategy covering all testing levels and edge cases",
        backstory="Senior QA architect with 12 years of experience in test planning and quality assurance strategies",
        llm=llm
    )

    unit_test_agent = Agent(
        role="Unit Test Specialist",
        goal="Create detailed unit test cases with mocks, assertions, and edge cases",
        backstory="Test automation engineer specializing in unit testing, TDD, and testing frameworks like Jest, PyTest, JUnit",
        llm=llm
    )

    integration_test_agent = Agent(
        role="Integration Test Specialist",
        goal="Design integration test scenarios covering API contracts, data flows, and component interactions",
        backstory="QA engineer experienced in integration testing, API testing, and service interactions",
        llm=llm
    )

    e2e_test_agent = Agent(
        role="E2E Test Specialist",
        goal="Create end-to-end test scenarios covering user journeys and business workflows",
        backstory="QA automation expert in E2E testing with Cypress, Playwright, and Selenium",
        llm=llm
    )

    code_context = f"\n\nRelated code:\n```\n{code}\n```" if code else ""
    
    strategy_task = Task(
        description=f"Create a test strategy for this feature:\n\n{data.feature_description}{code_context}\n\nIdentify all test levels needed, key scenarios, edge cases, and acceptance criteria.",
        agent=test_strategist,
        expected_output="Comprehensive test strategy document with test levels and key scenarios"
    )

    tasks = [strategy_task]
    agents = [test_strategist]

    if data.test_type in ["unit", "all"]:
        unit_task = Task(
            description=f"Generate unit test cases for:\n\n{data.feature_description}{code_context}\n\nInclude test setup, assertions, mocks/stubs needed, and edge cases. Provide actual test code examples.",
            agent=unit_test_agent,
            expected_output="Detailed unit test cases with code examples"
        )
        tasks.append(unit_task)
        agents.append(unit_test_agent)

    if data.test_type in ["integration", "all"]:
        integration_task = Task(
            description=f"Generate integration test scenarios for:\n\n{data.feature_description}{code_context}\n\nCover API contracts, data flows, error handling, and service interactions.",
            agent=integration_test_agent,
            expected_output="Integration test scenarios with API test examples"
        )
        tasks.append(integration_task)
        agents.append(integration_test_agent)

    if data.test_type in ["e2e", "all"]:
        e2e_task = Task(
            description=f"Generate E2E test scenarios for:\n\n{data.feature_description}\n\nCover complete user journeys, business workflows, and critical paths.",
            agent=e2e_test_agent,
            expected_output="E2E test scenarios with step-by-step user journey tests"
        )
        tasks.append(e2e_task)
        agents.append(e2e_test_agent)

    crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=True
    )

    result = crew.kickoff()
    return {"result": str(result)}


# ==================== Bug Analysis Workflow ====================

@app.post("/analyze-bug")
def analyze_bug(data: BugAnalysisInput):
    """Analyze bug reports and suggest fixes."""
    # Fetch code from source if provided
    code = None
    if data.code or data.file_path or data.github_url:
        code = fetch_code_from_source(data.code, data.file_path, data.github_url, None)
    
    llm = get_llm()
    
    bug_analyst = Agent(
        role="Bug Analyst",
        goal="Thoroughly analyze bug description and error logs to identify root cause",
        backstory="Debugging expert with deep experience in troubleshooting complex software issues across multiple tech stacks",
        llm=llm
    )

    code_investigator = Agent(
        role="Code Investigator",
        goal="Analyze related code to pinpoint exact location and cause of the bug",
        backstory="Senior developer skilled at reading and understanding codebases to trace bugs to their source",
        llm=llm
    )

    solution_architect = Agent(
        role="Solution Architect",
        goal="Propose comprehensive fix with implementation details and prevention strategies",
        backstory="Principal engineer experienced in bug fixing, patch development, and implementing robust solutions",
        llm=llm
    )

    error_context = f"\n\nError logs:\n```\n{data.error_logs}\n```" if data.error_logs else ""
    code_context = f"\n\nRelated code:\n```\n{code}\n```" if code else ""

    analysis_task = Task(
        description=f"Analyze this bug report:\n\n{data.bug_description}{error_context}\n\nIdentify the symptoms, potential causes, and impact of the bug.",
        agent=bug_analyst,
        expected_output="Detailed bug analysis with root cause identification"
    )

    investigation_task = Task(
        description=f"Investigate the bug in the code context:{code_context}\n\nBug: {data.bug_description}\n\nPinpoint the exact location and mechanism of the bug.",
        agent=code_investigator,
        expected_output="Code investigation report with bug location and mechanism"
    )

    solution_task = Task(
        description="Based on the analysis and investigation, propose a comprehensive fix. Include: 1) Step-by-step fix implementation, 2) Code changes needed, 3) Testing approach to verify fix, 4) Prevention strategies for similar bugs.",
        agent=solution_architect,
        expected_output="Complete solution with implementation steps, code changes, and prevention strategies"
    )

    crew = Crew(
        agents=[bug_analyst, code_investigator, solution_architect],
        tasks=[analysis_task, investigation_task, solution_task],
        verbose=True
    )

    result = crew.kickoff()
    return {"result": str(result)}


# ==================== Health Check ====================

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "CrewAI Dev & QA Automation"}
