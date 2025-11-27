"""
AI Code Remediation Microservice
FastAPI application for secure code fix generation
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict
from contextlib import asynccontextmanager
import time
import logging
import difflib
from model_handler import ModelHandler
from rag_retriever import RAGRetriever
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('remediation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global instances
model_handler = None
rag_retriever = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model_handler, rag_retriever
    
    logger.info("Starting AI Code Remediation Service...")
    
    # Initialize model handler
    # Using very small model by default for low-memory systems
    model_name = os.getenv("MODEL_NAME", "Salesforce/codegen-350M-mono")
    use_gpu = os.getenv("USE_GPU", "false").lower() == "true"
    
    logger.info(f"Loading model: {model_name}")
    model_handler = ModelHandler(model_name=model_name, use_gpu=use_gpu)
    
    # Initialize RAG retriever if recipes directory exists
    if os.path.exists("recipes"):
        logger.info("Initializing RAG retriever...")
        rag_retriever = RAGRetriever(recipes_dir="recipes")
    else:
        logger.warning("Recipes directory not found. RAG will be disabled.")
    
    logger.info("Service ready!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down service...")

# Initialize FastAPI app
app = FastAPI(
    title="AI Code Remediation Service",
    description="Local LLM-powered code security fix generator",
    version="1.0.0",
    lifespan=lifespan
)

# Input/Output models
class FixRequest(BaseModel):
    language: str = Field(..., description="Programming language (e.g., java, python)")
    cwe: str = Field(..., description="CWE identifier (e.g., CWE-89)")
    code: str = Field(..., description="Vulnerable code snippet")

class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int

class FixResponse(BaseModel):
    fixed_code: str
    diff: str
    explanation: str
    model_used: str
    token_usage: TokenUsage
    latency_ms: float
    rag_context_used: Optional[bool] = False
    
    class Config:
        protected_namespaces = ()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "AI Code Remediation Service",
        "status": "running",
        "model": model_handler.model_name if model_handler else "not loaded",
        "rag_enabled": rag_retriever is not None
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model_handler is not None,
        "rag_available": rag_retriever is not None
    }

def generate_diff(original: str, fixed: str) -> str:
    """Generate unified diff between original and fixed code"""
    original_lines = original.splitlines(keepends=True)
    fixed_lines = fixed.splitlines(keepends=True)
    
    diff = difflib.unified_diff(
        original_lines,
        fixed_lines,
        fromfile='vulnerable.code',
        tofile='fixed.code',
        lineterm=''
    )
    
    return ''.join(diff)

def build_prompt(language: str, cwe: str, code: str, context: Optional[str] = None) -> str:
    """Build a concise prompt for faster processing"""
    prompt = f"""Fix the {cwe} vulnerability in this {language} code:

```{language}
{code}
```

FIXED_CODE:
```
"""
    
    return prompt

@app.post("/local_fix", response_model=FixResponse)
async def local_fix(request: FixRequest):
    """
    Generate a secure fix for vulnerable code
    
    Args:
        request: FixRequest containing language, CWE, and vulnerable code
        
    Returns:
        FixResponse with fixed code, diff, explanation, and metrics
    """
    start_time = time.time()
    
    try:
        logger.info(f"Received fix request - Language: {request.language}, CWE: {request.cwe}")
        
        # Skip RAG for faster responses (context adds too many tokens)
        rag_context = None
        rag_used = False
        # if rag_retriever:
        #     rag_context = rag_retriever.retrieve(request.cwe, request.language)
        #     rag_used = rag_context is not None
        #     if rag_used:
        #         logger.info(f"RAG context retrieved for {request.cwe}")
        
        # Build prompt
        prompt = build_prompt(request.language, request.cwe, request.code, rag_context)
        
        # Generate fix using LLM
        logger.info("Generating fix with LLM...")
        result = model_handler.generate_fix(prompt)
        
        # Parse the response - robust extraction
        response_text = result['text'].strip()
        fixed_code = ""
        explanation = f"Fixed {request.cwe} vulnerability in {request.language} code."
        
        # Try multiple extraction strategies
        if "```" in response_text:
            # Strategy 1: Extract from code blocks
            code_parts = response_text.split("```")
            for i in range(1, len(code_parts), 2):
                code_candidate = code_parts[i].strip()
                # Remove language identifier if present
                lines = code_candidate.split('\n')
                if lines and lines[0].strip().lower() in ['python', 'java', 'javascript', 'cpp', 'c', request.language.lower()]:
                    code_candidate = '\n'.join(lines[1:])
                if code_candidate and len(code_candidate) > 10:
                    fixed_code = code_candidate
                    break
        
        # Strategy 2: If no code block found, look for code-like patterns
        if not fixed_code:
            # Try to extract anything that looks like code
            lines = response_text.split('\n')
            code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
            if code_lines:
                fixed_code = '\n'.join(code_lines[:20])  # Take first 20 lines
        
        # If still no code, use original as fallback
        if not fixed_code or len(fixed_code) < 10:
            fixed_code = request.code
            logger.warning("Failed to parse fixed code from response, using original")
        
        if not explanation:
            explanation = response_text  # Use full response as explanation
        
        # Generate diff
        diff = generate_diff(request.code, fixed_code)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Log metrics
        logger.info(f"Fix generated - Latency: {latency_ms:.2f}ms, "
                   f"Input tokens: {result['input_tokens']}, "
                   f"Output tokens: {result['output_tokens']}")
        
        # Log to CSV for metrics tracking
        log_metrics(request, result, latency_ms)
        
        return FixResponse(
            fixed_code=fixed_code,
            diff=diff,
            explanation=explanation,
            model_used=model_handler.model_name,
            token_usage=TokenUsage(
                input_tokens=result['input_tokens'],
                output_tokens=result['output_tokens']
            ),
            latency_ms=latency_ms,
            rag_context_used=rag_used
        )
        
    except Exception as e:
        logger.error(f"Error processing fix request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating fix: {str(e)}")

def log_metrics(request: FixRequest, result: Dict, latency_ms: float):
    """Log metrics to CSV file"""
    import csv
    from datetime import datetime
    
    metrics_file = 'metrics.csv'
    file_exists = os.path.exists(metrics_file)
    
    with open(metrics_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'timestamp', 'language', 'cwe', 'input_tokens', 
                'output_tokens', 'latency_ms', 'model'
            ])
        writer.writerow([
            datetime.now().isoformat(),
            request.language,
            request.cwe,
            result['input_tokens'],
            result['output_tokens'],
            f"{latency_ms:.2f}",
            model_handler.model_name
        ])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)