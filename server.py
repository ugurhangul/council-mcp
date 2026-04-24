import os
import asyncio
import httpx
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("LLM Council")

async def query_openai(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}]
    }
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data, timeout=90.0)
            resp.raise_for_status()
            return f"### OpenAI (GPT-4o) Perspective\n{resp.json()['choices'][0]['message']['content']}"
        except Exception as e:
            return f"### OpenAI Error\n{str(e)}"

async def query_anthropic(prompt: str) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    data = {
        "model": "claude-3-7-sonnet-20250219",
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": prompt}]
    }
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post("https://api.anthropic.com/v1/messages", headers=headers, json=data, timeout=90.0)
            resp.raise_for_status()
            return f"### Anthropic (Claude 3.7) Perspective\n{resp.json()['content'][0]['text']}"
        except Exception as e:
            return f"### Anthropic Error\n{str(e)}"

async def query_gemini(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts":[{"text": prompt}]}]
    }
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(url, headers=headers, json=data, timeout=90.0)
            resp.raise_for_status()
            return f"### Google (Gemini 2.5 Pro) Perspective\n{resp.json()['candidates'][0]['content']['parts'][0]['text']}"
        except Exception as e:
            return f"### Gemini Error\n{str(e)}"

async def query_ollama(prompt: str) -> str:
    model = os.getenv("OLLAMA_MODEL")
    if not model:
        return None
    
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    url = f"{host.rstrip('/')}/api/chat"
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(url, json=data, timeout=90.0)
            resp.raise_for_status()
            return f"### Ollama ({model}) Perspective\n{resp.json()['message']['content']}"
        except Exception as e:
            return f"### Ollama Error\n{str(e)}"

async def query_ollama_cloud(prompt: str) -> str:
    model = os.getenv("OLLAMA_CLOUD_MODEL")
    if not model:
        return None
    
    host = os.getenv("OLLAMA_CLOUD_HOST")
    if not host:
        return None
        
    url = f"{host.rstrip('/')}/api/chat"
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    
    # Check if there are specific auth headers for cloud (like Bearer tokens)
    headers = {}
    auth_token = os.getenv("OLLAMA_CLOUD_AUTH")
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
        
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(url, headers=headers, json=data, timeout=90.0)
            resp.raise_for_status()
            return f"### Ollama Cloud ({model}) Perspective\n{resp.json()['message']['content']}"
        except Exception as e:
            return f"### Ollama Cloud Error\n{str(e)}"

@mcp.tool()
async def consult_council(query: str) -> str:
    """Consult other AI models (ChatGPT, Claude, Gemini) for their perspectives. Use this when you want to brainstorm or get second opinions."""
    
    # Run queries in parallel
    tasks = [
        query_openai(query),
        query_anthropic(query),
        query_gemini(query),
        query_ollama(query),
        query_ollama_cloud(query)
    ]
    results = await asyncio.gather(*tasks)
    
    # Filter out None results (where API keys weren't configured)
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        return "Error: No API keys or models configured! Please set OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, OLLAMA_MODEL, or OLLAMA_CLOUD_MODEL in the .env file."
        
    return f"Here are the perspectives from the council:\n\n" + "\n\n---\n\n".join(valid_results)


if __name__ == "__main__":
    mcp.run()
