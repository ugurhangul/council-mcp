import os
import asyncio
import httpx
from typing import List, Dict, Optional
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("LLM Council")

async def query_openai(messages: List[Dict[str, str]]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-4o",
        "messages": messages
    }
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data, timeout=90.0)
            resp.raise_for_status()
            return f"### OpenAI (GPT-4o) Perspective\n{resp.json()['choices'][0]['message']['content']}"
        except Exception as e:
            return f"### OpenAI Error\n{str(e)}"

async def query_anthropic(messages: List[Dict[str, str]]) -> str:
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
        "messages": messages
    }
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post("https://api.anthropic.com/v1/messages", headers=headers, json=data, timeout=90.0)
            resp.raise_for_status()
            return f"### Anthropic (Claude 3.7) Perspective\n{resp.json()['content'][0]['text']}"
        except Exception as e:
            return f"### Anthropic Error\n{str(e)}"

async def query_gemini(messages: List[Dict[str, str]]) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    
    gemini_contents = []
    for msg in messages:
        role = "model" if msg["role"] == "assistant" else "user"
        gemini_contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": gemini_contents
    }
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(url, headers=headers, json=data, timeout=90.0)
            resp.raise_for_status()
            return f"### Google (Gemini 3 Flash Preview) Perspective\n{resp.json()['candidates'][0]['content']['parts'][0]['text']}"
        except Exception as e:
            return f"### Gemini Error\n{str(e)}"

async def query_ollama(messages: List[Dict[str, str]]) -> str:
    model = os.getenv("OLLAMA_MODEL")
    if not model:
        return None
    
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    url = f"{host.rstrip('/')}/api/chat"
    data = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(url, json=data, timeout=90.0)
            resp.raise_for_status()
            return f"### Ollama ({model}) Perspective\n{resp.json()['message']['content']}"
        except Exception as e:
            return f"### Ollama Error\n{str(e)}"

async def query_ollama_cloud(messages: List[Dict[str, str]]) -> str:
    model = os.getenv("OLLAMA_CLOUD_MODEL")
    if not model:
        return None
    
    host = os.getenv("OLLAMA_CLOUD_HOST", "http://localhost:11434")

    url = f"{host.rstrip('/')}/api/chat"
    data = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    
    headers = {}
    auth_token = os.getenv("OLLAMA_CLOUD_AUTH")
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
        
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(url, headers=headers, json=data, timeout=90.0)
            resp.raise_for_status()
            return f"### Ollama Secondary ({model}) Perspective\n{resp.json()['message']['content']}"
        except Exception as e:
            return f"### Ollama Secondary Error\n{str(e)}"


@mcp.tool()
async def consult_council(query: str, history: Optional[List[Dict[str, str]]] = None, synthesize_consensus: bool = False) -> str:
    """Consult other AI models (ChatGPT, Claude, Gemini) for their perspectives. 
    Use history parameter for conversational memory (list of dicts with 'role' and 'content').
    Set synthesize_consensus to True for the models to do a second round of debate and provide a final synthesis."""
    
    if history is None:
        history = []
        
    messages = history.copy()
    messages.append({"role": "user", "content": query})
    
    # Phase 1: Run queries in parallel
    tasks = [
        query_openai(messages),
        query_anthropic(messages),
        query_gemini(messages),
        query_ollama(messages),
        query_ollama_cloud(messages)
    ]
    results = await asyncio.gather(*tasks)
    
    # Filter out None results (where API keys weren't configured)
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        return "Error: No API keys or models configured! Please set OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, OLLAMA_MODEL, or OLLAMA_CLOUD_MODEL in the .env file."
        
    combined_perspectives = "Here are the initial perspectives from the council:\n\n" + "\n\n---\n\n".join(valid_results)
    
    # Phase 2: Synthesis / Cross-talk (if requested)
    if synthesize_consensus and len(valid_results) > 1:
        consensus_prompt = (
            "You are participating in an AI council debate. A user asked the following query:\n"
            f"<query>\n{query}\n</query>\n\n"
            "Here are the perspectives provided by various AI models on the council:\n"
            f"<perspectives>\n{combined_perspectives}\n</perspectives>\n\n"
            "Please act as the council leader. Synthesize these perspectives, resolve any conflicts, and provide a final, definitive consensus answer to the user's query."
        )
        
        # We append this follow-up as a new user message for the consensus round
        synthesis_messages = messages.copy()
        synthesis_messages[-1] = {"role": "user", "content": consensus_prompt}
        
        synthesis_tasks = [
            query_openai(synthesis_messages),
            query_anthropic(synthesis_messages),
            query_gemini(synthesis_messages),
            query_ollama(synthesis_messages),
            query_ollama_cloud(synthesis_messages)
        ]
        synthesis_results = await asyncio.gather(*synthesis_tasks)
        valid_synthesis = [r for r in synthesis_results if r is not None]
        
        final_output = combined_perspectives + "\n\n=================================\n### COUNCIL CONSENSUS ###\n=================================\n\n" + "\n\n---\n\n".join(valid_synthesis)
        return final_output
        
    return combined_perspectives

if __name__ == "__main__":
    mcp.run()
