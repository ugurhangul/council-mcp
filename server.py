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
            resp = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data, timeout=600.0)
            resp.raise_for_status()
            return f"### OpenAI (GPT-4o) Perspective\n{resp.json()['choices'][0]['message']['content']}"
        except httpx.HTTPStatusError as e:
            return f"### OpenAI Error\nHTTP {e.response.status_code}: {e.response.text}"
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
            resp = await client.post("https://api.anthropic.com/v1/messages", headers=headers, json=data, timeout=600.0)
            resp.raise_for_status()
            return f"### Anthropic (Claude 3.7) Perspective\n{resp.json()['content'][0]['text']}"
        except httpx.HTTPStatusError as e:
            return f"### Anthropic Error\nHTTP {e.response.status_code}: {e.response.text}"
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
            resp = await client.post(url, headers=headers, json=data, timeout=600.0)
            resp.raise_for_status()
            return f"### Google (Gemini 3 Flash Preview) Perspective\n{resp.json()['candidates'][0]['content']['parts'][0]['text']}"
        except httpx.HTTPStatusError as e:
            return f"### Gemini Error\nHTTP {e.response.status_code}: {e.response.text}"
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
            resp = await client.post(url, json=data, timeout=600.0)
            resp.raise_for_status()
            return f"### Ollama ({model}) Perspective\n{resp.json()['message']['content']}"
        except httpx.HTTPStatusError as e:
            return f"### Ollama Error\nHTTP {e.response.status_code}: {e.response.text}"
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
            resp = await client.post(url, headers=headers, json=data, timeout=600.0)
            resp.raise_for_status()
            return f"### Ollama Secondary ({model}) Perspective\n{resp.json()['message']['content']}"
        except httpx.HTTPStatusError as e:
            return f"### Ollama Secondary Error\nHTTP {e.response.status_code}: {e.response.text}"
        except Exception as e:
            return f"### Ollama Secondary Error\n{str(e)}"

async def query_nvidia(messages: List[Dict[str, str]]) -> str:
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        return None
    
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    model = os.getenv("NVIDIA_MODEL", "deepseek-ai/deepseek-v4-pro")
    
    data = {
        "model": model,
        "messages": messages,
        "temperature": 1,
        "top_p": 0.95,
        "max_tokens": 16384,
        "stream": False
    }
    
    # Apply specific reasoning kwargs based on the model family
    if "deepseek" in model.lower():
        data["chat_template_kwargs"] = {
            "thinking": True,
            "reasoning_effort": "high"
        }
    elif "glm" in model.lower():
        data["chat_template_kwargs"] = {
            "enable_thinking": True,
            "clear_thinking": False
        }

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post("https://integrate.api.nvidia.com/v1/chat/completions", headers=headers, json=data, timeout=600.0)
            resp.raise_for_status()
            message = resp.json()['choices'][0]['message']
            content = message.get('content', '')
            reasoning = message.get('reasoning_content', '')
            
            output = f"### NVIDIA NIM ({model}) Perspective\n"
            if reasoning:
                output += f"<reasoning>\n{reasoning}\n</reasoning>\n\n"
            output += content
            return output
        except httpx.HTTPStatusError as e:
            return f"### NVIDIA Error\nHTTP {e.response.status_code}: {e.response.text}"
        except Exception as e:
            return f"### NVIDIA Error\n{str(e)}"


@mcp.tool()
async def consult_council(query: str, history: Optional[List[Dict[str, str]]] = None, synthesize_consensus: bool = False, model_roles: Optional[Dict[str, str]] = None, target_models: Optional[List[str]] = None, files: Optional[List[str]] = None) -> str:
    """Consult other AI models (ChatGPT, Claude, Gemini, NVIDIA NIM) for their perspectives. 
    Use history parameter for conversational memory (list of dicts with 'role' and 'content').
    Set synthesize_consensus to True for the models to do a second round of debate and provide a final synthesis.
    Use model_roles to assign personas (e.g. {"openai": "Devil's Advocate"}). Valid keys: openai, anthropic, gemini, ollama_local, ollama_secondary, nvidia.
    Use target_models to route the query to specific models only (e.g. ["nvidia", "gemini"]). If None, queries all models.
    Use files to pass an array of absolute file paths. The council will read and review their contents."""
    
    if history is None:
        history = []
        
    file_contents = ""
    if files:
        for file_path in files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        file_contents += f"\n\n--- Content of {os.path.basename(file_path)} ---\n{content}\n"
                except Exception as e:
                    file_contents += f"\n\n--- Could not read {file_path}: {str(e)} ---\n"
            else:
                file_contents += f"\n\n--- File not found: {file_path} ---\n"
                
    if file_contents:
        query = f"{query}\n\nHere are the attached files for context:{file_contents}"
        
    messages = history.copy()
    messages.append({"role": "user", "content": query})
    
    if model_roles is None:
        model_roles = {}
        
    def get_messages_for_model(model_id: str):
        custom_role = model_roles.get(model_id)
        if not custom_role:
            return messages
        msgs = messages.copy()
        instruction = f"[COUNCIL ROLE ASSIGNMENT: You are acting as the {custom_role}. Please adopt this specific perspective for your response.]\n\n"
        msgs[-1] = {"role": "user", "content": instruction + msgs[-1]["content"]}
        return msgs
    
    # Phase 1: Run queries in parallel
    tasks = []
    if target_models is None or "openai" in target_models:
        tasks.append(query_openai(get_messages_for_model("openai")))
    if target_models is None or "anthropic" in target_models:
        tasks.append(query_anthropic(get_messages_for_model("anthropic")))
    if target_models is None or "gemini" in target_models:
        tasks.append(query_gemini(get_messages_for_model("gemini")))
    if target_models is None or "ollama_local" in target_models:
        tasks.append(query_ollama(get_messages_for_model("ollama_local")))
    if target_models is None or "ollama_secondary" in target_models:
        tasks.append(query_ollama_cloud(get_messages_for_model("ollama_secondary")))
    if target_models is None or "nvidia" in target_models:
        tasks.append(query_nvidia(get_messages_for_model("nvidia")))
        
    if not tasks:
        return "Error: target_models list is empty or contains invalid model names."
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
            "Please act as the council leader. Synthesize these perspectives, resolve any conflicts, and provide a final, definitive consensus answer to the user's query.\n"
            "CRITICAL REQUIREMENT: You must include a section titled '🔥 Disagreement Heatmap' where you explicitly list out any specific facts, logic, or opinions where the models disagreed. If there is high disagreement on a point, heavily warn the user!"
        )
        
        # We append this follow-up as a new user message for the consensus round
        synthesis_messages = messages.copy()
        synthesis_messages[-1] = {"role": "user", "content": consensus_prompt}
        
        synthesis_tasks = []
        if target_models is None or "openai" in target_models:
            synthesis_tasks.append(query_openai(synthesis_messages))
        if target_models is None or "anthropic" in target_models:
            synthesis_tasks.append(query_anthropic(synthesis_messages))
        if target_models is None or "gemini" in target_models:
            synthesis_tasks.append(query_gemini(synthesis_messages))
        if target_models is None or "ollama_local" in target_models:
            synthesis_tasks.append(query_ollama(synthesis_messages))
        if target_models is None or "ollama_secondary" in target_models:
            synthesis_tasks.append(query_ollama_cloud(synthesis_messages))
        if target_models is None or "nvidia" in target_models:
            synthesis_tasks.append(query_nvidia(synthesis_messages))
        synthesis_results = await asyncio.gather(*synthesis_tasks)
        valid_synthesis = [r for r in synthesis_results if r is not None]
        
        final_output = combined_perspectives + "\n\n=================================\n### COUNCIL CONSENSUS ###\n=================================\n\n" + "\n\n---\n\n".join(valid_synthesis)
        return final_output
        
    return combined_perspectives

@mcp.tool()
async def check_health() -> str:
    """Check the health and configuration of the LLM Council."""
    status = []
    
    # Check OpenAI
    if os.getenv("OPENAI_API_KEY"):
        status.append("✅ OpenAI: Configured")
    else:
        status.append("❌ OpenAI: Missing API Key")
        
    # Check Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        status.append("✅ Anthropic: Configured")
    else:
        status.append("❌ Anthropic: Missing API Key")
        
    # Check Gemini
    if os.getenv("GEMINI_API_KEY"):
        status.append("✅ Gemini: Configured (gemini-3-flash-preview)")
    else:
        status.append("❌ Gemini: Missing API Key")
        
    # Check NVIDIA NIM
    if os.getenv("NVIDIA_API_KEY"):
        nvidia_model = os.getenv("NVIDIA_MODEL", "deepseek-ai/deepseek-v4-pro")
        status.append(f"✅ NVIDIA NIM: Configured ({nvidia_model})")
    else:
        status.append("❌ NVIDIA NIM: Missing API Key")
        
    # Check Ollama Local
    ollama_model = os.getenv("OLLAMA_MODEL")
    if ollama_model:
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{host.rstrip('/')}/api/tags", timeout=3.0)
                if resp.status_code == 200:
                    status.append(f"✅ Ollama Local: Connected ({ollama_model})")
                else:
                    status.append(f"⚠️ Ollama Local: Configured ({ollama_model}) but host returned {resp.status_code}")
        except Exception:
            status.append(f"❌ Ollama Local: Configured ({ollama_model}) but host unreachable ({host})")
    else:
        status.append("➖ Ollama Local: Not Configured")
        
    # Check Ollama Secondary
    cloud_model = os.getenv("OLLAMA_CLOUD_MODEL")
    if cloud_model:
        host = os.getenv("OLLAMA_CLOUD_HOST", "http://localhost:11434")
        try:
            async with httpx.AsyncClient() as client:
                headers = {}
                auth = os.getenv("OLLAMA_CLOUD_AUTH")
                if auth:
                    headers["Authorization"] = f"Bearer {auth}"
                resp = await client.get(f"{host.rstrip('/')}/api/tags", headers=headers, timeout=3.0)
                if resp.status_code == 200:
                    status.append(f"✅ Ollama Secondary: Connected ({cloud_model})")
                else:
                    status.append(f"⚠️ Ollama Secondary: Configured ({cloud_model}) but host returned {resp.status_code}")
        except Exception:
            status.append(f"❌ Ollama Secondary: Configured ({cloud_model}) but host unreachable ({host})")
    else:
        status.append("➖ Ollama Secondary: Not Configured")

    return "### Council Health Check\n\n" + "\n".join(status)

if __name__ == "__main__":
    mcp.run()
