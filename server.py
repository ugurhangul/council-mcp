import os
import sys
import asyncio
import httpx
from typing import List, Dict, Optional
from mcp.server.fastmcp import FastMCP, Context
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("LLM Council")

LOG_FILE = os.path.join(os.path.dirname(__file__), "council_progress.md")

# Split timeouts: fail fast on connection (10s), allow long reads for thinking models (300s)
API_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)
# Extended timeout for deep-thinking NVIDIA models (GLM, DeepSeek reasoning) that can take 10+ minutes
API_TIMEOUT_EXTENDED = httpx.Timeout(connect=10.0, read=600.0, write=10.0, pool=10.0)

# Default model names — override via environment variables (set in mcp.json env block)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20250414")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")

def update_log(msg: str):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n\n")
    except Exception:
        pass

async def query_openai(messages: List[Dict[str, str]]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": OPENAI_MODEL,
        "messages": messages
    }
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
        try:
            resp = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
            resp.raise_for_status()
            return f"### OpenAI ({OPENAI_MODEL}) Perspective\n{resp.json()['choices'][0]['message']['content']}"
        except httpx.ConnectError:
            return f"### OpenAI Error\nConnection failed — server unreachable"
        except httpx.TimeoutException as e:
            return f"### OpenAI Error\nTimeout: {type(e).__name__}"
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
        "model": ANTHROPIC_MODEL,
        "max_tokens": 8192,
        "messages": messages
    }
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
        try:
            resp = await client.post("https://api.anthropic.com/v1/messages", headers=headers, json=data)
            resp.raise_for_status()
            return f"### Anthropic ({ANTHROPIC_MODEL}) Perspective\n{resp.json()['content'][0]['text']}"
        except httpx.ConnectError:
            return f"### Anthropic Error\nConnection failed — server unreachable"
        except httpx.TimeoutException as e:
            return f"### Anthropic Error\nTimeout: {type(e).__name__}"
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
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": gemini_contents
    }
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
        try:
            resp = await client.post(url, headers=headers, json=data)
            resp.raise_for_status()
            return f"### Google ({GEMINI_MODEL}) Perspective\n{resp.json()['candidates'][0]['content']['parts'][0]['text']}"
        except httpx.ConnectError:
            return f"### Gemini Error\nConnection failed — server unreachable"
        except httpx.TimeoutException as e:
            return f"### Gemini Error\nTimeout: {type(e).__name__}"
        except httpx.HTTPStatusError as e:
            return f"### Gemini Error\nHTTP {e.response.status_code}: {e.response.text}"
        except Exception as e:
            return f"### Gemini Error\n{str(e)}"

async def query_gemini_cli(messages: List[Dict[str, str]]) -> str:
    enabled = os.getenv("GEMINI_CLI_ENABLED", "").lower() == "true"
    if not enabled:
        return None
    
    # Build prompt from the last user message, prepend history as context
    prompt = messages[-1]["content"] if messages else ""
    if not prompt:
        return None
    
    if len(messages) > 1:
        history_text = ""
        for msg in messages[:-1]:
            role = msg["role"].capitalize()
            history_text += f"{role}: {msg['content']}\n\n"
        prompt = f"Previous conversation:\n{history_text}\nCurrent question:\n{prompt}"
    
    gemini_cli_path = os.getenv("GEMINI_CLI_PATH", "gemini")
    model = os.getenv("GEMINI_CLI_MODEL", "")
    
    cmd = [gemini_cli_path, "-p", prompt]
    if model:
        cmd.extend(["-m", model])
    
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300.0)
        
        if proc.returncode != 0:
            err_msg = stderr.decode("utf-8", errors="replace").strip()
            return f"### Gemini CLI Error\nExit code {proc.returncode}: {err_msg}"
        
        output = stdout.decode("utf-8", errors="replace").strip()
        if not output:
            return f"### Gemini CLI Error\nEmpty response from subprocess"
        
        display = f"Gemini CLI ({model})" if model else "Gemini CLI"
        return f"### {display} Perspective\n{output}"
    except asyncio.TimeoutError:
        return f"### Gemini CLI Error\nTimeout: subprocess exceeded 300s"
    except FileNotFoundError:
        return f"### Gemini CLI Error\n`gemini` command not found — install with: npm install -g @google/gemini-cli"
    except Exception as e:
        return f"### Gemini CLI Error\n{str(e)}"

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
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
        try:
            resp = await client.post(url, json=data)
            resp.raise_for_status()
            return f"### Ollama ({model}) Perspective\n{resp.json()['message']['content']}"
        except httpx.ConnectError:
            return f"### Ollama Error\nConnection failed — is Ollama running at {host}?"
        except httpx.TimeoutException as e:
            return f"### Ollama Error\nTimeout: {type(e).__name__}"
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
        
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
        try:
            resp = await client.post(url, headers=headers, json=data)
            resp.raise_for_status()
            return f"### Ollama Secondary ({model}) Perspective\n{resp.json()['message']['content']}"
        except httpx.ConnectError:
            return f"### Ollama Secondary Error\nConnection failed — is cloud host reachable?"
        except httpx.TimeoutException as e:
            return f"### Ollama Secondary Error\nTimeout: {type(e).__name__}"
        except httpx.HTTPStatusError as e:
            return f"### Ollama Secondary Error\nHTTP {e.response.status_code}: {e.response.text}"
        except Exception as e:
            return f"### Ollama Secondary Error\n{str(e)}"

async def query_nvidia(messages: List[Dict[str, str]], model: str, thinking_mode: bool = True) -> str:
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        return None
    
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    data = {
        "model": model,
        "messages": messages,
        "temperature": 1,
        "top_p": 0.95,
        "max_tokens": 16384,
        "stream": False
    }
    
    # Apply specific reasoning kwargs based on the model family if thinking mode is enabled
    if thinking_mode:
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
            
    if "mistral" in model.lower() or "devstral" in model.lower():
        data["max_tokens"] = 8192
        data["temperature"] = 0.15
        data["seed"] = 42

    # Use extended timeout for thinking-mode models that produce long reasoning chains
    timeout = API_TIMEOUT_EXTENDED if thinking_mode else API_TIMEOUT
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post("https://integrate.api.nvidia.com/v1/chat/completions", headers=headers, json=data)
            
            # Detect queue — NVIDIA returns 202 when the model is overloaded
            if resp.status_code == 202:
                req_id = resp.headers.get("nvcf-reqid", "unknown")
                return f"### NVIDIA NIM ({model}) — QUEUED, ABORTED\n⏳ Model is overloaded. Request was queued (ID: `{req_id}`). Skipping to avoid long wait."
            
            resp.raise_for_status()
            body = resp.json()
            
            # Defensive: ensure the response has the expected structure
            if 'choices' not in body or not body['choices']:
                return f"### NVIDIA Error ({model})\nUnexpected response — no 'choices' in payload:\n```json\n{resp.text[:500]}\n```"
            
            message = body['choices'][0]['message']
            content = message.get('content', '')
            reasoning = message.get('reasoning_content', '') or message.get('reasoning', '')
            
            output = f"### NVIDIA NIM ({model}) Perspective\n"
            if reasoning:
                output += f"<reasoning>\n{reasoning}\n</reasoning>\n\n"
            output += content
            return output
        except httpx.ConnectError:
            return f"### NVIDIA Error\nConnection failed — NVIDIA NIM unreachable"
        except httpx.TimeoutException as e:
            return f"### NVIDIA Error ({model})\nTimeout: {type(e).__name__} — model may need thinking_mode=False"
        except httpx.HTTPStatusError as e:
            return f"### NVIDIA Error\nHTTP {e.response.status_code}: {e.response.text}"
        except Exception as e:
            return f"### NVIDIA Error\n{str(e)}"

async def query_openrouter(messages: List[Dict[str, str]], model: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/ugurhangul/council-mcp",
        "X-Title": "LLM Council"
    }
    
    # Extract a short display name from the model id (e.g. "google/gemini-2.5-flash" -> "Gemini 2.5 Flash")
    display_name = model.split("/")[-1].replace("-", " ").title() if "/" in model else model
    
    data = {
        "model": model,
        "messages": messages
    }
    
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
        try:
            resp = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
            resp.raise_for_status()
            body = resp.json()
            
            # OpenRouter free models sometimes return error objects without 'choices'
            if 'choices' not in body or not body['choices']:
                error_msg = body.get('error', {}).get('message', '') if isinstance(body.get('error'), dict) else str(body.get('error', ''))
                if error_msg:
                    return f"### OpenRouter Error ({model})\nAPI returned error: {error_msg}"
                return f"### OpenRouter Error ({model})\nUnexpected response — no 'choices' in payload:\n```json\n{resp.text[:500]}\n```"
            
            content = body['choices'][0].get('message', {}).get('content', '')
            if not content:
                return f"### OpenRouter Error ({model})\nEmpty response from model (content was null/empty)"
            
            return f"### OpenRouter ({display_name}) Perspective\n{content}"
        except httpx.ConnectError:
            return f"### OpenRouter Error ({model})\nConnection failed — OpenRouter unreachable"
        except httpx.TimeoutException as e:
            return f"### OpenRouter Error ({model})\nTimeout: {type(e).__name__}"
        except httpx.HTTPStatusError as e:
            return f"### OpenRouter Error ({model})\nHTTP {e.response.status_code}: {e.response.text}"
        except Exception as e:
            return f"### OpenRouter Error ({model})\n{str(e)}"


@mcp.tool()
async def consult_council(query: str, ctx: Context = None, history: Optional[List[Dict[str, str]]] = None, synthesize_consensus: bool = False, model_roles: Optional[Dict[str, str]] = None, target_models: Optional[List[str]] = None, files: Optional[List[str]] = None, thinking_mode: bool = True) -> str:
    """Consult other AI models (ChatGPT, Claude, Gemini, Gemini CLI, NVIDIA NIM, OpenRouter) for their perspectives. 
    Use history parameter for conversational memory (list of dicts with 'role' and 'content').
    Set synthesize_consensus to True for the models to do a second round of debate and provide a final synthesis.
    Use model_roles to assign personas (e.g. {"openai": "Devil's Advocate"}). Valid keys: openai, anthropic, gemini, gemini_cli, ollama_local, ollama_secondary, nvidia, openrouter.
    Use target_models to route the query to specific models only (e.g. ["nvidia_1", "gemini", "gemini_cli", "openrouter_1"]). If None, queries all models.
    Use files to pass an array of absolute file paths. The council will read and review their contents.
    Use thinking_mode=False to explicitly disable reasoning passes on complex models like DeepSeek to dramatically speed up inference."""
    
    if history is None:
        history = []
        
    # File ingestion pipeline
    MAX_FILE_SIZE = 50_000  # 50KB per file to stay within token limits
    LANG_MAP = {
        ".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "tsx",
        ".jsx": "jsx", ".cs": "csharp", ".java": "java", ".go": "go",
        ".rs": "rust", ".rb": "ruby", ".php": "php", ".swift": "swift",
        ".kt": "kotlin", ".cpp": "cpp", ".c": "c", ".h": "c",
        ".html": "html", ".css": "css", ".scss": "scss",
        ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
        ".xml": "xml", ".sql": "sql", ".sh": "bash", ".ps1": "powershell",
        ".md": "markdown", ".txt": "text", ".env": "text", ".ini": "ini",
        ".dockerfile": "dockerfile", ".tf": "hcl",
    }
    BINARY_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
                         ".mp3", ".mp4", ".wav", ".avi", ".mov", ".pdf", ".zip",
                         ".tar", ".gz", ".exe", ".dll", ".so", ".bin", ".woff", ".woff2"}
    
    file_contents = ""
    if files:
        import glob as glob_mod
        
        # Expand globs and directories into individual file paths
        expanded_paths = []
        for file_path in files:
            if "*" in file_path or "?" in file_path:
                expanded_paths.extend(glob_mod.glob(file_path, recursive=True))
            elif os.path.isdir(file_path):
                for root, _, filenames in os.walk(file_path):
                    for fname in filenames:
                        expanded_paths.append(os.path.join(root, fname))
            else:
                expanded_paths.append(file_path)
        
        for file_path in expanded_paths:
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext in BINARY_EXTENSIONS:
                file_contents += f"\n\n--- Skipped binary file: {os.path.basename(file_path)} ---\n"
                continue
                
            if not os.path.exists(file_path):
                file_contents += f"\n\n--- File not found: {file_path} ---\n"
                continue
                
            try:
                file_size = os.path.getsize(file_path)
                if file_size > MAX_FILE_SIZE:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read(MAX_FILE_SIZE)
                    content += f"\n\n... [TRUNCATED — file is {file_size:,} bytes, showing first {MAX_FILE_SIZE:,}] ..."
                else:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                
                lang = LANG_MAP.get(ext, "")
                file_contents += f"\n\n--- {os.path.basename(file_path)} ({file_size:,} bytes) ---\n```{lang}\n{content}\n```\n"
            except Exception as e:
                file_contents += f"\n\n--- Could not read {file_path}: {str(e)} ---\n"
                
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
    
    disable_local = os.getenv("DISABLE_LOCAL", "").lower() == "true"
    disable_cloud = os.getenv("DISABLE_CLOUD", "").lower() == "true"
    
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("# 🏛️ AI Council Live Status\n\n")
    
    if ctx:
        ctx.info("🏛️ The AI Council is assembling...")
    sys.stderr.write("🏛️ The AI Council is assembling...\n")
    sys.stderr.flush()
    update_log("⏳ **The AI Council is assembling...**")
        
    async def fetch_and_notify(model_name: str, coro):
        import time
        if ctx:
            ctx.info(f"🧠 [{model_name}] started thinking...")
        sys.stderr.write(f"🧠 [{model_name}] started thinking...\n")
        sys.stderr.flush()
        update_log(f"🧠 `[{model_name}]` started thinking...")
        
        start_time = time.monotonic()
        
        try:
            result = await coro
            elapsed = time.monotonic() - start_time
            
            # Check only the first line for error markers to avoid false positives
            # from valid responses that happen to contain "Error" in their body text
            first_line = (result or "").split("\n")[0]
            if result and "Error" in first_line and result.startswith("###"):
                update_log(f"❌ `[{model_name}]` encountered an API ERROR ({elapsed:.1f}s):\n```text\n{result}\n```")
            else:
                update_log(f"✅ `[{model_name}]` has delivered its perspective! ({elapsed:.1f}s)")
                
            if ctx:
                ctx.info(f"✅ [{model_name}] finished! ({elapsed:.1f}s)")
            sys.stderr.write(f"✅ [{model_name}] finished! ({elapsed:.1f}s)\n")
            sys.stderr.flush()
            
            return result
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start_time
            update_log(f"⏱️ `[{model_name}]` TIMED OUT after {elapsed:.1f}s")
            return f"### {model_name} Error\nTimed out after {elapsed:.1f}s"
        except Exception as e:
            elapsed = time.monotonic() - start_time
            update_log(f"❌ `[{model_name}]` FATAL ERROR ({elapsed:.1f}s):\n```text\n{str(e)}\n```")
            return f"### {model_name} Error\n{str(e)}"
    
    # Phase 1: Run queries in parallel
    tasks = []
    if (target_models is None or "openai" in target_models) and not disable_cloud:
        tasks.append(fetch_and_notify("OpenAI", query_openai(get_messages_for_model("openai"))))
    if (target_models is None or "anthropic" in target_models) and not disable_cloud:
        tasks.append(fetch_and_notify("Anthropic", query_anthropic(get_messages_for_model("anthropic"))))
    if (target_models is None or "gemini" in target_models) and not disable_cloud:
        tasks.append(fetch_and_notify("Gemini", query_gemini(get_messages_for_model("gemini"))))
    if (target_models is None or "gemini_cli" in target_models) and not disable_cloud:
        tasks.append(fetch_and_notify("Gemini CLI", query_gemini_cli(get_messages_for_model("gemini_cli"))))
    if (target_models is None or "ollama_local" in target_models) and not disable_local:
        tasks.append(fetch_and_notify("Ollama Local", query_ollama(get_messages_for_model("ollama_local"))))
    if (target_models is None or "ollama_secondary" in target_models) and not disable_cloud:
        tasks.append(fetch_and_notify("Ollama Cloud", query_ollama_cloud(get_messages_for_model("ollama_secondary"))))
        
    nvidia_models_str = os.getenv("NVIDIA_MODELS")
    nvidia_models_list = []
    if nvidia_models_str:
        nvidia_models_list = [m.strip() for m in nvidia_models_str.split(",") if m.strip()]
    elif os.getenv("NVIDIA_MODEL"):
        nvidia_models_list = [os.getenv("NVIDIA_MODEL")]
        
    for i, model in enumerate(nvidia_models_list):
        slot_name = f"nvidia_{i+1}" if len(nvidia_models_list) > 1 else "nvidia"
        if (target_models is None or slot_name in target_models) and not disable_cloud:
            tasks.append(fetch_and_notify(f"NVIDIA {model}", query_nvidia(get_messages_for_model(slot_name), model, thinking_mode)))
    
    # OpenRouter models
    openrouter_models_str = os.getenv("OPENROUTER_MODELS")
    openrouter_models_list = []
    if openrouter_models_str:
        openrouter_models_list = [m.strip() for m in openrouter_models_str.split(",") if m.strip()]
    elif os.getenv("OPENROUTER_MODEL"):
        openrouter_models_list = [os.getenv("OPENROUTER_MODEL")]
        
    for i, model in enumerate(openrouter_models_list):
        slot_name = f"openrouter_{i+1}" if len(openrouter_models_list) > 1 else "openrouter"
        if (target_models is None or slot_name in target_models) and not disable_cloud:
            tasks.append(fetch_and_notify(f"OpenRouter {model}", query_openrouter(get_messages_for_model(slot_name), model)))
            
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
        
        if ctx:
            ctx.info("⚖️ The Council is reviewing all perspectives for synthesis...")
        sys.stderr.write("⚖️ The Council is reviewing all perspectives for synthesis...\n")
        sys.stderr.flush()
        update_log("⚖️ **The Council is reviewing all perspectives for synthesis...**")
            
        synthesis_tasks = []
        if (target_models is None or "openai" in target_models) and not disable_cloud:
            synthesis_tasks.append(fetch_and_notify("OpenAI (Synthesis)", query_openai(synthesis_messages)))
        if (target_models is None or "anthropic" in target_models) and not disable_cloud:
            synthesis_tasks.append(fetch_and_notify("Anthropic (Synthesis)", query_anthropic(synthesis_messages)))
        if (target_models is None or "gemini" in target_models) and not disable_cloud:
            synthesis_tasks.append(fetch_and_notify("Gemini (Synthesis)", query_gemini(synthesis_messages)))
        if (target_models is None or "gemini_cli" in target_models) and not disable_cloud:
            synthesis_tasks.append(fetch_and_notify("Gemini CLI (Synthesis)", query_gemini_cli(synthesis_messages)))
        if (target_models is None or "ollama_local" in target_models) and not disable_local:
            synthesis_tasks.append(fetch_and_notify("Ollama Local (Synthesis)", query_ollama(synthesis_messages)))
        if (target_models is None or "ollama_secondary" in target_models) and not disable_cloud:
            synthesis_tasks.append(fetch_and_notify("Ollama Cloud (Synthesis)", query_ollama_cloud(synthesis_messages)))
            
        for i, model in enumerate(nvidia_models_list):
            slot_name = f"nvidia_{i+1}" if len(nvidia_models_list) > 1 else "nvidia"
            if (target_models is None or slot_name in target_models) and not disable_cloud:
                synthesis_tasks.append(fetch_and_notify(f"NVIDIA {model} (Synthesis)", query_nvidia(synthesis_messages, model, thinking_mode)))
        
        for i, model in enumerate(openrouter_models_list):
            slot_name = f"openrouter_{i+1}" if len(openrouter_models_list) > 1 else "openrouter"
            if (target_models is None or slot_name in target_models) and not disable_cloud:
                synthesis_tasks.append(fetch_and_notify(f"OpenRouter {model} (Synthesis)", query_openrouter(synthesis_messages, model)))
                
        synthesis_results = await asyncio.gather(*synthesis_tasks)
        valid_synthesis = [r for r in synthesis_results if r is not None]
        
        final_output = combined_perspectives + "\n\n=================================\n### COUNCIL CONSENSUS ###\n=================================\n\n" + "\n\n---\n\n".join(valid_synthesis)
        return final_output
        
    return combined_perspectives

@mcp.tool()
async def check_health() -> str:
    """Check the health and configuration of the LLM Council."""
    status = []
    
    disable_local = os.getenv("DISABLE_LOCAL", "").lower() == "true"
    disable_cloud = os.getenv("DISABLE_CLOUD", "").lower() == "true"

    if disable_cloud:
        status.append("🛑 CLOUD MODELS GLOBALLY DISABLED")
    if disable_local:
        status.append("🛑 LOCAL MODELS GLOBALLY DISABLED")
        
    # Check OpenAI
    if os.getenv("OPENAI_API_KEY"):
        status.append(f"✅ OpenAI: Configured ({OPENAI_MODEL})")
    else:
        status.append("❌ OpenAI: Missing API Key")
        
    # Check Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        status.append(f"✅ Anthropic: Configured ({ANTHROPIC_MODEL})")
    else:
        status.append("❌ Anthropic: Missing API Key")
        
    # Check Gemini API
    if os.getenv("GEMINI_API_KEY"):
        status.append(f"✅ Gemini API: Configured ({GEMINI_MODEL})")
    else:
        status.append("❌ Gemini API: Missing API Key")
        
    # Check Gemini CLI
    if os.getenv("GEMINI_CLI_ENABLED", "").lower() == "true":
        gemini_cli_path = os.getenv("GEMINI_CLI_PATH", "gemini")
        gemini_cli_model = os.getenv("GEMINI_CLI_MODEL", "default")
        try:
            proc = await asyncio.create_subprocess_exec(
                gemini_cli_path, "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            version = stdout.decode("utf-8", errors="replace").strip().split("\n")[0]
            status.append(f"✅ Gemini CLI: Installed ({version}, model: {gemini_cli_model})")
        except FileNotFoundError:
            status.append(f"❌ Gemini CLI: Enabled but `{gemini_cli_path}` not found — install with: npm install -g @google/gemini-cli")
        except Exception as e:
            status.append(f"⚠️ Gemini CLI: Enabled but version check failed ({str(e)})")
    else:
        status.append("➖ Gemini CLI: Not Enabled (set GEMINI_CLI_ENABLED=true)")
        
    # Check NVIDIA NIM
    if os.getenv("NVIDIA_API_KEY"):
        nvidia_models_str = os.getenv("NVIDIA_MODELS")
        if nvidia_models_str:
            models = [m.strip() for m in nvidia_models_str.split(",") if m.strip()]
            for i, model in enumerate(models):
                status.append(f"✅ NVIDIA NIM {i+1}: Configured ({model})")
        else:
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

    # Check OpenRouter
    if os.getenv("OPENROUTER_API_KEY"):
        or_models_str = os.getenv("OPENROUTER_MODELS")
        if or_models_str:
            models = [m.strip() for m in or_models_str.split(",") if m.strip()]
            for i, model in enumerate(models):
                status.append(f"✅ OpenRouter {i+1}: Configured ({model})")
        elif os.getenv("OPENROUTER_MODEL"):
            status.append(f"✅ OpenRouter: Configured ({os.getenv('OPENROUTER_MODEL')})")
        else:
            status.append("⚠️ OpenRouter: API Key set but no models configured (set OPENROUTER_MODELS)")
    else:
        status.append("➖ OpenRouter: Not Configured")

    return "### Council Health Check\n\n" + "\n".join(status)

if __name__ == "__main__":
    mcp.run()
