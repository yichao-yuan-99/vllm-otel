# Harbor Agent Model Endpoint Configuration Analysis

## Summary

**No, `--ek` / `--environment-kwarg` CANNOT be used to specify the language model endpoint.** These options are for environment (Docker/Daytona/etc.) configuration only, not for agent LLM configuration.

## Key Findings

### 1. `--ek` / `--environment-kwarg` Usage

Located in `src/harbor/cli/jobs.py` (lines 361-370, 616-617):

```python
environment_kwargs: Annotated[
    list[str] | None,
    Option(
        "--ek",
        "--environment-kwarg",
        help="Environment kwarg in key=value format (can be used multiple times)",
        rich_help_panel="Environment",
        show_default=False,
    ),
] = None,
```

These kwargs are passed to `config.environment.kwargs` and used by the **execution environment** (Docker, Daytona, E2B, Modal, etc.), NOT the agent's LLM configuration.

---

## Universal Custom Endpoint Setup (Agent-Agnostic)

To ensure your custom endpoint works with **any agent that supports custom endpoints**, you can set **ALL** environment variables and pass **BOTH** `--ak api_base` and `--ak base_url`. There are no conflicts since each agent only uses its supported method.

### Recommended Universal Command

```bash
# Set all environment variables (for agents that require env vars)
export ANTHROPIC_BASE_URL="http://your-endpoint:8000/v1"
export OPENAI_BASE_URL="http://your-endpoint:8000/v1"
export LLM_BASE_URL="http://your-endpoint:8000/v1"
export BASE_URL="http://your-endpoint:8000/v1"
export OPENAI_API_BASE="http://your-endpoint:8000/v1"

# Run with both agent kwargs (for agents that support kwargs)
harbor run \
  --agent <agent-name> \
  --model hosted_vllm/your-model \
  --ak api_base="http://your-endpoint:8000/v1" \
  --ak base_url="http://your-endpoint:8000/v1" \
  --dataset terminal-bench@2.0
```

### Why This Works (No Conflicts)

| Agent | Uses `--ak api_base` | Uses `--ak base_url` | Uses Env Vars | Notes |
|-------|----------------------|----------------------|---------------|-------|
| terminus-2 | ✅ Yes | ❌ Ignored | ❌ Ignored | Only uses `api_base` kwarg |
| openhands | ✅ Yes | ❌ Ignored | ✅ Fallback | Prefers kwarg, falls back to env |
| claude-code | ❌ Ignored | ❌ Ignored | ✅ Only `ANTHROPIC_BASE_URL` | Doesn't accept kwargs |
| codex | ❌ Ignored | ❌ Ignored | ✅ Only `OPENAI_BASE_URL` | Doesn't accept kwargs |
| qwen-code | ❌ Ignored | ✅ Yes | ✅ Fallback | Prefers `base_url` kwarg |
| cline-cli | ❌ Ignored | ❌ Ignored | ✅ Only `BASE_URL` (openai) | Doesn't accept kwargs |
| mini-swe-agent | ❌ Ignored | ❌ Ignored | ✅ Only `OPENAI_API_BASE` | Doesn't accept kwargs |
| swe-agent | ❌ Ignored | ❌ Ignored | ✅ Only `OPENAI_BASE_URL` | Doesn't accept kwargs |

### One-Liner Version

```bash
ANTHROPIC_BASE_URL=http://localhost:8000/v1 \
OPENAI_BASE_URL=http://localhost:8000/v1 \
LLM_BASE_URL=http://localhost:8000/v1 \
BASE_URL=http://localhost:8000/v1 \
OPENAI_API_BASE=http://localhost:8000/v1 \
harbor run \
  --agent terminus-2 \
  --model hosted_vllm/Qwen/Qwen2.5-Coder-32B-Instruct \
  --ak api_base=http://localhost:8000/v1 \
  --ak base_url=http://localhost:8000/v1 \
  --dataset terminal-bench@2.0
```

---

## Complete Agent Model/Endpoint Configuration Reference

### Table Summary

| Agent | Model Flag (`--model`) | Endpoint via Agent Kwargs (`--ak`) | Endpoint via Environment Variable |
|-------|------------------------|------------------------------------|-----------------------------------|
| **terminus-2** | ✅ Yes | ✅ `api_base` | ❌ No |
| **openhands** | ✅ Yes | ✅ `api_base` | ✅ `LLM_BASE_URL` |
| **claude-code** | ✅ Yes | ❌ No | ✅ `ANTHROPIC_BASE_URL` |
| **aider** | ✅ Yes (`provider/model`) | ❌ No | ⚠️ Provider-specific keys only |
| **codex** | ✅ Yes | ❌ No | ✅ `OPENAI_BASE_URL` |
| **goose** | ✅ Yes (`provider/model`) | ❌ No | ⚠️ Provider-specific config |
| **gemini-cli** | ✅ Yes | ❌ No | ❌ No (Google only) |
| **cursor-cli** | ✅ Yes | ❌ No | ❌ No (Cursor only) |
| **opencode** | ✅ Yes (`provider/model`) | ❌ No | ⚠️ Provider-specific keys only |
| **qwen-code** | ✅ Yes | ✅ `base_url` | ✅ `OPENAI_BASE_URL` |
| **cline-cli** | ✅ Yes (`provider:model`) | ❌ No | ✅ `BASE_URL` (openai only) |
| **mini-swe-agent** | ✅ Yes | ❌ No | ✅ `OPENAI_API_BASE` |
| **swe-agent** | ✅ Yes | ❌ No | ✅ `OPENAI_BASE_URL` |

---

## Detailed Agent Configuration

### 1. Terminus-2 (`terminus-2`)

**File**: `src/harbor/agents/terminus_2/terminus_2.py`

| Option | Type | How to Set |
|--------|------|------------|
| Model | `--model` | `-m anthropic/claude-opus-4-1` |
| API Base | `--ak api_base` | `--ak api_base=https://api.openai.com/v1` |
| Model Info | `--ak model_info` | `--ak model_info='{"max_input_tokens": 32768}'` |

**Example:**
```bash
harbor run \
  --agent terminus-2 \
  --model hosted_vllm/Qwen/Qwen2.5-Coder-32B-Instruct \
  --ak api_base=http://localhost:8000/v1 \
  --ak model_info='{"max_input_tokens": 32768, "max_output_tokens": 4096}' \
  --dataset terminal-bench@2.0
```

---

### 2. OpenHands (`openhands`)

**File**: `src/harbor/agents/installed/openhands.py`

| Option | Type | How to Set |
|--------|------|------------|
| Model | `--model` | `-m openai/gpt-4o` |
| API Base | `--ak api_base` | `--ak api_base=https://api.openai.com/v1` |
| API Base (env) | Env Var | `LLM_BASE_URL` |

**Example:**
```bash
# Using agent kwarg
harbor run \
  --agent openhands \
  --model hosted_vllm/Qwen/Qwen2.5-Coder-32B-Instruct \
  --ak api_base=http://localhost:8000/v1 \
  --dataset terminal-bench@2.0

# Using environment variable
LLM_BASE_URL=http://localhost:8000/v1 harbor run \
  --agent openhands \
  --model hosted_vllm/Qwen/Qwen2.5-Coder-32B-Instruct \
  --dataset terminal-bench@2.0
```

**Fallback Environment Variables:**
- `HOSTED_VLLM_API_BASE`
- `VLLM_API_BASE`
- `OPENAI_API_BASE`

---

### 3. Claude Code (`claude-code`)

**File**: `src/harbor/agents/installed/claude_code.py`

| Option | Type | How to Set |
|--------|------|------------|
| Model | `--model` | `-m anthropic/claude-opus-4-1` |
| API Base | Env Var only | `ANTHROPIC_BASE_URL` |

**Example:**
```bash
# Using OpenRouter or custom endpoint
ANTHROPIC_BASE_URL=https://openrouter.ai/api/v1 harbor run \
  --agent claude-code \
  --model anthropic/claude-opus-4-1 \
  --dataset terminal-bench@2.0
```

---

### 4. Aider (`aider`)

**File**: `src/harbor/agents/installed/aider.py`

| Option | Type | How to Set |
|--------|------|------------|
| Model | `--model` | `-m openai/gpt-4o` (provider/model format) |
| API Base | ❌ Not supported | Provider-specific only |

**Note:** Aider does not support custom API base URLs directly. It infers the provider from the model name format (`provider/model`) and uses provider-specific API keys.

**Example:**
```bash
harbor run \
  --agent aider \
  --model openai/gpt-4o \
  --dataset terminal-bench@2.0
```

---

### 5. Codex (`codex`)

**File**: `src/harbor/agents/installed/codex.py`

| Option | Type | How to Set |
|--------|------|------------|
| Model | `--model` | `-m openai/gpt-4o` |
| API Base | Env Var only | `OPENAI_BASE_URL` |

**Example:**
```bash
OPENAI_BASE_URL=https://api.openai.com/v1 harbor run \
  --agent codex \
  --model openai/gpt-4o \
  --dataset terminal-bench@2.0
```

---

### 6. Goose (`goose`)

**File**: `src/harbor/agents/installed/goose.py`

| Option | Type | How to Set |
|--------|------|------------|
| Model | `--model` | `-m openai/gpt-4o` (provider/model format) |
| API Base | Env Var (provider-specific) | Provider-specific |

**Supported Providers:**
- `openai` → `OPENAI_API_KEY`
- `anthropic` → `ANTHROPIC_API_KEY`
- `databricks` → `DATABRICKS_HOST`, `DATABRICKS_TOKEN`
- `tetrate` → `TETRATE_API_KEY`, `TETRATE_HOST` (optional)

**Example:**
```bash
harbor run \
  --agent goose \
  --model openai/gpt-4o \
  --dataset terminal-bench@2.0
```

---

### 7. Gemini CLI (`gemini-cli`)

**File**: `src/harbor/agents/installed/gemini_cli.py`

| Option | Type | How to Set |
|--------|------|------------|
| Model | `--model` | `-m google/gemini-1.5-pro` |
| API Base | ❌ Not supported | Google API only |

**Note:** Gemini CLI only works with Google's Gemini API. No custom endpoint support.

**Environment Variables:**
- `GEMINI_API_KEY`
- `GOOGLE_APPLICATION_CREDENTIALS`
- `GOOGLE_CLOUD_PROJECT`
- `GOOGLE_CLOUD_LOCATION`
- `GOOGLE_GENAI_USE_VERTEXAI`
- `GOOGLE_API_KEY`

**Example:**
```bash
harbor run \
  --agent gemini-cli \
  --model google/gemini-1.5-pro \
  --dataset terminal-bench@2.0
```

---

### 8. Cursor CLI (`cursor-cli`)

**File**: `src/harbor/agents/installed/cursor_cli.py`

| Option | Type | How to Set |
|--------|------|------------|
| Model | `--model` | `-m cursor/gpt-4o` |
| API Base | ❌ Not supported | Cursor API only |

**Note:** Cursor CLI only works with Cursor's API.

**Environment Variables:**
- `CURSOR_API_KEY` (required)

**Example:**
```bash
harbor run \
  --agent cursor-cli \
  --model cursor/gpt-4o \
  --dataset terminal-bench@2.0
```

---

### 9. OpenCode (`opencode`)

**File**: `src/harbor/agents/installed/opencode.py`

| Option | Type | How to Set |
|--------|------|------------|
| Model | `--model` | `-m openai/gpt-4o` (provider/model format) |
| API Base | ❌ Not supported | Provider-specific |

**Supported Providers:**
- `amazon-bedrock` → `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`
- `anthropic` → `ANTHROPIC_API_KEY`
- `azure` → `AZURE_RESOURCE_NAME`, `AZURE_API_KEY`
- `deepseek` → `DEEPSEEK_API_KEY`
- `github-copilot` → `GITHUB_TOKEN`
- `google` → `GEMINI_API_KEY`, `GOOGLE_GENERATIVE_AI_API_KEY`, etc.
- `groq` → `GROQ_API_KEY`
- `huggingface` → `HF_TOKEN`
- `llama` → `LLAMA_API_KEY`
- `mistral` → `MISTRAL_API_KEY`
- `openai` → `OPENAI_API_KEY`
- `xai` → `XAI_API_KEY`

**Example:**
```bash
harbor run \
  --agent opencode \
  --model openai/gpt-4o \
  --dataset terminal-bench@2.0
```

---

### 10. Qwen Code (`qwen-code`)

**File**: `src/harbor/agents/installed/qwen_code.py`

| Option | Type | How to Set |
|--------|------|------------|
| Model | `--model` | `-m qwen/qwen3-coder-plus` |
| API Base | `--ak base_url` | `--ak base_url=https://api.openai.com/v1` |
| API Base (env) | Env Var | `OPENAI_BASE_URL` |
| API Key | `--ak api_key` | `--ak api_key=sk-...` |
| API Key (env) | Env Var | `OPENAI_API_KEY` |

**Example:**
```bash
# Using agent kwargs
harbor run \
  --agent qwen-code \
  --model qwen/qwen3-coder-plus \
  --ak base_url=http://localhost:8000/v1 \
  --ak api_key=dummy-key \
  --dataset terminal-bench@2.0

# Using environment variables
OPENAI_BASE_URL=http://localhost:8000/v1 harbor run \
  --agent qwen-code \
  --model qwen/qwen3-coder-plus \
  --dataset terminal-bench@2.0
```

---

### 11. Cline CLI (`cline-cli`)

**File**: `src/harbor/agents/installed/cline/cline.py`

| Option | Type | How to Set |
|--------|------|------------|
| Model | `--model` | `-m openrouter:anthropic/claude-opus-4.5` (provider:model format) |
| API Base | Env Var (openai only) | `BASE_URL` |
| API Key | Env Var | `API_KEY` |

**Supported Providers:**
- `anthropic`
- `openai` / `openai-compatible` / `openai-native`
- `openrouter`
- `xai`
- `bedrock`
- `gemini`
- `ollama`
- `cerebras`
- `cline`
- `oca`
- `hicap`
- `nousresearch`

**Example:**
```bash
# Standard provider
API_KEY=sk-... harbor run \
  --agent cline-cli \
  --model anthropic:claude-opus-4-1 \
  --dataset terminal-bench@2.0

# OpenAI with custom endpoint
API_KEY=dummy BASE_URL=http://localhost:8000/v1 harbor run \
  --agent cline-cli \
  --model openai:gpt-4o \
  --dataset terminal-bench@2.0
```

---

### 12. Mini SWE Agent (`mini-swe-agent`)

**File**: `src/harbor/agents/installed/mini_swe_agent.py`

| Option | Type | How to Set |
|--------|------|------------|
| Model | `--model` | `-m openai/gpt-4o` |
| API Base | Env Var only | `OPENAI_API_BASE` |
| API Key | Env Var | `MSWEA_API_KEY` or provider-specific |

**Example:**
```bash
OPENAI_API_BASE=http://localhost:8000/v1 harbor run \
  --agent mini-swe-agent \
  --model openai/gpt-4o \
  --dataset terminal-bench@2.0
```

---

### 13. SWE Agent (`swe-agent`)

**File**: `src/harbor/agents/installed/swe_agent.py`

| Option | Type | How to Set |
|--------|------|------------|
| Model | `--model` | `-m openai/gpt-4o` |
| API Base | Env Var only | `OPENAI_BASE_URL` |
| API Key | Env Var | `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `TOGETHER_API_KEY` |

**Example:**
```bash
OPENAI_BASE_URL=http://localhost:8000/v1 harbor run \
  --agent swe-agent \
  --model openai/gpt-4o \
  --dataset terminal-bench@2.0
```

---

## Conclusion

### Agent Support for Custom Endpoints

1. **Agents supporting `--ak api_base` or `--ak base_url`**:
   - `terminus-2`: `--ak api_base=...`
   - `openhands`: `--ak api_base=...`
   - `qwen-code`: `--ak base_url=...`

2. **Agents requiring environment variables**:
   - `claude-code` (`ANTHROPIC_BASE_URL`)
   - `codex` (`OPENAI_BASE_URL`)
   - `cline-cli` (`BASE_URL` for openai)
   - `mini-swe-agent` (`OPENAI_API_BASE`)
   - `swe-agent` (`OPENAI_BASE_URL`)

3. **Agents with no custom endpoint support**:
   - `aider`
   - `goose`
   - `gemini-cli`
   - `cursor-cli`
   - `opencode`

### Universal Configuration Strategy

To ensure maximum compatibility, use this pattern:

```bash
# Set all environment variables
export ANTHROPIC_BASE_URL="http://your-endpoint/v1"
export OPENAI_BASE_URL="http://your-endpoint/v1"
export LLM_BASE_URL="http://your-endpoint/v1"
export BASE_URL="http://your-endpoint/v1"
export OPENAI_API_BASE="http://your-endpoint/v1"

# Pass both agent kwargs
harbor run \
  --agent <agent> \
  --model hosted_vllm/your-model \
  --ak api_base="http://your-endpoint/v1" \
  --ak base_url="http://your-endpoint/v1" \
  ...
```

This approach ensures that no matter which agent you're using, as long as it supports custom endpoints, it will work without you having to remember the specific configuration method for each agent.
