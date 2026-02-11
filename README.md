![Header image](assets/Header.png)

# Agentica Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/discord/1470799122717085941?logo=discord&label=Discord)](https://discord.gg/bddGs8bb)
[![Twitter](https://img.shields.io/twitter/follow/symbolica?style=flat&logo=x&label=Follow)](https://x.com/symbolica)

[Agentica](https://agentica.symbolica.ai) is a type-safe AI framework that lets LLM agents integrate with your code—functions, classes, live objects, even entire SDKs. Instead of building MCP wrappers or brittle schemas, you pass references directly; the framework enforces your types at runtime, constrains return types, and manages agent lifecycle.

This repository contains the session management backend (the "server"). It handles session orchestration, state persistence, sandboxed code execution, and coordination between clients and the Agentica runtime.

## Documentation

The full documentation for the SDKs can be found at [docs.symbolica.ai](https://docs.symbolica.ai).

## Getting Started

### Required Platforms and Tools

At present, building and running the server requires one of the following platforms

- macOS (arm64, x86_64)
- Linux (x86_64, aarch64)

and the presence of the following tools
- [uv](https://github.com/astral-sh/uv)
- `curl`, `tar`, `shasum`, `bash` 

### Building the server

```bash
uv sync
```

### Running the server

The most common configuration for usage will be
```bash
export OPENROUTER_API_KEY="your-api-key"
uv run agentica-server
```

While for development of the server itself we suggest
```bash
export OPENROUTER_API_KEY="your-api-key"
export AGENTICA_LOG_TAGS=CORE
uv run agentica-server --log-level=INFO
```

The server starts on port 2345 by default. Use `--port` to change this.

To see all available options:

```bash
uv run agentica-server --help
```

### Configuring Inference Providers

By default, the server uses `https://openrouter.ai/api` as the base URL and environment variable `OPENROUTER_API_KEY` as the api key.

You can configure multiple inference providers to route different models to different endpoints. This is useful for:
- Using OpenAI's native API for OpenAI models (better performance, access to latest features)
- Routing specific models to specific providers
- Using different API keys for different model families

#### Using a Config File

Create a `inference_providers.yml` file:

```yaml
# First matching provider wins (order matters)
- endpoint: https://api.openai.com/v1/responses
  token: ${oc.env:OPENAI_API_KEY}
  model_pattern: "openai/*"

- endpoint: https://openrouter.ai/api/v1/chat/completions
  token: ${oc.env:OPENROUTER_API_KEY}
  # model_pattern defaults to "*" (matches all)
```

Then run:

```bash
uv run agentica-server --inference-providers inference_providers.yml
```

The `${oc.env:VAR_NAME}` syntax reads environment variables at startup.


#### Provider Configuration Options

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `endpoint` | Yes | - | Full URL ending with `/responses` or `/chat/completions`. API type is inferred from the suffix. |
| `token` | Yes | - | API key for authentication |
| `model_pattern` | No | `*` | fnmatch pattern for model matching (e.g., `openai/*`, `anthropic/*`, `*`) |

#### Legacy Mode

For backward compatibility, you can still use the legacy CLI arguments:

```bash
uv run agentica-server \
  --inference-endpoint https://openrouter.ai/api/v1/chat/completions \
  --inference-token $OPENROUTER_API_KEY
```

This creates a single provider that matches all models.

## Issues

Please report bugs, feature requests, and other issues in the [symbolica/agentica-issues](https://github.com/symbolica-ai/agentica-issues) repository.

## License

This project is licensed under the [MIT License](./LICENSE).
