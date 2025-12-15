# Agentica Server

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
export AGENTICA_NO_SANDBOX=1
export FORCE_ENABLE_LOGGING=CORE
uv run agentica-server --log-level=INFO
```

The server starts on port 2345 by default. Use `--port` to change this.

To see all available options:

```bash
uv run agentica-server --help
```

## Issues

Please report bugs, feature requests, and other issues in the [symbolica/agentica-issues](https://github.com/symbolica-ai/agentica-issues) repository.

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines. All contributors must agree to our [CLA](./CLA.md).

## Code of Conduct

This project adheres to a [Code of Conduct](./CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## License

See [LICENSE](./LICENSE).
