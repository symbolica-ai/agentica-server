import asyncio
import logging
import os
import socket
import sys
import threading
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import dotenv
import uvicorn
from agentica_internal.core.log import add_log_stream
from agentica_internal.telemetry import get_tracer, initialize_tracing
from litestar import Litestar
from litestar._openapi.schema_generation.schema import TYPE_MAP
from litestar.config.cors import CORSConfig
from litestar.exceptions import HTTPException
from litestar.logging import LoggingConfig
from litestar.openapi.spec import Schema
from litestar.openapi.spec.enums import OpenAPIFormat, OpenAPIType
from litestar.types import ASGIApp

from application.defaults import (
    DEFAULT_DISABLE_OTEL,
    DEFAULT_INFERENCE_TOKEN,
    DEFAULT_LOG_POSTER_URL,
    DEFAULT_MAX_CONCURRENT_INVOCATIONS,
    DEFAULT_OTEL_ENDPOINT,
    DEFAULT_PORT,
    DEFAULT_SANDBOX_LOG_PATH,
    DEFAULT_SANDBOX_LOG_TAGS,
    ORGANIZATION_ID,
)
from application.http_client import close_client, init_client
from application.routes import get_routes
from auth import RequestLoggingMiddleware
from messages import Poster
from server_session_manager import InferenceProvider, ServerSessionManager
from server_session_manager.provider import build_providers

if TYPE_CHECKING:
    import httpx

# Patch Litestar's bytes field handling for OpenAPI schema generation
# Fix: bytes should be "string" with "format: byte" (base64-encoded) instead of plain "string"
TYPE_MAP[bytes] = Schema(type=OpenAPIType.STRING, format=OpenAPIFormat.BINARY)

dotenv.load_dotenv()

logger = logging.getLogger('session_manager_application')

# FOR OPENAPI SCHEMA GENERATION, DO NOT REMOVE
# litestar CLI requires top level variable called `app` so it can crawl over all the types
cors_config = CORSConfig(allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app = Litestar(route_handlers=get_routes(), cors_config=cors_config)
assert app is not None

type SandboxMode = Literal['from_env', 'wasm', 'no_sandbox']


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Session manager",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument(
        "--inference-providers",
        type=str,
        default=None,
        help="Path to YAML file for inference providers. Must be list of {endpoint, token, model_pattern} objects.",
    )
    _ = parser.add_argument(
        "--log-poster-url",
        type=str,
        default=DEFAULT_LOG_POSTER_URL,
        help="Log poster URL",
    )
    _ = parser.add_argument(
        "--inference-token",
        type=str,
        default=DEFAULT_INFERENCE_TOKEN,
        help="Infra token",
    )
    _ = parser.add_argument(
        "--inference-endpoint",
        type=str,
        default=None,  # default is set in ../server_session_manager/provider.py
        help="Inference endpoint (must end with '/responses' or '/chat/completions')",
    )
    _ = parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port to bind to",
    )
    _ = parser.add_argument(
        "--log-level",
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='WARNING',
        help="Log level",
    )
    _ = parser.add_argument(
        "--max-concurrent-invocations",
        type=int,
        default=DEFAULT_MAX_CONCURRENT_INVOCATIONS,
        help="Max concurrent invocations (omitted = unlimited)",
    )
    _ = parser.add_argument(
        "--user-id",
        type=str,
        default=None,
        help="User ID",
    )
    _ = parser.add_argument(
        "--print-stacks-after-n-seconds",
        type=int,
        default=0,
        help="For debugging, will print asyncio stack traces after N seconds",
    )
    _ = parser.add_argument(
        "--otel-endpoint",
        type=str,
        default=DEFAULT_OTEL_ENDPOINT,
        help="OpenTelemetry OTLP gRPC endpoint for distributed tracing (e.g., http://localhost:4317)",
    )
    _ = parser.add_argument(
        "--disable-otel",
        action='store_true',
        default=DEFAULT_DISABLE_OTEL,
        help="Whether to disable OTEL.",
    )
    _ = parser.add_argument(
        "--sandbox-mode",
        choices=['from_env', 'wasm', 'no_sandbox'],
        default='from_env',
        help="What sandbox mode to use, where 'from_env' checks the AGENTICA_NO_SANDBOX envar.",
    )
    _ = parser.add_argument(
        "--recursion-limit",
        type=int,
        default=sys.getrecursionlimit(),
        help="Recursion limit",
    )
    _ = parser.add_argument(
        "--log-tags",
        type=str,
        default='DEFAULT',
        help="Log tags to enable, if DEFAULT use AGENTICA_LOG_TAGS",
    )
    _ = parser.add_argument(
        "--log-file",
        type=str,
        default='',
        help="Where to write logtag logs, by default only to stdout.",
    )
    _ = parser.add_argument(
        "--sandbox-log-path",
        type=str,
        default=DEFAULT_SANDBOX_LOG_PATH,
        help="Templated path to write sandbox logs to.",
    )
    _ = parser.add_argument(
        "--sandbox-log-tags",
        type=str,
        default=DEFAULT_SANDBOX_LOG_TAGS,
        help="Sandbox log tags to enable (if unset, will inherit from --log-tags)",
    )
    _ = parser.add_argument(
        "--tui",
        action="store_true",
        help="Enable interactive TUI with eval() input prompt (requires terminal)",
    )
    _ = parser.add_argument(
        "--no-tui",
        action="store_true",
        help="Explicitly disable TUI even if running in a terminal",
    )
    _ = parser.add_argument(
        "--silent-for-testing",
        action='store_true',
        default=False,
        help="Avoid some noisy printing that pollutes integration testing output.",
    )
    _ = parser.add_argument(
        "--print-pid",
        action="store_true",
        help="Print the PID of the session manager process every 5 seconds",
    )
    _ = parser.add_argument(
        "--version",
        action="store_true",
        help="Print the version of the session manager",
    )
    _ = parser.add_argument(
        "--no-version-check",
        action="store_true",
        help="Disable version check",
    )

    args = parser.parse_args()

    if args.version:
        from agentic.version_policy import _SESSION_MANAGER_VERSION

        print(f"{_SESSION_MANAGER_VERSION}")
        exit(0)

    if args.no_version_check:
        os.environ['AGENTICA_SERVER_DISABLE_VERSION_CHECK'] = '1'

    return args


class SessionManager:
    log_poster_url: str
    providers: list[InferenceProvider]
    max_concurrent_invocations: int | None
    otel_endpoint: str
    disable_otel: bool
    sandbox_mode: SandboxMode
    sandbox_log_path: str | None

    _ssm: ServerSessionManager

    _app: Litestar
    _config: uvicorn.Config
    _server: uvicorn.Server
    _thread: threading.Thread | None

    def __init__(
        self,
        *,
        providers: list[InferenceProvider],
        log_poster_url: str = DEFAULT_LOG_POSTER_URL,
        user_id: str | None = None,
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "WARNING",
        port: int = DEFAULT_PORT,
        max_concurrent_invocations: int | None = DEFAULT_MAX_CONCURRENT_INVOCATIONS,
        otel_endpoint: str = DEFAULT_OTEL_ENDPOINT,
        disable_otel: bool = False,
        sandbox_mode: SandboxMode = 'from_env',
        sandbox_log_path: str | None = None,
        sandbox_log_tags: str | None = None,
        silent_for_testing: bool = False,
    ):
        self.log_poster_url = log_poster_url
        self.providers = providers
        self.user_id = user_id
        self.max_concurrent_invocations = max_concurrent_invocations
        self.port = port
        self.otel_endpoint = otel_endpoint
        self.disable_otel = disable_otel
        self.sandbox_mode = sandbox_mode
        self.organization_id = ORGANIZATION_ID

        # Initialize OpenTelemetry tracing
        environment = os.getenv("ENVIRONMENT", "local")
        initialize_tracing(
            service_name="session-manager",
            environment=environment,
            tempo_endpoint=otel_endpoint,
            organization_id=self.organization_id,
            log_level=log_level,
        ) if not disable_otel else None

        logger.info(
            "OpenTelemetry tracing initialized with endpoint: %s and organization_id: %s",
            otel_endpoint,
            self.organization_id,
        )

        # Litestar stuff
        logging_config = LoggingConfig(
            root={"level": log_level.upper(), "handlers": ["console"]},
            loggers={
                "httpx": {"level": "WARNING"},
                "httpcore": {"level": "WARNING"},
                "auth.request_logging_middleware": {},  # inherits the specified log level
            },
            formatters={
                "standard": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
            },
            disable_stack_trace={HTTPException},
        )

        # Configure middleware
        middleware = []

        # Add request logging middleware
        def create_logging_middleware(app: ASGIApp) -> ASGIApp:
            return RequestLoggingMiddleware(app=app)

        middleware.append(create_logging_middleware)
        logger.info("Request logging middleware added")

        cors_config = CORSConfig(allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

        self._app = Litestar(
            route_handlers=get_routes(),
            logging_config=logging_config,
            middleware=middleware,
            cors_config=cors_config,
            debug=True,  # TODO: should this be True in production?
            on_startup=[
                self._init_http_client,
                self._setup_otel_logging,
                self._log_startup_message,
            ],
            on_shutdown=[self._shutdown_http_client],
        )

        # Get tracer for session manager
        tracer = get_tracer(__name__, not disable_otel)

        self._ssm = self._app.state.session_manager = ServerSessionManager(
            log_poster=Poster(url=log_poster_url),
            providers=providers,
            user_id=user_id,
            tracer=tracer,
            max_concurrent_invocations=max_concurrent_invocations,
            sandbox_mode=sandbox_mode,
            sandbox_log_path=sandbox_log_path or None,
            sandbox_log_tags=sandbox_log_tags or None,
            silent_for_testing=silent_for_testing,
        )

        self._config = uvicorn.Config(
            app=self._app,
            host="0.0.0.0",
            port=port,
            log_level="warning",
            ws_ping_interval=20,
            ws_ping_timeout=None,
            ws_max_size=268435456,
        )
        self._server = uvicorn.Server(self._config)
        self._thread = None

    def _log_server_start(self) -> None:
        logging.getLogger(__name__).info(
            "Starting Session Manager server on http://%s:%s",
            self._config.host,
            self._config.port,
        )

    def start_threaded(self):
        if self._thread and self._thread.is_alive():
            return
        self._log_server_start()
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

    def start_sync(self):
        self._log_server_start()
        return self._server.run()

    async def start(self):
        self._log_server_start()
        return await self._server.serve()

    def stop(self, timeout: float = 5.0):
        self._ssm.close()
        self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=timeout)

    async def _setup_otel_logging(self, app: Litestar) -> None:
        """Set up OpenTelemetry logging after Litestar has configured logging."""

        if self.disable_otel:
            return

        # Use hostname as instance ID, fallback to "session-manager-1" if unavailable
        try:
            instance_id = socket.gethostname()
        except Exception:
            instance_id = "session-manager-1"

        from agentica_internal.otel_logging import CustomLogFW

        logFW = CustomLogFW(
            service_name="session-manager",
            instance_id=instance_id,
            endpoint=self.otel_endpoint,
            organization_id=self.organization_id,
        )
        otel_handler = logFW.setup_logging()

        # Add OTEL handler to root logger (after Litestar's config)
        root_logger = logging.getLogger()
        root_logger.addHandler(otel_handler)

        logger.info(f"OpenTelemetry logging initialized with instance: {instance_id}")

    async def _log_startup_message(
        self, app: Litestar
    ) -> None:  # pragma: no cover - startup logging
        logging.getLogger(__name__).info(
            "Session Manager server startup complete on http://%s:%s",
            self._config.host,
            self._config.port,
        )

    async def _init_http_client(self, app: Litestar) -> None:
        """Initialize the shared HTTP client on startup."""
        init_client()

    async def _shutdown_http_client(self, app: Litestar) -> None:
        """Close the shared HTTP client on shutdown."""
        await close_client()

    def inference_endpoint_client(
        self,
        base_url: str = "https://openrouter.ai/",
    ) -> 'httpx.Client':
        import httpx

        return httpx.Client(
            base_url=f"{base_url}",
            headers={"Authorization": f"Bearer {self.providers[0].client.api_key}"},
        )


async def main():
    args = parse_args()

    log_level = args.log_level

    logger.setLevel(log_level)
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.addFilter(lambda record: "/logs" not in record.getMessage())

    if n := args.print_stacks_after_n_seconds:
        from agentica_internal.core.print import print_asyncio_stacks_in_n_seconds

        print_asyncio_stacks_in_n_seconds(n)

    recursion_limit = args.recursion_limit
    sys.setrecursionlimit(recursion_limit)

    log_tags = args.log_tags
    log_file = args.log_file
    if log_tags != 'DEFAULT':
        os.environ['AGENTICA_LOG_TAGS'] = log_tags

    if log_tags := os.getenv('AGENTICA_LOG_TAGS') and log_file:
        log_path = Path(log_file)
        if log_path.exists():
            pid = os.getpid()
            log_path = log_path.with_suffix(f'.{pid}.{log_path.suffix}')
        add_log_stream(log_path)
        msg = f'LOGGING TAGS={log_tags!r} TO FILE={log_path!r}'
        print(f'SESSION_MANAGER: {msg}\n')
        logger.info(msg)

    sandbox_log_path = args.sandbox_log_path
    if sandbox_log_path:
        logger.info(f'sandbox_log_path={sandbox_log_path!r}')
    else:
        sandbox_log_path = None

    # Build providers from configuration sources
    providers = build_providers(
        inference_providers_path=args.inference_providers,
        legacy_endpoint=args.inference_endpoint,
        legacy_token=args.inference_token,
    )

    sm = SessionManager(
        log_poster_url=args.log_poster_url,
        providers=providers,
        port=args.port,
        user_id=args.user_id,
        log_level=log_level,
        max_concurrent_invocations=args.max_concurrent_invocations or None,
        otel_endpoint=args.otel_endpoint,
        disable_otel=args.disable_otel,
        sandbox_mode=args.sandbox_mode,
        sandbox_log_path=sandbox_log_path,
        sandbox_log_tags=args.sandbox_log_tags or None,
        silent_for_testing=args.silent_for_testing,
    )

    # Determine if TUI should be used
    use_tui = args.tui and not args.no_tui

    if use_tui:
        # Check if we're in a terminal
        if not sys.stdin.isatty():
            logger.warning("TUI requested but stdin is not a terminal, falling back to normal mode")
            use_tui = False

    if use_tui:
        from application.tui import run_with_tui

        await run_with_tui(sm)
    else:

        async def print_pid():
            while True:
                logger.info(f"[***] PID: {os.getpid()}")
                await asyncio.sleep(5)

        if args.print_pid:
            asyncio.create_task(print_pid())
        await sm.start()


def main_sync():
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
