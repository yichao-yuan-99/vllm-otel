import argparse
import asyncio
import logging
import os
import socket
import sys
from pathlib import Path

import uvicorn

from gateway_lite.app import GatewayConfig, create_app, create_gateway_service
from gateway_lite.port_profiles import load_port_profile
from gateway_lite.runtime_config import load_runtime_settings


LOGGER = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _create_listen_socket(host: str, port: int) -> socket.socket:
    return socket.create_server((host, port), reuse_port=False)


async def _serve_gateway(app_instance, host: str, ports: list[int]) -> None:
    """Serve the gateway on multiple ports using a single uvicorn server."""
    sockets = [_create_listen_socket(host, port) for port in ports]
    config = uvicorn.Config(app_instance, host=host, port=ports[0])
    server = uvicorn.Server(config)
    try:
        await server.serve(sockets=sockets)
    finally:
        for listen_socket in sockets:
            listen_socket.close()


def cmd_run(args: argparse.Namespace) -> int:
    setup_logging(args.verbose)

    settings = load_runtime_settings()
    selected_profile_id = (
        args.port_profile_id
        if args.port_profile_id is not None
        else settings.port_profile_id
    )
    profile = load_port_profile(selected_profile_id)

    upstream_base_url = (
        args.upstream_base_url
        or os.environ.get("GATEWAY_LITE_VLLM_BASE_URL", "")
    ).strip()
    upstream_api_key = (
        args.upstream_api_key
        or os.environ.get("GATEWAY_LITE_UPSTREAM_API_KEY", "")
        or settings.upstream_api_key
        or ""
    ).strip()

    LOGGER.info(
        "Starting gateway-lite with port_profile_id=%s (vllm_port=%s, gateway_port=%s, gateway_parse_port=%s)",
        selected_profile_id,
        profile.vllm_port,
        profile.gateway_port,
        profile.gateway_parse_port,
    )

    # Create config with settings
    config = GatewayConfig.from_port_profile(
        selected_profile_id,
        runtime_settings=settings,
    )
    if upstream_base_url:
        config.vllm_base_url = upstream_base_url
    config.upstream_api_key = upstream_api_key or None

    LOGGER.info("Upstream base URL: %s", config.vllm_base_url)
    if config.upstream_api_key:
        LOGGER.info("Upstream API key override is configured")

    # Create app - both ports behave identically (no response transformation)
    app = create_app(
        service=create_gateway_service(config=config),
        gateway_parse_port=profile.gateway_parse_port,
    )

    # Determine which ports to listen on
    ports = [profile.gateway_port]
    if profile.gateway_parse_port != profile.gateway_port:
        ports.append(profile.gateway_parse_port)

    LOGGER.info(
        "Gateway-lite listening on: %s",
        ", ".join([f"http://{args.host}:{p}" for p in ports])
    )
    LOGGER.info(
        "  - Port %s: regular (forward only)",
        profile.gateway_port
    )
    if profile.gateway_parse_port != profile.gateway_port:
        LOGGER.info(
            "  - Port %s: parsed (same behavior as regular port)",
            profile.gateway_parse_port
        )

    asyncio.run(_serve_gateway(app, args.host, ports))
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    import httpx

    settings = load_runtime_settings()
    selected_profile_id = (
        args.port_profile_id
        if args.port_profile_id is not None
        else settings.port_profile_id
    )
    profile = load_port_profile(selected_profile_id)

    url = f"http://{args.host}:{profile.gateway_port}/healthz"
    try:
        response = httpx.get(url, timeout=5.0)
        response.raise_for_status()
        print(f"Gateway-lite is healthy: {response.json()}")
        return 0
    except Exception as exc:
        print(f"Gateway-lite health check failed: {exc}")
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gateway-lite",
        description="Lightweight vLLM Gateway without Jaeger/OpenTelemetry",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # run command
    run_parser = subparsers.add_parser("run", help="Run the gateway-lite server")
    run_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    run_parser.add_argument(
        "--port-profile-id",
        type=int,
        default=None,
        help="Override runtime config and select a specific port profile ID.",
    )
    run_parser.add_argument(
        "--upstream-base-url",
        default=None,
        help=(
            "Override the upstream OpenAI-compatible base URL "
            "(for example https://api.openai.com/v1). "
            "Can also be set via GATEWAY_LITE_VLLM_BASE_URL."
        ),
    )
    run_parser.add_argument(
        "--upstream-api-key",
        default=None,
        help=(
            "Set a fixed API key to inject into forwarded upstream requests. "
            "Can also be set via GATEWAY_LITE_UPSTREAM_API_KEY."
        ),
    )
    run_parser.set_defaults(func=cmd_run)

    # health command
    health_parser = subparsers.add_parser("health", help="Check gateway-lite health")
    health_parser.add_argument(
        "--host",
        default="localhost",
        help="Gateway-lite host (default: localhost)",
    )
    health_parser.add_argument(
        "--port-profile-id",
        type=int,
        default=None,
        help="Override runtime config and select a specific port profile ID.",
    )
    health_parser.set_defaults(func=cmd_health)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
