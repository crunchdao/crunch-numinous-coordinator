import json
import os
import re
from contextvars import ContextVar

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from lru import LRU
from neurons.miner.gateway import cache as gateway_cache
from neurons.miner.gateway.app import ChutesClient, DesearchClient, LightningRodClient, LunarCrushClient, NuminousIndiciaClient, NuminousSignalsClient, OpenAIClient, OpenRouterClient, PerplexityClient, PublicDataProxyClient, UnusualWhalesClient, VericoreClient, app
from neurons.miner.gateway.providers import public_data as public_data_provider

from crunch_node.bt_compat.path_validator import CrunchNodeAsyncValidatorSigningProxy, UnknownRunIdException
from crunch_node.config import CrunchNodeConfig

CLIENTS = {
    "chutes": (ChutesClient, "CHUTES_API_KEY"),
    "desearch": (DesearchClient, "DESEARCH_API_KEY"),
    "lightning-rod": (LightningRodClient, "LIGHTNING_ROD_API_KEY"),
    "lunar-crush": (LunarCrushClient, "LUNAR_CRUSH_API_KEY"),
    "numinous-indicia": (NuminousIndiciaClient, None),
    "numinous-signals": (NuminousSignalsClient, "NUMINOUS_SIGNALS_API_KEY"),
    "openai": (OpenAIClient, "OPENAI_API_KEY"),
    "openrouter": (OpenRouterClient, "OPENROUTER_API_KEY"),
    "perplexity": (PerplexityClient, "PERPLEXITY_API_KEY"),
    "public-data": (None, None),
    "unusual-whales": (UnusualWhalesClient, "UNUSUAL_WHALES_API_KEY"),
    "vericore": (VericoreClient, "VERICORE_API_KEY"),
    "openrouter": (OpenRouterClient, "OPENROUTER_API_KEY"),
}


def apply_path_validator_patch(config: CrunchNodeConfig):
    path_validator = CrunchNodeAsyncValidatorSigningProxy(
        run_registry_dir=config.run_registry_dir
    )

    @app.middleware("http")
    async def body_middleware(request: Request, call_next):
        path = request.url.path
        if not path.startswith("/api/gateway/"):
            return await call_next(request)

        if not request.method in ("POST", "PUT", "PATCH"):
            return await call_next(request)

        body = await request.body()

        # TODO: body is parsed twice into json...
        try:
            blocked = path_validator._check_track_access(path, body)
        except UnknownRunIdException as exception:
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": str(exception)},
            )

        if blocked is not None:
            return JSONResponse(
                status_code=blocked.status,
                content={"detail": blocked.text},
            )

        try:
            parsed = json.loads(body)
            run_id = parsed.get("run_id")
        except (json.JSONDecodeError, AttributeError):
            run_id = None

        if run_id is not None and run_id.startswith("public-"):
            run_id = "00000000-0000-0000-0000-000000000000"

            parsed["run_id"] = run_id
            body = json.dumps(parsed).encode()

        request._body = body

        response = await call_next(request)
        return response


def apply_client_api_key_patches():
    request_ctx: ContextVar[Request] = ContextVar("request_ctx")

    @app.middleware("http")
    async def set_request_context(request: Request, call_next):
        token = request_ctx.set(request)
        response = await call_next(request)
        request_ctx.reset(token)
        return response

    def _get_api_key_from_current_request(client_name: str) -> str:
        header_name = f"x-{client_name}-api-key"

        api_key = request_ctx.get().headers.get(header_name)
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"{header_name} header is required for {client_name}",
            )

        return api_key

    def patch_init(client_name: str, client_class: type):
        original = client_class.__init__

        def patched(self, **kwargs):
            kwargs["api_key"] = _get_api_key_from_current_request(client_name)
            original(self, **kwargs)

        return patched

    for client_name, (client_class, api_key_env_var) in CLIENTS.items():
        if api_key_env_var is None:
            continue

        os.environ[api_key_env_var] = "invalid"

        client_class.__init__ = patch_init(client_name, client_class)

    def patched_data_get_api_key_for_source(source: public_data_provider.PublicDataSourceInfo) -> str | None:
        if not source.requires_auth:
            return None

        client_name = source.name.lower().replace("_", "-")
        return _get_api_key_from_current_request(client_name)

    public_data_provider._get_api_key_for_source = patched_data_get_api_key_for_source


def apply_cache_patch():
    gateway_cache._cache = LRU(10_000)


def apply_openapi_patch():
    for route in app.router.routes:
        match = re.match(r"/api/gateway/(.+?)/", route.path)
        if not match:
            continue

        client_name = match.group(1)
        _, api_key_env_var = CLIENTS[client_name]

        if api_key_env_var is None:
            continue

        if route.openapi_extra is None:
            route.openapi_extra = {}

        route.openapi_extra.setdefault("parameters", []).append({
            "name": f"x-{client_name}-api-key",
            "in": "header",
            "required": True,
            "schema": {"type": "string"},
            "description": f"{client_name} API key header",
        })
