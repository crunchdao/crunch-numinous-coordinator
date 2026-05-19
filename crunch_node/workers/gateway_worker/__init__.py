"""
Gateway worker — FastAPI app serving the Gateway.

Reads predictions, scores, agent runs directly from PG
and exposes them via REST endpoints.
"""

if True:
    # bt_compat MUST be imported before any neurons.validator.* module
    import crunch_node.bt_compat  # noqa: F401

from crunch_node.config import CrunchNodeConfig

config = CrunchNodeConfig()


if True:
    from ._error import apply_aiohttp_patch, apply_gateway_error_handling_patch

    apply_aiohttp_patch()
    apply_gateway_error_handling_patch()


if True:
    from ._app import apply_cache_patch, apply_client_api_key_patches, apply_openapi_patch, apply_path_validator_patch

    apply_path_validator_patch(config)
    apply_client_api_key_patches()
    apply_cache_patch()
    apply_openapi_patch()

if True:
    from ._app import app
