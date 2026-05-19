import asyncio
import functools
import json
from typing import Any, Callable

import aiohttp
import nest_asyncio2
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from neurons.miner.gateway import error_handler as gateway_error_handler

nest_asyncio2.apply()

original_raise_for_status = aiohttp.ClientResponse.raise_for_status


def apply_aiohttp_patch():

    def patched_raise_for_status(self):
        try:
            return original_raise_for_status(self)
        except aiohttp.ClientResponseError as error:
            response_body = asyncio.run(self.text())
            error.body = response_body

            raise error

    aiohttp.ClientResponse.raise_for_status = patched_raise_for_status


def apply_gateway_error_handling_patch():

    def patched_handle_provider_errors(provider: str) -> Callable[[Callable], Callable]:
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await func(*args, **kwargs)
                except HTTPException:
                    raise
                except Exception as e:
                    error_message = f"{provider} API error: {str(e)}"

                    if isinstance(e, aiohttp.ClientResponseError):
                        response_body = getattr(e, "body", None)
                        try:
                            response_body = json.loads(response_body)
                        except (json.JSONDecodeError, TypeError):
                            pass

                        status_code = e.status
                    else:
                        response_body = None
                        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

                    gateway_error_handler.logger.error(
                        error_message,
                        exc_info=e if status_code >= 500 else None
                    )

                    return JSONResponse(
                        status_code=status_code,
                        content={
                            "detail": error_message,
                            "upstream_error": response_body,
                        },
                    )

            return wrapper

        return decorator

    gateway_error_handler.handle_provider_errors = patched_handle_provider_errors
