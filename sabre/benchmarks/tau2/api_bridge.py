"""
OpenAI-compatible API server that bridges tau2-bench to SABRE.

This runs as a standalone FastAPI server that tau2-bench can call.
"""

import os
from typing import Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from .agent import SabreTau2Agent


def create_app(sabre_port: int = 8011) -> FastAPI:
    """
    Create FastAPI application for the bridge.

    Args:
        sabre_port: Port where SABRE server is running

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="SABRE tau2-bench Bridge",
        description="OpenAI-compatible API bridge for evaluating SABRE with tau2-bench",
        version="0.1.0"
    )

    # Global agent instance
    agent: Optional[SabreTau2Agent] = None

    @app.on_event("startup")
    async def startup():
        nonlocal agent
        try:
            agent = SabreTau2Agent(sabre_port=sabre_port)
            print("✓ SABRE agent initialized and ready for tau2-bench evaluation")
            print(f"✓ Connected to SABRE server at port {sabre_port}")
        except Exception as e:
            print(f"✗ Failed to initialize SABRE agent: {e}")
            raise

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """
        OpenAI-compatible chat completions endpoint.

        This endpoint receives requests from tau2-bench and forwards them to SABRE.
        """
        if agent is None:
            raise HTTPException(
                status_code=503,
                detail="Agent not initialized"
            )

        try:
            data = await request.json()

            # Extract messages and tools to avoid duplicate keyword arguments
            messages = data.pop('messages', [])
            tools = data.pop('tools', None)

            result = await agent.process_chat_completion(
                messages=messages,
                tools=tools,
                **data  # Pass remaining parameters (temperature, max_tokens, etc.)
            )

            # Check if result is an error
            if 'error' in result:
                return JSONResponse(content=result, status_code=500)

            return JSONResponse(content=result)

        except Exception as e:
            return JSONResponse(
                content={
                    "error": {
                        "message": str(e),
                        "type": "bridge_error",
                        "code": "internal_error"
                    }
                },
                status_code=500
            )

    @app.get("/v1/models")
    async def list_models():
        """List available models."""
        return {
            "object": "list",
            "data": [{
                "id": "sabre",
                "object": "model",
                "created": 1677610602,
                "owned_by": "sabre",
                "permission": [],
                "root": "sabre",
                "parent": None
            }]
        }

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        status = "healthy" if agent is not None else "not_initialized"
        return {
            "status": status,
            "agent": "sabre",
            "sabre_port": sabre_port
        }

    return app


def run_bridge(port: int = 8765, sabre_port: int = 8011):
    """
    Start the API bridge server.

    Args:
        port: Port to run the bridge on
        sabre_port: Port where SABRE server is running
    """
    print(f"Starting SABRE-tau2 API bridge...")
    print(f"  Bridge port: {port}")
    print(f"  SABRE server port: {sabre_port}")
    print(f"\nTo use with tau2-bench:")
    print(f"  tau2 run --agent-llm sabre \\")
    print(f"           --agent-api-base http://localhost:{port}/v1 \\")
    print(f"           --domain retail \\")
    print(f"           --num-trials 5")
    print()

    app = create_app(sabre_port=sabre_port)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    run_bridge()
