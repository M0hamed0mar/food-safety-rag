"""
API package.

Exposes the FastAPI app factory.
"""

from app.api.main import create_app, app

__all__ = ["create_app", "app"]
