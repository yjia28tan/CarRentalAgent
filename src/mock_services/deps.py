from collections.abc import Callable

import numpy as np
from fastapi import Request

from src.mock_services.repository import Repository
from src.mock_services.services.previews import PreviewStore

Embedder = Callable[[list[str]], np.ndarray]


def get_repo(request: Request) -> Repository:
    return request.app.state.repo


def get_previews(request: Request) -> PreviewStore:
    return request.app.state.previews


def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder
