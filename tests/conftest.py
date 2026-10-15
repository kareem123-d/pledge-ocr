import os

os.environ["OCR_PRELOAD"] = "false"

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from helpers import FakeEngine


@pytest.fixture
def fake_engine():
    return FakeEngine()


@pytest.fixture
def client(fake_engine):
    with TestClient(create_app(engine_factory=lambda: fake_engine)) as c:
        yield c