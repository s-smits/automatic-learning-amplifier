from __future__ import annotations

from pathlib import Path

import pytest

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "src" / "app.py"
DOCS_DIR = Path(__file__).resolve().parents[1] / "data" / "documents"


@pytest.fixture()
def app_test():
    at = AppTest.from_file(str(APP_PATH))
    at.run()
    yield at
    for path in DOCS_DIR.glob("test_upload_*"):
        if path.is_file():
            path.unlink()


def test_initial_state_shows_upload_warning(app_test: AppTest) -> None:
    assert any("upload documents" in warning.value.lower() for warning in app_test.warning)


def test_default_widget_values(app_test: AppTest) -> None:
    assert app_test.slider("Number of Questions").value == 5
    assert app_test.slider("Word Limit").value == 500
    assert app_test.slider("Overlap").value == 0.1


def test_uploading_file_persists_to_documents_folder(app_test: AppTest) -> None:
    sample_name = "test_upload_sample.txt"
    app_test.file_uploader("Upload Documents").set_value(
        [(sample_name, b"sample contents", "text/plain")]
    )
    app_test.run()
    uploaded_path = DOCS_DIR / sample_name
    assert uploaded_path.exists()
    assert not app_test.warning
