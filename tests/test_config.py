import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import sys

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from ala.config.setup import parse_arguments
from ala.data.processing import load_prepared_data
from ala.config.paths import FolderPaths


class TestFolderPaths(unittest.TestCase):
    def setUp(self) -> None:
        self.args = SimpleNamespace(ft_type="qlora")
        self.folders = FolderPaths(self.args)

    def tearDown(self) -> None:
        # Remove any artifacts created during tests inside the documents/data directories.
        for folder in (
            self.folders.documents_folder,
            self.folders.prepared_data_folder,
        ):
            for path in Path(folder).glob("test_unittest_*"):
                if path.is_file():
                    path.unlink()

    def test_required_directories_exist(self):
        expected = [
            self.folders.documents_folder,
            self.folders.prepared_data_folder,
            self.folders.image_folder,
            self.folders.json_data_folder,
            self.folders.finetune_data_folder,
            self.folders.summaries_folder,
            self.folders.ft_folder,
        ]

        for directory in expected:
            self.assertTrue(Path(directory).exists(), f"Missing directory: {directory}")


class TestDataProcessing(unittest.TestCase):
    def setUp(self) -> None:
        self.args = SimpleNamespace(
            ft_type="qlora",
            word_limit=10,
            overlap=0.0,
            starting_index=0,
            ending_index=1,
            images=False,
        )
        self.temp_dir = tempfile.TemporaryDirectory()
        base_path = Path(self.temp_dir.name)
        data_dir = base_path / "data"
        models_dir = base_path / "models"

        self.folders = SimpleNamespace(
            documents_folder=data_dir / "documents",
            prepared_data_folder=data_dir / "data_prepared",
            image_folder=data_dir / "images",
            json_data_folder=data_dir / "qa_json",
            finetune_data_folder=data_dir / "data_ft",
            summaries_folder=data_dir / "summaries",
            ft_folder=models_dir / self.args.ft_type,
        )

        for path in (
            self.folders.documents_folder,
            self.folders.prepared_data_folder,
            self.folders.image_folder,
            self.folders.json_data_folder,
            self.folders.finetune_data_folder,
            self.folders.summaries_folder,
            self.folders.ft_folder,
        ):
            Path(path).mkdir(parents=True, exist_ok=True)

        self.document_path = Path(self.folders.documents_folder) / "test_unittest_document.txt"
        self.document_path.write_text(
            "This is a simple test document with enough words to create chunks."
        )

        self.prepared_before = set(Path(self.folders.prepared_data_folder).glob("*"))

    def tearDown(self) -> None:
        if self.document_path.exists():
            self.document_path.unlink()

        prepared_dir = Path(self.folders.prepared_data_folder)
        for path in prepared_dir.glob("test_unittest_*"):
            path.unlink()

        current_files = set(prepared_dir.glob("*"))
        for path in current_files - self.prepared_before:
            if path.is_file():
                path.unlink()

        self.temp_dir.cleanup()

    def test_load_prepared_data_creates_chunks(self):
        text_chunks, chunk_count = load_prepared_data(self.args, self.folders)

        self.assertGreater(chunk_count, 0, "Expected at least one chunk to be created")
        self.assertEqual(len(text_chunks), chunk_count)


class TestArgumentParsing(unittest.TestCase):
    def test_local_and_claude_conflict(self):
        with self.assertRaises(ValueError):
            parse_arguments(["--claude"])

    def test_openrouter_and_claude_conflict(self):
        with self.assertRaises(ValueError):
            parse_arguments(["--openrouter", "--claude"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
