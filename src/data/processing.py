"""Prepared data loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

from natsort import natsorted

from .prepare import file_processor


def _slice_files(files: Sequence[Path], start: int, end: int | None) -> Sequence[Path]:
    if end is None:
        return files[start:]
    return files[start:end]


def load_prepared_data(args, folders) -> Tuple[List[str], int]:
    file_processor(folders, args)

    data_files = natsorted(folders.prepared_data_folder.glob("*"), key=lambda path: path.name)
    selected_files = _slice_files(data_files, args.starting_index, args.ending_index)

    text_chunks: List[str] = []
    chunk_count = 0
    for file_path in selected_files:
        print("current file:", file_path)
        text_chunk = file_path.read_text()
        text_chunks.append(text_chunk)
        chunk_count += 1

    return text_chunks, chunk_count
