"""Helpers for extracting JSON payloads from language model outputs."""

from __future__ import annotations

import json
from typing import Any, List, Optional


def extract_json(output: str) -> Optional[List[Any]]:
    try:
        start_index = output.find('{')
        end_index = output.rfind('}') + 1
        if start_index == -1 or end_index <= start_index:
            return None
        json_str = output[start_index:end_index]
        data = json.loads(json_str)
        if isinstance(data, dict) and 'text' in data:
            print('Succesfully extracted JSON')
            return data['text']
        if isinstance(data, list):
            print('Succesfully extracted JSON list')
            return data
        return None
    except (json.JSONDecodeError, TypeError):
        return None
