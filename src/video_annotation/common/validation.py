"""Dataset metadata dependency guard. Source: Event/scripts/dataset.py.

Implementations extracted from the Event repository; source files are retained there.
"""
import re


_OUTPUT_PATH = re.compile(r'(?:^|[/\\\s\"\'])outputs[/\\]')


def strings(value, key=''):
    """Yield nested JSON strings with a location suitable for error messages."""
    if isinstance(value, dict):
        for name, item in value.items():
            yield from strings(item, f'{key}.{name}' if key else name)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from strings(item, f'{key}[{index}]')
    elif isinstance(value, str):
        yield key, value


def require_no_output_references(value, label='Dataset metadata'):
    for key, text in strings(value):
        if _OUTPUT_PATH.search(text):
            raise ValueError(f'{label}:{key} references outputs: {text}. '
                             'Store the required asset under data before publishing.')
