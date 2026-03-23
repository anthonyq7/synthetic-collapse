import re

# Match [Author], [year] or [Author], year or Author, [year] or Author, year
_CITE_PATTERN = re.compile(
    r"\[?\s*([A-Z][a-z]+)\s*\]?\s*,\s*\[?\s*(\d{4})\s*\]?"
)


def extract_citations_from_body(text: str) -> set[tuple[str, int]]:
    results: set[tuple[str, int]] = set()
    if not text:
        return results
    for group in re.findall(r"\(([^)]+)\)", text):
        segments = [s.strip() for s in group.split(";")]
        for segment in segments:
            for m in _CITE_PATTERN.finditer(segment):
                author = m.group(1).strip()
                year = int(m.group(2))
                results.add((author, year))
    return results
