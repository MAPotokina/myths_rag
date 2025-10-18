#!/usr/bin/env python3
"""
normalize_prose_edda.py
========================

This script automates the process of extracting and normalising the text
from a PDF copy of **The Prose Edda**.  Normalising a text prior to
ingestion into a retrieval‑augmented generation (RAG) pipeline is
important because most PDFs include irregular line breaks, hyphenated
words at the end of lines, and various kinds of fancy punctuation.  If
left unprocessed, these artefacts will produce poor vector
representations when embedded.

### Enhanced Features

* **OCR Error Correction** – Fixes common OCR errors like "bom" → "born"
  and removes soft hyphens that weren't caught by the initial processing.

* **Advanced Formatting Cleanup** – Handles spaced-out text, form feeds,
  and other PDF formatting artifacts that can interfere with text processing.

* **Structural Segmentation** – Identifies and preserves document structure
  including chapters, sections, and logical breaks for better context.

* **Semantic Segmentation** – Improves chunking by respecting sentence and
  paragraph boundaries rather than arbitrary character limits.

* **Metadata Enrichment** – Adds structural metadata and generates multiple
  output formats (JSON, Markdown, plain text) for different use cases.

* **Quality Control** – Validates output and flags potential issues for
  manual review.

### Usage

Run the script from the command line and specify the path to your
Prose Edda PDF and an output directory:

```bash
python3 scripts/normalize_prose_edda.py \
    --input_pdf path/to/prose_edda.pdf \
    --output_dir data/processed
```

Multiple files will be created in the output directory:

1. `normalized.txt` – the full normalised text of the Prose Edda.
2. `segments.txt` – one segment per line, suitable for loading into
   downstream vectorisation pipelines.
3. `segments.json` – structured segments with metadata.
4. `structure.json` – document structure and navigation.
5. `quality_report.txt` – validation report and flagged issues.

"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class DocumentStructure:
    """Represents the structural elements of the document."""

    title: str
    author: str
    chapters: list[dict[str, Any]]
    sections: list[dict[str, Any]]
    metadata: dict[str, Any]


@dataclass
class TextSegment:
    """Represents a segment of text with metadata."""

    content: str
    segment_id: str
    chapter: Optional[str] = None
    section: Optional[str] = None
    page_ref: Optional[str] = None
    word_count: int = 0
    char_count: int = 0


@dataclass
class QualityReport:
    """Represents quality control findings."""

    ocr_errors: list[str]
    formatting_issues: list[str]
    segmentation_issues: list[str]
    validation_warnings: list[str]


def extract_text_with_pdftotext(pdf_path: Path) -> str:
    """Extract the text of a PDF using the ``pdftotext`` CLI.

    ``pdftotext`` prints the extracted text to stdout when the output
    filename is ``-``.  If the command is not installed, a
    ``FileNotFoundError`` will be raised.

    Args:
        pdf_path: Path to the input PDF.

    Returns:
        A string containing all extracted text.
    """
    # Ensure the input file exists
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Call pdftotext to convert the PDF to text.  We pass '-' as the
    # output filename so that the text is written to stdout.  The
    # `-layout` option preserves a rough layout of the original, which
    # helps maintain paragraph structure.
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    text = result.stdout.decode("utf-8", errors="ignore")
    return text


def normalise_text(text: str) -> tuple[str, QualityReport, list[tuple[int, str]]]:
    """Perform a series of normalisation steps on the extracted text.

    Enhanced steps include:
    - Reassembling words broken across line breaks with a hyphen
    - Converting curly quotation marks and dashes to ASCII equivalents
    - Normalising Unicode code points and removing combining accents
    - Fixing common OCR errors (bom -> born, etc.)
    - Removing soft hyphens and other invisible characters
    - Normalizing spaced-out text (A N T H O N Y -> ANTHONY)
    - Converting form feeds to paragraph breaks
    - Collapsing multiple spaces and trimming whitespace
    - Filtering to only include valuable chapters (PROLOGUE, GYLFAGINNING, SKALDSKAPARMAL, HATTATAL)
    - Removing page headers and footers while preserving page numbers for metadata

    Args:
        text: Raw text extracted from the PDF.

    Returns:
        A tuple of (normalised string, quality report).
    """
    quality_report = QualityReport(
        ocr_errors=[],
        formatting_issues=[],
        segmentation_issues=[],
        validation_warnings=[],
    )

    # Remove hyphen followed by newline: e.g. "loca-\nted" -> "located"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Remove soft hyphens (Unicode U+00AD) that weren't caught
    soft_hyphen_count = text.count("\u00ad")
    if soft_hyphen_count > 0:
        text = text.replace("\u00ad", "")
        quality_report.formatting_issues.append(
            f"Removed {soft_hyphen_count} soft hyphens"
        )

    # Fix common OCR errors
    ocr_corrections = {
        r"\bbom\b": "born",
        r"\b(\w+)­(\w+)\b": r"\1\2",  # Remove soft hyphens in words
    }

    for pattern, replacement in ocr_corrections.items():
        matches = re.findall(pattern, text)
        if matches:
            text = re.sub(pattern, replacement, text)
            quality_report.ocr_errors.append(
                f"Fixed OCR error: {pattern} -> {replacement}"
            )

    # Replace curly quotes, dashes and ellipses with ASCII equivalents
    replacements = {
        """: '"',
        """: '"',
        "'": "'",
        "—": "-",
        "–": "-",
        "…": "...",
    }
    for old, new in replacements.items():
        if old in text:
            text = text.replace(old, new)

    # Normalize spaced-out text (e.g., "A N T H O N Y" -> "ANTHONY")
    # This pattern matches sequences of single letters with spaces
    spaced_text_matches = re.findall(
        r"\b([A-Z])\s+([A-Z])(?:\s+([A-Z]))*(?:\s+([A-Z]))*", text
    )
    if spaced_text_matches:
        # More sophisticated pattern for spaced text
        text = re.sub(
            r"\b([A-Z])\s+([A-Z])(?:\s+([A-Z]))*(?:\s+([A-Z]))*",
            lambda m: "".join([g for g in m.groups() if g]),
            text,
        )
        quality_report.formatting_issues.append("Normalized spaced-out text")

    # Convert form feeds to paragraph breaks
    form_feed_count = text.count("\f")
    if form_feed_count > 0:
        text = re.sub(r"\f+", "\n\n", text)
        quality_report.formatting_issues.append(
            f"Converted {form_feed_count} form feeds to paragraph breaks"
        )

    # Convert to Unicode decomposition (NFKD) and strip accents
    nfkd_form = unicodedata.normalize("NFKD", text)
    text_no_accents = "".join(c for c in nfkd_form if not unicodedata.combining(c))

    # Replace multiple whitespace with a single space and normalise line endings
    # First normalise line endings to '\n'
    text_no_accents = text_no_accents.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of spaces and tabs
    text_no_accents = re.sub(r"[ \t]+", " ", text_no_accents)
    # Strip trailing spaces on each line
    lines = [line.rstrip() for line in text_no_accents.split("\n")]

    # Filter to only include valuable chapters and clean page headers/footers
    filtered_lines, page_numbers = _filter_chapters_and_clean_pages(
        lines, quality_report
    )

    return "\n".join(filtered_lines), quality_report, page_numbers


def _filter_chapters_and_clean_pages(
    lines: list[str], quality_report: QualityReport
) -> tuple[list[str], list[tuple[int, str]]]:
    """Filter text to only include valuable chapters and clean page headers/footers.

    Args:
        lines: list of text lines to process.
        quality_report: Quality report to update with findings.

    Returns:
        tuple of (filtered lines, list of (filtered_line_number, page_number) tuples).
    """
    # Define the valuable chapters
    valuable_chapters = {"PROLOGUE", "GYLFAGINNING", "SKALDSKAPARMAL", "HATTATAL"}

    filtered_lines = []
    page_numbers = []
    current_chapter = None
    in_valuable_content = False
    filtered_line_count = 0

    # Patterns for page headers and footers
    # Handle various formats like [14 -15 ] G y lfa g in n in g, [4,9-10] Prologue, [1-3], etc.
    page_header_pattern = re.compile(r"^\[([^\]]+)\](?:\s+([A-Za-z\s]+))?$")
    page_footer_pattern = re.compile(r"^\s*(\d+)\s*$")

    for i, line in enumerate(lines):
        line_clean = line.strip().upper()

        # Check if this line starts a valuable chapter
        if line_clean in valuable_chapters:
            current_chapter = line_clean
            in_valuable_content = True
            filtered_lines.append(line)  # Keep the chapter header
            filtered_line_count += 1
            quality_report.formatting_issues.append(
                f"Started chapter: {current_chapter}"
            )
            continue

        # Skip everything before we reach a valuable chapter
        if not in_valuable_content:
            continue

        # Check for page headers like "[14 -15 ] G y lfa g in n in g" or "[1-3]"
        header_match = page_header_pattern.match(line)
        if header_match:
            page_num_raw = header_match.group(1).strip()

            # Clean up the page number (remove extra spaces, normalize format)
            page_num_clean = re.sub(r"\s+", "", page_num_raw)  # Remove all spaces
            page_num_clean = re.sub(
                r"[^\d\-]", "", page_num_clean
            )  # Keep only digits and dashes

            page_numbers.append((filtered_line_count, page_num_clean))
            quality_report.formatting_issues.append(f"Removed page header: {line}")
            continue  # Skip the header line

        # Check for page footers (just page numbers)
        footer_match = page_footer_pattern.match(line)
        if footer_match:
            page_num = footer_match.group(1)
            page_numbers.append((filtered_line_count, page_num))
            quality_report.formatting_issues.append(f"Removed page footer: {line}")
            if page_num == "220":
                break  # Skip the footer line
            continue  # Skip the footer line

        # Skip other common non-content elements
        if _is_non_content_line(line):
            quality_report.formatting_issues.append(
                f"Removed non-content line: {line[:50]}..."
            )
            continue

        # Clean embedded page references from the line content
        cleaned_line = _clean_embedded_page_references(
            line, page_numbers, filtered_line_count, quality_report
        )

        # Keep the line if we're in valuable content
        if cleaned_line.strip():  # Only add non-empty lines
            filtered_lines.append(cleaned_line)
            filtered_line_count += 1

    quality_report.formatting_issues.append(
        f"Filtered to {len(filtered_lines)} lines from {len(lines)} original lines"
    )
    quality_report.formatting_issues.append(
        f"Extracted {len(page_numbers)} page references"
    )

    return filtered_lines, page_numbers


def _is_non_content_line(line: str) -> bool:
    """Check if a line is non-content that should be skipped.

    Args:
        line: Line to check.

    Returns:
        True if the line should be skipped.
    """
    line_clean = line.strip()

    # Skip empty lines (they'll be handled by paragraph splitting)
    if not line_clean:
        return True

    # Skip lines that are just page numbers or roman numerals
    if re.match(r"^\s*[ivxlcdm]+\s*$", line_clean, re.IGNORECASE):
        return True

    # Skip lines that are just numbers
    if re.match(r"^\s*\d+\s*$", line_clean):
        return True

    # Skip very short lines that are likely headers/footers
    if len(line_clean) < 3:
        return True

    # Skip lines that are all caps and very short (likely headers)
    if line_clean.isupper() and len(line_clean) < 20:
        return True

    # Skip lines with only punctuation
    if re.match(r"^[^\w\s]*$", line_clean):
        return True

    return False


def _clean_embedded_page_references(
    line: str,
    page_numbers: list[tuple[int, str]],
    line_num: int,
    quality_report: QualityReport,
) -> str:
    """Clean embedded page references from line content.

    Args:
        line: Line to clean.
        page_numbers: list to add extracted page numbers to.
        line_num: Line number for tracking.
        quality_report: Quality report to update.

    Returns:
        Cleaned line with page references removed.
    """
    # Pattern to match embedded page references like "Prologue [1-3]" or "text [14-15]"
    embedded_pattern = re.compile(r"(\w+)\s*\[([^\]]+)\]")

    def replace_page_ref(match: re.Match[str]) -> str:
        word_before = match.group(1)
        page_ref = match.group(2)

        # Clean up the page number
        page_num_clean = re.sub(r"\s+", "", page_ref)  # Remove all spaces
        page_num_clean = re.sub(
            r"[^\d\-]", "", page_num_clean
        )  # Keep only digits and dashes

        # Add to page numbers list
        page_numbers.append((line_num, page_num_clean))
        quality_report.formatting_issues.append(
            f"Removed embedded page reference: [{page_ref}] from '{word_before}'"
        )

        # Return just the word before the page reference
        return word_before

    # Replace embedded page references
    cleaned_line = embedded_pattern.sub(replace_page_ref, line)

    return cleaned_line


def _create_paragraphs_from_sentences(text: str) -> list[str]:
    """Create paragraphs from sentences when no paragraph breaks exist.

    Args:
        text: Text to split into paragraphs.

    Returns:
        list of paragraph strings.
    """
    # Split text into lines first
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    paragraphs = []
    current_paragraph = []

    for line in lines:
        # If line ends with sentence-ending punctuation, it's likely the end of a sentence
        if line.endswith((".", "!", "?", ":")):
            current_paragraph.append(line)
            # Join the current paragraph and add it
            if current_paragraph:
                paragraphs.append(" ".join(current_paragraph))
                current_paragraph = []
        else:
            current_paragraph.append(line)

    # Add any remaining content as a paragraph
    if current_paragraph:
        paragraphs.append(" ".join(current_paragraph))

    # If we still have very few paragraphs, try splitting by length
    if len(paragraphs) <= 2:
        # Split long paragraphs into smaller chunks
        new_paragraphs = []
        for para in paragraphs:
            if len(para) > 1000:  # If paragraph is very long
                # Split by sentences
                sentences = re.split(r"(?<=[.!?])\s+", para)
                current_chunk: list[str] = []
                current_length = 0

                for sentence in sentences:
                    if current_length + len(sentence) > 800:  # Target chunk size
                        if current_chunk:
                            new_paragraphs.append(" ".join(current_chunk))
                        current_chunk = [sentence]
                        current_length = len(sentence)
                    else:
                        current_chunk.append(sentence)
                        current_length += len(sentence)

                if current_chunk:
                    new_paragraphs.append(" ".join(current_chunk))
            else:
                new_paragraphs.append(para)

        paragraphs = new_paragraphs

    return paragraphs


def identify_document_structure(text: str) -> DocumentStructure:
    """Identify and extract document structure from the filtered normalized text.

    Args:
        text: Filtered normalized text to analyze (only valuable chapters).

    Returns:
        DocumentStructure object with identified elements.
    """
    lines = text.split("\n")

    # Extract title and author
    title = "Prose Edda"
    author = "Snorri Sturluson"

    # Identify chapters - only the valuable ones should be present
    chapters = []
    valuable_chapters = ["PROLOGUE", "GYLFAGINNING", "SKALDSKAPARMAL", "HATTATAL"]

    for i, line in enumerate(lines):
        line_clean = line.strip().upper()

        # Check for valuable chapter markers
        if line_clean in valuable_chapters:
            chapters.append(
                {
                    "title": line.strip(),
                    "line_number": i,
                    "type": "chapter",
                    "chapter_key": line_clean,
                }
            )

    # Identify sections within chapters (subheadings, etc.)
    sections = []
    for i, line in enumerate(lines):
        line_clean = line.strip()

        # Look for section markers (often italicized or indented in original)
        # These might be subsection titles or topic headers
        if (
            len(line_clean) > 10
            and len(line_clean) < 100
            and not line_clean.isupper()  # Not chapter titles
            and not re.match(r"^\d+", line_clean)  # Not numbered lists
            and line_clean.endswith(":")
        ):  # Often end with colon
            sections.append({"title": line_clean, "line_number": i, "type": "section"})

    metadata = {
        "total_lines": len(lines),
        "total_chapters": len(chapters),
        "total_sections": len(sections),
        "valuable_chapters_only": True,
        "filtered_content": True,
    }

    return DocumentStructure(
        title=title,
        author=author,
        chapters=chapters,
        sections=sections,
        metadata=metadata,
    )


def split_into_segments(
    text: str,
    max_length: int = 1500,
    overlap: int = 200,
    document_structure: Optional[DocumentStructure] = None,
    page_numbers: Optional[list[tuple[int, str]]] = None,
) -> list[TextSegment]:
    """Split normalised text into structured segments with metadata.

    Enhanced segmentation that:
    - Respects sentence and paragraph boundaries
    - Includes structural metadata
    - Provides better context preservation
    - Generates unique segment IDs

    Args:
        text: Normalised text to segment.
        max_length: Maximum number of characters in a segment.
        overlap: Number of characters from the end of the previous
            segment to include at the start of the next segment.
        document_structure: Optional document structure for metadata.

    Returns:
        A list of TextSegment objects with metadata.
    """
    # Split into paragraphs first
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # If no paragraphs found (no double newlines), create paragraphs from sentences
    if len(paragraphs) <= 1:
        paragraphs = _create_paragraphs_from_sentences(text)

    segments: list[TextSegment] = []
    current_content = ""
    segment_counter = 0

    for paragraph in paragraphs:
        # Check if adding this paragraph would exceed the limit
        if current_content and len(current_content) + len(paragraph) + 2 > max_length:
            # Create segment from current content
            segment_id = f"segment_{segment_counter:04d}"
            segment = TextSegment(
                content=current_content.strip(),
                segment_id=segment_id,
                word_count=len(current_content.split()),
                char_count=len(current_content),
            )
            segments.append(segment)
            segment_counter += 1

            # Determine overlapping context for next segment
            if overlap > 0 and len(current_content) > overlap:
                # Find a good break point within the overlap
                context = current_content[-overlap:]
                # Try to break at sentence or word boundary
                sentence_break = context.rfind(". ")
                if sentence_break > overlap // 2:
                    context = context[sentence_break + 2 :]
                else:
                    word_break = context.rfind(" ")
                    if word_break > overlap // 2:
                        context = context[word_break + 1 :]
                current_content = context + paragraph
            else:
                current_content = paragraph
        else:
            # Append paragraph to the current segment
            if current_content:
                current_content += "\n\n" + paragraph
            else:
                current_content = paragraph

    # Add the final segment
    if current_content:
        segment_id = f"segment_{segment_counter:04d}"
        segment = TextSegment(
            content=current_content.strip(),
            segment_id=segment_id,
            word_count=len(current_content.split()),
            char_count=len(current_content),
        )
        segments.append(segment)

    # Add structural metadata if available
    if document_structure:
        _add_structural_metadata(segments, document_structure)

    # Add page references if available
    if page_numbers:
        _add_page_references(segments, page_numbers)

    return segments


def _add_structural_metadata(
    segments: list[TextSegment], structure: DocumentStructure
) -> None:
    """Add structural metadata to segments based on document structure."""
    lines = []
    for segment in segments:
        lines.extend(segment.content.split("\n"))

    for segment in segments:
        segment_lines = segment.content.split("\n")
        if not segment_lines:
            continue

        # Find which chapter this segment belongs to
        segment_start_line = sum(
            len(s.content.split("\n")) for s in segments[: segments.index(segment)]
        )

        # Find the most recent chapter before this segment
        current_chapter = None
        for chapter in structure.chapters:
            if chapter["line_number"] <= segment_start_line:
                current_chapter = chapter["title"]
            else:
                break

        if current_chapter:
            segment.chapter = current_chapter


def _add_page_references(
    segments: list[TextSegment], page_numbers: list[tuple[int, str]]
) -> None:
    """Add page references to segments based on line numbers.

    Args:
        segments: list of segments to update.
        page_numbers: list of (line_number, page_number) tuples from the original text.
    """
    if not page_numbers:
        return

    # Create a mapping from line numbers to page numbers
    page_map = {}
    for line_num, page_num in page_numbers:
        if page_num.strip():  # Only add non-empty page numbers
            page_map[line_num] = page_num

    # For each segment, find the most relevant page number
    for segment in segments:
        # Count lines in previous segments to find this segment's starting line
        segment_start_line = 0
        for prev_segment in segments:
            if prev_segment == segment:
                break
            segment_start_line += len(prev_segment.content.split("\n"))

        # Find the closest page number before or at this segment
        closest_page = None
        for line_num in sorted(page_map.keys()):
            if line_num <= segment_start_line:
                closest_page = page_map[line_num]
            else:
                break

        if closest_page:
            segment.page_ref = closest_page


def write_output_files(
    output_dir: Path,
    normalized_text: str,
    segments: list[TextSegment],
    document_structure: DocumentStructure,
    quality_report: QualityReport,
) -> None:
    """Write all output files in various formats."""

    # Write normalized text
    normalized_path = output_dir / "normalized.txt"
    with normalized_path.open("w", encoding="utf-8") as fh:
        fh.write(normalized_text)
    print(f"Wrote normalised text to {normalized_path}")

    # Write segments as plain text
    segments_path = output_dir / "segments.txt"
    with segments_path.open("w", encoding="utf-8") as fh:
        for segment in segments:
            fh.write(segment.content.strip() + "\n\n")
    print(f"Wrote {len(segments)} segments to {segments_path}")

    # Write segments as JSON with metadata
    segments_json_path = output_dir / "segments.json"
    segments_data = []
    for segment in segments:
        segments_data.append(
            {
                "segment_id": segment.segment_id,
                "content": segment.content,
                "chapter": segment.chapter,
                "section": segment.section,
                "page_ref": segment.page_ref,
                "word_count": segment.word_count,
                "char_count": segment.char_count,
            }
        )

    with segments_json_path.open("w", encoding="utf-8") as fh:
        json.dump(segments_data, fh, indent=2, ensure_ascii=False)
    print(f"Wrote structured segments to {segments_json_path}")

    # Write document structure
    structure_path = output_dir / "structure.json"
    structure_data = {
        "title": document_structure.title,
        "author": document_structure.author,
        "chapters": document_structure.chapters,
        "sections": document_structure.sections,
        "metadata": document_structure.metadata,
    }

    with structure_path.open("w", encoding="utf-8") as fh:
        json.dump(structure_data, fh, indent=2, ensure_ascii=False)
    print(f"Wrote document structure to {structure_path}")

    # Write quality report
    quality_path = output_dir / "quality_report.txt"
    with quality_path.open("w", encoding="utf-8") as fh:
        fh.write("QUALITY CONTROL REPORT\n")
        fh.write("=" * 50 + "\n\n")

        fh.write("OCR ERRORS FIXED:\n")
        if quality_report.ocr_errors:
            for error in quality_report.ocr_errors:
                fh.write(f"  - {error}\n")
        else:
            fh.write("  - No OCR errors detected\n")
        fh.write("\n")

        fh.write("FORMATTING ISSUES FIXED:\n")
        if quality_report.formatting_issues:
            for issue in quality_report.formatting_issues:
                fh.write(f"  - {issue}\n")
        else:
            fh.write("  - No formatting issues detected\n")
        fh.write("\n")

        fh.write("SEGMENTATION ISSUES:\n")
        if quality_report.segmentation_issues:
            for issue in quality_report.segmentation_issues:
                fh.write(f"  - {issue}\n")
        else:
            fh.write("  - No segmentation issues detected\n")
        fh.write("\n")

        fh.write("VALIDATION WARNINGS:\n")
        if quality_report.validation_warnings:
            for warning in quality_report.validation_warnings:
                fh.write(f"  - {warning}\n")
        else:
            fh.write("  - No validation warnings\n")
        fh.write("\n")

        fh.write("SUMMARY:\n")
        fh.write(f"  - Total segments: {len(segments)}\n")
        fh.write(f"  - Total chapters identified: {len(document_structure.chapters)}\n")
        fh.write(f"  - Total sections identified: {len(document_structure.sections)}\n")
        fh.write(
            f"  - Average segment length: {sum(s.char_count for s in segments) // len(segments) if segments else 0} characters\n"
        )

    print(f"Wrote quality report to {quality_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and normalise the Prose Edda PDF for RAG ingestion with enhanced processing."
    )
    parser.add_argument(
        "--input_pdf",
        type=Path,
        required=True,
        help="Path to the Prose Edda PDF file",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where the normalised outputs will be saved",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1500,
        help="Maximum number of characters per segment (default: 1500)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Number of overlapping characters between consecutive segments (default: 200)",
    )
    parser.add_argument(
        "--min_segment_length",
        type=int,
        default=100,
        help="Minimum segment length to avoid very short segments (default: 100)",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Extract text from the PDF
    print(f"Extracting text from {args.input_pdf}…")
    raw_text = extract_text_with_pdftotext(args.input_pdf)

    # Normalise the extracted text
    print("Normalising text…")
    normalised, quality_report, page_numbers = normalise_text(raw_text)

    # Identify document structure
    print("Identifying document structure…")
    document_structure = identify_document_structure(normalised)

    # Create segments with enhanced processing
    print("Creating structured segments…")
    segments = split_into_segments(
        normalised, args.max_length, args.overlap, document_structure, page_numbers
    )

    # Filter out very short segments
    segments = [s for s in segments if s.char_count >= args.min_segment_length]

    # Add segmentation quality checks
    for i, segment in enumerate(segments):
        if segment.char_count < args.min_segment_length:
            quality_report.segmentation_issues.append(
                f"Segment {segment.segment_id} is very short ({segment.char_count} chars)"
            )

        # Check for incomplete sentences at segment boundaries
        if not segment.content.endswith((".", "!", "?", '"', "'")):
            if i < len(segments) - 1:  # Not the last segment
                quality_report.validation_warnings.append(
                    f"Segment {segment.segment_id} may end mid-sentence"
                )

    # Write all output files
    print("Writing output files…")
    write_output_files(
        args.output_dir, normalised, segments, document_structure, quality_report
    )

    print("\nProcessing complete!")
    print(
        f"Generated {len(segments)} segments from {len(document_structure.chapters)} chapters"
    )
    if quality_report.ocr_errors or quality_report.formatting_issues:
        print(
            "Quality issues were detected and fixed - see quality_report.txt for details"
        )


if __name__ == "__main__":
    main()
