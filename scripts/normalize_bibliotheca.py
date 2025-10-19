#!/usr/bin/env python3
"""
normalize_bibliotheca.py
========================

This script processes the clean bibliotheca.txt file to extract and normalize
the text from **Bibliotheca** (Library of Greek Mythology by Apollodorus).
The text is already clean and well-formatted, so this script focuses on:

- Filtering to only include Book 1, Book 2, and Book 3 content
- Parsing section markers (e.g., § 1.1.1, § 2.3.4)
- Creating structured segments with metadata
- Generating multiple output formats for RAG ingestion

### Usage

Run the script from the command line and specify the path to your
bibliotheca.txt file and an output directory:

```bash
python3 scripts/normalize_bibliotheca.py \
    --input_txt path/to/bibliotheca.txt \
    --output_dir data/processed
```

Multiple files will be created in the output directory:

1. `bibliotheca_normalized.txt` – the filtered and normalized text.
2. `bibliotheca_segments.txt` – one segment per line for vectorization.
3. `bibliotheca_segments.json` – structured segments with metadata.
4. `bibliotheca_structure.json` – document structure and navigation.
5. `bibliotheca_quality_report.txt` – processing report.

"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DocumentStructure:
    """Represents the structural elements of the document."""

    title: str
    author: str
    books: List[Dict[str, Any]]
    sections: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass
class TextSegment:
    """Represents a segment of text with metadata."""

    content: str
    segment_id: str
    book: Optional[str] = None
    section: Optional[str] = None
    section_number: Optional[str] = None
    word_count: int = 0
    char_count: int = 0


@dataclass
class QualityReport:
    """Represents quality control findings."""

    processing_notes: List[str]
    filtering_notes: List[str]
    segmentation_notes: List[str]
    validation_warnings: List[str]


def read_clean_text(txt_path: Path) -> str:
    """Read the clean bibliotheca.txt file.

    Args:
        txt_path: Path to the bibliotheca.txt file.

    Returns:
        The text content as a string.
    """
    if not txt_path.is_file():
        raise FileNotFoundError(f"Text file not found: {txt_path}")

    with txt_path.open("r", encoding="utf-8") as fh:
        return fh.read()


def filter_books_and_parse_sections(
    text: str,
) -> Tuple[str, QualityReport, List[Dict[str, Any]]]:
    """Filter text to only include Books 1, 2, and 3, and parse section structure.

    Args:
        text: The full text content.

    Returns:
        Tuple of (filtered text, quality report, section metadata).
    """
    quality_report = QualityReport(
        processing_notes=[],
        filtering_notes=[],
        segmentation_notes=[],
        validation_warnings=[],
    )

    lines = text.split("\n")
    filtered_lines: List[str] = []
    sections = []
    in_valuable_content = False

    # Pattern to match section markers like "§ 1.1.1" or "§ 2.3.4"
    section_pattern = re.compile(r"^§\s*(\d+)\.(\d+)\.(\d+)\s+(.+)$")

    for i, line in enumerate(lines):
        line = line.strip()

        # Check for section markers
        section_match = section_pattern.match(line)
        if section_match:
            book_num = int(section_match.group(1))
            chapter_num = int(section_match.group(2))
            section_num = int(section_match.group(3))
            content = section_match.group(4)

            # Only include Books 1, 2, and 3
            if book_num in [1, 2, 3]:
                in_valuable_content = True

                # Create section metadata
                section_info = {
                    "section_id": f"{book_num}.{chapter_num}.{section_num}",
                    "book": book_num,
                    "chapter": chapter_num,
                    "section": section_num,
                    "line_number": len(filtered_lines),
                    "content_preview": content[:100] + "..."
                    if len(content) > 100
                    else content,
                }
                sections.append(section_info)

                # Add the full line to filtered content
                filtered_lines.append(line)
                quality_report.processing_notes.append(
                    f"Added section {section_info['section_id']}"
                )

            else:
                # Skip sections from other books
                quality_report.filtering_notes.append(
                    f"Skipped section from Book {book_num}: {line[:50]}..."
                )
                in_valuable_content = False
        else:
            # Regular content line
            if in_valuable_content and line:
                filtered_lines.append(line)
            elif not in_valuable_content and line:
                # This is content before we reach the valuable books
                quality_report.filtering_notes.append(
                    f"Skipped pre-Book content: {line[:50]}..."
                )

    quality_report.processing_notes.append(
        f"Filtered to {len(filtered_lines)} lines from {len(lines)} original lines"
    )
    quality_report.processing_notes.append(
        f"Found {len(sections)} sections in Books 1-3"
    )

    return "\n".join(filtered_lines), quality_report, sections


def identify_document_structure(
    text: str, sections: List[Dict[str, Any]]
) -> DocumentStructure:
    """Identify and extract document structure from the filtered text.

    Args:
        text: Filtered text content.
        sections: List of section metadata.

    Returns:
        DocumentStructure object with identified elements.
    """
    lines = text.split("\n")

    # Extract title and author
    title = "Bibliotheca"
    author = "Apollodorus"

    # Group sections by book
    books = []
    for book_num in [1, 2, 3]:
        book_sections = [s for s in sections if s["book"] == book_num]
        if book_sections:
            books.append(
                {
                    "book_number": book_num,
                    "title": f"Book {book_num}",
                    "section_count": len(book_sections),
                    "first_section": book_sections[0]["section_id"],
                    "last_section": book_sections[-1]["section_id"],
                }
            )

    # Identify chapters within books
    chapters = []
    for book_num in [1, 2, 3]:
        book_sections = [s for s in sections if s["book"] == book_num]
        chapter_nums = sorted(set(s["chapter"] for s in book_sections))
        for chapter_num in chapter_nums:
            chapter_sections = [s for s in book_sections if s["chapter"] == chapter_num]
            chapters.append(
                {
                    "book": book_num,
                    "chapter": chapter_num,
                    "section_count": len(chapter_sections),
                    "first_section": chapter_sections[0]["section_id"],
                    "last_section": chapter_sections[-1]["section_id"],
                }
            )

    metadata = {
        "total_lines": len(lines),
        "total_books": len(books),
        "total_chapters": len(chapters),
        "total_sections": len(sections),
        "books_included": [1, 2, 3],
        "filtered_content": True,
    }

    return DocumentStructure(
        title=title,
        author=author,
        books=books,
        sections=chapters,  # Using chapters as the main sections
        metadata=metadata,
    )


def split_into_segments(
    text: str,
    max_length: int = 1500,
    overlap: int = 200,
    document_structure: Optional[DocumentStructure] = None,
    sections: Optional[List[Dict[str, Any]]] = None,
) -> List[TextSegment]:
    """Split text into structured segments with metadata.

    Args:
        text: Text to segment.
        max_length: Maximum number of characters in a segment.
        overlap: Number of overlapping characters between segments.
        document_structure: Optional document structure for metadata.
        sections: Optional section metadata.

    Returns:
        A list of TextSegment objects with metadata.
    """
    # Split into sections first (each section is a natural boundary)
    section_pattern = re.compile(r"^§\s*(\d+)\.(\d+)\.(\d+)\s+(.+)$", re.MULTILINE)
    section_matches = list(section_pattern.finditer(text))

    segments: List[TextSegment] = []
    segment_counter = 0

    for i, match in enumerate(section_matches):
        section_id = f"{match.group(1)}.{match.group(2)}.{match.group(3)}"

        # Get the full content for this section (until next section or end)
        start_pos = match.start()
        if i + 1 < len(section_matches):
            end_pos = section_matches[i + 1].start()
        else:
            end_pos = len(text)

        full_section_text = text[start_pos:end_pos].strip()

        # Remove section marker from content
        clean_content = re.sub(r"^§\s*\d+\.\d+\.\d+\s+", "", full_section_text).strip()

        # If section is too long, split it into smaller chunks
        if len(clean_content) > max_length:
            # Split by sentences within the section
            sentences = re.split(r"(?<=[.!?])\s+", clean_content)
            current_chunk = ""

            for sentence in sentences:
                if (
                    len(current_chunk) + len(sentence) + 1 > max_length
                    and current_chunk
                ):
                    # Create segment from current chunk
                    segment_id = f"segment_{segment_counter:04d}"
                    segment = TextSegment(
                        content=current_chunk.strip(),
                        segment_id=segment_id,
                        book=f"Book {match.group(1)}",
                        section=f"Chapter {match.group(2)}",
                        section_number=section_id,
                        word_count=len(current_chunk.split()),
                        char_count=len(current_chunk),
                    )
                    segments.append(segment)
                    segment_counter += 1

                    # Start new chunk with overlap
                    if overlap > 0 and len(current_chunk) > overlap:
                        overlap_text = current_chunk[-overlap:]
                        # Try to break at sentence boundary
                        sentence_break = overlap_text.rfind(". ")
                        if sentence_break > overlap // 2:
                            overlap_text = overlap_text[sentence_break + 2 :]
                        current_chunk = overlap_text + " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    current_chunk += " " + sentence if current_chunk else sentence

            # Add final chunk
            if current_chunk.strip():
                segment_id = f"segment_{segment_counter:04d}"
                segment = TextSegment(
                    content=current_chunk.strip(),
                    segment_id=segment_id,
                    book=f"Book {match.group(1)}",
                    section=f"Chapter {match.group(2)}",
                    section_number=section_id,
                    word_count=len(current_chunk.split()),
                    char_count=len(current_chunk),
                )
                segments.append(segment)
                segment_counter += 1
        else:
            # Section fits in one segment
            segment_id = f"segment_{segment_counter:04d}"
            segment = TextSegment(
                content=clean_content,
                segment_id=segment_id,
                book=f"Book {match.group(1)}",
                section=f"Chapter {match.group(2)}",
                section_number=section_id,
                word_count=len(clean_content.split()),
                char_count=len(clean_content),
            )
            segments.append(segment)
            segment_counter += 1

    return segments


def write_output_files(
    output_dir: Path,
    normalized_text: str,
    segments: List[TextSegment],
    document_structure: DocumentStructure,
    quality_report: QualityReport,
) -> None:
    """Write all output files in various formats."""

    # Write normalized text
    normalized_path = output_dir / "bibliotheca_normalized.txt"
    with normalized_path.open("w", encoding="utf-8") as fh:
        fh.write(normalized_text)
    print(f"Wrote normalised text to {normalized_path}")

    # Write segments as plain text
    segments_path = output_dir / "bibliotheca_segments.txt"
    with segments_path.open("w", encoding="utf-8") as fh:
        for segment in segments:
            fh.write(segment.content.strip() + "\n\n")
    print(f"Wrote {len(segments)} segments to {segments_path}")

    # Write segments as JSON with metadata
    segments_json_path = output_dir / "bibliotheca_segments.json"
    segments_data = []
    for segment in segments:
        segments_data.append(
            {
                "segment_id": segment.segment_id,
                "content": segment.content,
                "book": segment.book,
                "section": segment.section,
                "section_number": segment.section_number,
                "word_count": segment.word_count,
                "char_count": segment.char_count,
            }
        )

    with segments_json_path.open("w", encoding="utf-8") as fh:
        json.dump(segments_data, fh, indent=2, ensure_ascii=False)
    print(f"Wrote structured segments to {segments_json_path}")

    # Write document structure
    structure_path = output_dir / "bibliotheca_structure.json"
    structure_data = {
        "title": document_structure.title,
        "author": document_structure.author,
        "books": document_structure.books,
        "chapters": document_structure.sections,  # chapters are stored in sections field
        "metadata": document_structure.metadata,
    }

    with structure_path.open("w", encoding="utf-8") as fh:
        json.dump(structure_data, fh, indent=2, ensure_ascii=False)
    print(f"Wrote document structure to {structure_path}")

    # Write quality report
    quality_path = output_dir / "bibliotheca_quality_report.txt"
    with quality_path.open("w", encoding="utf-8") as fh:
        fh.write("BIBLIOTHECA PROCESSING REPORT\n")
        fh.write("=" * 50 + "\n\n")

        fh.write("PROCESSING NOTES:\n")
        if quality_report.processing_notes:
            for note in quality_report.processing_notes:
                fh.write(f"  - {note}\n")
        else:
            fh.write("  - No processing notes\n")
        fh.write("\n")

        fh.write("FILTERING NOTES:\n")
        if quality_report.filtering_notes:
            for note in quality_report.filtering_notes[:20]:  # Limit to first 20
                fh.write(f"  - {note}\n")
            if len(quality_report.filtering_notes) > 20:
                fh.write(f"  ... and {len(quality_report.filtering_notes) - 20} more\n")
        else:
            fh.write("  - No filtering notes\n")
        fh.write("\n")

        fh.write("SEGMENTATION NOTES:\n")
        if quality_report.segmentation_notes:
            for note in quality_report.segmentation_notes:
                fh.write(f"  - {note}\n")
        else:
            fh.write("  - No segmentation notes\n")
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
        fh.write(f"  - Total books: {len(document_structure.books)}\n")
        fh.write(f"  - Total chapters: {len(document_structure.sections)}\n")
        fh.write(
            f"  - Average segment length: {sum(s.char_count for s in segments) // len(segments) if segments else 0} characters\n"
        )

    print(f"Wrote quality report to {quality_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process the clean bibliotheca.txt file for RAG ingestion."
    )
    parser.add_argument(
        "--input_txt",
        type=Path,
        required=True,
        help="Path to the bibliotheca.txt file",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where the processed outputs will be saved",
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

    # Read the clean text file
    print(f"Reading text from {args.input_txt}…")
    raw_text = read_clean_text(args.input_txt)

    # Filter books and parse sections
    print("Filtering books and parsing sections…")
    filtered_text, quality_report, sections = filter_books_and_parse_sections(raw_text)

    # Identify document structure
    print("Identifying document structure…")
    document_structure = identify_document_structure(filtered_text, sections)

    # Create segments
    print("Creating structured segments…")
    segments = split_into_segments(
        filtered_text, args.max_length, args.overlap, document_structure, sections
    )

    # Filter out very short segments
    segments = [s for s in segments if s.char_count >= args.min_segment_length]

    # Add quality checks
    for i, segment in enumerate(segments):
        if segment.char_count < args.min_segment_length:
            quality_report.validation_warnings.append(
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
        args.output_dir, filtered_text, segments, document_structure, quality_report
    )

    print("\nProcessing complete!")
    print(
        f"Generated {len(segments)} segments from {len(document_structure.books)} books"
    )
    print(
        f"Books included: {[book['book_number'] for book in document_structure.books]}"
    )


if __name__ == "__main__":
    main()
