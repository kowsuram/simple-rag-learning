"""Section-aware document chunker with sliding window overlap and rich logging."""

import re
from typing import List, Dict, Any
from src.config import config
from src.logger import log_chunk_details


class MedicationChunker:
    """Chunks structured healthcare medication monographs while preserving clinical context."""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or config.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunk_overlap

    def split_text_with_overlap(self, text: str, max_chars: int, overlap_chars: int) -> List[Dict[str, Any]]:
        """Split a block of text using sentence-aware sliding window with overlap."""
        text = text.strip()
        if len(text) <= max_chars:
            return [{"start": 0, "end": len(text), "text": text, "overlap": 0}]

        # Try to break on sentence terminators (. ! ? \n)
        chunks = []
        start = 0
        total_len = len(text)

        while start < total_len:
            end = min(start + max_chars, total_len)

            if end < total_len:
                # Look backwards for a sentence boundary or comma/space within the last 80 chars
                boundary = -1
                for char in [". ", "? ", "! ", "\n", "; ", ", "]:
                    pos = text.rfind(char, start + int(max_chars * 0.6), end)
                    if pos != -1:
                        boundary = max(boundary, pos + len(char))
                
                if boundary != -1:
                    end = boundary

            chunk_str = text[start:end].strip()
            overlap_amount = self.chunk_overlap if start > 0 else 0
            
            chunks.append({
                "start": start,
                "end": end,
                "text": chunk_str,
                "overlap": overlap_amount
            })

            if end >= total_len:
                break

            # Slide window forward, subtracting overlap
            start = max(start + 1, end - overlap_chars)

        return chunks

    def chunk_monograph(self, monograph: Dict[str, Any], verbose: bool = True) -> List[Dict[str, Any]]:
        """Process a medication document into contextualized chunks."""
        med_name = monograph.get("medication_name", "Unknown Medication")
        brand_names = ", ".join(monograph.get("brand_names", []))
        med_class = monograph.get("therapeutic_class", "")
        
        # Clinical sections to ingest
        clinical_sections = [
            ("therapeutic_class", "Therapeutic Class & Classification"),
            ("indications", "Indications & Approved Uses"),
            ("dosage_and_administration", "Dosage & Administration"),
            ("contraindications", "Contraindications & High Risk"),
            ("drug_interactions", "Drug-Drug & Food Interactions"),
            ("side_effects", "Adverse Effects & Side Effects"),
            ("critical_warnings", "Critical & Black Box Warnings")
        ]

        all_chunks = []
        global_chunk_idx = 0

        for section_key, section_display in clinical_sections:
            section_content = monograph.get(section_key, "")
            if not section_content:
                continue

            # Split long sections if needed
            sub_chunks = self.split_text_with_overlap(
                section_content,
                max_chars=self.chunk_size,
                overlap_chars=self.chunk_overlap
            )

            for sc in sub_chunks:
                # Add contextual header to the chunk text to ground vector embeddings
                contextualized_text = (
                    f"Medication: {med_name} (Brands: {brand_names})\n"
                    f"Class: {med_class}\n"
                    f"Section: {section_display}\n"
                    f"Content: {sc['text']}"
                )

                chunk_data = {
                    "medication_name": med_name,
                    "section": section_display,
                    "section_key": section_key,
                    "chunk_index": global_chunk_idx,
                    "raw_text": sc["text"],
                    "content": contextualized_text,
                    "char_start": sc["start"],
                    "char_end": sc["end"],
                    "overlap_chars": sc["overlap"],
                    "metadata": {
                        "medication_name": med_name,
                        "brand_names": monograph.get("brand_names", []),
                        "therapeutic_class": med_class,
                        "section": section_display,
                        "char_length": len(sc["text"])
                    }
                }
                all_chunks.append(chunk_data)

                if verbose:
                    log_chunk_details(
                        medication=med_name,
                        section=section_display,
                        chunk_index=global_chunk_idx,
                        total_chunks=len(sub_chunks),
                        start_char=sc["start"],
                        end_char=sc["end"],
                        text=sc["text"],
                        overlap_chars=sc["overlap"]
                    )

                global_chunk_idx += 1

        return all_chunks


# Global chunker instance
chunker = MedicationChunker()
