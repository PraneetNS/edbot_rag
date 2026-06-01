import re
from typing import List

def split_into_sentences(text: str) -> List[str]:
    """
    Splits text by punctuation sentences (. or ?), avoiding decimal values
    and common short abbreviations.
    """
    if not text.strip():
        return []
    # Split on sentence boundaries, keeping the punctuation
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def custom_sentence_splitter(text: str) -> List[str]:
    """
    Custom sentence splitter for the SemanticSplitterNodeParser.
    Iterates through text, segments out code blocks (```), numbered lists (1.), 
    bullet items (-, *), and treats them as atomic blocks.
    """
    segments = []
    # Match code blocks explicitly
    code_block_pattern = r'(```[\s\S]*?```)'
    parts = re.split(code_block_pattern, text)

    for part in parts:
        part_strip = part.strip()
        if not part_strip:
            continue
        
        # If it is a code block, keep it fully intact
        if part_strip.startswith('```') and part_strip.endswith('```'):
            segments.append(part_strip)
        else:
            # Handle standard mentoring explanations and step guides
            lines = part.split('\n')
            current_paragraph = []
            
            for line in lines:
                stripped_line = line.strip()
                if not stripped_line:
                    if current_paragraph:
                        segments.extend(split_into_sentences('\n'.join(current_paragraph)))
                        current_paragraph = []
                    continue
                
                # Check for list step prefixes like "1. ", "Step 1:", "- ", "• ", "* "
                is_step_marker = re.match(r'^(\d+\.|-|•|\*|Step\s+\d+:)\s+', stripped_line, re.IGNORECASE)
                if is_step_marker:
                    # Flush accumulated normal paragraph first
                    if current_paragraph:
                        segments.extend(split_into_sentences('\n'.join(current_paragraph)))
                        current_paragraph = []
                    # Treat the list/step item as an individual cohesive sentence/idea
                    segments.append(line)
                else:
                    current_paragraph.append(line)
            
            if current_paragraph:
                segments.extend(split_into_sentences('\n'.join(current_paragraph)))

    return [s.strip() for s in segments if s.strip()]
