"""File utility functions"""

from pathlib import Path
from typing import Optional
from rag_eval.core.logging import get_logger

logger = get_logger("utils.file")


def read_prompt_file(version: str) -> str:
    """
    Read a prompt file from the prompts directory.
    
    Args:
        version: Prompt version (e.g., "v1", "v2")
        
    Returns:
        Prompt text content
    """
    prompt_path = Path(__file__).parent.parent / "prompts" / f"prompt_{version}.md"
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    logger.info(f"Reading prompt file: {prompt_path}")
    return prompt_path.read_text()

