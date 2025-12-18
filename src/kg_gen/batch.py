"""Batch API support for KGGen.

This module provides functionality for building OpenAI Batch API requests
for knowledge graph extraction. The batch API offers 50% cost savings
with a 24-hour completion window.

Usage:
    >>> from kg_gen import KGGen, BatchRequestConfig
    >>> kg = KGGen(model="openai/gpt-4o")
    >>> config = BatchRequestConfig(model="gpt-4o")
    >>> requests = kg.build_entity_extraction_requests(
    ...     texts=[{"content_id": "c1", "text": "Link is the hero..."}],
    ...     config=config,
    ... )
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from kg_gen.utils.chunk_text import chunk_text


@dataclass
class BatchRequestConfig:
    """Configuration for batch API requests.

    Attributes:
        model: OpenAI model to use (without 'openai/' prefix)
        entity_temperature: Temperature for entity extraction (deterministic)
        relation_temperature: Temperature for relation extraction (deterministic)
        dedup_temperature: Temperature for deduplication (slightly variable)
        max_tokens: Maximum tokens for responses
    """

    model: str = "gpt-4o"
    entity_temperature: float = 0.0
    relation_temperature: float = 0.0
    dedup_temperature: float = 0.1
    max_tokens: int = 4096


# System prompts matching DSPy signatures
ENTITY_EXTRACTION_SYSTEM_PROMPT = """You are a knowledge graph entity extractor. Extract key entities from the source text. Extracted entities are subjects or objects.

This is for an extraction task, please be THOROUGH and accurate to the reference text.

Return your response as JSON in this exact format:
{"entities": ["entity1", "entity2", ...]}

Only return the JSON, no other text."""

ENTITY_EXTRACTION_CONVERSATION_SYSTEM_PROMPT = """You are a knowledge graph entity extractor. Extract key entities from the conversation. Extracted entities are subjects or objects. Consider both explicit entities and participants in the conversation.

This is for an extraction task, please be THOROUGH and accurate.

Return your response as JSON in this exact format:
{"entities": ["entity1", "entity2", ...]}

Only return the JSON, no other text."""

RELATION_EXTRACTION_SYSTEM_PROMPT = """You are a knowledge graph relation extractor. Extract subject-predicate-object triples from the source text.

Subject and object must be from the entities list provided. Entities were previously extracted from the same source text.

This is for an extraction task, please be thorough, accurate, and faithful to the reference text.

Return your response as JSON in this exact format:
{"relations": [{"subject": "entity1", "predicate": "relation", "object": "entity2"}, ...]}

Only return the JSON, no other text."""

RELATION_EXTRACTION_CONVERSATION_SYSTEM_PROMPT = """You are a knowledge graph relation extractor. Extract subject-predicate-object triples from the conversation, including:
1. Relations between concepts discussed
2. Relations between speakers and concepts (e.g. user asks about X)
3. Relations between speakers (e.g. assistant responds to user)

Subject and object must be from the entities list provided. Entities were previously extracted from the same source text.

This is for an extraction task, please be thorough, accurate, and faithful to the reference text.

Return your response as JSON in this exact format:
{"relations": [{"subject": "entity1", "predicate": "relation", "object": "entity2"}, ...]}

Only return the JSON, no other text."""

DEDUPLICATION_SYSTEM_PROMPT = """You are a deduplication assistant. Given an item and a list of candidate duplicates, identify which candidates are duplicates of the item.

Two items are duplicates if they refer to the same real-world entity, even if spelled differently or using synonyms.

Return your response as JSON in this exact format:
{"duplicates": ["candidate1", "candidate2"], "alias": "preferred_name"}

- "duplicates": List of candidates that are duplicates of the item (can be empty)
- "alias": The preferred canonical name for all duplicates (use the most common/official form)

Only return the JSON, no other text."""


def _create_custom_id(
    request_type: str,
    content_id: str | None = None,
    chunk_index: int = 0,
    item: str | None = None,
    item_type: str = "entity",
) -> str:
    """Create a custom_id for batch requests.

    Args:
        request_type: 'entity', 'relation', or 'dedup'
        content_id: Content UUID (for entity/relation)
        chunk_index: Chunk index (for entity/relation)
        item: Item being deduplicated (for dedup)
        item_type: 'entity' or 'edge' (for dedup)

    Returns:
        Custom ID string
    """
    if request_type == "entity":
        return f"kg-entity-{content_id}-{chunk_index}"
    elif request_type == "relation":
        return f"kg-relation-{content_id}-{chunk_index}"
    elif request_type == "dedup":
        item_hash = hashlib.md5(item.encode()).hexdigest()[:16]
        return f"kg-dedup-{item_type}-{item_hash}"
    else:
        raise ValueError(f"Unknown request type: {request_type}")


def build_entity_extraction_requests(
    texts: list[dict[str, Any]],
    config: BatchRequestConfig,
    chunk_size: int | None = None,
) -> list[dict[str, Any]]:
    """Build entity extraction requests for the OpenAI Batch API.

    Args:
        texts: List of content dictionaries with:
            - content_id: UUID of the content
            - text: Text to extract entities from
            - is_conversation: Whether text is conversational (optional)
        config: Batch request configuration
        chunk_size: Optional chunk size for splitting large texts

    Returns:
        List of request dictionaries ready for JSONL serialization:
        [{"custom_id": "...", "body": {...}}, ...]
    """
    requests = []

    for text_item in texts:
        content_id = text_item["content_id"]
        text = text_item["text"]
        is_conversation = text_item.get("is_conversation", False)

        # Select appropriate system prompt
        system_prompt = (
            ENTITY_EXTRACTION_CONVERSATION_SYSTEM_PROMPT if is_conversation else ENTITY_EXTRACTION_SYSTEM_PROMPT
        )

        # Chunk text if needed
        if chunk_size:
            chunks = chunk_text(text, chunk_size)
        else:
            chunks = [text]

        for chunk_idx, chunk in enumerate(chunks):
            custom_id = _create_custom_id("entity", content_id, chunk_idx)

            request = {
                "custom_id": custom_id,
                "body": {
                    "model": config.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Source text:\n{chunk}"},
                    ],
                    "temperature": config.entity_temperature,
                    "max_tokens": config.max_tokens,
                    "response_format": {"type": "json_object"},
                },
            }
            requests.append(request)

    return requests


def build_relation_extraction_requests(
    texts: list[dict[str, Any]],
    entities_by_content: dict[str, list[str]],
    config: BatchRequestConfig,
    chunk_size: int | None = None,
    context: str = "",
) -> list[dict[str, Any]]:
    """Build relation extraction requests for the OpenAI Batch API.

    Args:
        texts: List of content dictionaries with:
            - content_id: UUID of the content
            - text: Text to extract relations from
            - is_conversation: Whether text is conversational (optional)
        entities_by_content: Map of content_id to entities (from Phase 1)
        config: Batch request configuration
        chunk_size: Optional chunk size for splitting large texts
        context: Additional extraction context

    Returns:
        List of request dictionaries ready for JSONL serialization
    """
    requests = []

    for text_item in texts:
        content_id = text_item["content_id"]
        text = text_item["text"]
        is_conversation = text_item.get("is_conversation", False)

        # Get entities for this content
        entities = entities_by_content.get(content_id, [])
        if not entities:
            continue  # Skip if no entities

        # Select appropriate system prompt and add context
        base_prompt = (
            RELATION_EXTRACTION_CONVERSATION_SYSTEM_PROMPT if is_conversation else RELATION_EXTRACTION_SYSTEM_PROMPT
        )
        system_prompt = f"{base_prompt}\n\n{context}" if context else base_prompt

        # Chunk text if needed
        if chunk_size:
            chunks = chunk_text(text, chunk_size)
        else:
            chunks = [text]

        entities_str = json.dumps(entities)

        for chunk_idx, chunk in enumerate(chunks):
            custom_id = _create_custom_id("relation", content_id, chunk_idx)

            request = {
                "custom_id": custom_id,
                "body": {
                    "model": config.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": f"Entities: {entities_str}\n\nSource text:\n{chunk}",
                        },
                    ],
                    "temperature": config.relation_temperature,
                    "max_tokens": config.max_tokens,
                    "response_format": {"type": "json_object"},
                },
            }
            requests.append(request)

    return requests


def build_deduplication_requests(
    items: list[str],
    candidates_per_item: dict[str, list[str]],
    item_type: str,
    config: BatchRequestConfig,
) -> list[dict[str, Any]]:
    """Build deduplication requests for the OpenAI Batch API.

    Args:
        items: List of items to deduplicate
        candidates_per_item: Map of item to potential duplicate candidates
        item_type: Type of items ('entity' or 'edge')
        config: Batch request configuration

    Returns:
        List of request dictionaries ready for JSONL serialization
    """
    requests = []

    for item in items:
        candidates = candidates_per_item.get(item, [])
        if not candidates:
            continue  # Skip if no candidates

        custom_id = _create_custom_id("dedup", item=item, item_type=item_type)

        user_content = f"Item: {item}\n\nCandidate duplicates:\n"
        user_content += "\n".join(f"- {c}" for c in candidates)

        request = {
            "custom_id": custom_id,
            "body": {
                "model": config.model,
                "messages": [
                    {"role": "system", "content": DEDUPLICATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "temperature": config.dedup_temperature,
                "max_tokens": config.max_tokens,
                "response_format": {"type": "json_object"},
            },
        }
        requests.append(request)

    return requests
