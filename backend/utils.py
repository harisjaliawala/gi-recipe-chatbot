from __future__ import annotations

"""Utility helpers for the recipe chatbot backend.

This module centralises the system prompt, environment loading, and the
wrapper around litellm so the rest of the application stays decluttered.
"""

import os
from typing import Final, List, Dict

import braintrust as bt
import litellm  # type: ignore
from dotenv import load_dotenv

# Ensure the .env file is loaded as early as possible.
load_dotenv(override=False)

bt.init_logger(os.environ.get("BRAINTRUST_PARENT"))
litellm_wrapper = bt.wrap_litellm(litellm)

# --- Constants -------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT: str = (
    "You are a culinary and nutritional expert that is going to help craft recipes based on the requesters needs, dietary restrictions and preferences. \n"
    "Your mission is to create crystal clear, easy to follow recipes that anyone can cook \n\n"

)

try:
    bt_sys_prompt = bt.load_prompt(project=os.environ.get("BRAINTRUST_PARENT"), slug="recipe-bot-system-prompt-f85c")
    SYSTEM_PROMPT = bt_sys_prompt.prompt.messages[0].content
except Exception as e:
    print(f"Error loading system prompt from braintrust: {e}")
    SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT

# Fetch configuration *after* we loaded the .env file.
MODEL_NAME: Final[str] = os.environ.get("MODEL_NAME", "gpt-4o-mini")


# --- Agent wrapper ---------------------------------------------------------------

@bt.traced(notrace_io=True)
def get_agent_response(
    messages: List[Dict[str, str]],
    metadata: dict | None = None,
    ) -> List[Dict[str, str]]:  # noqa: WPS231
    """Call the underlying large-language model via *litellm*.

    Parameters
    ----------
    messages:
        The full conversation history. Each item is a dict with "role" and "content".

    Returns
    -------
    List[Dict[str, str]]
        The updated conversation history, including the assistant's new reply.
    """

    # litellm is model-agnostic; we only need to supply the model name and key.
    # The first message is assumed to be the system prompt if not explicitly provided
    # or if the history is empty. We'll ensure the system prompt is always first.
    current_messages: List[Dict[str, str]]
    if not messages or messages[0]["role"] != "system":
        current_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    else:
        current_messages = messages

    completion = litellm_wrapper.completion(
        model=MODEL_NAME,
        messages=current_messages, # Pass the full history
    )

    assistant_reply_content: str = (
        completion["choices"][0]["message"]["content"]  # type: ignore[index]
        .strip()
    )
    
    # Append assistant's response to the history
    updated_messages = current_messages + [{"role": "assistant", "content": assistant_reply_content}]
    
    bt.current_span().log(
        input=current_messages,
        output=[updated_messages[-1]],
        metadata=metadata,
    )
    return updated_messages 