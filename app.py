"""
# pip install gradio requests
# python app.py

A single-file, zero-config AI chatbot using Gradio + a free open-source model.

- Tries a local Ollama instance first: http://localhost:11434
- Falls back to the public Hugging Face Inference API for Zephyr 7B Beta
- Uses rolling chat memory so the conversation stays contextual
- Gracefully handles timeout/network failures without crashing
"""

import json
import os
import re
import sys
import time
from typing import List, Tuple, Optional

import requests
import gradio as gr

SYSTEM_PROMPT = (
    "You are a helpful, intelligent, friendly assistant. "
    "Answer clearly, concisely, and thoughtfully. "
    "Use the conversation context to maintain continuity, be accurate, and avoid making up facts. "
    "If you are unsure, say so honestly."
)

MAX_HISTORY = 8
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
HF_MODEL_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
HF_HEADERS = {"Content-Type": "application/json"}
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    HF_HEADERS["Authorization"] = f"Bearer {HF_TOKEN}"


def clean_response_text(text: str) -> str:
    """Normalize raw model output and strip common prompt duplication."""
    if not text:
        return ""
    cleaned = text.strip()
    cleaned = cleaned.replace("\r", "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def build_messages(history: List[Tuple[str, str]], user_message: str) -> List[dict]:
    """Convert the chat state into a list of messages for the model."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for past_user, past_assistant in history:
        if past_user:
            messages.append({"role": "user", "content": past_user})
        if past_assistant:
            messages.append({"role": "assistant", "content": past_assistant})
    messages.append({"role": "user", "content": user_message})
    return messages


def ollama_available() -> bool:
    """Check whether a local Ollama server is running."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return response.status_code == 200
    except requests.RequestException:
        return False


def build_ollama_prompt(messages: List[dict]) -> str:
    """Convert chat messages into a plain-text prompt for Ollama."""
    prompt_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            prompt_parts.append(f"[System]\n{content}\n")
        elif role == "user":
            prompt_parts.append(f"[User]\n{content}\n")
        elif role == "assistant":
            prompt_parts.append(f"[Assistant]\n{content}\n")
    prompt_parts.append("[Assistant]\n")
    return "\n".join(prompt_parts)


def generate_with_ollama(history: List[Tuple[str, str]], user_message: str) -> str:
    """Generate a response via the local Ollama server if available."""
    messages = build_messages(history, user_message)
    prompt = build_ollama_prompt(messages)
    payload = {
        "model": DEFAULT_OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 512,
        },
    }
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=90,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Ollama request failed ({response.status_code}): {response.text[:200]}")
        data = response.json()
        return clean_response_text(data.get("response", "").strip())
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama timed out while generating a response. Please try again.")
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Ollama connection error: {exc}")
    except Exception as exc:
        raise RuntimeError(f"Unexpected Ollama error: {exc}")


def generate_with_hf(history: List[Tuple[str, str]], user_message: str) -> str:
    """Generate a response via the public Hugging Face Inference API."""
    messages = build_messages(history, user_message)
    prompt = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages
    ) + "\nAssistant:"

    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 512, "temperature": 0.7, "top_p": 0.9, "do_sample": True},
        "options": {"wait_for_model": True},
    }
    try:
        response = requests.post(HF_MODEL_URL, headers=HF_HEADERS, json=payload, timeout=90)
        if response.status_code >= 400:
            raise RuntimeError(f"Hugging Face request failed ({response.status_code}): {response.text[:200]}")

        result = response.json()
        if isinstance(result, list):
            if not result:
                raise RuntimeError("Hugging Face returned an empty response.")
            text = result[0].get("generated_text", "") if isinstance(result[0], dict) else str(result[0])
        elif isinstance(result, dict):
            text = result.get("generated_text", "")
            if not text and "error" in result:
                raise RuntimeError(result["error"])
            if not text and "choices" in result and result["choices"]:
                text = result["choices"][0].get("text", "")
        else:
            text = str(result)

        if not text:
            raise RuntimeError("Hugging Face returned no usable text.")

        if text.startswith(prompt):
            text = text[len(prompt):]
        return clean_response_text(text)
    except requests.exceptions.Timeout:
        raise RuntimeError("Hugging Face timed out while generating a response. Please try again.")
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Hugging Face connection error: {exc}")
    except Exception as exc:
        raise RuntimeError(f"Unexpected Hugging Face error: {exc}")


def generate_chat_response(history: List[Tuple[str, str]], user_message: str) -> str:
    """Try local Ollama first, then fallback to Hugging Face public endpoint."""
    if not user_message or not user_message.strip():
        return "Please type a message before sending."

    try:
        if ollama_available():
            return generate_with_ollama(history, user_message)
        raise RuntimeError("Local Ollama not detected")
    except Exception as ollama_error:
        try:
            return generate_with_hf(history, user_message)
        except Exception as hf_error:
            return (
                "I couldn't reach the local Ollama server or the public Hugging Face endpoint. "
                "If you have Ollama installed, start it with: `ollama serve`. "
                "Otherwise, try again in a moment as the public model may be temporarily unavailable. "
                f"Details: {hf_error}"
            )


def response_fn(message: str, history: Optional[List[Tuple[str, str]]] = None):
    """Gradio-compatible function that updates chat history and returns the full display list."""
    if history is None:
        history = []

    if not message or not message.strip():
        return history

    trimmed_history = history[-MAX_HISTORY:]
    assistant_reply = generate_chat_response(trimmed_history, message)
    updated_history = trimmed_history + [(message, assistant_reply)]
    return updated_history


with gr.Blocks(theme=gr.themes.Soft(), title="Open-Source AI Chatbot") as demo:
    gr.Markdown(
        """
        # Open-Source AI Chatbot

        This app uses a local Ollama instance when available and falls back to a free public Hugging Face model.
        """
    )

    chatbot = gr.Chatbot(
        type="tuples",
        height=520,
        show_copy_button=True,
        bubble_full_width=False,
        avatar_images=(None, None),
    )

    msg = gr.Textbox(
        placeholder="Type your message here...",
        show_label=False,
        scale=8,
    )

    with gr.Row():
        send_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.ClearButton([msg, chatbot])

    def submit_message(message, history):
        return response_fn(message, history)

    msg.submit(submit_message, [msg, chatbot], chatbot)
    send_btn.click(submit_message, [msg, chatbot], chatbot)

    demo.load(
        fn=lambda: (
            "Hi! I’m ready to help. If Ollama is installed locally, I’ll use it automatically. "
            "Otherwise I’ll fall back to a public open-source model."
        ),
        inputs=[],
        outputs=[chatbot],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=False)
