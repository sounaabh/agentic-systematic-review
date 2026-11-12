"""LLM factory + a robust helper for structured (Pydantic) outputs.

Switch provider/model in configs/config.yaml. Small local models sometimes return
slightly broken JSON, so `ask_structured` repairs and retries before failing.
"""
import json
import re
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

from src.config import CONFIG

T = TypeVar("T", bound=BaseModel)


def get_llm(model: str | None = None, json_mode: bool = False):
    cfg = CONFIG["llm"]
    provider = cfg["provider"]
    model = model or cfg["model"]
    temp = cfg.get("temperature", 0)

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        kwargs = dict(model=model, temperature=temp, num_ctx=cfg.get("num_ctx", 8192), seed=42)
        if json_mode:
            kwargs["format"] = "json"
        return ChatOllama(**kwargs)
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=temp)
    if provider == "nvidia":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        return ChatNVIDIA(model=model, temperature=temp)
    raise ValueError(f"Unknown LLM provider: {provider}")


def model_name(model: str | None = None) -> str:
    return f"{CONFIG['llm']['provider']}:{model or CONFIG['llm']['model']}"


def _extract_json(text: str) -> str:
    """Pull the first {...} block out of a model reply (handles ```json fences)."""
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output")
    return text[start : end + 1]


def _schema_hint(schema: Type[BaseModel]) -> str:
    """Compact description of the fields (small models follow this better than full JSON Schema)."""
    props = schema.model_json_schema().get("properties", {})
    lines = []
    for name, p in props.items():
        typ = p.get("type") or ("enum " + str(p["enum"]) if "enum" in p else "object")
        if "anyOf" in p:
            typ = " | ".join(x.get("type", "object") for x in p["anyOf"])
        desc = p.get("description", "")
        lines.append(f'  "{name}": {typ}{"  // " + desc if desc else ""}')
    return "{\n" + ",\n".join(lines) + "\n}"


def ask_structured(llm, schema: Type[T], system: str, user: str, retries: int = 2) -> T:
    """Ask the LLM for JSON matching `schema`. On failure, retries with the error message."""
    messages = [
        ("system", f"{system}\n\nReply ONLY with one JSON object with these fields:\n{_schema_hint(schema)}"),
        ("user", user),
    ]
    last_err = None
    for _ in range(retries + 1):
        reply = llm.invoke(messages).content
        try:
            return schema.model_validate_json(_extract_json(reply))
        except (ValueError, ValidationError) as e:
            last_err = e
            messages += [("assistant", reply),
                         ("user", f"Invalid JSON for the required fields. Error: {str(e)[:500]}. "
                                  "Reply again with JSON only.")]
    raise RuntimeError(f"LLM failed to return valid {schema.__name__}: {last_err}")


def ask_text(llm, system: str, user: str) -> str:
    return llm.invoke([("system", system), ("user", user)]).content.strip()
