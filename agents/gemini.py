import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types


MODEL = "gemma-3-12b-it"

MAX_FILE_BYTES = 400_000
MAX_TOOL_CALLS_PER_TURN = 8

MAX_TOOL_RESULT_CHARS = 20_000
LIST_FILES_MAX_ITEMS = 250

DEFAULT_IGNORED_DIRS = {
    ".git",
    ".idea",
    ".pytest_cache",
    ".mypy_cache",
    "__pycache__",
    "node_modules",
    "venv",
    ".venv",
    "dist",
    "build",
}

BLUE = "\033[94m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RESET = "\033[0m"


@dataclass
class ToolDefinition:
    name: str
    description: str
    input_schema: Dict[str, Any]
    func: Callable[[Dict[str, Any]], str]


def safe_resolve_rel_path(base_dir: Path, rel_path: str) -> Path:
    base_dir = base_dir.resolve()
    candidate = (base_dir / rel_path).resolve()
    if candidate == base_dir or base_dir in candidate.parents:
        return candidate
    raise ValueError("path escapes working directory")


def tool_read_file(args: Dict[str, Any], base_dir: Path) -> str:
    path = args.get("path", "")
    if not isinstance(path, str) or not path:
        raise ValueError("missing path")

    full_path = safe_resolve_rel_path(base_dir, path)
    if full_path.is_dir():
        raise ValueError("path is a directory")

    data = full_path.read_bytes()
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("file too large")

    return data.decode("utf-8", errors="replace")


def tool_list_files(args: Dict[str, Any], base_dir: Path) -> str:
    rel = args.get("path", "")
    if rel is None:
        rel = ""
    if not isinstance(rel, str):
        raise ValueError("path must be a string")

    start = safe_resolve_rel_path(base_dir, rel) if rel else base_dir.resolve()
    if not start.exists():
        raise ValueError("path not found")

    if start.is_file():
        return json.dumps([str(Path(rel))])

    items: List[str] = []
    try:
        with os.scandir(start) as it:
            for entry in it:
                name = entry.name

                if entry.is_dir(follow_symlinks=False) and name in DEFAULT_IGNORED_DIRS:
                    continue

                if entry.is_dir(follow_symlinks=False):
                    items.append(name + "/")
                else:
                    items.append(name)

                if len(items) >= LIST_FILES_MAX_ITEMS:
                    break
    except PermissionError:
        raise ValueError("permission denied")

    items = sorted(items)
    return json.dumps(items, ensure_ascii=False)


def tool_edit_file(args: Dict[str, Any], base_dir: Path) -> str:
    path = args.get("path", "")
    old_str = args.get("old_str", None)
    new_str = args.get("new_str", None)

    if not isinstance(path, str) or not path:
        raise ValueError("missing path")
    if not isinstance(old_str, str) or not isinstance(new_str, str):
        raise ValueError("old_str and new_str must be strings")
    if old_str == new_str:
        raise ValueError("old_str and new_str must be different")

    full_path = safe_resolve_rel_path(base_dir, path)
    full_path.parent.mkdir(parents=True, exist_ok=True)

    if not full_path.exists():
        if old_str != "":
            raise ValueError("file does not exist, old_str must be empty to create")
        full_path.write_text(new_str, encoding="utf-8")
        return f"created {path}"

    content = full_path.read_text(encoding="utf-8", errors="replace")

    if old_str == "":
        raise ValueError("old_str must not be empty when editing an existing file")

    count = content.count(old_str)
    if count == 0:
        raise ValueError("old_str not found")
    if count > 1:
        raise ValueError("old_str has more than one match")

    updated = content.replace(old_str, new_str, 1)
    full_path.write_text(updated, encoding="utf-8")
    return "OK"


def _extract_fenced_payload(text: str) -> Optional[str]:
    m = re.search(r"```(?:json)?\s*", text, flags=re.IGNORECASE)
    if not m:
        return None

    after = text[m.end() :]
    close_idx = after.find("```")
    if close_idx == -1:
        return after.strip()

    return after[:close_idx].strip()


def _extract_first_json_object(s: str) -> Optional[str]:
    start = s.find("{")
    if start == -1:
        return None

    in_str = False
    esc = False
    depth = 0

    for i in range(start, len(s)):
        ch = s[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            depth += 1
            continue

        if ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1].strip()
            continue

    return s[start:].strip()


def _repair_unbalanced_json(s: str) -> str:
    start = s.find("{")
    if start == -1:
        return s.strip()

    s = s[start:]

    in_str = False
    esc = False
    stack: List[str] = []
    out: List[str] = []

    for ch in s:
        out.append(ch)

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            stack.append("{")
            continue

        if ch == "[":
            stack.append("[")
            continue

        if ch == "}" and stack and stack[-1] == "{":
            stack.pop()
            continue

        if ch == "]" and stack and stack[-1] == "[":
            stack.pop()
            continue

    for opener in reversed(stack):
        out.append("}" if opener == "{" else "]")

    return "".join(out).strip()


def extract_json_object(text: str) -> Optional[str]:
    payload = _extract_fenced_payload(text)
    if payload is None:
        payload = text

    blob = _extract_first_json_object(payload)
    if blob:
        return blob

    return None


def _json_loads_lenient(blob: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(blob)
        if isinstance(obj, dict):
            return obj
        return None
    except Exception:
        repaired = _repair_unbalanced_json(blob)
        try:
            obj = json.loads(repaired)
            if isinstance(obj, dict):
                return obj
            return None
        except Exception:
            return None


def parse_tool_calls(model_text: str) -> Optional[List[Dict[str, Any]]]:
    blob = extract_json_object(model_text)
    if not blob:
        return None

    obj = _json_loads_lenient(blob)
    if not obj:
        return None

    calls = obj.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return None

    out: List[Dict[str, Any]] = []
    for c in calls:
        if not isinstance(c, dict):
            continue
        name = c.get("name")
        args = c.get("args")
        if isinstance(name, str) and isinstance(args, dict):
            out.append({"name": name, "args": args})

    return out or None


def _extract_retry_seconds(err_text: str) -> Optional[float]:
    m = re.search(r"Please retry in ([0-9.]+)s", err_text)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None

    m = re.search(r"retryDelay['\"]:\s*['\"]([0-9]+)s['\"]", err_text)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None

    return None


def _truncate_tool_result(text: str) -> str:
    if len(text) <= MAX_TOOL_RESULT_CHARS:
        return text
    head = text[:MAX_TOOL_RESULT_CHARS]
    return head + "\n\n[TRUNCATED TOOL RESULT]"


SYSTEM_INSTRUCTION = """
You are a code editing assistant running inside a local folder.

You have access to these tools:

1) read_file
Read a text file at a relative path.

2) list_files
List files and directories at an optional relative path.
Returns a JSON list of strings. Directories end with "/".

3) edit_file
Edit a text file by replacing old_str with new_str.
Rules:
- old_str and new_str must be different
- if the file does not exist, you may create it only when old_str is an empty string
- if editing an existing file, old_str must match exactly once

How to call tools:
If you want to use tools, respond with ONLY valid JSON and nothing else, in this exact shape:
{"tool_calls":[{"name":"read_file","args":{"path":"main.py"}}]}

Do not wrap JSON in markdown or code fences.

You may request multiple tool calls at once:
{"tool_calls":[{"name":"list_files","args":{}},{"name":"read_file","args":{"path":"main.py"}}]}

When you receive tool results, they will come as a user message containing JSON:
{"tool_results":[{"name":"read_file","ok":true,"result":"..."}]}

After tool results, either call more tools (using the same JSON format) or answer normally.
""".strip()


class Agent:
    def __init__(self, client: genai.Client, base_dir: Path):
        self.client = client
        self.base_dir = base_dir.resolve()

        self.conversation: List[types.Content] = [
            types.Content(role="user", parts=[types.Part(text=SYSTEM_INSTRUCTION)])
        ]

        self.tools: Dict[str, ToolDefinition] = {
            "read_file": ToolDefinition(
                name="read_file",
                description="Read a text file at a relative path.",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                func=lambda args: tool_read_file(args, self.base_dir),
            ),
            "list_files": ToolDefinition(
                name="list_files",
                description="List files and directories at an optional relative path.",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
                func=lambda args: tool_list_files(args, self.base_dir),
            ),
            "edit_file": ToolDefinition(
                name="edit_file",
                description="Replace old_str with new_str in a text file, or create when old_str is empty.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old_str": {"type": "string"},
                        "new_str": {"type": "string"},
                    },
                    "required": ["path", "old_str", "new_str"],
                },
                func=lambda args: tool_edit_file(args, self.base_dir),
            ),
        }

    def _trim_conversation(self, keep_last: int = 18) -> None:
        if len(self.conversation) > keep_last + 1:
            first = self.conversation[0]
            self.conversation = [first] + self.conversation[-keep_last:]

    def _generate_with_retry(self) -> types.GenerateContentResponse:
        attempts = 0
        while True:
            try:
                return self.client.models.generate_content(
                    model=MODEL,
                    contents=self.conversation,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        max_output_tokens=1024,
                    ),
                )
            except Exception as e:
                attempts += 1
                text = str(e)

                is_429 = (" 429 " in text) or ("RESOURCE_EXHAUSTED" in text) or text.startswith("429")
                if (not is_429) or attempts >= 6:
                    raise

                retry_s = _extract_retry_seconds(text)
                if retry_s is None:
                    retry_s = min(2.0 ** attempts, 20.0)

                time.sleep(retry_s)

    def run(self) -> None:
        print("Chat with Gemma (ctrl c to quit)")
        read_user_input = True

        while True:
            if read_user_input:
                try:
                    user_input = input(f"{BLUE}You{RESET}: ")
                except EOFError:
                    break
                self.conversation.append(types.Content(role="user", parts=[types.Part(text=user_input)]))

            self._trim_conversation()

            response = self._generate_with_retry()

            model_text = response.text or ""
            self.conversation.append(types.Content(role="model", parts=[types.Part(text=model_text)]))

            tool_calls = parse_tool_calls(model_text)
            if not tool_calls:
                print(f"{YELLOW}Gemma{RESET}: {model_text}")
                read_user_input = True
                continue

            tool_calls = tool_calls[:MAX_TOOL_CALLS_PER_TURN]
            tool_results: List[Dict[str, Any]] = []

            for call in tool_calls:
                name = call["name"]
                args = call["args"]

                print(f"{GREEN}tool{RESET}: {name}({json.dumps(args, ensure_ascii=False)})")

                tool = self.tools.get(name)
                if not tool:
                    tool_results.append({"name": name, "ok": False, "result": "tool not found"})
                    continue

                try:
                    result = tool.func(args)
                    result = _truncate_tool_result(result)
                    tool_results.append({"name": name, "ok": True, "result": result})
                except Exception as e:
                    tool_results.append({"name": name, "ok": False, "result": str(e)})

            self.conversation.append(
                types.Content(
                    role="user",
                    parts=[types.Part(text=json.dumps({"tool_results": tool_results}, ensure_ascii=False))],
                )
            )
            read_user_input = False


def main() -> None:
    repo_root = (Path(__file__).resolve().parent / "..").resolve()
    dotenv_path = repo_root / ".env"

    load_dotenv(dotenv_path)

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print(f"Missing GEMINI_API_KEY. Expected it in {dotenv_path}")
        sys.exit(1)

    if len(api_key) < 20:
        print("GEMINI_API_KEY looks too short, check your .env file, it may be truncated")
        sys.exit(1)

    print(f"Using GEMINI_API_KEY: {api_key[:4]}...{api_key[-4:]} (len {len(api_key)})")

    os.environ.pop("GOOGLE_API_KEY", None)

    client = genai.Client(api_key=api_key)
    agent = Agent(client=client, base_dir=repo_root)
    agent.run()


if __name__ == "__main__":
    main()

