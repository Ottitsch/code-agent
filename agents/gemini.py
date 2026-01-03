import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from google import genai
from google.genai import types


MODEL = "gemma-3-12b-it"
MAX_FILE_BYTES = 400_000
MAX_TOOL_CALLS_PER_TURN = 8

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


def _safe_resolve_rel_path(base_dir: Path, rel_path: str) -> Path:
    base_dir = base_dir.resolve()
    candidate = (base_dir / rel_path).resolve()
    if base_dir not in candidate.parents and candidate != base_dir:
        raise ValueError("path escapes working directory")
    return candidate


def tool_read_file(args: Dict[str, Any], base_dir: Path) -> str:
    path = args.get("path", "")
    if not isinstance(path, str) or not path:
        raise ValueError("missing path")

    full_path = _safe_resolve_rel_path(base_dir, path)
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

    start = _safe_resolve_rel_path(base_dir, rel) if rel else base_dir.resolve()
    if not start.exists():
        raise ValueError("path not found")
    if start.is_file():
        return json.dumps([str(Path(rel))])

    items: List[str] = []
    for root, dirs, files in os.walk(start):
        root_path = Path(root)
        rel_root = root_path.relative_to(start)
        if str(rel_root) != ".":
            items.append(str(rel_root) + "/")
        for d in dirs:
            p = (rel_root / d)
            items.append(str(p) + "/")
        for f in files:
            p = (rel_root / f)
            items.append(str(p))
    items = sorted(set(items))
    return json.dumps(items)


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

    full_path = _safe_resolve_rel_path(base_dir, path)
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


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return t


def _extract_json_object(text: str) -> Optional[str]:
    t = _strip_code_fences(text)
    if t.startswith("{") and t.endswith("}"):
        return t
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start : end + 1].strip()
    return None


def parse_tool_calls(model_text: str) -> Optional[List[Dict[str, Any]]]:
    blob = _extract_json_object(model_text)
    if not blob:
        return None
    try:
        obj = json.loads(blob)
    except Exception:
        return None

    if not isinstance(obj, dict):
        return None
    calls = obj.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return None

    normalized: List[Dict[str, Any]] = []
    for c in calls:
        if not isinstance(c, dict):
            continue
        name = c.get("name")
        args = c.get("args")
        if isinstance(name, str) and isinstance(args, dict):
            normalized.append({"name": name, "args": args})
    return normalized or None


SYSTEM_INSTRUCTION = """
You are a code editing assistant running inside a local folder.

You have access to these tools:

1) read_file
Description: Read a text file at a relative path.
Input JSON schema: {"type":"object","properties":{"path":{"type":"string","description":"relative file path"}},"required":["path"]}

2) list_files
Description: List files and directories at an optional relative path. Returns a JSON list of strings. Directories end with "/".
Input JSON schema: {"type":"object","properties":{"path":{"type":"string","description":"optional relative path"}}}

3) edit_file
Description: Edit a text file by replacing old_str with new_str.
Rules:
- old_str and new_str must be different
- if the file does not exist, you may create it only when old_str is an empty string
- if editing an existing file, old_str must match exactly once
Input JSON schema: {"type":"object","properties":{"path":{"type":"string"},"old_str":{"type":"string"},"new_str":{"type":"string"}},"required":["path","old_str","new_str"]}

How to call tools:
If you want to use tools, respond with ONLY valid JSON and nothing else, in this exact shape:
{"tool_calls":[{"name":"read_file","args":{"path":"main.py"}}]}

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
        self.conversation: List[types.Content] = []

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

    def _gen_config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,
            max_output_tokens=2048,
        )

    def _trim_conversation(self, keep_last: int = 24) -> None:
        if len(self.conversation) > keep_last:
            self.conversation = self.conversation[-keep_last:]

    def _model_turn(self) -> types.GenerateContentResponse:
        self._trim_conversation()
        return self.client.models.generate_content(
            model=MODEL,
            contents=self.conversation,
            config=self._gen_config(),
        )

    def _append_user_text(self, text: str) -> None:
        self.conversation.append(types.Content(role="user", parts=[types.Part(text=text)]))

    def run(self) -> None:
        print("Chat with Gemma (ctrl c to quit)")
        read_user_input = True

        while True:
            if read_user_input:
                try:
                    user_input = input(f"{BLUE}You{RESET}: ")
                except EOFError:
                    break
                self._append_user_text(user_input)

            resp = self._model_turn()
            content = resp.candidates[0].content
            self.conversation.append(content)

            text = resp.text or ""
            tool_calls = parse_tool_calls(text)

            if not tool_calls:
                print(f"{YELLOW}Gemma{RESET}: {text}")
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
                    tool_results.append(
                        {"name": name, "ok": False, "result": "tool not found"}
                    )
                    continue

                try:
                    result = tool.func(args)
                    tool_results.append({"name": name, "ok": True, "result": result})
                except Exception as e:
                    tool_results.append({"name": name, "ok": False, "result": str(e)})

            self._append_user_text(json.dumps({"tool_results": tool_results}, ensure_ascii=False))
            read_user_input = False


def main() -> None:
    if not os.environ.get("GEMINI_API_KEY"):
        print("Missing GEMINI_API_KEY in environment")
        sys.exit(1)

    client = genai.Client()
    agent = Agent(client=client, base_dir=Path.cwd())
    agent.run()


if __name__ == "__main__":
    main()

