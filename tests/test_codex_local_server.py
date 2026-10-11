"""Exercise the real Codex CLI against a disposable loopback Responses API.

The test requires no external model, account or network service. On machines
without Codex, the stub-based agent tests cover the same configuration path.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import threading

import pytest

from hillclimb.agents.base import AgentRequest
from hillclimb.agents.codex_cli import CodexCliAgent


@pytest.mark.skipif(shutil.which("codex") is None, reason="requires an installed Codex CLI")
@pytest.mark.parametrize("api_key", [None, "test-prefix:test-suffix"])
def test_real_codex_uses_loopback_responses_provider(tmp_path: Path, monkeypatch, api_key):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    if api_key is None:
        monkeypatch.delenv("HILLCLIMB_LOCAL_API_KEY", raising=False)
    else:
        monkeypatch.setenv("HILLCLIMB_LOCAL_API_KEY", api_key)
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            if self.path != "/v1/responses":
                self.send_error(404)
                return
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((payload, self.headers.get("Authorization")))
            message = {"id": "msg_test", "type": "message", "role": "assistant",
                       "status": "completed", "content": [
                           {"type": "output_text", "text": "HARNESS-PONG", "annotations": []},
                       ]}
            response = {"id": "resp_test", "object": "response", "created_at": 0,
                        "model": "test-responses-model", "status": "completed",
                        "output": [message], "parallel_tool_calls": False,
                        "usage": {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13,
                                  "input_tokens_details": {"cached_tokens": 0},
                                  "output_tokens_details": {"reasoning_tokens": 0}}}
            events = [
                ("response.created", {"response": {**response, "status": "in_progress", "output": [], "usage": None}}),
                ("response.in_progress", {"response": {**response, "status": "in_progress", "output": [], "usage": None}}),
                ("response.output_item.added", {"output_index": 0, "item": {**message, "status": "in_progress", "content": []}}),
                ("response.content_part.added", {"item_id": message["id"], "output_index": 0,
                                                 "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}}),
                ("response.output_text.delta", {"item_id": message["id"], "output_index": 0,
                                                 "content_index": 0, "delta": "HARNESS-PONG"}),
                ("response.output_text.done", {"item_id": message["id"], "output_index": 0,
                                                "content_index": 0, "text": "HARNESS-PONG"}),
                ("response.content_part.done", {"item_id": message["id"], "output_index": 0,
                                                "content_index": 0, "part": message["content"][0]}),
                ("response.output_item.done", {"output_index": 0, "item": message}),
                ("response.completed", {"response": response}),
            ]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for sequence, (name, data) in enumerate(events):
                event = {"type": name, "sequence_number": sequence, **data}
                self.wfile.write(f"event: {name}\ndata: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    except PermissionError:
        pytest.skip("sandbox disallows binding a localhost test server")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        agent = CodexCliAgent(auth="local", local_base_url=f"http://127.0.0.1:{server.server_port}/v1")
        candidate_dir = tmp_path / "candidate"
        candidate_dir.mkdir()
        result = agent.invoke(AgentRequest(
            operator="draft", prompt="Reply with exactly HARNESS-PONG. Do not use tools.",
            candidate_dir=candidate_dir, timeout_s=60, model="test-responses-model",
        ))
        assert result.ok, result.error_message
        assert requests and all(payload["model"] == "test-responses-model" for payload, _ in requests)
        assert all(tool["type"] == "function" for payload, _ in requests for tool in payload["tools"])
        assert all(auth == (f"Bearer {api_key}" if api_key else None) for _, auth in requests)
        assert "HARNESS-PONG" in (candidate_dir / "agent_stream.jsonl").read_text()
        raw = (candidate_dir / "agent_raw.json").read_text()
        assert "model_provider=local" in json.loads(raw)["cmd"]
        if api_key:
            assert api_key not in raw
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
