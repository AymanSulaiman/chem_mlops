import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from app.scripts.flows.eval import eval_hosted


class FakeChat(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers["content-length"]))
        events = [
            {"tool": {"tool": "get_mechanism", "args": {"name": "SIROLIMUS"}}},
            {"toolResult": {"tool": "get_mechanism", "ok": True}},
            {"message": {"content": "It targets Peptidyl-prolyl "}},
            {"message": {"content": "cis-trans isomerase FKBP1A."}},
            {"done": True},
        ]
        self.send_response(200)
        self.send_header("x-model", "chembl-drug-chat:1b")
        self.end_headers()
        self.wfile.write("".join(json.dumps(e) + "\n" for e in events).encode())

    def log_message(self, format, *args):
        pass


def test_run_scores_streamed_answers(tmp_path, monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), FakeChat)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        json.dumps(
            {
                "question": "q1",
                "drug": "SIROLIMUS",
                "must_contain": ["peptidyl-prolyl cis-trans isomerase fkbp1a"],
            }
        )
        + "\n"
        + json.dumps({"question": "q2", "drug": "X", "must_contain": ["hiv protease"]})
        + "\n"
    )
    monkeypatch.setattr(eval_hosted, "EVAL_DIR", tmp_path / "eval")
    try:
        metrics = eval_hosted.run(f"http://127.0.0.1:{server.server_port}", golden_path=golden)
    finally:
        server.shutdown()

    assert metrics["model"] == "chembl-drug-chat:1b"
    assert metrics["pass_rate"] == 0.5
    assert metrics["tool_call_rate"] == 1.0
    assert metrics["error_rate"] == 0.0
    rows = [
        json.loads(line)
        for line in next((tmp_path / "eval").iterdir())
        .joinpath("results.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [r["passed"] for r in rows] == [True, False]
    assert rows[0]["tools"] == ["get_mechanism"]
