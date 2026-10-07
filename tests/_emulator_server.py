"""One verified fake engine served over HTTP: the T4 supervision re-run's engine process (test-side).

``python tests/_emulator_server.py <recipe-id>`` builds the recipe's emulator through the one wiring home
(``tests._engines.emulator_for``: the recipe, its real tokenizer, its current corpus) and serves it on
``127.0.0.1:$PORT`` with the standard library's HTTP server, every request answered by
:meth:`rcp_ndcg.testing.engines.VllmEmulator.handle`. Like the supervision stub engine it replaces, it records
its pid and start in ``$STUBS`` (phase 2 records whether phase 1's engine was already stopped) and marks
itself ready once it listens; it runs until the supervision stops it.
"""

from __future__ import annotations

import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_HOP_BY_HOP = frozenset({"content-length", "transfer-encoding", "connection", "content-encoding"})


def _record_start(stubs: Path, phase: str) -> None:
    (stubs / f"engine-{phase}.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    lines = [f"engine-{phase} start"]
    if phase == "2":
        previous = int((stubs / "engine-1.pid").read_text(encoding="utf-8"))
        try:
            os.kill(previous, 0)
            lines.append("engine-2 saw phase 1 running")
        except OSError:
            lines.append("engine-2 saw phase 1 stopped")
    with (stubs / "order").open("a", encoding="utf-8") as handle:
        handle.write("".join(f"{line}\n" for line in lines))


def main(recipe_id: str) -> None:
    """Serve ``recipe_id``'s emulator on ``$PORT`` until stopped."""
    import httpx

    from tests._engines import emulator_for

    stubs, phase, port = Path(os.environ["STUBS"]), os.environ.get("PHASE", "1"), int(os.environ["PORT"])
    _record_start(stubs, phase)
    emulator = emulator_for(recipe_id)

    class Handler(BaseHTTPRequestHandler):
        def _answer(self) -> None:
            length = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(length) if length else b""
            headers = {key: value for key, value in self.headers.items() if key.lower() not in _HOP_BY_HOP}
            request = httpx.Request(self.command, f"http://engine{self.path}", content=body, headers=headers)
            response = emulator.handle(request)
            self.send_response(response.status_code)
            for key, value in response.headers.items():
                if key.lower() not in _HOP_BY_HOP:
                    self.send_header(key, value)
            self.send_header("content-length", str(len(response.content)))
            self.end_headers()
            self.wfile.write(response.content)

        do_GET = do_POST = _answer

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - the stdlib's signature
            return

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    (stubs / f"ready-{port}").touch()  # listening: the supervision's readiness probe may pass
    server.serve_forever()


if __name__ == "__main__":
    main(sys.argv[1])
