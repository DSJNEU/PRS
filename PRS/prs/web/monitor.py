from __future__ import annotations

import argparse
from pathlib import Path
from threading import Lock
from typing import Optional

from flask import Flask, jsonify, render_template, request

from prs.engine import PRSEngine, create_demo_engine

app = Flask(__name__)
_engine: Optional[PRSEngine] = None
_lock = Lock()


def get_engine() -> PRSEngine:
    global _engine
    with _lock:
        if _engine is None:
            # Web 监控台使用本地 PRSEngine，展示论文方法的控制闭环状态。
            _engine = create_demo_engine(Path(".prs_runtime/web_checkpoints"), nodes=4)
            _engine.run_steps(3)
        return _engine


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/snapshot")
def api_snapshot():
    # 前端只读 snapshot，不直接触碰 engine 内部对象。
    return jsonify(get_engine().snapshot())


@app.route("/api/training/step", methods=["POST"])
def api_training_step():
    steps = int((request.get_json(silent=True) or {}).get("steps", 1))
    outputs = get_engine().run_steps(max(1, min(steps, 50)))
    return jsonify({"status": "ok", "outputs": outputs, "snapshot": get_engine().snapshot()})


@app.route("/api/failure", methods=["POST"])
def api_failure():
    payload = request.get_json(silent=True) or {}
    node_id = payload.get("node_id")
    event = get_engine().simulate_failure(int(node_id) if node_id is not None else None)
    return jsonify({"status": "ok", "event": event.to_dict(), "snapshot": get_engine().snapshot()})


@app.route("/api/scheduler/step", methods=["POST"])
def api_scheduler_step():
    decision = get_engine().scheduler_step()
    return jsonify({"status": "ok", "decision": decision, "snapshot": get_engine().snapshot()})


@app.route("/api/checkpoints/<int:step>", methods=["DELETE"])
def api_delete_checkpoint(step: int):
    engine = get_engine()
    deleted = engine.checkpoint_manager.delete_checkpoint(step) if engine.checkpoint_manager else False
    return jsonify({"status": "ok" if deleted else "missing", "snapshot": engine.snapshot()})


def run_server(host: str = "127.0.0.1", port: int = 5000, debug: bool = False) -> None:
    get_engine()
    app.run(host=host, port=port, debug=debug, threaded=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="PRS Web 监控界面")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    run_server(args.host, args.port, args.debug)


if __name__ == "__main__":
    main()
