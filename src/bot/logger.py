import threading
import time
import cv2
import os
import json


class RunLogger:
    def __init__(self, out_dir: str = "logs/", run_metadata: dict | None = None) -> None:
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        pid = os.getpid()
        self.path = os.path.join(out_dir, f"bot_debug_{ts}_{pid}.jsonl")
        self._lock = threading.Lock()
        self._fh = open(self.path, "a", encoding="utf-8")
        if run_metadata:
            self.log_event("run_start", run_metadata)

    def log_event(self, event_type: str, payload: dict) -> None:
        event = {
            "event_type": event_type,
            "timestamp": time.time(),
            "payload": payload,
        }
        with self._lock:
            json.dump(event, self._fh, ensure_ascii=False)
            self._fh.write("\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            self._fh.close()


def save_debug_image(vis, reason: str, seq: int, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    safe_reason = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in reason)
    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = f"frame_{seq:06d}_{safe_reason}_{ts}.png"
    path = os.path.join(out_dir, filename)
    cv2.imwrite(path, vis)
    return path