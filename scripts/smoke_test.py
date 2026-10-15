# scripts/smoke_test.py
"""
Smoke test for a running API (local, Docker or Render)

What it checks
1. GET /health returns 200 (model loaded)
2. Optional: /health reports the expected git commit (proves the NEW release is live)
3. POST /predict with scripts/sample_request.json returns 200 and one prediction per record

Usage
- Local:   python scripts/smoke_test.py --url http://127.0.0.1:8000
- Render:  python scripts/smoke_test.py --url https://<your-service>.onrender.com
- CI/CD:   python scripts/smoke_test.py --url "$RENDER_SERVICE_URL" --expect-commit "$GITHUB_SHA" --timeout 1200

Only the Python standard library is used, so it runs anywhere without installing packages
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SAMPLE_REQUEST = Path(__file__).resolve().parent / "sample_request.json"


def _request(url: str, payload: dict = None, timeout: int = 60):
    """Return (status_code, parsed_json_or_text). Never raises on HTTP errors."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        return 0, str(e)


def wait_until_healthy(base_url: str, expect_commit: str, timeout_s: int, interval_s: int) -> dict:
    """Poll /health until it is 200 (and on the expected commit) or the timeout is reached."""
    deadline = time.time() + timeout_s
    while True:
        status, body = _request(f"{base_url}/health")
        commit = body.get("git_commit", "") if isinstance(body, dict) else ""
        on_expected_commit = not expect_commit or commit == expect_commit
        print(f"/health -> {status} | git_commit={commit or '?'}", flush=True)

        if status == 200 and on_expected_commit:
            return body
        if time.time() > deadline:
            sys.exit(f"FAILED: service not healthy on the expected commit after {timeout_s}s. Last body: {body}")
        # Free Render services sleep when idle and the first request wakes them up (can take ~1 minute)
        time.sleep(interval_s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the prediction API")
    parser.add_argument("--url", required=True, help="Base URL, e.g. http://127.0.0.1:8000")
    parser.add_argument("--expect-commit", default="", help="Git SHA that /health must report (optional)")
    parser.add_argument("--timeout", type=int, default=120, help="Seconds to wait for a healthy service")
    parser.add_argument("--interval", type=int, default=15, help="Seconds between /health checks")
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    health = wait_until_healthy(base_url, args.expect_commit, args.timeout, args.interval)
    print(f"Healthy | model_version={health.get('model_version')}")

    payload = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
    status, body = _request(f"{base_url}/predict", payload)
    if status != 200:
        sys.exit(f"FAILED: /predict returned {status}: {body}")

    predictions = body.get("predictions", [])
    if len(predictions) != len(payload["records"]):
        sys.exit(f"FAILED: expected {len(payload['records'])} predictions, got {len(predictions)}")

    print(f"/predict -> 200 | {predictions}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
