"""Client for the Microburbs Hackathon API.

Reads the team key from MICROBURBS_API_KEY (or .env). Retries 429 and 5xx,
honouring Retry-After, and unwraps the {"data": ...} envelope: a no-data
answer (still HTTP 200) comes back as None. Every 200 response is cached on
disk so reruns cost nothing; the cache is API output, so it stays out of git.
"""
import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import requests
from dotenv import load_dotenv

DEFAULT_BASE_URL = "https://api-ht.microburbs.com.au"


class MicroburbsError(Exception):
    def __init__(self, status: int, message: str, path: str):
        super().__init__(f"HTTP {status} for {path}: {message}")
        self.status = status
        self.path = path


class Microburbs:
    def __init__(self, cache_dir: Path | None = None, max_rps: float = 8.0,
                 max_retries: int = 5, timeout: float = 30):
        load_dotenv()
        self.api_key = os.environ.get("MICROBURBS_API_KEY")
        if not self.api_key or self.api_key == "YOUR_TEAM_KEY":
            raise RuntimeError("No API key. Copy .env.example to .env and paste your team key.")
        self.base_url = (os.environ.get("MICROBURBS_API_BASE") or DEFAULT_BASE_URL).rstrip("/")
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {self.api_key}"
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_retries = max_retries
        self.timeout = timeout
        self._min_interval = 1.0 / max_rps  # stay under the ~10 req/s per-key limit
        self._next_slot = 0.0
        self._lock = threading.Lock()

    def __repr__(self):
        return f"Microburbs(base_url={self.base_url!r})"

    def get(self, path: str, **params):
        """GET a /v1/... path. Returns the `data` payload, or None when there is no data."""
        body = self.get_body(path, **params)
        return body.get("data") if isinstance(body, dict) and "data" in body else body

    def get_body(self, path: str, **params):
        """GET and return the whole JSON body (for paged endpoints that carry `meta`)."""
        params = {k: v for k, v in params.items() if v is not None}
        cache_file = self._cache_file(path, params)
        if cache_file and cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))["body"]

        url = self.base_url + quote(path, safe="/")
        for attempt in range(self.max_retries + 1):
            self._throttle()
            resp = self.session.get(url, params=params, timeout=self.timeout)
            retryable = resp.status_code == 429 or resp.status_code >= 500
            if retryable and attempt < self.max_retries:
                time.sleep(_wait_seconds(resp, attempt))
                continue
            if resp.status_code != 200:
                raise MicroburbsError(resp.status_code, _error_message(resp), path)
            body = resp.json()
            if cache_file:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps({"path": path, "params": params, "body": body}),
                                      encoding="utf-8")
            return body

    def get_many(self, requests_: list[tuple[str, dict]], workers: int = 4, errors: str = "none"):
        """Run many GETs in parallel. errors="none" turns failures (e.g. 403) into None."""
        def one(item):
            path, params = item
            try:
                return self.get(path, **params)
            except MicroburbsError:
                if errors == "raise":
                    raise
                return None
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(one, requests_))

    def get_all_pages(self, path: str, page_size: int = 200, max_pages: int = 10_000, **params):
        """Follow limit/offset paging until a short page. Returns the concatenated rows."""
        rows = []
        for page in range(max_pages):
            data = self.get(path, limit=page_size, offset=page * page_size, **params)
            batch = _rows(data)
            rows.extend(batch)
            if len(batch) < page_size:
                break
        return rows

    def _throttle(self):
        with self._lock:
            now = time.monotonic()
            wait = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + self._min_interval
        if wait > 0:
            time.sleep(wait)

    def _cache_file(self, path: str, params: dict) -> Path | None:
        if not self.cache_dir:
            return None
        key = json.dumps([path, sorted(params.items())], default=str)
        digest = hashlib.sha1(key.encode()).hexdigest()
        return self.cache_dir / "api" / digest[:2] / f"{digest}.json"


def _rows(data) -> list:
    """Paged endpoints return either a list or a dict holding the list under some key."""
    if data is None:
        return []
    if isinstance(data, list):
        return data
    for key in ("results", "properties", "items", "rows", "data"):
        if isinstance(data.get(key), list):
            return data[key]
    lists = [v for v in data.values() if isinstance(v, list)]
    return lists[0] if len(lists) == 1 else []


def _wait_seconds(resp, attempt: int) -> float:
    retry_after = resp.headers.get("Retry-After")
    if retry_after:
        try:
            return max(float(retry_after), 0.5)
        except ValueError:
            pass
    return min(2 ** attempt, 30) + random.uniform(0, 0.5)


def _error_message(resp) -> str:
    try:
        body = resp.json()
        return str(body.get("detail") or body.get("message") or body)
    except ValueError:
        return resp.text[:300]
