"""Stack smoke test, run INSIDE the compose network (standard library only).

    docker compose exec -T api python - < scripts/service_smoke.py

1. Egress check: a TCP connect to 1.1.1.1:443 and a DNS lookup of pypi.org must fail.
2. Waits for /readyz (database + worker heartbeat).
3. Registers the demo scene (scripts/make_demo_scene.py) as optical and as fused
   (look direction from /data/demo_scene.json), then runs one job per sensor mode over
   the whole scene and waits for each.
4. Writes /data/outputs/smoke.json (job ids, status, tiles, timings, building counts) and
   exits non-zero if any job fails or the network is reachable.
"""

from __future__ import annotations

import json
import socket
import sys
import time
import urllib.error
import urllib.request

API = "http://127.0.0.1:8000"


def call(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        API + path, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def egress_blocked() -> bool:
    ok = True
    try:
        socket.create_connection(("1.1.1.1", 443), timeout=3).close()
        print("FAIL: TCP to 1.1.1.1:443 succeeded")
        ok = False
    except OSError as e:
        print(f"egress TCP blocked ({e.__class__.__name__})")
    try:
        socket.getaddrinfo("pypi.org", 443)
        print("FAIL: DNS lookup of pypi.org succeeded")
        ok = False
    except OSError as e:
        print(f"egress DNS blocked ({e.__class__.__name__})")
    return ok


def main() -> int:
    offline = egress_blocked()
    for _ in range(120):
        st, body = call("GET", "/readyz")
        if st == 200:
            break
        time.sleep(1)
    else:
        print(f"FAIL: /readyz never became ready: {body}")
        return 1
    print("ready:", body)
    with open("/data/demo_scene.json") as f:
        orient = int(json.load(f)["orientation"])
    scenes = {
        "demo_opt": {"path": "demo_optical.tif", "sensor": "optical"},
        "demo_fused": {
            "path": "demo_optical.tif",
            "sar_path": "demo_sar.tif",
            "sensor": "fused",
            "orientation": orient,
        },
    }
    bounds = None
    for sid, body in scenes.items():
        st, out = call("POST", "/v1/scenes", {"id": sid, **body})
        if st not in (201, 409):
            print(f"FAIL: register {sid}: {st} {out}")
            return 1
        if st == 201:
            bounds = out["bounds"]
    if bounds is None:
        _, lst = call("GET", "/v1/scenes")
        bounds = next(s["bounds"] for s in lst if s["id"] == "demo_opt")
    lon0, lat0, lon1, lat1 = bounds
    e = 1e-6  # stay just inside the footprint
    aoi = {
        "type": "Polygon",
        "coordinates": [
            [
                [lon0 + e, lat0 + e],
                [lon1 - e, lat0 + e],
                [lon1 - e, lat1 - e],
                [lon0 + e, lat1 - e],
                [lon0 + e, lat0 + e],
            ]
        ],
    }
    runs = [("demo_opt", "optical"), ("demo_fused", "sar"), ("demo_fused", "fused")]
    results = []
    failed = False
    for sid, mode in runs:
        st, job = call(
            "POST",
            "/v1/jobs",
            {"scene_id": sid, "task": "segment_buildings", "aoi": aoi, "sensor_mode": mode},
        )
        if st != 202:
            print(f"FAIL: submit {sid}/{mode}: {st} {job}")
            return 1
        jid = job["id"]
        t0 = time.time()
        while time.time() - t0 < 900:
            _, j = call("GET", f"/v1/jobs/{jid}")
            if j["status"] in ("ready", "failed"):
                break
            time.sleep(2)
        n = None
        if j["status"] == "ready":
            st, gj = call("GET", f"/v1/jobs/{jid}/result?format=geojson")
            n = len(gj.get("features", []))
        else:
            failed = True
        rec = {
            "scene": sid,
            "mode": mode,
            "job": jid,
            "status": j["status"],
            "model": j.get("model_id"),
            "n_tiles": j.get("n_tiles"),
            "buildings": n,
            "timings": j.get("timings"),
            "error": j.get("error"),
        }
        print(json.dumps(rec))
        results.append(rec)
    with open("/data/outputs/smoke.json", "w") as f:
        json.dump({"offline": offline, "jobs": results}, f, indent=2)
    return 1 if failed or not offline else 0


if __name__ == "__main__":
    sys.exit(main())
