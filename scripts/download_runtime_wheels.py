#!/usr/bin/env python3
"""Resumable, hash-verified wheel downloads for an annotated uv requirements file.

This prepares a wheelhouse; uv remains responsible for dependency resolution and
installation. Curl resumes interrupted transfers instead of restarting large wheels.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

METADATA_PROXY = None


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.extend(v for k, v in attrs if k == "href")


def requirements(path):
    items = []
    for line in path.read_text().splitlines():
        match = re.match(r"^([a-zA-Z0-9_-]+)==(\S+)", line)
        if match:
            items.append({"name": match[1], "version": match[2]})
        elif "# from https://" in line:
            items[-1]["index"] = line.split("# from ", 1)[1].strip()
    if not items or any("index" not in item for item in items):
        raise ValueError("Compile requirements with uv --emit-index-annotation first")
    return items


def get_url(url):
    error = None
    for attempt in range(4):
        try:
            if METADATA_PROXY and attempt % 2:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler(
                    {"http": METADATA_PROXY, "https": METADATA_PROXY}))
            else:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(url, timeout=30) as response:
                return response.read()
        except Exception as exc:
            error = exc
            time.sleep(min(attempt+1, 3))
    raise error


def select_artifact(item):
    rank = {tag: i for i, tag in enumerate(sys_tags())}
    candidates = []
    if item["index"].rstrip("/") == "https://pypi.org/simple":
        metadata = json.loads(get_url(f'https://pypi.org/pypi/{item["name"]}/{item["version"]}/json'))
        sources = [{"url": entry["url"], "sha256": entry["digests"]["sha256"],
                    "size": entry["size"]} for entry in metadata["urls"] if not entry["yanked"]]
    else:
        base = item["index"].rstrip("/") + "/" + canonicalize_name(item["name"]) + "/"
        parser = Links()
        parser.feed(get_url(base).decode())
        sources = []
        for href in parser.links:
            url, _, fragment = urllib.parse.urljoin(base, href).partition("#")
            digest = urllib.parse.parse_qs(fragment).get("sha256", [None])[0]
            if digest:
                sources.append({"url": url, "sha256": digest, "size": None})
    for source in sources:
        filename = urllib.parse.unquote(urllib.parse.urlsplit(source["url"]).path.rsplit("/", 1)[-1])
        if filename.endswith(".whl"):
            name, version, _, tags = parse_wheel_filename(filename)
            matching = set(tags).intersection(rank)
            if name == canonicalize_name(item["name"]) and version == Version(item["version"]) and matching:
                candidates.append((min(rank[tag] for tag in matching), filename, source))
        elif item["index"].rstrip("/") == "https://pypi.org/simple":
            candidates.append((10**9, filename, source))
    if not candidates:
        raise ValueError(f"No compatible artifact: {item}")
    _, filename, source = sorted(candidates, key=lambda x: (x[0], x[1]))[0]
    return {**item, **source, "filename": filename}


def cached_packages():
    compatible = {str(tag) for tag in sys_tags()}
    found = set()
    for path in (Path.home()/".cache/uv/archive-v0").glob("*/*.dist-info"):
        wheel = path / "WHEEL"
        if not wheel.is_file() or not any(line[5:].strip() in compatible
                                         for line in wheel.read_text().splitlines()
                                         if line.startswith("Tag: ")):
            continue
        metadata = (path/"METADATA").read_text(errors="replace")
        name = re.search(r"^Name: (.+)$", metadata, re.M)
        version = re.search(r"^Version: (.+)$", metadata, re.M)
        if name and version:
            found.add((canonicalize_name(name[1]), Version(version[1])))
    return found


def digest_file(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    global METADATA_PROXY
    parser = argparse.ArgumentParser()
    parser.add_argument("requirements", type=Path)
    parser.add_argument("--directory", type=Path, default=Path(".runtime/wheels"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--proxy", help="Optional existing HTTP proxy; no system settings changed")
    parser.add_argument("--pypi-mirror", help="Optional mirror base such as https://mirrors.aliyun.com/pypi")
    parser.add_argument("--attempts", type=int, default=1000)
    args = parser.parse_args()
    METADATA_PROXY = args.proxy
    args.directory.mkdir(parents=True, exist_ok=True)
    items = requirements(args.requirements)
    cached = cached_packages()
    manifest_path = args.directory / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        resolved = {(x["name"], x["version"], x["index"]): x for x in old}
    else:
        resolved = {}
    manifest = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for item in items:
            key = (item["name"], item["version"], item["index"])
            if key in resolved:
                manifest.append(resolved[key])
            else:
                futures[pool.submit(select_artifact, item)] = item
        failures = []
        for future in as_completed(futures):
            try:
                artifact = future.result()
                manifest.append(artifact)
                manifest_path.write_text(json.dumps(manifest, indent=2))
                print("Resolved", artifact["filename"], flush=True)
            except Exception as exc:
                failures.append((futures[future], str(exc)))
                print("Resolve failed", futures[future]["name"], str(exc), flush=True)
    if failures:
        raise RuntimeError(f"Metadata lookup failed; successful entries retained in {manifest_path}: {failures}")
    manifest.sort(key=lambda x: x["filename"])
    manifest_path.write_text(json.dumps(manifest, indent=2))
    status = {x["filename"]: {"state": "pending"} for x in manifest}
    lock = threading.Lock()
    status_path = args.directory / "status.json"

    def save_status():
        with lock:
            for name, state in status.items():
                partial = args.directory / (name + ".part")
                if partial.exists():
                    state["bytes_downloaded"] = partial.stat().st_size
            temporary = status_path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"updated_unix": time.time(), "files": status}, indent=2))
            temporary.replace(status_path)

    def download(item):
        filename = item["filename"]
        target = args.directory / filename
        partial = target.with_name(filename + ".part")
        if target.exists() and digest_file(target) == item["sha256"]:
            status[filename] = {"state": "verified"}
            return
        if (canonicalize_name(item["name"]), Version(item["version"])) in cached:
            status[filename] = {"state": "uv_cache_available"}
            return
        status[filename] = {"state": "downloading", "expected_bytes": item["size"]}
        log = target.with_name(filename + ".log")
        routes = []
        if args.pypi_mirror and item["url"].startswith("https://files.pythonhosted.org/"):
            mirror_url = args.pypi_mirror.rstrip("/") + "/" + item["url"].split(
                "https://files.pythonhosted.org/", 1)[1]
            routes.append((mirror_url, False))
        if args.proxy:
            routes.append((item["url"], True))
        routes.append((item["url"], False))
        for attempt in range(args.attempts):
            url, use_proxy = routes[attempt % len(routes)]
            cmd = ["curl", "--silent", "--show-error", "--fail", "--location",
                   "--connect-timeout", "15", "--max-time", "60", "--continue-at", "-",
                   "--output", str(partial), url]
            if use_proxy:
                cmd[1:1] = ["--proxy", args.proxy]
            with log.open("a") as f:
                result = subprocess.run(cmd, stdout=f, stderr=f)
            if result.returncode == 0:
                if digest_file(partial) != item["sha256"]:
                    status[filename] = {"state": "hash_mismatch", "attempt": attempt+1}
                    raise ValueError(f"SHA256 mismatch: {partial}")
                partial.replace(target)
                status[filename] = {"state": "verified", "bytes": target.stat().st_size}
                print("Verified", filename, flush=True)
                return
            if result.returncode == 33:
                status[filename] = {"state": "server_cannot_resume"}
                raise RuntimeError(f"Server refused resume; preserved {partial}")
            status[filename]["attempt"] = attempt+1
            time.sleep(min(1+attempt, 5))
        status[filename]["state"] = "retry_limit"
        raise RuntimeError(f"Retry limit reached: {filename}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(download, item) for item in manifest]
        while not all(job.done() for job in jobs):
            save_status()
            time.sleep(5)
        save_status()
        for job in jobs:
            job.result()
    print("Wheelhouse ready; install with uv --find-links", args.directory, flush=True)


if __name__ == "__main__":
    main()
