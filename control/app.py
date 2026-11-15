"""aiohttp control panel: list / start / stop workers and stream logs."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from aiohttp import web

from . import process_manager as pm
from .registry import get_product, load_products

STATIC_DIR = Path(__file__).resolve().parent / "static"
LOGGER = logging.getLogger("control")


def _json_error(message: str, status: int = 400) -> web.Response:
    return web.json_response({"error": message}, status=status)


async def index(_request: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_DIR / "index.html")


async def api_products(_request: web.Request) -> web.Response:
    products = []
    for product in load_products():
        st = await asyncio.to_thread(pm.status, product)
        item = product.to_dict()
        item["status"] = st.to_dict()
        products.append(item)
    # Running first, then startable, then by title
    products.sort(
        key=lambda p: (
            not (p.get("status") or {}).get("running"),
            not p.get("startable", False),
            str(p.get("title") or p.get("id") or "").lower(),
        )
    )
    return web.json_response({"products": products})


async def api_product_status(request: web.Request) -> web.Response:
    product_id = request.match_info["id"]
    product = get_product(product_id)
    if product is None:
        return _json_error("Unknown product", 404)
    st = await asyncio.to_thread(pm.status, product)
    item = product.to_dict()
    item["status"] = st.to_dict()
    return web.json_response(item)


async def api_start(request: web.Request) -> web.Response:
    product_id = request.match_info["id"]
    product = get_product(product_id)
    if product is None:
        return _json_error("Unknown product", 404)
    if not product.startable:
        return _json_error("Product is not startable", 400)
    try:
        body: Dict[str, Any] = await request.json()
    except (json.JSONDecodeError, TypeError):
        body = {}
    hub = str(body.get("hub") or "cloud").strip().lower()
    if hub not in ("local", "cloud"):
        return _json_error('hub must be "local" or "cloud"')
    st = await asyncio.to_thread(pm.start, product, hub)
    status_code = 200
    if st.error == "Already running":
        status_code = 409
    elif st.error:
        status_code = 500
    return web.json_response(
        {"product": product.to_dict(), "status": st.to_dict()},
        status=status_code,
    )


async def api_stop(request: web.Request) -> web.Response:
    product_id = request.match_info["id"]
    product = get_product(product_id)
    if product is None:
        return _json_error("Unknown product", 404)
    st = await asyncio.to_thread(pm.stop, product)
    status_code = 500 if st.error else 200
    return web.json_response(
        {"product": product.to_dict(), "status": st.to_dict()},
        status=status_code,
    )


async def api_logs(request: web.Request) -> web.Response:
    product_id = request.match_info["id"]
    product = get_product(product_id)
    if product is None:
        return _json_error("Unknown product", 404)
    try:
        tail = int(request.query.get("tail", "200"))
    except ValueError:
        tail = 200
    tail = max(0, min(tail, 5000))
    text, path = await asyncio.to_thread(pm.read_log_tail, product_id, tail)
    return web.json_response({"product_id": product_id, "path": path, "text": text})


async def api_logs_stream(request: web.Request) -> web.StreamResponse:
    product_id = request.match_info["id"]
    product = get_product(product_id)
    if product is None:
        return _json_error("Unknown product", 404)

    response = web.StreamResponse(
        status=200,
        reason="OK",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await response.prepare(request)

    # Start from current end; send a short recent tail first.
    text, _path = await asyncio.to_thread(pm.read_log_tail, product_id, 100)
    offset = await asyncio.to_thread(pm.log_file_size, product_id)
    if text:
        payload = json.dumps({"text": text})
        await response.write(f"event: log\ndata: {payload}\n\n".encode("utf-8"))

    try:
        while True:
            if request.transport is None or request.transport.is_closing():
                break
            chunk, offset = await asyncio.to_thread(
                pm.read_log_since, product_id, offset
            )
            if chunk:
                payload = json.dumps({"text": chunk})
                await response.write(
                    f"event: log\ndata: {payload}\n\n".encode("utf-8")
                )
            else:
                await response.write(b": ping\n\n")
            await asyncio.sleep(1.0)
    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
        pass
    return response


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/api/products", api_products)
    app.router.add_get("/api/products/{id}", api_product_status)
    app.router.add_post("/api/products/{id}/start", api_start)
    app.router.add_post("/api/products/{id}/stop", api_stop)
    app.router.add_get("/api/products/{id}/logs", api_logs)
    app.router.add_get("/api/products/{id}/logs/stream", api_logs_stream)
    app.router.add_static("/static/", STATIC_DIR, name="static")
    return app


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Localhost worker control panel (start/stop/status/logs)."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Listen port (default: 8765)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args(argv)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        LOGGER.warning(
            "Binding to %s — this panel can start/stop workers; prefer 127.0.0.1",
            args.host,
        )
    app = create_app()
    LOGGER.info("Worker control panel at http://%s:%s", args.host, args.port)
    web.run_app(app, host=args.host, port=args.port, print=None)
    return 0
