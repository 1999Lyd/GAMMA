"""Wire-protocol smoke test for mme_rma_adapter.py -- no GPU, no real policy.

Spins up a mock websocket policy server that speaks the same protocol as
mme_vla_suite.serving.websocket_policy_server (reset / add_buffer / infer),
loads our adapter through THEIR `load_policy_adapter`, and replays the exact
call order the harness uses:

    reset() -> [observe() x N, infer_actions()] x M

then asserts that what reached the "server" is what the RoboMME eval client
would have sent (incremental per-step buffer whose tail is the current frame,
lowercased prompt, 8-d state, [20,7] chunk back).

Usage:  .venv/bin/python smoke_adapter_protocol.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading

import numpy as np

BENCH = os.path.expandvars("${RMA_BENCH_ROOT}")
sys.path.insert(0, f"{BENCH}/scripts")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openpi_client import msgpack_numpy  # noqa: E402
import websockets.asyncio.server as _server  # noqa: E402

from policy_adapter import load_policy_adapter  # noqa: E402

PORT = int(os.environ.get("SMOKE_PORT", "8399"))
HORIZON, ADIM = 20, 7

RECEIVED: list[dict] = []


class MockServer:
    def __init__(self, port: int) -> None:
        self.port = port
        self.ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    async def _handler(self, ws) -> None:
        packer = msgpack_numpy.Packer()
        await ws.send(packer.pack({"mock": True}))
        while True:
            try:
                obs = msgpack_numpy.unpackb(await ws.recv())
            except Exception:  # noqa: BLE001 - client went away
                return
            if obs.get("reset", False):
                RECEIVED.append({"kind": "reset"})
                await ws.send(packer.pack({"reset_finished": True}))
            elif obs.get("add_buffer", False):
                RECEIVED.append(
                    {
                        "kind": "add_buffer",
                        "images_shape": tuple(np.asarray(obs["images"]).shape),
                        "images_dtype": str(np.asarray(obs["images"]).dtype),
                        "state_shape": tuple(np.asarray(obs["state"]).shape),
                        "exec_start_idx": obs.get("exec_start_idx"),
                        "tail_image": np.asarray(obs["images"])[-1, 0].copy(),
                    }
                )
                await ws.send(packer.pack({"add_buffer_finished": True}))
            else:
                RECEIVED.append(
                    {
                        "kind": "infer",
                        "keys": sorted(obs.keys()),
                        "prompt": obs["prompt"],
                        "image_shape": tuple(np.asarray(obs["observation/image"]).shape),
                        "state": np.asarray(obs["observation/state"]).copy(),
                        "image": np.asarray(obs["observation/image"]).copy(),
                    }
                )
                await ws.send(
                    packer.pack({"actions": np.zeros((HORIZON, ADIM), dtype=np.float32)})
                )

    async def _run(self) -> None:
        async with _server.serve(self._handler, "127.0.0.1", self.port, compression=None,
                                 max_size=None):
            self.ready.set()
            await asyncio.Event().wait()

    def start(self) -> None:
        def target() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._run())

        threading.Thread(target=target, daemon=True).start()
        self.ready.wait(timeout=15)


def fake_obs(step: int) -> dict:
    rng = np.random.default_rng(step)
    return {
        "observation/image": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        "observation/state": rng.random(8).astype(np.float32),
    }


def main() -> int:
    MockServer(PORT).start()

    adapter = load_policy_adapter(
        f"{os.path.dirname(os.path.abspath(__file__))}/mme_rma_adapter.py:build_adapter",
        host="127.0.0.1",
        port=PORT,
    )
    print(f"[smoke] adapter loaded via their load_policy_adapter: {type(adapter).__name__}")

    prompt = "Pick and place butter into the basket, then pick and place popcorn into the same basket."
    replan = 10
    step = 0
    for ep in range(2):
        adapter.reset()
        for _ in range(2):  # two replan cycles
            last = None
            for _ in range(replan):
                obs = fake_obs(step)
                step += 1
                adapter.observe(obs, prompt, 256)
                last = obs
                if _ == 0:
                    actions = adapter.infer_actions(obs=obs, prompt=prompt, resize_size=256)
                    assert actions.shape == (HORIZON, ADIM), actions.shape
            del last
    adapter.close()

    kinds = [r["kind"] for r in RECEIVED]
    print(f"[smoke] server saw {len(RECEIVED)} messages: {kinds}")

    resets = [r for r in RECEIVED if r["kind"] == "reset"]
    buffers = [r for r in RECEIVED if r["kind"] == "add_buffer"]
    infers = [r for r in RECEIVED if r["kind"] == "infer"]
    assert len(resets) == 2, len(resets)
    assert len(buffers) == 4, len(buffers)
    assert len(infers) == 4, len(infers)

    # first infer of an episode: only the current frame has been observed;
    # second infer: the 10 frames since the previous infer.
    got = [b["images_shape"] for b in buffers]
    want = [(1, 1, 256, 256, 3), (10, 1, 256, 256, 3)] * 2
    assert got == want, f"buffer shapes {got} != {want}"
    assert all(b["images_dtype"] == "uint8" for b in buffers)
    assert [b["state_shape"] for b in buffers] == [(1, 8), (10, 8)] * 2
    assert all(b["exec_start_idx"] == 0 for b in buffers)
    print(f"[smoke] add_buffer shapes {got}; exec_start_idx all 0")

    # buffer tail must equal the frame the infer is conditioned on
    for i, (b, inf) in enumerate(zip(buffers, infers)):
        assert np.array_equal(b["tail_image"], inf["image"]), f"buffer tail != infer frame at {i}"
    print("[smoke] buffer tail == infer frame for all 4 infers")

    assert all(inf["prompt"] == prompt.lower() for inf in infers), infers[0]["prompt"]
    print(f"[smoke] prompt lowercased -> {infers[0]['prompt']!r}")
    assert infers[0]["keys"] == [
        "observation/image",
        "observation/state",
        "observation/wrist_image",
        "prompt",
    ], infers[0]["keys"]
    print(f"[smoke] infer element keys {infers[0]['keys']}")
    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
