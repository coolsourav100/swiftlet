"""
CLI entrypoint. Implements end-to-end request proxying: forwards chat
requests to the routed llama-server and streams the response back, with
tok/s measurements fed back into the learned config store.
"""

import argparse
import atexit
import json
import os
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
from dotenv import load_dotenv

from .classifier import classify
from .config_store import LearnedConfigStore
from .orchestrator import Orchestrator, ServerPool, ServerHandle


def demo(store_path: str):
    """
    Runs a handful of representative requests through the real classifier +
    config store + pool logic, using a fake launcher (no actual model load),
    so you can see the routing decisions and learning behavior directly.
    """
    store = LearnedConfigStore(store_path, seed=42)

    def fake_launcher(config, port):
        print(f"  [would launch] llama-server --n-gpu-layers {config.n_gpu_layers} "
              f"--n-cpu-moe {config.n_cpu_moe} --batch-size {config.batch_size} --port {port}")
        return ServerHandle(config=config, port=port, started_at=0.0)

    pool = ServerPool(max_size=3, launcher=fake_launcher)
    orch = Orchestrator(store, pool)

    scenarios = [
        ("short chat turn", 80, 150),
        ("long document Q&A", 4000, 100),
        ("open-ended long generation", 100, 1500),
        ("short chat turn (again)", 90, 140),
    ]

    for name, prompt_tokens, gen_tokens in scenarios:
        sig, config, handle, exploring = orch.route(prompt_tokens, gen_tokens)
        tag = "EXPLORE" if exploring else "EXPLOIT"
        print(f"\n[{name}] prompt={prompt_tokens} gen={gen_tokens} -> signature={sig} phase={sig.phase.value}")
        print(f"  decision: {tag} -> n_gpu_layers={config.n_gpu_layers} n_cpu_moe={config.n_cpu_moe}")

        # Simulated measurement — replace with a real tok/s reading from
        # llama-server's response once wired up to an actual instance.
        fake_measured_tps = 20 + config.n_cpu_moe * 0.3
        orch.record(sig, config, fake_measured_tps)
        print(f"  recorded (simulated) {fake_measured_tps:.1f} tok/s")

    print(f"\nLearned store saved to: {store_path}")


def make_proxy_handler(orch: Orchestrator, ctx_size: int = 8192):
    class ProxyHandler(BaseHTTPRequestHandler):
        def _count_tokens(self, body: dict) -> int:
            """
            Accurate token count via /tokenize on any running server.
            Falls back to a conservative heuristic if no server is up yet.
            """
            prompt = body.get("prompt", body.get("messages", ""))

            if isinstance(prompt, list):
                parts = []
                for msg in prompt:
                    if isinstance(msg, dict):
                        parts.append(msg.get("content", str(msg)))
                    else:
                        parts.append(str(msg))
                text = "\n".join(parts)
                template_overhead = len(prompt) * 5
            else:
                text = str(prompt)
                template_overhead = 0

            with orch.pool._lock:
                handles = list(orch.pool._pool.values())

            for handle in handles:
                try:
                    resp = httpx.post(
                        f"http://127.0.0.1:{handle.port}/tokenize",
                        json={"content": text},
                        timeout=10.0,
                    )
                    if resp.status_code == 200:
                        return len(resp.json().get("tokens", [])) + template_overhead
                except Exception:
                    continue

            if isinstance(prompt, list):
                return sum(len(str(p)) // 3 for p in prompt) + template_overhead
            else:
                return len(str(prompt)) // 3

        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)

            try:
                body = json.loads(post_data)
            except json.JSONDecodeError:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Invalid JSON")
                return

            prompt_tokens = self._count_tokens(body)

            expected_gen_tokens = body.get("n_predict", body.get("max_tokens", 128))
            is_stream = bool(body.get("stream", False))

            if prompt_tokens >= ctx_size:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                error_body = json.dumps({
                    "error": (
                        f"Prompt has {prompt_tokens} tokens but context size is "
                        f"{ctx_size}. Restart swiftlet with --ctx-size "
                        f"{min(prompt_tokens + 256, 32768)} (or larger)."
                    )
                }).encode()
                self.wfile.write(error_body)
                print(
                    f"  [reject] prompt ({prompt_tokens} tokens) exceeds "
                    f"ctx_size ({ctx_size}) — tell user to increase --ctx-size"
                )
                return

            try:
                sig, config, handle, exploring = orch.route(prompt_tokens, expected_gen_tokens)
                tag = "EXPLORE" if exploring else "EXPLOIT"
                print(
                    f"\n[Proxy] Routed {prompt_tokens} prompt / {expected_gen_tokens} gen "
                    f"to port {handle.port} (decision: {tag}, config: "
                    f"{config.n_gpu_layers} GPU / {config.n_cpu_moe} CPU)"
                )

                target_url = f"http://127.0.0.1:{handle.port}{self.path}"
                headers = {
                    k: v for k, v in self.headers.items()
                    if k.lower() not in ("host", "content-length")
                }

                with httpx.Client(timeout=None) as client:
                    if is_stream:
                        self._proxy_stream(client, target_url, post_data, headers, orch, sig, config, prompt_tokens)
                    else:
                        self._proxy_once(client, target_url, post_data, headers, orch, sig, config, prompt_tokens)

            except BrokenPipeError:
                print("Proxy warning: client disconnected mid-stream (BrokenPipe).")
            except Exception as e:
                print(f"Proxy error: {e}")
                try:
                    self.send_response(502)
                    self.end_headers()
                    self.wfile.write(f"Bad Gateway: {e}".encode())
                except Exception:
                    pass

        def _proxy_stream(self, client, target_url, post_data, headers, orch, sig, config, prompt_tokens):
            final_tok_per_sec = None
            with client.stream("POST", target_url, content=post_data, headers=headers) as response:
                self.send_response(response.status_code)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("X-Swiftlet-GPU-Layers", str(config.n_gpu_layers))
                self.send_header("X-Swiftlet-CPU-MoE", str(config.n_cpu_moe))
                self.send_header("X-Swiftlet-Prompt-Tokens", str(prompt_tokens))
                self.end_headers()

                # iter_lines() buffers internally and only yields complete
                # lines, so a JSON payload split across two TCP reads still
                # parses correctly — iter_raw() does NOT guarantee that, and
                # would intermittently fail to parse timings on truncated
                # chunk boundaries.
                for line in response.iter_lines():
                    self.wfile.write((line + "\n").encode("utf-8"))
                    self.wfile.flush()

                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            continue
                        try:
                            data = json.loads(data_str)
                            timings = data.get("timings", {})
                            if "predicted_per_second" in timings:
                                final_tok_per_sec = timings["predicted_per_second"]
                        except json.JSONDecodeError:
                            pass

            if final_tok_per_sec is not None:
                orch.record(sig, config, final_tok_per_sec)
                print(f"  Recorded {final_tok_per_sec:.2f} tok/s")
            else:
                print("  Warning: could not parse timings from stream.")
            
            # Post-response memory check event loop
            orch.enforce_memory_limit(threshold=85.0)

        def _proxy_once(self, client, target_url, post_data, headers, orch, sig, config, prompt_tokens):
            response = client.post(target_url, content=post_data, headers=headers)
            self.send_response(response.status_code)
            for k, v in response.headers.items():
                if k.lower() not in ("content-encoding", "content-length", "transfer-encoding"):
                    self.send_header(k, v)
            self.send_header("X-Swiftlet-GPU-Layers", str(config.n_gpu_layers))
            self.send_header("X-Swiftlet-CPU-MoE", str(config.n_cpu_moe))
            self.send_header("X-Swiftlet-Prompt-Tokens", str(prompt_tokens))
            self.end_headers()
            self.wfile.write(response.content)

            try:
                resp_json = response.json()
                timings = resp_json.get("timings", {})
                if "predicted_per_second" in timings:
                    tok_per_sec = timings["predicted_per_second"]
                    orch.record(sig, config, tok_per_sec)
                    print(f"  Recorded {tok_per_sec:.2f} tok/s")
            except Exception:
                print("  Warning: could not parse timings from response.")
                
            # Post-response memory check event loop
            orch.enforce_memory_limit(threshold=85.0)

        def log_message(self, format, *args):
            pass  # suppress BaseHTTPRequestHandler's default request logging; our own prints cover it

    return ProxyHandler

import psutil

def hunt_zombies(cache_dir: str):
    """
    Finds and kills any orphaned llama-server processes from previous crashed sessions
    that share our specific cache directory, freeing up GPU VRAM.
    """
    killed_count = 0
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            name = p.info.get('name', '')
            cmdline = p.info.get('cmdline', [])
            if name and 'llama-server' in name and cmdline:
                # Only kill it if it is using our specific cache directory
                if '--slot-save-path' in cmdline and cache_dir in cmdline:
                    p.kill()
                    killed_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
            
    if killed_count > 0:
        print(f"  [zombie hunter] 🔫 Terminated {killed_count} orphaned llama-server process(es). VRAM cleared.")

def serve(model_path: str, store_path: str, llama_server_bin: str, startup_timeout: int = 300, pool_size: int = 1, threads: int = 8, ctx_size: int = 8192):
    cache_dir = os.path.join(os.getcwd(), ".swiftlet_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Hunt and kill orphaned processes from prior crashed runs before allocating new servers
    hunt_zombies(cache_dir)
    
    store = LearnedConfigStore(store_path, seed=42)

    def real_launcher(config, port) -> ServerHandle:
        print(
            f"  [launching] {llama_server_bin} --n-gpu-layers {config.n_gpu_layers} "
            f"--n-cpu-moe {config.n_cpu_moe} --batch-size {config.batch_size} --threads {threads} --port {port}"
        )

        proc = subprocess.Popen([
            llama_server_bin,
            "-m", model_path,
            "--n-gpu-layers", str(config.n_gpu_layers),
            "--n-cpu-moe", str(config.n_cpu_moe),
            "--batch-size", str(config.batch_size),
            "--threads", str(threads),
            "-np", "1",
            "-c", str(ctx_size),
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0",
            "--slot-save-path", cache_dir,
            "--port", str(port),
        ])

        poll_interval = 2
        elapsed = 0

        while elapsed < startup_timeout:
            # Fail fast if the process already died, instead of waiting out
            # the full timeout on a crash — a dead process will never pass
            # the health check no matter how long you wait.
            if proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server on port {port} exited early (code {proc.returncode}) "
                    f"during startup — check its logs above for the actual error."
                )

            try:
                res = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
                if res.status_code == 200:
                    print(f"  [ready] port {port} came up after {elapsed}s")
                    break
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RequestError):
                pass

            time.sleep(poll_interval)
            elapsed += poll_interval
            if elapsed % 20 == 0:
                print(f"  [waiting] still loading on port {port}... ({elapsed}s elapsed)")
        else:
            proc.terminate()
            raise RuntimeError(
                f"llama-server on port {port} didn't come up within {startup_timeout}s. "
                f"If your model is very large, pass --startup-timeout with a bigger value."
            )

        return ServerHandle(config=config, port=port, started_at=time.time(), process=proc)

    pool = ServerPool(max_size=pool_size, launcher=real_launcher)
    orch = Orchestrator(store, pool)

    # Real cleanup now that ServerPool.shutdown() actually exists and
    # terminates every process it launched (see orchestrator.py).
    atexit.register(pool.shutdown)

    httpd = ThreadingHTTPServer(("", 8000), make_proxy_handler(orch, ctx_size))
    print("Starting swiftlet proxy on http://localhost:8000")
    print(f"Routing to {model_path} via {llama_server_bin}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        pool.shutdown()


def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="swiftlet — adaptive CPU/GPU config learning for llama.cpp")
    parser.add_argument("--model", default=os.getenv("MODEL_PATH"), help="Path to a GGUF model")
    parser.add_argument("--store", default=os.getenv("STORE_PATH", "swiftlet_learned_config.json"), help="Path to the learned config store")
    parser.add_argument("--demo", action="store_true", help="Run the routing/learning demo with a fake launcher")
    parser.add_argument("--llama-server", default=os.getenv("LLAMA_SERVER_PATH", "llama-server"), help="Path to the llama-server binary")
    parser.add_argument("--startup-timeout", type=int, default=int(os.getenv("STARTUP_TIMEOUT", 300)),
                         help="Seconds to wait for llama-server to become ready (default: 300)")
    parser.add_argument("--pool-size", type=int, default=int(os.getenv("POOL_SIZE", 1)),
                         help="Max concurrent llama-server instances (default: 1 — "
                              "each instance holds a FULL model copy in memory, so "
                              "on a 16GB Mac running a large model, keep this at 1 "
                              "unless you've confirmed you have headroom for more.")
    parser.add_argument("--threads", type=int, default=int(os.getenv("THREADS", 8)), help="CPU threads for llama-server")
    parser.add_argument("--ctx-size", type=int, default=int(os.getenv("CTX_SIZE", 8192)),
                         help="Context window size (default: 8192 — larger uses more memory, compensated by q8_0 KV cache)")

    args = parser.parse_args()
    
    if args.demo or not args.model:
        if not args.demo:
            print("No --model given — running --demo instead.\n")
        demo(args.store)
        return

    if args.model:
        serve(args.model, args.store, args.llama_server, args.startup_timeout, args.pool_size, args.threads, args.ctx_size)


if __name__ == "__main__":
    main()
