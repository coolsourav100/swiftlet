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
import psutil
from dotenv import load_dotenv
from pathlib import Path

from .classifier import classify
from .config_store import LearnedConfigStore, BayesianConfigStore
from .orchestrator import Orchestrator, ServerPool, ServerHandle
from .logging_config import setup_logging, get_logger
from .toml_config import load_config, generate_default_toml

from .backends.llamacpp import LlamaCppLauncher
from .backends.mlx import MLXLauncher
from .backends.ollama import OllamaLauncher
from .web_tools import build_web_context


def demo(store_path: str, learning_type: str = "eps-greedy"):
    """
    Runs a handful of representative requests through the real classifier +
    config store + pool logic, using a fake launcher (no actual model load),
    so you can see the routing decisions and learning behavior directly.
    """
    log = get_logger("demo")

    if learning_type == "bayesian":
        store = BayesianConfigStore(store_path, seed=42)
    else:
        store = LearnedConfigStore(store_path, seed=42)

    def fake_launcher(config, port):
        log.info(f"  [would launch] llama-server --n-gpu-layers {config.n_gpu_layers} "
              f"--n-cpu-moe {config.n_cpu_moe} --batch-size {config.batch_size} --port {port}")
        return ServerHandle(config=config, port=port, started_at=0.0)

    pool = ServerPool(max_size=3, launcher=fake_launcher)
    orch = Orchestrator(store, pool)

    scenarios = [
        ("short chat turn", 80, 150),
        ("long document Q&A", 4000, 100),
        ("open-ended long generation", 100, 1500),
        ("short chat turn (again)", 90, 140),
        ("code refactoring", 2000, 800),
        ("quick question", 50, 60),
        ("deep analysis", 3000, 2000),
        ("follow-up chat", 120, 200),
    ]

    for name, prompt_tokens, gen_tokens in scenarios:
        sig, config, handle, exploring = orch.route(prompt_tokens, gen_tokens)
        tag = "EXPLORE" if exploring else "EXPLOIT"
        log.info(f"\n[{name}] prompt={prompt_tokens} gen={gen_tokens} -> signature={sig} phase={sig.phase.value}")
        log.info(f"  decision: {tag} -> n_gpu_layers={config.n_gpu_layers} n_cpu_moe={config.n_cpu_moe}")

        # Simulated measurement — replace with a real tok/s reading from
        # llama-server's response once wired up to an actual instance.
        fake_measured_tps = 20 + config.n_cpu_moe * 0.3
        orch.record(sig, config, fake_measured_tps)
        log.info(f"  recorded (simulated) {fake_measured_tps:.1f} tok/s")

    log.info(f"\nLearned store saved to: {store_path}")


def make_proxy_handler(orch: Orchestrator, ctx_size: int = 8192, web_search: bool = False):
    log = get_logger("proxy")

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

        # ── NEW: Web UI + Dashboard API ────────────────────────

        def do_GET(self):
            if self.path in ('/', '/ui', '/ui.html'):
                self._serve_ui()
            elif self.path == '/api/state':
                self._serve_state()
            elif self.path == '/api/export-config':
                self._serve_export()
            else:
                self.send_response(404)
                self.end_headers()

        def _serve_ui(self):
            """Serve the single-page Swiftlet UI."""
            ui_path = Path(__file__).parent / "ui.html"
            if not ui_path.exists():
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"ui.html not found")
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(ui_path.read_bytes())

        def _serve_state(self):
            """Return the learned config store as JSON for the dashboard."""
            from .config_store import FULL_CONFIG_SPACE

            store = orch.store
            signatures = {}

            for sig_str, configs in store._data.items():
                signatures[sig_str] = {}
                for cfg_key, stats in configs.items():
                    signatures[sig_str][cfg_key] = {
                        "config": {
                            "n_gpu_layers": stats.config.n_gpu_layers,
                            "n_cpu_moe": stats.config.n_cpu_moe,
                            "batch_size": stats.config.batch_size,
                        },
                        "trials": stats.trials,
                        "total_tok_per_sec": stats.total_tok_per_sec,
                        "mean_tok_per_sec": stats.mean_tok_per_sec,
                    }

            # Config space for the matrix columns
            config_space = [
                {
                    "n_gpu_layers": c.n_gpu_layers,
                    "n_cpu_moe": c.n_cpu_moe,
                    "batch_size": c.batch_size,
                    "key": c.key(),
                }
                for c in FULL_CONFIG_SPACE
            ]

            # Active servers
            active = []
            for handle in orch.pool._pool.values():
                active.append({
                    "port": handle.port,
                    "config": {
                        "n_gpu_layers": handle.config.n_gpu_layers,
                        "n_cpu_moe": handle.config.n_cpu_moe,
                        "batch_size": handle.config.batch_size,
                    },
                    "backend": getattr(handle, "backend_name", "llama.cpp"),
                    "uptime_s": round(time.time() - handle.started_at, 0),
                })

            body = json.dumps({
                "signatures": signatures,
                "config_space": config_space,
                "active_servers": active,
                "ctx_size": ctx_size,
                "last_signature": getattr(orch, "_last_signature", None),
                "last_config_key": getattr(orch, "_last_config_key", None),
                "cpu_usage": psutil.cpu_percent(percpu=True)
            })

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())

        def _serve_export(self):
            """Export the learned config store as a downloadable JSON file with hardware metadata."""
            import platform
            
            store = orch.store
            export_data = {
                "swiftlet_version": "2.0.1",
                "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "hardware": {
                    "machine": platform.machine(),
                    "system": platform.system(),
                    "processor": platform.processor(),
                    "ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
                },
                "store_data": {},
            }

            for sig_str, configs in store._data.items():
                export_data["store_data"][sig_str] = {}
                for cfg_key, stats in configs.items():
                    export_data["store_data"][sig_str][cfg_key] = {
                        "config": {
                            "n_gpu_layers": stats.config.n_gpu_layers,
                            "n_cpu_moe": stats.config.n_cpu_moe,
                            "batch_size": stats.config.batch_size,
                        },
                        "trials": stats.trials,
                        "total_tok_per_sec": stats.total_tok_per_sec,
                        "observations": stats.observations,
                    }

            body = json.dumps(export_data, indent=2)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Disposition", 'attachment; filename="swiftlet_profile.json"')
            self.end_headers()
            self.wfile.write(body.encode())

        # ── Existing POST handler (unchanged) ──────────────────

        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)

            # Handle import endpoint
            if self.path == '/api/import-config':
                self._handle_import(post_data)
                return

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

            # ── Web Context Injection ────────────────────────────────
            # Enabled globally (--web-search) OR per-request (frontend toggle).
            # The LLM never touches the internet — only the proxy does.
            request_web_search = self.headers.get("X-Swiftlet-Web-Search", "").lower() == "true"
            if web_search or request_web_search:
                messages = body.get("messages", [])
                last_user_msg = ""
                for m in reversed(messages):
                    if m.get("role") == "user":
                        last_user_msg = m.get("content", "")
                        break
                
                if last_user_msg:
                    web_context = build_web_context(last_user_msg, enable_search=True)
                    if web_context:
                        # Inject as a system message at position 0
                        body.setdefault("messages", [])
                        body["messages"].insert(0, {
                            "role": "system",
                            "content": web_context,
                        })
                        post_data = json.dumps(body).encode()
                        log.info(f"[web] Injected web context for: {last_user_msg[:60]}")
            # ─────────────────────────────────────────────────────────

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
                log.warning(
                    f"[reject] prompt ({prompt_tokens} tokens) exceeds "
                    f"ctx_size ({ctx_size}) — tell user to increase --ctx-size"
                )
                return

            try:
                sig, config, handle, exploring = orch.route(prompt_tokens, expected_gen_tokens)
                orch._last_signature = str(sig)
                orch._last_config_key = config.key()
                
                tag = "EXPLORE" if exploring else "EXPLOIT"
                log.info(
                    f"Routed {prompt_tokens} prompt / {expected_gen_tokens} gen "
                    f"to port {handle.port} (decision: {tag}, config: "
                    f"{config.n_gpu_layers} GPU / {config.n_cpu_moe} CPU)"
                )

                target_url = f"http://127.0.0.1:{handle.port}{self.path}"

                # Structured output: forward response_format as grammar if present
                if isinstance(body.get('response_format'), dict):
                    fmt = body['response_format']
                    if fmt.get('type') == 'json_object':
                        body['grammar'] = 'root ::= "{" [^}]* "}"'
                        post_data = json.dumps(body).encode()
                    elif fmt.get('type') == 'json_schema' and 'json_schema' in fmt:
                        # Pass through the schema for backends that support it
                        body['json_schema'] = fmt['json_schema']
                        post_data = json.dumps(body).encode()

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
                log.warning("Client disconnected mid-stream (BrokenPipe).")
            except Exception as e:
                log.error(f"Proxy error: {e}")
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
                log.info(f"Recorded {final_tok_per_sec:.2f} tok/s")
            else:
                log.warning("Could not parse timings from stream.")
            
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
                    log.info(f"Recorded {tok_per_sec:.2f} tok/s")
            except Exception:
                log.warning("Could not parse timings from response.")
                
            # Post-response memory check event loop
            orch.enforce_memory_limit(threshold=85.0)

        def _handle_import(self, post_data: bytes):
            """Import a learned config profile from uploaded JSON."""
            try:
                import_data = json.loads(post_data)
                store_data = import_data.get("store_data", import_data)
                
                from .config_store import EngineConfig, ConfigStats
                
                imported_count = 0
                for sig_str, configs in store_data.items():
                    if sig_str.startswith("_") or sig_str in ("swiftlet_version", "exported_at", "hardware"):
                        continue
                    orch.store._data.setdefault(sig_str, {})
                    for cfg_key, stats in configs.items():
                        cfg = EngineConfig(**stats["config"])
                        orch.store._data[sig_str][cfg_key] = ConfigStats(
                            config=cfg,
                            trials=stats["trials"],
                            total_tok_per_sec=stats["total_tok_per_sec"],
                            observations=stats.get("observations", []),
                        )
                        imported_count += 1
                
                orch.store.save()
                
                body = json.dumps({"status": "ok", "imported": imported_count})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body.encode())
                log.info(f"Imported {imported_count} config entries")
                
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
                log.error(f"Import failed: {e}")

        def log_message(self, format, *args):
            pass  # suppress BaseHTTPRequestHandler's default request logging

    return ProxyHandler


def hunt_zombies(cache_dir: str):
    """
    Finds and kills any orphaned llama-server processes from previous crashed sessions
    that share our specific cache directory, freeing up GPU VRAM.
    """
    log = get_logger("cleanup")
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
        log.info(f"[zombie hunter] 🔫 Terminated {killed_count} orphaned llama-server process(es). VRAM cleared.")

def serve(model_path: str, store_path: str, backend_type: str, learning_type: str, llama_server_bin: str, startup_timeout: int = 300, pool_size: int = 1, threads: int = 8, ctx_size: int = 8192, log_level: str = "INFO", log_dir: str = "", web_search: bool = False):
    setup_logging(level=log_level, log_dir=log_dir or None)
    log = get_logger("server")

    cache_dir = os.path.join(os.getcwd(), ".swiftlet_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    hunt_zombies(cache_dir)
    
    if learning_type == "bayesian":
        store = BayesianConfigStore(store_path, seed=42)
    else:
        store = LearnedConfigStore(store_path, seed=42)

    if backend_type == "ollama":
        backend = OllamaLauncher(model_path=model_path, startup_timeout=startup_timeout)
    elif backend_type == "mlx":
        backend = MLXLauncher(model_path=model_path, startup_timeout=startup_timeout)
    else:
        backend = LlamaCppLauncher(
            model_path=model_path, 
            bin_path=llama_server_bin, 
            ctx_size=ctx_size, 
            threads=threads, 
            startup_timeout=startup_timeout, 
            cache_dir=cache_dir
        )

    def real_launcher(config, port) -> ServerHandle:
        bh = backend.launch(config, port)
        return ServerHandle(config=bh.config, port=bh.port, started_at=bh.started_at, process=bh.process)

    pool = ServerPool(max_size=pool_size, launcher=real_launcher)
    orch = Orchestrator(store, pool)

    atexit.register(pool.shutdown)

    httpd = ThreadingHTTPServer(("", 8000), make_proxy_handler(orch, ctx_size, web_search=web_search))
    print(r"""
  ███████╗██╗    ██╗██╗███████╗████████╗██╗     ███████╗████████╗
  ██╔════╝██║    ██║██║██╔════╝╚══██╔══╝██║     ██╔════╝╚══██╔══╝
  ███████╗██║ █╗ ██║██║█████╗     ██║   ██║     █████╗     ██║   
  ╚════██║██║███╗██║██║██╔══╝     ██║   ██║     ██╔══╝     ██║   
  ███████║╚███╔███╔╝██║██║        ██║   ███████╗███████╗   ██║   
  ╚══════╝ ╚══╝╚══╝ ╚═╝╚═╝        ╚═╝   ╚══════╝╚══════╝   ╚═╝   

  v2.1.0

  ⏳ Starting server...
""")
    log.info("✔ Swiftlet is running!")
    log.info(f"  Dashboard:  http://localhost:8000")
    log.info(f"  API Base:   http://localhost:8000/v1")
    log.info(f"  Backend:    {backend_type}")
    log.info(f"  Learning:   {learning_type}")
    log.info(f"  Model:      {model_path}")
    log.info(f"  CTX Size:   {ctx_size}")
    log.info(f"  Web Search: {'ON \U0001F310' if web_search else 'OFF'}")
    log.info(f"  Press Ctrl+C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down...")
        pool.shutdown()


def main():
    load_dotenv()
    
    # Load TOML config as defaults
    toml_cfg = load_config()
    model_cfg = toml_cfg.get("model", {})
    server_cfg = toml_cfg.get("server", {})
    learn_cfg = toml_cfg.get("learning", {})
    log_cfg = toml_cfg.get("logging", {})
    
    parser = argparse.ArgumentParser(description="swiftlet — adaptive CPU/GPU config learning for llama.cpp")
    parser.add_argument("--model", default=os.getenv("MODEL_PATH", model_cfg.get("path", "")), help="Path to a GGUF model (or Ollama model name)")
    parser.add_argument("--backend", default=os.getenv("BACKEND", server_cfg.get("backend", "llamacpp")), choices=["llamacpp", "mlx", "ollama"], help="Inference backend (default: llamacpp)")
    parser.add_argument("--learning", default=os.getenv("LEARNING", learn_cfg.get("algorithm", "eps-greedy")), choices=["eps-greedy", "bayesian"], help="Learning algorithm for the config store")
    parser.add_argument("--store", default=os.getenv("STORE_PATH", learn_cfg.get("store_path", "swiftlet_learned_config.json")), help="Path to the learned config store")
    parser.add_argument("--demo", action="store_true", help="Run the routing/learning demo with a fake launcher")
    parser.add_argument("--init-config", action="store_true", help="Generate a default swiftlet.toml config file")
    parser.add_argument("--llama-server", default=os.getenv("LLAMA_SERVER_PATH", server_cfg.get("llama_server", "llama-server")), help="Path to the llama-server binary")
    parser.add_argument("--startup-timeout", type=int, default=int(os.getenv("STARTUP_TIMEOUT", server_cfg.get("startup_timeout", 300))),
                         help="Seconds to wait for llama-server to become ready (default: 300)")
    parser.add_argument("--pool-size", type=int, default=int(os.getenv("POOL_SIZE", server_cfg.get("pool_size", 1))),
                         help="Max concurrent llama-server instances (default: 1)")
    parser.add_argument("--threads", type=int, default=int(os.getenv("THREADS", server_cfg.get("threads", 8))), help="CPU threads for llama-server")
    parser.add_argument("--ctx-size", type=int, default=int(os.getenv("CTX_SIZE", model_cfg.get("ctx_size", 8192))),
                         help="Context window size (default: 8192)")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", log_cfg.get("level", "INFO")), 
                         choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    parser.add_argument("--log-dir", default=os.getenv("LOG_DIR", log_cfg.get("log_dir", "")), help="Directory for log files")
    parser.add_argument("--web-search", action="store_true", default=os.getenv("WEB_SEARCH", "").lower() in ("1", "true", "yes"),
                         help="Enable privacy-safe web search (DuckDuckGo) for current info")

    args = parser.parse_args()
    
    if args.init_config:
        config_path = Path.cwd() / "swiftlet.toml"
        if config_path.exists():
            print(f"swiftlet.toml already exists at {config_path}")
        else:
            config_path.write_text(generate_default_toml())
            print(f"Created {config_path}")
        return
    
    if args.demo or not args.model:
        if not args.demo:
            print("No --model given — running --demo instead.\n")
        setup_logging(level=args.log_level)
        demo(args.store, args.learning)
        return

    if args.model:
        serve(
            model_path=os.path.expanduser(args.model), 
            store_path=args.store, 
            backend_type=args.backend,
            learning_type=args.learning,
            llama_server_bin=os.path.expanduser(args.llama_server), 
            startup_timeout=args.startup_timeout, 
            pool_size=args.pool_size, 
            threads=args.threads, 
            ctx_size=args.ctx_size,
            log_level=args.log_level,
            log_dir=args.log_dir,
            web_search=args.web_search,
        )


if __name__ == "__main__":
    main()
