import argparse
import json
import sys
import time
import subprocess
import os
import psutil
from dotenv import load_dotenv

import httpx
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll, Vertical
from textual.widgets import Input, Markdown, Collapsible, Static

class UserTurn(Static):
    DEFAULT_CSS = """
    UserTurn {
        margin: 1 0;
        padding: 1;
        border: solid green;
    }
    """
    def __init__(self, content: str, **kwargs):
        super().__init__(**kwargs)
        self.content = content
        
    def compose(self) -> ComposeResult:
        yield Markdown(f"**You:**\n\n{self.content}")

class AITurn(Static):
    DEFAULT_CSS = """
    AITurn {
        margin: 1 0;
        padding: 1;
        border: solid blue;
    }
    .ai-header {
        text-style: bold;
        margin-bottom: 1;
    }
    Collapsible {
        color: $text-muted;
        text-style: italic;
    }
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reasoning = ""
        self.final_answer = ""
        self.has_reasoning = False
        self.has_started = False
        
        self.spinner_frames = ["", ".", "..", "...", "....", ".....", "......", ".......", "........"]
        self.frame_idx = 0
        self.header = Static("Swiftlet:", classes="ai-header")
        
        self.reasoning_widget = Markdown()
        self.content_widget = Markdown("")
        self.collapsible = Collapsible(self.reasoning_widget, title="thinking (in progress)")
        self.collapsible.collapsed = True
        self.collapsible.display = False
        
    def on_mount(self) -> None:
        self.animation_timer = self.set_interval(0.2, self.animate_header)
        
    def animate_header(self) -> None:
        if self.has_started:
            self.animation_timer.stop()
            self.header.update("[b]Swiftlet:[/b]")
        else:
            self.frame_idx = (self.frame_idx + 1) % len(self.spinner_frames)
            self.header.update(f"[b]Swiftlet:[/b] {self.spinner_frames[self.frame_idx]}")
            
    def compose(self) -> ComposeResult:
        yield self.header
        yield self.collapsible
        yield self.content_widget
        
    def update_reasoning(self, r_content: str):
        self.has_started = True
        if not self.has_reasoning:
            self.has_reasoning = True
            self.collapsible.display = True
            
        self.reasoning += r_content
        self.reasoning_widget.update(self.reasoning)
            
    def update_answer(self, c_content: str):
        self.has_started = True
        if self.has_reasoning and self.collapsible.title != "thinking (finished, click to expand)":
            self.collapsible.title = "thinking (finished, click to expand)"
            
        self.final_answer += c_content
        self.content_widget.update(self.final_answer)

class Sidebar(Static):
    DEFAULT_CSS = """
    Sidebar {
        border-left: solid blue;
        padding: 1;
    }
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.display_text = Static("[b]Hardware Dashboard[/b]\n\nWaiting...")
        
    def compose(self) -> ComposeResult:
        yield self.display_text
        
    def update_metrics(self, gpu_layers, cpu_moe, speed_tok_s, prompt_tokens, output_tokens, cpu_usage, is_thinking):
        status = "⠴ Thinking..." if is_thinking else "Generating"
        if not is_thinking and output_tokens == 0:
            status = "Idle"
            
        speed = f"{speed_tok_s:.2f}" if speed_tok_s else "---"
        gpu = gpu_layers if gpu_layers is not None else "---"
        cpu = cpu_moe if cpu_moe is not None else "---"
        p_tok = prompt_tokens if prompt_tokens is not None else "---"
        o_tok = output_tokens if output_tokens is not None else "---"
        
        content = f"""[b]Hardware Dashboard[/b]
        
[b]Status[/b]: {status}

[b]Routing[/b]
GPU Layers: {gpu}
CPU MoE Cores: {cpu}

[b]Performance[/b]
Tokens/sec: {speed}
Input Tokens: {p_tok}
Output Tokens: {o_tok}

[b]CPU Usage[/b]
"""
        if isinstance(cpu_usage, list):
            for i, p in enumerate(cpu_usage):
                color = "green"
                if p >= 85:
                    color = "red"
                elif p >= 50:
                    color = "yellow"
                content += f"Core {i:02d}: [{color}]{p:5.1f}%[/{color}]\n"
        else:
            content += f"Aggregate: {cpu_usage:.1f}%\n"
            
        self.display_text.update(content)

class SwiftletApp(App):
    CSS = """
    Screen {
        layout: horizontal;
    }
    #chat-pane {
        width: 3fr;
        height: 100%;
        padding: 0 1;
    }
    #sidebar {
        width: 26;
        height: 100%;
    }
    Input {
        dock: bottom;
        margin: 1;
    }
    """
    
    def __init__(self, endpoint: str):
        super().__init__()
        self.endpoint = endpoint
        self.chat_history = []
        
        self.gpu_layers = None
        self.cpu_moe = None
        self.speed = None
        self.prompt_tokens = None
        self.output_tokens = 0
        self.is_thinking = False
        self.cpu_usage = []

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="chat-pane")
        yield Sidebar(id="sidebar")
        yield Input(placeholder="Type a message and press Enter... ('quit' to exit)")

    def on_mount(self) -> None:
        psutil.cpu_percent(interval=None, percpu=True) # Prime baseline
        self.set_interval(0.5, self.update_cpu)
        self.update_sidebar()
        
    def update_cpu(self) -> None:
        self.cpu_usage = psutil.cpu_percent(interval=None, percpu=True)
        self.update_sidebar()
        
    def update_sidebar(self):
        sidebar = self.query_one(Sidebar)
        sidebar.update_metrics(
            self.gpu_layers, self.cpu_moe, self.speed,
            self.prompt_tokens, self.output_tokens, self.cpu_usage, self.is_thinking
        )

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        user_input = event.value.strip()
        if not user_input:
            return
            
        if user_input.lower() in ['quit', 'exit']:
            self.exit()
            return
            
        event.input.value = ""
        event.input.disabled = True
        
        self.chat_history.append({"role": "user", "content": user_input})
        
        chat_pane = self.query_one("#chat-pane", VerticalScroll)
        await chat_pane.mount(UserTurn(user_input))
        
        ai_turn = AITurn()
        await chat_pane.mount(ai_turn)
        chat_pane.scroll_end(animate=False)
        
        self.is_thinking = True
        self.output_tokens = 0
        self.speed = None
        self.update_sidebar()
        
        payload = {
            "messages": self.chat_history,
            "stream": True,
            "max_tokens": 4096,
            "chat_template_kwargs": {"enable_thinking": True}
        }
        
        self.stream_response(ai_turn, payload)

    @work(exclusive=True)
    async def stream_response(self, ai_turn: AITurn, payload: dict):
        try:
            start_time = None
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", self.endpoint, json=payload) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        ai_turn.update_answer(f"**Error {response.status_code}**: {error_text.decode()}")
                        self.is_thinking = False
                        self.update_sidebar()
                        return
                        
                    self.gpu_layers = response.headers.get("X-Swiftlet-GPU-Layers")
                    self.cpu_moe = response.headers.get("X-Swiftlet-CPU-MoE")
                    self.prompt_tokens = response.headers.get("X-Swiftlet-Prompt-Tokens")
                    
                    async for line in response.aiter_lines():
                        if line.startswith('data: '):
                            data_str = line[6:].strip()
                            if data_str == '[DONE]':
                                self.is_thinking = False
                                self.update_sidebar()
                                continue
                            try:
                                data = json.loads(data_str)
                                if 'choices' in data and len(data['choices']) > 0:
                                    delta = data['choices'][0].get('delta', {})
                                    
                                    r_content = delta.get('reasoning_content')
                                    if r_content:
                                        ai_turn.update_reasoning(r_content)
                                        self.output_tokens += 1
                                        if start_time is None: start_time = time.time()
                                        
                                    c_content = delta.get('content')
                                    if c_content:
                                        self.is_thinking = False
                                        ai_turn.update_answer(c_content)
                                        self.output_tokens += 1
                                        if start_time is None: start_time = time.time()
                                        
                                elif 'content' in data and data['content'] is not None:
                                    self.is_thinking = False
                                    ai_turn.update_answer(data['content'])
                                    self.output_tokens += 1
                                    if start_time is None: start_time = time.time()
                                    
                                if start_time is not None and self.output_tokens > 0:
                                    elapsed = time.time() - start_time
                                    if elapsed > 0:
                                        self.speed = self.output_tokens / elapsed
                                        
                                if 'timings' in data:
                                    timings = data['timings']
                                    if 'predicted_per_second' in timings:
                                        self.speed = timings['predicted_per_second']
                                    if 'prompt_n' in timings:
                                        self.prompt_tokens = timings['prompt_n']
                                    if 'predicted_n' in timings:
                                        self.output_tokens = timings['predicted_n']
                                        
                                self.update_sidebar()
                                self.query_one("#chat-pane", VerticalScroll).scroll_end(animate=False)
                            except json.JSONDecodeError:
                                pass
            
            self.chat_history.append({"role": "assistant", "content": ai_turn.final_answer})
            
        except Exception as e:
            ai_turn.update_answer(f"\n**Connection Error**: {str(e)}")
            self.is_thinking = False
            self.update_sidebar()
        finally:
            input_widget = self.query_one(Input)
            input_widget.disabled = False
            input_widget.focus()

if __name__ == "__main__":
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Textual Interactive Chat CLI for swiftlet")
    parser.add_argument("--endpoint", default="http://localhost:8000/v1/chat/completions", help="Proxy endpoint URL")
    parser.add_argument("--model", type=str, default=os.getenv("MODEL_PATH"), help="Path to the GGUF model")
    parser.add_argument("--llama-server", type=str, default=os.getenv("LLAMA_SERVER_PATH", "/Applications/Ollama.app/Contents/Resources/llama-server"), help="Path to llama-server binary")
    parser.add_argument("--ctx-size", type=int, default=int(os.getenv("CTX_SIZE", 8192)), help="Context size for the model")
    args = parser.parse_args()
    
    proxy_proc = None
    if args.model:
        print("Starting Swiftlet proxy server in background...")
        proxy_log = open("swiftlet_proxy.log", "w")
        proxy_proc = subprocess.Popen([
            sys.executable, "-m", "swiftlet.cli",
            "--model", os.path.expanduser(args.model),
            "--llama-server", os.path.expanduser(args.llama_server),
            "--ctx-size", str(args.ctx_size)
        ], stdout=proxy_log, stderr=proxy_log)
        time.sleep(2)  # Wait for proxy to bind port 8000
        
        if proxy_proc.poll() is not None:
            print(f"\nProxy exited immediately (code {proxy_proc.returncode}) — check the model/llama-server paths above.")
            with open("swiftlet_proxy.log", "r") as f:
                lines = f.readlines()
                for line in lines[-15:]:
                    print(line.rstrip())
            sys.exit(1)

    try:
        app = SwiftletApp(args.endpoint)
        app.run()
    finally:
        if proxy_proc:
            print("\nShutting down proxy server...")
            proxy_proc.terminate()
            proxy_proc.wait()
