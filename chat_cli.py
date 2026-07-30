import argparse
import json
import sys
import threading
import time
import itertools
import httpx


def chat_loop(endpoint: str, threshold: float):
    print(f"Connecting to swiftlet proxy at {endpoint}...")
    print(f"Performance threshold set to {threshold} tok/s.")
    print("Type 'quit' or 'exit' to stop.\n")

    messages = []
    
    with httpx.Client(timeout=None) as client:
        while True:
            try:
                user_input = input("You: ")
                if user_input.strip().lower() in ['quit', 'exit']:
                    break
                if not user_input.strip():
                    continue

                messages.append({"role": "user", "content": user_input})
                
                payload = {
                    "messages": messages,
                    "stream": True,
                    "n_predict": 2048,
                    "chat_template_kwargs": {"enable_thinking": False}
                }

                stop_event = threading.Event()
                def animate():
                    spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
                    while not stop_event.is_set():
                        sys.stdout.write(f"\rAI: \033[90mThinking {next(spinner)}\033[0m")
                        sys.stdout.flush()
                        time.sleep(0.1)
                    sys.stdout.write("\r\033[K")
                    sys.stdout.write("AI: ")
                    sys.stdout.flush()

                t = threading.Thread(target=animate)
                t.daemon = True
                t.start()

                final_tok_per_sec = None
                ai_content = ""
                
                try:
                    with client.stream("POST", endpoint, json=payload) as response:
                        stop_event.set()
                        t.join()
                        
                        if response.status_code != 200:
                            print(f"\n[Error: Proxy returned {response.status_code}]")
                            print(response.read().decode())
                            continue
                            
                        for line in response.iter_lines():
                            if line.startswith('data: '):
                                data_str = line[6:].strip()
                                if data_str == '[DONE]':
                                    continue
                                try:
                                    data = json.loads(data_str)
                                    # Standard OAI chat completion delta
                                    if 'choices' in data and len(data['choices']) > 0:
                                        delta = data['choices'][0].get('delta', {})
                                        
                                        # Print reasoning content if present (dimmed/italicized usually, but just bracketed for CLI)
                                        if 'reasoning_content' in delta and delta['reasoning_content'] is not None:
                                            r_content = delta['reasoning_content']
                                            print(f"\033[90m{r_content}\033[0m", end="", flush=True)
                                            
                                        if 'content' in delta and delta['content'] is not None:
                                            content = delta['content']
                                            ai_content += content
                                            print(content, end="", flush=True)
                                    # Standard completion text
                                    elif 'content' in data and data['content'] is not None:
                                        content = data['content']
                                        ai_content += content
                                        print(content, end="", flush=True)
                                        
                                    # Extract timings
                                    if 'timings' in data and 'predicted_per_second' in data['timings']:
                                        final_tok_per_sec = data['timings']['predicted_per_second']
                                except json.JSONDecodeError:
                                    pass
                except httpx.ConnectError:
                    stop_event.set()
                    t.join()
                    print("\n[Error: Could not connect to the proxy. Is it running on port 8000?]")
                    break
                except Exception as e:
                    stop_event.set()
                    t.join()
                    print(f"\n[Error: {e}]")
                    break
                
                print() # Newline after response
                messages.append({"role": "assistant", "content": ai_content})

                # Log metrics
                if final_tok_per_sec is not None:
                    status = ">= threshold (FAST)" if final_tok_per_sec >= threshold else "< threshold (SLOW)"
                    print(f"\n[Metrics] Speed: {final_tok_per_sec:.2f} tok/s ({status})")
                else:
                    print("\n[Metrics] Speed: Unknown (no timings received)")
                
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n[Interrupted]")
                break
            except EOFError:
                print()
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Chat CLI for swiftlet")
    parser.add_argument("--endpoint", default="http://localhost:8000/v1/chat/completions", help="Proxy endpoint URL")
    parser.add_argument("--threshold", type=float, default=20.0, help="Performance threshold in tok/s")
    args = parser.parse_args()
    
    chat_loop(args.endpoint, args.threshold)
