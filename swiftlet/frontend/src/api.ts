export interface Message {
  role: 'user' | 'assistant';
  content: string;
  reasoning?: string;
  prompt_tokens?: number;
  completion_tokens?: number;
}

export interface ConfigSpaceItem {
  n_gpu_layers: number;
  n_cpu_moe: number;
  batch_size: number;
  key: string;
}

export interface LearningState {
  signatures: Record<string, Record<string, any>>;
  config_space: ConfigSpaceItem[];
  active_servers: any[];
  ctx_size: number;
  last_signature: string | null;
  last_config_key: string | null;
  cpu_usage?: number[];
}

export const API = {
  async chat(
    messages: Message[], 
    onToken: (delta: any) => void, 
    onDone: () => void, 
    onHeaders?: (h: any) => void
  ) {
    const res = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'swiftlet',
        messages: messages.map(m => ({ role: m.role, content: m.content })),
        stream: true,
        max_tokens: 4096,
        chat_template_kwargs: { enable_thinking: true },
      }),
    });

    if (onHeaders) {
      onHeaders({
        gpuLayers: res.headers.get('X-Swiftlet-GPU-Layers'),
        cpuMoe: res.headers.get('X-Swiftlet-CPU-MoE'),
        promptTokens: res.headers.get('X-Swiftlet-Prompt-Tokens'),
      });
    }

    if (!res.body) return onDone();

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6).trim();
        if (data === '[DONE]') {
          onDone();
          return;
        }
        try {
          const json = JSON.parse(data);
          const delta = json.choices?.[0]?.delta;
          if (delta) onToken(delta);
          
          // Llama.cpp standard OpenAI usage block
          if (json.usage) {
            onToken({ _usage: json.usage });
          }
          // Llama.cpp timings block
          if (json.timings) {
            onToken({ 
              _tps: json.timings.predicted_per_second,
              _usage: { prompt_tokens: json.timings.prompt_n, completion_tokens: json.timings.predicted_n }
            });
          }
        } catch (e) {
          // ignore parse errors for partial chunks
        }
      }
    }
    onDone();
  },

  async fetchState(): Promise<LearningState | null> {
    try {
      const res = await fetch('/api/state');
      if (res.ok) return await res.json();
    } catch {}
    return null;
  }
};
