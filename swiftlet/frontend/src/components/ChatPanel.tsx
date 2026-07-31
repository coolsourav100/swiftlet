import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { API, Message } from '../api';

interface ChatPanelProps {
  onStatusChange: (status: 'connected' | 'disconnected') => void;
  onNewRequest: (tag: string, explore: boolean, tps: number | null, pTokens: number, cTokens: number) => void;
  onTokensUpdate?: (prompt: number, completion: number) => void;
}

export function ChatPanel({ onStatusChange, onNewRequest, onTokensUpdate }: ChatPanelProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [connected, setConnected] = useState(false);
  
  // Streaming state for the current AI response
  const [currentReasoning, setCurrentReasoning] = useState('');
  const [currentContent, setCurrentContent] = useState('');
  const [routingTag, setRoutingTag] = useState('');
  
  const [thinkStartTime, setThinkStartTime] = useState<number | null>(null);
  const [thinkDuration, setThinkDuration] = useState<number>(0);
  
  // Timer effect
  useEffect(() => {
    let interval: any;
    if (isStreaming && !currentContent && (currentReasoning || thinkStartTime)) {
      if (!thinkStartTime) setThinkStartTime(Date.now());
      interval = setInterval(() => {
        setThinkDuration((Date.now() - (thinkStartTime || Date.now())) / 1000);
      }, 100);
    } else if (currentContent) {
      if (interval) clearInterval(interval);
    }
    return () => { if (interval) clearInterval(interval); };
  }, [isStreaming, currentReasoning, currentContent, thinkStartTime]);

  const scrollContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollContainerRef.current) {
      scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
    }
  }, [messages, currentReasoning, currentContent]);

  const handleSend = async () => {
    if (!input.trim() || isStreaming) return;
    
    const userMsg: Message = { role: 'user', content: input.trim() };
    const newMessages = [...messages, userMsg];
    
    setMessages(newMessages);
    setInput('');
    setIsStreaming(true);
    setCurrentReasoning('');
    setCurrentContent('');
    setRoutingTag('');
    setThinkStartTime(null);
    setThinkDuration(0);

    let tps: number | null = null;
    let localReasoning = '';
    let localContent = '';
    let finalTag = '';
    let pTokens = 0;
    let cTokens = 0;

    await API.chat(
      newMessages,
      (delta) => {
        let tokensUpdated = false;
        if (delta.reasoning_content) {
          localReasoning += delta.reasoning_content;
          setCurrentReasoning(localReasoning);
          cTokens += 1;
          tokensUpdated = true;
        }
        if (delta.content) {
          localContent += delta.content;
          setCurrentContent(localContent);
          cTokens += 1;
          tokensUpdated = true;
        }
        if (delta._tps) {
          tps = delta._tps;
        }
        if (delta._usage) {
          pTokens = delta._usage.prompt_tokens;
          cTokens = delta._usage.completion_tokens;
          tokensUpdated = true;
        }
        if (tokensUpdated) {
          onTokensUpdate?.(pTokens, cTokens);
        }
      },
      () => {
        setMessages([...newMessages, { 
          role: 'assistant', 
          content: localContent, 
          reasoning: localReasoning,
          prompt_tokens: pTokens,
          completion_tokens: cTokens
        }]);
        setIsStreaming(false);
        setCurrentContent('');
        setCurrentReasoning('');
        onNewRequest(finalTag || routingTag, (finalTag || routingTag).includes('EXPLORE'), tps, pTokens, cTokens);
      },
      (headers) => {
        setConnected(true);
        onStatusChange('connected');
        const tag = `GPU ${headers.gpuLayers || '?'} / MoE ${headers.cpuMoe || '?'}`;
        finalTag = tag;
        setRoutingTag(tag);
        if (headers.promptTokens) {
          pTokens = parseInt(headers.promptTokens, 10);
          onTokensUpdate?.(pTokens, cTokens);
        }
      }
    );
  };

  return (
    <div className="flex flex-col h-full overflow-hidden relative min-w-0 bg-surface-container rounded-lg border border-outline-variant">
      {/* Chat Header */}
      <div className="px-3 py-1.5 border-b border-outline-variant flex justify-between items-center bg-surface-container-low shrink-0">
        <h2 className="font-label-caps text-[10px] text-on-surface-variant uppercase tracking-widest">Swiftlet Chat</h2>
        <div className="flex items-center gap-1.5">
          <span className={`w-1.5 h-1.5 rounded-full ${connected ? 'bg-primary-container' : 'bg-error'}`}></span>
          <span className={`font-label-sm text-[9px] ${connected ? 'text-primary-container' : 'text-error'}`}>
            {connected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>

      <div 
        ref={scrollContainerRef}
        className="flex-1 p-3 overflow-y-auto overflow-x-hidden flex flex-col gap-3 min-w-0 scrollbar-thin"
      >
        {messages.map((msg, i) => (
          <div key={i} className={`flex w-full ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            {msg.role === 'user' ? (
              <div className="text-primary-container py-1 px-2 border border-outline-variant/20 rounded text-[12px] font-['Roboto'] font-medium tracking-wide leading-relaxed whitespace-pre-wrap max-w-[90%] text-right shadow-sm">
                {msg.content}
              </div>
            ) : (
              <div className="py-0.5 flex flex-col gap-1 min-w-0 max-w-full w-full">
                {msg.reasoning && (
                  <details className="group flex flex-col gap-1 text-on-surface-variant">
                    <summary className="flex items-center gap-1 cursor-pointer list-none [&::-webkit-details-marker]:hidden outline-none">
                      <span className="material-symbols-outlined text-[14px] text-on-surface-variant/50 transition-transform group-open:rotate-90">chevron_right</span>
                      <span className="material-symbols-outlined text-[12px] text-primary-container">psychology</span>
                      <span className="text-[9px] font-semibold text-primary-container/80 uppercase tracking-widest">Thinking Process</span>
                    </summary>
                    <div className="text-[9px] leading-relaxed font-mono text-on-surface-variant pl-6 opacity-70 border-l border-outline-variant/30 ml-1.5 mt-1 whitespace-pre-wrap max-h-48 overflow-y-auto scrollbar-thin">
                      {msg.reasoning}
                    </div>
                  </details>
                )}
                <div className="text-[12px] font-['Roboto'] font-light tracking-wide leading-relaxed text-gray-300 prose prose-invert max-w-none prose-p:my-0.5 prose-pre:my-1 prose-pre:p-1.5 prose-pre:text-[9px] prose-headings:my-1 prose-a:text-primary-container prose-p:text-gray-300 prose-li:text-gray-300">
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                </div>
              </div>
            )}
          </div>
        ))}

        {/* Streaming Placeholder */}
        {isStreaming && (
          <div className="flex w-full justify-start">
            <div className="py-0.5 flex flex-col gap-1 min-w-0 max-w-full w-full">
              {(currentReasoning || thinkStartTime) && (
                <details open className="group flex flex-col gap-1 text-on-surface-variant">
                  <summary className="flex items-center gap-1 cursor-pointer list-none [&::-webkit-details-marker]:hidden outline-none">
                    <span className="material-symbols-outlined text-[14px] text-on-surface-variant/50 transition-transform group-open:rotate-90">chevron_right</span>
                    <span className="material-symbols-outlined text-[12px] text-primary-container animate-spin" style={{ animationDuration: '2s' }}>sync</span>
                    <div className="flex items-center gap-1 font-label-sm text-[9px] tracking-widest text-primary-container/80 uppercase">
                      <span className="animate-pulse">Thinking...</span>
                      <span className="font-label-sm text-tertiary-container opacity-80">[{thinkDuration.toFixed(1)}s]</span>
                    </div>
                  </summary>
                  {currentReasoning && (
                    <div className="text-[9px] leading-relaxed font-mono text-on-surface-variant pl-6 opacity-70 border-l border-outline-variant/30 ml-1.5 mt-1 whitespace-pre-wrap max-h-48 overflow-y-auto scrollbar-thin">
                      {currentReasoning}
                    </div>
                  )}
                </details>
              )}
              {currentContent && (
                <div className="text-[12px] font-['Roboto'] font-light tracking-wide leading-relaxed text-gray-300 prose prose-invert max-w-none prose-p:my-0.5 prose-pre:my-1 prose-pre:p-1.5 prose-pre:text-[9px] prose-headings:my-1 prose-a:text-primary-container prose-p:text-gray-300 prose-li:text-gray-300">
                  <ReactMarkdown>{currentContent}</ReactMarkdown>
                </div>
              )}
              {!currentContent && !currentReasoning && !thinkStartTime && (
                <div className="flex gap-1 py-1">
                  <div className="w-1.5 h-1.5 bg-primary-container rounded-full animate-bounce"></div>
                  <div className="w-1.5 h-1.5 bg-primary-container rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                  <div className="w-1.5 h-1.5 bg-primary-container rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Chat Input Area */}
      <div className="p-2 border-t border-outline-variant bg-surface-container-low shrink-0">
        <div className="flex gap-2">
          <input 
            type="text"
            className="flex-1 bg-surface-container-highest border border-outline-variant rounded-md px-3 py-1.5 text-on-surface focus:outline-none focus:border-primary-container focus:ring-1 focus:ring-primary-container font-body-md text-[11px] transition-all placeholder-on-surface-variant/50"
            placeholder="Type a message..."
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSend()}
            disabled={isStreaming}
          />
          <button 
            onClick={handleSend}
            disabled={isStreaming || !input.trim()}
            className="bg-surface-container hover:bg-surface-bright text-on-surface p-1.5 rounded-md border border-outline-variant transition-colors flex items-center justify-center w-8 h-8 disabled:opacity-50"
          >
            <span className="material-symbols-outlined text-[16px]">send</span>
          </button>
        </div>
      </div>
    </div>
  );
}
