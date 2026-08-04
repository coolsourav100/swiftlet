import { useState, useEffect } from 'react';

interface SettingsModalProps {
  open: boolean;
  onClose: () => void;
  systemPrompt: string;
  onSystemPromptChange: (prompt: string) => void;
  webSearchEnabled: boolean;
  onWebSearchChange: (enabled: boolean) => void;
}

const PRESETS: { label: string; prompt: string }[] = [
  { label: 'Default', prompt: 'You are a helpful AI assistant.' },
  { label: 'Senior Dev', prompt: 'You are a senior software engineer. Write clean, well-documented code. Explain your reasoning concisely.' },
  { label: 'Concise', prompt: 'Be extremely concise. No filler. Answer in as few words as possible.' },
  { label: 'Teacher', prompt: 'You are a patient teacher. Explain concepts step-by-step with examples. Ask if the student needs clarification.' },
];

export function SettingsModal({ open, onClose, systemPrompt, onSystemPromptChange, webSearchEnabled, onWebSearchChange }: SettingsModalProps) {
  const [draft, setDraft] = useState(systemPrompt);
  const [webSearch, setWebSearch] = useState(webSearchEnabled);

  useEffect(() => { setDraft(systemPrompt); setWebSearch(webSearchEnabled); }, [systemPrompt, webSearchEnabled, open]);

  if (!open) return null;

  const handleSave = () => {
    onSystemPromptChange(draft);
    onWebSearchChange(webSearch);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div
        className="bg-surface-container border border-outline-variant rounded-xl w-full max-w-lg mx-4 shadow-2xl animate-in fade-in zoom-in-95 duration-200"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-outline-variant">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-[16px] text-primary-container">settings</span>
            <h2 className="font-label-caps text-[11px] text-on-surface uppercase tracking-widest">Settings</h2>
          </div>
          <button onClick={onClose} className="text-on-surface-variant hover:text-on-surface transition-colors p-1 rounded">
            <span className="material-symbols-outlined text-[16px]">close</span>
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-3">
          <div>
            <label className="font-label-caps text-[9px] text-on-surface-variant uppercase tracking-widest block mb-1.5">
              System Prompt
            </label>
            <textarea
              className="w-full h-28 bg-surface-container-highest border border-outline-variant rounded-md px-3 py-2 text-on-surface text-[11px] font-['Roboto'] focus:outline-none focus:border-primary-container focus:ring-1 focus:ring-primary-container resize-none placeholder-on-surface-variant/50 scrollbar-thin"
              placeholder="e.g. You are a helpful AI assistant..."
              value={draft}
              onChange={e => setDraft(e.target.value)}
            />
          </div>

          {/* Presets */}
          <div>
            <label className="font-label-caps text-[9px] text-on-surface-variant uppercase tracking-widest block mb-1.5">
              Quick Presets
            </label>
            <div className="flex flex-wrap gap-1.5">
              {PRESETS.map(p => (
                <button
                  key={p.label}
                  onClick={() => setDraft(p.prompt)}
                  className={`px-2 py-1 rounded text-[9px] font-label-caps border transition-all ${
                    draft === p.prompt
                      ? 'bg-primary-container/20 text-primary-container border-primary-container/40'
                      : 'bg-surface-container-low text-on-surface-variant border-outline-variant/30 hover:border-primary-container/30'
                  }`}
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>

          {/* Web Search Toggle */}
          <div className="border-t border-outline-variant/50 pt-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-[16px] text-primary-container">travel_explore</span>
                <div>
                  <label className="font-label-caps text-[9px] text-on-surface uppercase tracking-widest block">
                    Web Search
                  </label>
                  <span className="text-[8px] text-on-surface-variant">
                    Privacy-safe · DuckDuckGo · No data leaks
                  </span>
                </div>
              </div>
              <button
                onClick={() => setWebSearch(!webSearch)}
                className={`relative w-10 h-5 rounded-full transition-colors duration-200 ${
                  webSearch
                    ? 'bg-primary-container/40'
                    : 'bg-surface-container-highest border border-outline-variant'
                }`}
              >
                <div
                  className={`absolute top-0.5 w-4 h-4 rounded-full transition-all duration-200 ${
                    webSearch
                      ? 'left-[22px] bg-primary-container shadow-md'
                      : 'left-0.5 bg-on-surface-variant/50'
                  }`}
                />
              </button>
            </div>
            {webSearch && (
              <div className="mt-2 px-0.5 py-1.5 bg-primary-container/5 border border-primary-container/15 rounded-md">
                <div className="flex items-start gap-1.5">
                  <span className="material-symbols-outlined text-[12px] text-primary-container mt-px">info</span>
                  <span className="text-[8px] text-on-surface-variant leading-relaxed">
                    When enabled, Swiftlet automatically detects questions about current events, 
                    time, news, or weather and searches DuckDuckGo for answers. 
                    Only the search query is sent — your conversation stays 100% local.
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-2 px-4 py-3 border-t border-outline-variant">
          <button
            onClick={onClose}
            className="px-3 py-1.5 text-[10px] font-label-caps text-on-surface-variant border border-outline-variant rounded-md hover:bg-surface-container-low transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="px-3 py-1.5 text-[10px] font-label-caps text-primary-container bg-primary-container/10 border border-primary-container/30 rounded-md hover:bg-primary-container/20 transition-colors"
          >
            Save
          </button>
        </div>
      </div>
    </div>
  );
}
