import { useMemo, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { LearningState, API } from '../api';
import { CpuActivityShader } from './CpuActivityShader';
import { Sparkline } from './Sparkline';

const SIG_LABELS: Record<string, { label: string; phase: string }> = {
  'prompt_b0_gen_b0': { label: 'Quick Q&A', phase: 'balanced' },
  'prompt_b0_gen_b1': { label: 'Short Chat', phase: 'balanced' },
  'prompt_b0_gen_b2': { label: 'Long Gen', phase: 'decode' },
  'prompt_b1_gen_b0': { label: 'Medium In', phase: 'balanced' },
  'prompt_b1_gen_b1': { label: 'Balanced', phase: 'balanced' },
  'prompt_b1_gen_b2': { label: 'Med+Gen', phase: 'decode' },
  'prompt_b2_gen_b0': { label: 'Doc Q&A', phase: 'prefill' },
  'prompt_b2_gen_b1': { label: 'Long+Gen', phase: 'balanced' },
  'prompt_b2_gen_b2': { label: 'Deep Analysis', phase: 'decode' },
};

interface DashboardPanelProps {
  state: LearningState | null;
  history: Array<{ tag: string; config: string; tps: number | null }>;
  stats: { totalReqs: number; exploreCount: number; tokens?: { prompt: number; completion: number } };
}

export function DashboardPanel({ state, history, stats }: DashboardPanelProps) {
  const importRef = useRef<HTMLInputElement>(null);

  const tpsHistory = useMemo(() => {
    return history.filter(h => h.tps !== null).map(h => h.tps!).reverse();
  }, [history]);

  const handleImport = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const result = await API.importConfig(file);
    if (result) {
      alert(`Imported ${result.imported} config entries. Dashboard will update on next poll.`);
    } else {
      alert('Import failed. Check the file format.');
    }
    e.target.value = '';
  };

  const [minTps, maxTps] = useMemo(() => {
    if (!state) return [0, 40];
    let min = Infinity;
    let max = -Infinity;
    
    Object.values(state.signatures).forEach(configs => {
      Object.values(configs).forEach((st: any) => {
        if (st.trials > 0) {
          const mean = st.total_tok_per_sec / st.trials;
          if (mean < min) min = mean;
          if (mean > max) max = mean;
        }
      });
    });
    
    if (min === Infinity) return [0, 40];
    return [min, max];
  }, [state]);

  const range = maxTps - minTps || 1;
  const lastTps = history[0]?.tps;

  return (
    <div className="flex flex-col h-full text-sm">
      {/* Dashboard Header */}
      <div className="px-3 py-1.5 border-b border-outline-variant flex justify-between items-center bg-surface-container shrink-0 rounded-t-lg">
        <h2 className="font-label-caps text-[10px] text-on-surface-variant uppercase tracking-widest">Learning Dashboard</h2>
        <div className="flex items-center gap-2">
          <button
            onClick={() => API.exportConfig()}
            title="Export learned profile"
            className="p-0.5 rounded hover:bg-surface-container-highest text-on-surface-variant hover:text-primary-container transition-colors"
          >
            <span className="material-symbols-outlined text-[14px]">download</span>
          </button>
          <button
            onClick={() => importRef.current?.click()}
            title="Import learned profile"
            className="p-0.5 rounded hover:bg-surface-container-highest text-on-surface-variant hover:text-primary-container transition-colors"
          >
            <span className="material-symbols-outlined text-[14px]">upload</span>
          </button>
          <input ref={importRef} type="file" accept=".json" className="hidden" onChange={handleImport} />
          <span className="material-symbols-outlined text-[12px] text-tertiary-container">vital_signs</span>
          <span className="font-label-sm text-[10px] text-tertiary-container">Real-time</span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-3 bg-surface-container rounded-b-lg scrollbar-thin">
        {/* Stat Cards */}
        <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-2">
          <StatCard 
            label="Tok/s" 
            value={lastTps ? lastTps.toFixed(1) : '—'} 
            color={lastTps && lastTps > 28 ? 'text-primary' : lastTps && lastTps > 20 ? 'text-tertiary-container' : lastTps ? 'text-error' : 'text-on-surface'} 
          />
          <StatCard label="Requests" value={stats.totalReqs} />
          <StatCard label="Configs" value={state ? Object.keys(state.config_space).length : 0} />
          <StatCard label="Explore %" value={stats.totalReqs > 0 ? Math.round((stats.exploreCount / stats.totalReqs) * 100) + '%' : '0%'} />
          <StatCard label="In Tokens" value={stats.tokens ? stats.tokens.prompt : 0} color="text-tertiary-container" />
          <StatCard label="Out Tokens" value={stats.tokens ? stats.tokens.completion : 0} color="text-tertiary-container" />
        </div>

        {/* Performance Sparkline */}
        {tpsHistory.length > 1 && (
          <div className="bg-surface-container-low border border-outline-variant/50 rounded-lg p-2">
            <Sparkline data={tpsHistory} width={580} height={48} label="Tokens/sec over time" />
          </div>
        )}

        {/* Learning Matrix */}
        <div className="space-y-1.5">
          <h3 className="font-label-caps text-[9px] text-on-surface-variant uppercase">Bayesian Policy Matrix</h3>
          
          {!state || Object.keys(state.signatures).length === 0 ? (
            <div className="h-32 flex items-center justify-center border border-dashed border-outline-variant/50 rounded-lg bg-surface-container-low">
              <span className="text-on-surface-variant text-[11px] font-body-md">Waiting for routing data...</span>
            </div>
          ) : (
            <div className="overflow-x-auto bg-surface-container-low border border-outline-variant/50 rounded-lg p-1.5">
              <table className="w-full border-separate" style={{ borderSpacing: '2px' }}>
                <thead>
                  <tr>
                    <th></th>
                    {state.config_space.map(cfg => {
                      const g = cfg.n_gpu_layers === 99 ? '99' : cfg.n_gpu_layers;
                      const m = cfg.n_cpu_moe > 0 ? `/${cfg.n_cpu_moe}` : '';
                      return (
                        <th key={cfg.key} className="font-mono-label text-[9px] text-on-surface-variant p-0.5">
                          g{g}{m}
                        </th>
                      );
                    })}
                  </tr>
                </thead>
                <tbody>
                  {Object.keys(state.signatures).sort().map(sigKey => {
                    const meta = SIG_LABELS[sigKey] || { label: sigKey, phase: '?' };
                    
                    return (
                      <tr key={sigKey}>
                        <td className="text-left pr-2 py-0.5 whitespace-nowrap w-20">
                          <div className="font-body-sm text-[10px] font-medium text-on-surface truncate">{meta.label}</div>
                          <div className="text-[8px] text-on-surface-variant font-mono uppercase tracking-wider">{meta.phase}</div>
                        </td>
                        {state.config_space.map(cfg => {
                          const obs = state.signatures[sigKey]?.[cfg.key];
                          const isActive = state.last_signature === sigKey && state.last_config_key === cfg.key;
                          
                          if (obs && obs.trials > 0) {
                            const mean = obs.total_tok_per_sec / obs.trials;
                            const ratio = (mean - minTps) / range;
                            const r = Math.round(13 + (79 - 13) * ratio);
                            const g = Math.round(19 + (209 - 19) * ratio);
                            const b = Math.round(31 + (197 - 31) * ratio);
                            
                            return (
                              <td 
                                key={cfg.key}
                                className="relative text-center p-0.5 rounded min-w-[36px] font-mono-label text-[9px] font-medium transition-all duration-300"
                                style={{ 
                                  backgroundColor: isActive ? 'rgba(79, 209, 197, 0.2)' : `rgba(${r}, ${g}, ${b}, 0.15)`,
                                  color: isActive ? '#4fd1c5' : '#bbc9c7',
                                  border: isActive ? '1px solid #4fd1c5' : '1px solid transparent',
                                }}
                                title={`${mean.toFixed(1)} tok/s (${obs.trials} trials)`}
                              >
                                {isActive && (
                                  <motion.div 
                                    layoutId="active-cell"
                                    className="absolute inset-0 rounded pointer-events-none"
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1, boxShadow: '0 0 8px rgba(79, 209, 197, 0.4)' }}
                                    transition={{ duration: 0.3 }}
                                  />
                                )}
                                {mean.toFixed(0)}
                              </td>
                            );
                          }
                          
                          return (
                            <td key={cfg.key} className="text-center p-0.5 rounded min-w-[36px] font-mono-label text-[9px] text-on-surface-variant/30 bg-surface-container-highest border border-dashed border-outline-variant/30">
                              —
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
          {/* Routing History Log */}
          <div className="space-y-1.5">
            <h3 className="font-label-caps text-[9px] text-on-surface-variant uppercase">Routing History</h3>
            <div className="space-y-1 max-h-48 overflow-y-auto pr-1 scrollbar-thin">
              <AnimatePresence initial={false}>
                {history.length === 0 ? (
                  <div className="text-on-surface-variant text-[10px] font-body-md">No requests yet</div>
                ) : (
                  history.slice(0, 10).map((entry, i) => (
                    <motion.div 
                      key={i + '-' + entry.config}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      className="flex items-center gap-2 py-1 px-2 rounded-md bg-surface-container-low border border-outline-variant/30"
                    >
                      <span className={`px-1.5 py-0.5 rounded font-label-caps text-[8px] border ${
                        entry.tag === 'EXPLORE' 
                          ? 'bg-tertiary-container/10 text-tertiary-container border-tertiary-container/20'
                          : 'bg-primary-container/10 text-primary-container border-primary-container/20'
                      }`}>
                        {entry.tag}
                      </span>
                      <span className="text-[10px] text-on-surface font-mono-label flex-1 truncate">{entry.config}</span>
                      <span className="text-[10px] font-semibold text-primary">{entry.tps ? entry.tps.toFixed(1) + ' t/s' : '—'}</span>
                    </motion.div>
                  ))
                )}
              </AnimatePresence>
            </div>
          </div>

          {/* CPU Cores */}
          <div className="space-y-1.5">
            <h3 className="font-label-caps text-[9px] text-on-surface-variant uppercase">CPU Usage</h3>
            <div className="grid grid-cols-2 gap-1.5 max-h-48 overflow-y-auto pr-1 scrollbar-thin">
              {!state?.cpu_usage ? (
                <div className="col-span-2 text-on-surface-variant text-[10px] font-body-md italic">Waiting for metrics...</div>
              ) : (
                state.cpu_usage.map((usage, i) => {
                  const coreId = i.toString().padStart(2, '0');
                  const percent = usage.toFixed(1);
                  let colorClass = 'text-primary';
                  if (usage > 50) colorClass = 'text-tertiary-container';
                  if (usage > 85) colorClass = 'text-error';
                  
                  return (
                    <div key={i} className="bg-surface-container-low border border-outline-variant/30 rounded-md px-2 flex items-center justify-between relative overflow-hidden h-7 group">
                      <div className="absolute inset-0 z-0 opacity-40">
                        <CpuActivityShader usage={usage} />
                      </div>
                      <div className="font-mono-label text-on-surface-variant z-10 text-[9px]">Core {coreId}</div>
                      <div className={`font-mono-label ${colorClass} z-10 text-right w-10 drop-shadow-md text-[9px]`}>{percent}%</div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}

function StatCard({ label, value, color = "text-on-surface" }: { label: string, value: string | number, color?: string }) {
  return (
    <div className="bg-surface-container-low border border-outline-variant/30 rounded-md py-1.5 px-2 text-center flex flex-col justify-center shadow-sm">
      <div className={`text-xs md:text-sm font-bold font-mono-label ${color}`}>{value}</div>
      <div className="font-label-caps text-[8px] text-on-surface-variant mt-0.5 leading-none">{label}</div>
    </div>
  );
}
