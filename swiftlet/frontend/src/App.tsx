import { useState, useEffect } from 'react';
import { ChatPanel } from './components/ChatPanel';
import { DashboardPanel } from './components/DashboardPanel';
import { API, LearningState } from './api';

function App() {
  const [, setStatus] = useState<'connected' | 'disconnected'>('disconnected');
  const [state, setState] = useState<LearningState | null>(null);
  
  // Dashboard state
  const [history, setHistory] = useState<Array<{ tag: string; config: string; tps: number | null }>>([]);
  const [totalReqs, setTotalReqs] = useState(0);
  const [exploreCount, setExploreCount] = useState(0);
  const [latestTokens, setLatestTokens] = useState({ prompt: 0, completion: 0 });

  // Poll state
  useEffect(() => {
    const poll = async () => {
      const data = await API.fetchState();
      if (data) {
        setState(data);
        setStatus('connected');
      } else {
        setStatus('disconnected');
      }
    };
    poll();
    const interval = setInterval(poll, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="bg-background text-on-background font-body-md h-screen overflow-hidden flex flex-col">
      {/* Top Navigation */}
      <header className="flex justify-between items-center w-full px-margin h-16 bg-surface-dim border-b border-outline-variant shrink-0">
        <div className="flex items-center gap-4">
          <img alt="Swiftlet Logo" className="object-contain h-12" src="https://lh3.googleusercontent.com/aida-public/AB6AXuDHApJVYPcWVb2cdIe1VW3RG1rcKEalhsOIiohGobzvJmFFkkBdng3A2rQ11N0prQ2YESfyWFikVwcX9gK8oQwmnzn13CxzOfujHf9wJ5UMlocX3hxJqwDpIGrGD7eEabptjeNFFI-VyfOerqwJSaJ7rkp2aIY3YzNoqHPuPuX2v8BIYeYQnZfv2Ib3haCFXDURGVOHfmMESJg3dw7JRVZ9WHEUwhd0m65z2hjb9Y9OI8zLtn9-cH-WLnzRK6mUEgJM5w"/>
          <pre className="font-mono text-[4px] sm:text-[5px] md:text-[7px] lg:text-[8px] xl:text-[9px] leading-[1.1] text-primary-container hidden sm:block opacity-90 select-none tracking-tighter">
{`███████╗██╗    ██╗██╗███████╗████████╗██╗     ███████╗████████╗
██╔════╝██║    ██║██║██╔════╝╚══██╔══╝██║     ██╔════╝╚══██╔══╝
███████╗██║ █╗ ██║██║█████╗     ██║   ██║     █████╗     ██║   
╚════██║██║███╗██║██║██╔══╝     ██║   ██║     ██╔══╝     ██║   
███████║╚███╔███╔╝██║██║        ██║   ███████╗███████╗   ██║   
╚══════╝ ╚══╝╚══╝ ╚═╝╚═╝        ╚═╝   ╚══════╝╚══════╝   ╚═╝`}
          </pre>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 mr-2">
            <span className={`w-1.5 h-1.5 rounded-full ${status === 'connected' ? 'bg-primary-container' : 'bg-error'}`}></span>
            <span className={`font-label-sm text-[9px] uppercase tracking-widest ${status === 'connected' ? 'text-primary-container' : 'text-error'}`}>
              {status === 'connected' ? 'API Online' : 'Offline'}
            </span>
          </div>

          <div className="flex flex-col text-left font-mono-label text-[9px] text-on-surface-variant leading-relaxed border-l border-outline-variant/30 pl-4 py-1">
            {state?.active_servers?.[0] ? (
              <>
                <div><span className="text-primary-container opacity-80 mr-1">BACKEND:</span> {state.active_servers[0].backend}</div>
                <div><span className="text-primary-container opacity-80 mr-1">GPU LYS:</span> {state.active_servers[0].config.n_gpu_layers}</div>
                <div><span className="text-primary-container opacity-80 mr-1">CPU MOE:</span> {state.active_servers[0].config.n_cpu_moe}</div>
                <div><span className="text-primary-container opacity-80 mr-1">CTX SZE:</span> {state.ctx_size}</div>
              </>
            ) : (
              <>
                <div><span className="text-primary-container opacity-80 mr-1">BACKEND:</span> STANDBY</div>
                <div><span className="text-primary-container opacity-80 mr-1">GPU LYS:</span> ---</div>
                <div><span className="text-primary-container opacity-80 mr-1">CPU MOE:</span> ---</div>
                <div><span className="text-primary-container opacity-80 mr-1">CTX SZE:</span> {state?.ctx_size || '---'}</div>
              </>
            )}
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 flex overflow-hidden p-gutter gap-gutter max-w-[1920px] mx-auto w-full">
        <section className="flex-1 flex flex-col bg-surface-container rounded-lg border border-outline-variant overflow-hidden min-w-[320px]">
          <ChatPanel 
            onStatusChange={setStatus} 
            onTokensUpdate={(p, c) => setLatestTokens({ prompt: p, completion: c })}
            onNewRequest={(tag, isExplore, tps, pTokens, cTokens) => {
              setTotalReqs(prev => prev + 1);
              if (isExplore) setExploreCount(prev => prev + 1);
              
              setHistory(prev => [
                { tag: isExplore ? 'EXPLORE' : 'EXPLOIT', config: tag, tps },
                ...prev
              ]);
              setLatestTokens({ prompt: pTokens, completion: cTokens });
            }} 
          />
        </section>

        <section className="flex-1 flex flex-col gap-gutter overflow-y-auto pr-sm pb-sm min-w-[320px]">
          <DashboardPanel 
            state={state} 
            history={history} 
            stats={{ totalReqs, exploreCount, tokens: latestTokens }} 
          />
        </section>
      </main>
    </div>
  );
}

export default App;
