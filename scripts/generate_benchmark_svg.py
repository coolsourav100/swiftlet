import json
import os

def generate_svg():
    with open("swiftlet_benchmark_results.json", "r") as f:
        data = json.load(f)
        
    results = data.get("results", [])
    
    # SVG parameters
    width = 850
    height = 450
    bar_height = 30
    gap = 25
    start_y = 100
    
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">
    <defs>
        <style>
            .bg {{ fill: #1E1E1E; }}
            .text-main {{ fill: #E0E0E0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
            .text-title {{ font-size: 24px; font-weight: bold; fill: #FFFFFF; }}
            .text-subtitle {{ font-size: 14px; fill: #888888; }}
            .bar-bg {{ fill: #2A2A2A; rx: 4; }}
            .bar-fill {{ fill: #4fd1c5; rx: 4; }}
            .bar-fill-oracle {{ fill: #007ACC; rx: 4; }}
            .bar-fill-static {{ fill: #FF5F56; rx: 4; }}
            .text-label {{ font-size: 14px; font-weight: 500; text-anchor: end; }}
            .text-value {{ font-size: 13px; font-weight: bold; fill: #1E1E1E; text-anchor: end; }}
            .text-value-out {{ font-size: 13px; font-weight: bold; fill: #E0E0E0; text-anchor: start; }}
        </style>
    </defs>
    <rect width="100%" height="100%" class="bg" rx="12"/>
    <text x="40" y="50" class="text-main text-title">Routing Strategy Benchmark (Avg Tokens/sec)</text>
    <text x="40" y="75" class="text-main text-subtitle">Higher is better. Evaluated across 100 varying requests.</text>
    '''
    
    max_tps = max(r["avg_tps"] for r in results)
    
    for i, r in enumerate(results):
        strategy = r["strategy"]
        tps = r["avg_tps"]
        
        y = start_y + i * (bar_height + gap)
        
        # Color logic
        if "Oracle" in strategy:
            bar_class = "bar-fill-oracle"
        elif "Static" in strategy or "Random" in strategy:
            bar_class = "bar-fill-static"
        else:
            bar_class = "bar-fill"
            
        bar_width = (tps / max_tps) * 450
        
        svg += f'''
        <text x="270" y="{y + 20}" class="text-main text-label">{strategy}</text>
        <rect x="290" y="{y}" width="450" height="{bar_height}" class="bar-bg" />
        <rect x="290" y="{y}" width="{bar_width}" height="{bar_height}" class="{bar_class}" />
        '''
        
        if bar_width > 50:
            svg += f'<text x="{290 + bar_width - 10}" y="{y + 20}" class="text-value">{tps:.2f} t/s</text>'
        else:
            svg += f'<text x="{290 + bar_width + 10}" y="{y + 20}" class="text-value-out">{tps:.2f} t/s</text>'
            
    svg += '</svg>'
    
    os.makedirs("assets", exist_ok=True)
    with open("assets/benchmark_results.svg", "w") as f:
        f.write(svg)
        
if __name__ == "__main__":
    generate_svg()
