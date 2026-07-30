import os

def create_matrix_svg(filepath: str):
    svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 400" width="100%" height="100%">
    <defs>
        <style>
            .bg { fill: #1E1E1E; }
            .text-main { fill: #E0E0E0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
            .text-title { font-size: 24px; font-weight: bold; fill: #FFFFFF; }
            .text-subtitle { font-size: 14px; fill: #888888; }
            .grid-line { stroke: #333333; stroke-width: 1; }
            .bar-bg { fill: #2A2A2A; rx: 4; }
            .bar-fill-gpu { fill: #007ACC; rx: 4; }
            .bar-fill-cpu { fill: #4CAF50; rx: 4; }
            .text-label { font-size: 13px; font-weight: 500; }
            .text-value { font-size: 13px; font-weight: bold; }
        </style>
    </defs>
    
    <!-- Background -->
    <rect width="100%" height="100%" class="bg" rx="12"/>
    
    <!-- Title -->
    <text x="40" y="50" class="text-main text-title">Real-Time Performance Matrix</text>
    <text x="40" y="75" class="text-main text-subtitle">Hardware: Apple M5 (16GB Unified Memory) | Phase: Decode (Tokens/sec)</text>

    <!-- Legend -->
    <rect x="580" y="40" width="12" height="12" class="bar-fill-gpu" rx="2"/>
    <text x="600" y="51" class="text-main text-label">GPU Dominant</text>
    <rect x="695" y="40" width="12" height="12" class="bar-fill-cpu" rx="2"/>
    <text x="715" y="51" class="text-main text-label">CPU Offload</text>

    <!-- Chart Area -->
    <line x1="120" y1="120" x2="750" y2="120" class="grid-line" />
    <line x1="120" y1="170" x2="750" y2="170" class="grid-line" />
    <line x1="120" y1="220" x2="750" y2="220" class="grid-line" />
    <line x1="120" y1="270" x2="750" y2="270" class="grid-line" />
    <line x1="120" y1="320" x2="750" y2="320" class="grid-line" />

    <!-- Y-Axis Labels -->
    <text x="100" y="125" class="text-main text-label" text-anchor="end">26 t/s</text>
    <text x="100" y="175" class="text-main text-label" text-anchor="end">24 t/s</text>
    <text x="100" y="225" class="text-main text-label" text-anchor="end">22 t/s</text>
    <text x="100" y="275" class="text-main text-label" text-anchor="end">20 t/s</text>
    <text x="100" y="325" class="text-main text-label" text-anchor="end">18 t/s</text>

    <!-- Bars & X-Axis (CPU MoE Splits) -->
    <!-- 0 CPU -->
    <text x="180" y="350" class="text-main text-label" text-anchor="middle">0 CPU</text>
    <rect x="160" y="165" width="40" height="155" class="bar-fill-gpu" />
    <text x="180" y="155" class="text-main text-value" text-anchor="middle">23.93</text>

    <!-- 4 CPU -->
    <text x="310" y="350" class="text-main text-label" text-anchor="middle">4 CPU</text>
    <rect x="290" y="162" width="40" height="158" class="bar-fill-gpu" />
    <text x="310" y="152" class="text-main text-value" text-anchor="middle">24.05</text>

    <!-- 8 CPU -->
    <text x="440" y="350" class="text-main text-label" text-anchor="middle">8 CPU</text>
    <rect x="420" y="156" width="40" height="164" class="bar-fill-cpu" />
    <text x="440" y="146" class="text-main text-value" text-anchor="middle">24.28</text>

    <!-- 12 CPU -->
    <text x="570" y="350" class="text-main text-label" text-anchor="middle">12 CPU</text>
    <rect x="550" y="152" width="40" height="168" class="bar-fill-cpu" />
    <text x="570" y="142" class="text-main text-value" text-anchor="middle">24.44</text>

    <!-- 16 CPU -->
    <text x="700" y="350" class="text-main text-label" text-anchor="middle">16 CPU</text>
    <rect x="680" y="161" width="40" height="159" class="bar-fill-cpu" />
    <text x="700" y="151" class="text-main text-value" text-anchor="middle">24.09</text>

    <!-- Bottom Line -->
    <line x1="120" y1="320" x2="750" y2="320" style="stroke: #555; stroke-width: 2;" />
</svg>"""
    with open(filepath, "w") as f:
        f.write(svg_content)
    print(f"Created {filepath}")

def create_demo_svg(filepath: str):
    svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 300" width="100%" height="100%">
    <defs>
        <style>
            .bg { fill: #1E1E1E; }
            .mac-bar { fill: #333333; }
            .mac-btn-red { fill: #FF5F56; }
            .mac-btn-yellow { fill: #FFBD2E; }
            .mac-btn-green { fill: #27C93F; }
            .text { font-family: "Menlo", "Monaco", "Courier New", monospace; font-size: 14px; fill: #E0E0E0; }
            .prompt { fill: #007ACC; font-weight: bold; }
            .ai { fill: #4CAF50; font-weight: bold; }
            
            /* Animations */
            @keyframes type {
                from { width: 0; }
                to { width: 300px; }
            }
            @keyframes blink {
                50% { opacity: 0; }
            }
            @keyframes appear {
                0% { opacity: 0; }
                100% { opacity: 1; }
            }
            @keyframes loading {
                0% { content: "."; }
                25% { content: ".."; }
                50% { content: "..."; }
                75% { content: "...."; }
                100% { content: "....."; }
            }
            
            .typing-container {
                display: inline-block;
                overflow: hidden;
                white-space: nowrap;
                animation: type 2s steps(40, end);
            }
            .cursor {
                display: inline-block;
                width: 8px;
                height: 15px;
                background-color: #E0E0E0;
                animation: blink 1s step-end infinite;
                vertical-align: text-bottom;
            }
            .response {
                opacity: 0;
                animation: appear 0.1s linear forwards;
                animation-delay: 2.5s;
            }
            .response-text {
                opacity: 0;
                animation: appear 0.1s linear forwards;
                animation-delay: 4.5s;
            }
            .loading-dots::after {
                content: "";
                animation: loading 2s infinite;
            }
            .loading-container {
                opacity: 0;
                animation: appear 0.1s linear forwards, disappear 0.1s linear forwards 4.5s;
                animation-delay: 2.5s;
            }
            @keyframes disappear {
                to { opacity: 0; display: none; }
            }
        </style>
    </defs>
    
    <!-- Window -->
    <rect width="100%" height="100%" class="bg" rx="8"/>
    
    <!-- Mac Title Bar -->
    <rect width="100%" height="24" class="mac-bar" rx="8" />
    <rect x="0" y="12" width="100%" height="12" class="mac-bar" />
    <circle cx="20" cy="12" r="6" class="mac-btn-red" />
    <circle cx="40" cy="12" r="6" class="mac-btn-yellow" />
    <circle cx="60" cy="12" r="6" class="mac-btn-green" />
    
    <!-- Terminal Content -->
    <foreignObject x="20" y="40" width="760" height="240">
        <div xmlns="http://www.w3.org/1999/xhtml" style="font-family: 'Menlo', 'Monaco', 'Courier New', monospace; font-size: 14px; color: #E0E0E0;">
            <div>
                <span style="color: #007ACC; font-weight: bold;">You: </span>
                <span class="typing-container">write a rate limiter function in node js</span>
                <span class="cursor"></span>
            </div>
            
            <div style="margin-top: 15px; position: relative;">
                <div class="loading-container" style="position: absolute; top: 0;">
                    <span style="color: #4CAF50; font-weight: bold;">Swiftlet: </span>
                    <span class="loading-dots"></span>
                </div>
                
                <div class="response-text" style="position: absolute; top: 0;">
                    <span style="color: #4CAF50; font-weight: bold;">Swiftlet: </span>
                    <br/><br/>
                    Here is a simple sliding window rate limiter in Node.js...<br/>
                    <br/>
                    <span style="color: #888888; font-size: 12px;"><i>[Proxy] Recorded 24.44 tok/s (Decision: EXPLORE, CPU: 12)</i></span>
                </div>
            </div>
        </div>
    </foreignObject>
</svg>"""
    with open(filepath, "w") as f:
        f.write(svg_content)
    print(f"Created {filepath}")

if __name__ == "__main__":
    assets_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
    os.makedirs(assets_dir, exist_ok=True)
    create_matrix_svg(os.path.join(assets_dir, "performance_matrix.svg"))
    create_demo_svg(os.path.join(assets_dir, "demo.svg"))
