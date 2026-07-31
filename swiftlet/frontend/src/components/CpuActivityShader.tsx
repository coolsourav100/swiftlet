import { useEffect, useRef } from 'react';

interface CpuActivityShaderProps {
  usage: number; // 0 to 100
}

export function CpuActivityShader({ usage }: CpuActivityShaderProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const usageRef = useRef(usage);

  // Keep usageRef up to date without re-triggering the WebGL context setup
  useEffect(() => {
    usageRef.current = usage;
  }, [usage]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let animationFrameId: number;
    let seedVal = Math.random() * 100.0;
    
    // Smooth transition for the intensity target
    let currentIntensity = Math.max(0.1, usageRef.current / 100);

    const syncSize = () => {
      const w = canvas.clientWidth || 300;
      const h = canvas.clientHeight || 20;
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
    };

    let resizeObserver: ResizeObserver | null = null;
    if (typeof ResizeObserver !== 'undefined') {
      resizeObserver = new ResizeObserver(syncSize);
      resizeObserver.observe(canvas);
    }
    syncSize();

    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl') as WebGLRenderingContext | null;
    if (!gl) return;

    const vs = `attribute vec2 a_position;
varying vec2 v_texCoord;
void main() {
  v_texCoord = a_position * 0.5 + 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}`;
    const fs = `precision highp float;
varying vec2 v_texCoord;
uniform float u_time;
uniform vec2 u_resolution;
uniform float u_seed;
uniform float u_intensity;

float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
}

void main() {
    vec2 st = v_texCoord;
    float rows = 1.0;
    float cols = 20.0;
    vec2 grid = vec2(cols, rows);
    vec2 ipos = floor(st * grid);
    vec2 fpos = fract(st * grid);

    // Speed scales with intensity
    float speed = 1.0 + (u_intensity * 3.0);
    float noise = random(ipos + floor(u_time * speed) + u_seed);
    
    vec3 colorA = vec3(0.17, 0.48, 0.48); // Idle teal
    vec3 colorB = vec3(0.31, 0.82, 0.77); // Active cyan
    vec3 colorDanger = vec3(0.9, 0.2, 0.2); // High load red
    vec3 colorWarn = vec3(0.9, 0.6, 0.1); // Med load amber

    // Blend base colors based on intensity
    vec3 activeColor = colorB;
    if (u_intensity > 0.8) {
        activeColor = mix(colorWarn, colorDanger, (u_intensity - 0.8) * 5.0);
        colorA = mix(colorA, vec3(0.5, 0.1, 0.1), (u_intensity - 0.8) * 5.0);
    } else if (u_intensity > 0.5) {
        activeColor = mix(colorB, colorWarn, (u_intensity - 0.5) * 3.33);
        colorA = mix(colorA, vec3(0.4, 0.3, 0.1), (u_intensity - 0.5) * 3.33);
    }
    
    // The threshold controls how many "blocks" are lit up
    float baseThreshold = clamp(1.0 - u_intensity, 0.1, 0.9);
    float threshold = baseThreshold * 0.8 + 0.2 * sin(u_time * 0.5 + ipos.x * 0.2 + u_seed);
    float mask = step(noise, threshold);
    
    float border = 0.1;
    float box = step(border, fpos.x) * step(border, fpos.y) * step(fpos.x, 1.0 - border) * step(fpos.y, 1.0 - border);
    
    vec3 finalColor = mix(vec3(0.05, 0.08, 0.13), mix(colorA, activeColor, noise), mask * box);
    
    gl_FragColor = vec4(finalColor, 1.0);
}`;

    const createShader = (type: number, src: string) => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, src);
      gl.compileShader(s);
      return s;
    };

    const prog = gl.createProgram()!;
    gl.attachShader(prog, createShader(gl.VERTEX_SHADER, vs));
    gl.attachShader(prog, createShader(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(prog);
    gl.useProgram(prog);

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);

    const pos = gl.getAttribLocation(prog, 'a_position');
    gl.enableVertexAttribArray(pos);
    gl.vertexAttribPointer(pos, 2, gl.FLOAT, false, 0, 0);

    const uTime = gl.getUniformLocation(prog, 'u_time');
    const uRes = gl.getUniformLocation(prog, 'u_resolution');
    const uSeed = gl.getUniformLocation(prog, 'u_seed');
    const uIntensity = gl.getUniformLocation(prog, 'u_intensity');

    const render = (t: number) => {
      if (typeof ResizeObserver === 'undefined') syncSize();
      gl.viewport(0, 0, canvas.width, canvas.height);
      
      // Smoothly approach the target usage
      const targetIntensity = Math.max(0.1, usageRef.current / 100);
      currentIntensity += (targetIntensity - currentIntensity) * 0.1;

      if (uTime) gl.uniform1f(uTime, t * 0.001);
      if (uRes) gl.uniform2f(uRes, canvas.width, canvas.height);
      if (uSeed) gl.uniform1f(uSeed, seedVal);
      if (uIntensity) gl.uniform1f(uIntensity, currentIntensity);
      
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      animationFrameId = requestAnimationFrame(render);
    };

    animationFrameId = requestAnimationFrame(render);

    return () => {
      cancelAnimationFrame(animationFrameId);
      if (resizeObserver) resizeObserver.disconnect();
    };
  }, []); // Run once on mount

  return <canvas ref={canvasRef} className="w-full h-full block" />;
}
