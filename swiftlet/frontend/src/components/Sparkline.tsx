import { useMemo } from 'react';

interface SparklineProps {
  data: number[];
  width?: number;
  height?: number;
  color?: string;
  fillColor?: string;
  label?: string;
}

export function Sparkline({ 
  data, 
  width = 200, 
  height = 40, 
  color = '#4fd1c5', 
  fillColor = 'rgba(79, 209, 197, 0.1)',
  label 
}: SparklineProps) {
  const path = useMemo(() => {
    if (data.length < 2) return '';
    
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    const padding = 2;
    const w = width - padding * 2;
    const h = height - padding * 2;
    
    const points = data.map((v, i) => ({
      x: padding + (i / (data.length - 1)) * w,
      y: padding + h - ((v - min) / range) * h,
    }));

    const linePath = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' ');
    const fillPath = `${linePath} L ${points[points.length - 1].x.toFixed(1)} ${height} L ${points[0].x.toFixed(1)} ${height} Z`;
    
    return { linePath, fillPath };
  }, [data, width, height]);

  if (data.length < 2 || !path) {
    return (
      <div className="flex items-center justify-center text-[9px] text-on-surface-variant" style={{ width, height }}>
        Waiting for data...
      </div>
    );
  }

  const latest = data[data.length - 1];
  const avg = data.reduce((a, b) => a + b, 0) / data.length;

  return (
    <div className="relative" style={{ width, height: height + (label ? 16 : 0) }}>
      {label && (
        <div className="flex justify-between items-center mb-0.5">
          <span className="font-label-caps text-[8px] text-on-surface-variant uppercase tracking-widest">{label}</span>
          <span className="font-mono-label text-[9px]" style={{ color }}>
            {latest.toFixed(1)} <span className="text-on-surface-variant text-[7px]">avg {avg.toFixed(1)}</span>
          </span>
        </div>
      )}
      <svg width={width} height={height} className="block">
        <path d={path.fillPath} fill={fillColor} />
        <path d={path.linePath} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        {/* Latest value dot */}
        <circle
          cx={width - 2}
          cy={2 + (height - 4) - ((latest - Math.min(...data)) / (Math.max(...data) - Math.min(...data) || 1)) * (height - 4)}
          r="2.5"
          fill={color}
        />
      </svg>
    </div>
  );
}
