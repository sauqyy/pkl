interface HeatmapProps {
  data: {
    index: string[];
    data: number[][];
  };
  colorScheme?: 'blue' | 'red';
}

export function Heatmap({ data, colorScheme = 'blue' }: HeatmapProps) {
  if (!data || !data.index || !data.data) {
    return <div className="text-sm text-muted-foreground text-center py-8">No data available</div>;
  }

  const days = data.index;
  const values = data.data;

  // Find max for color scaling
  let maxVal = 0;
  values.forEach(row => row.forEach(val => maxVal = Math.max(maxVal, val)));

  const getColor = (val: number) => {
    if (val === 0) return '#0A0A0A'; // Same as background
    
    const intensity = Math.min((val / maxVal) * 0.8 + 0.2, 1);
    
    if (colorScheme === 'blue') {
      return `rgba(59, 130, 246, ${intensity})`;
    } else {
      return `rgba(239, 68, 68, ${intensity})`;
    }
  };

  const formatValue = (val: number) => {
    if (val > 1000) return `${(val / 1000).toFixed(1)}k`;
    return val.toString();
  };

  return (
    <div className="space-y-2">
      <div className="grid gap-[2px]" style={{ gridTemplateColumns: '80px repeat(24, 1fr)' }}>
        {/* Header Row (Hours) */}
        <div></div>
        {Array.from({ length: 24 }, (_, h) => (
          <div key={h} className="text-xs text-muted-foreground text-center">
            {h}
          </div>
        ))}

        {/* Data Rows */}
        {days.map((day, i) => (
          <>
            <div key={`label-${i}`} className="text-xs text-muted-foreground flex items-center">
              {day.substring(0, 3)}
            </div>
            {Array.from({ length: 24 }, (_, h) => {
              const val = values[i][h];
              const bgColor = getColor(val);
              const textColor = val > 0 && (val / maxVal) > 0.5 ? '#fff' : 'transparent';
              
              return (
                <div
                  key={`cell-${i}-${h}`}
                  className="aspect-square rounded flex items-center justify-center text-[10px] hover:scale-110 hover:z-10 transition-transform cursor-pointer border border-border/50"
                  style={{ backgroundColor: bgColor, color: textColor }}
                  title={`${day} ${h}:00 - ${val} ${colorScheme === 'blue' ? 'Calls' : 'Errors'}`}
                >
                  {formatValue(val)}
                </div>
              );
            })}
          </>
        ))}
      </div>

      {/* Time Labels */}
      <div className="flex justify-between text-xs text-muted-foreground px-20">
        <span>0:00 (Midnight)</span>
        <span>12:00 (Noon)</span>
        <span>23:00</span>
      </div>
    </div>
  );
}
