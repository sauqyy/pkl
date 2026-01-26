interface HeatmapProps {
  data: {
    index: string[];
    data: number[][];
  };
  colorScheme?: 'blue' | 'red';
}


export function Heatmap({ data, colorScheme = 'blue' }: HeatmapProps) {
  if (!data || !data.index || !data.data || !data.data[0]) {
    return <div className="text-sm text-muted-foreground text-center py-8">No data available</div>;
  }

  const days = data.index;
  const values = data.data;
  const columnsCount = values[0].length; // 24 or 60

  // Find max for color scaling
  let maxVal = 0;
  values.forEach(row => row.forEach(val => maxVal = Math.max(maxVal, val)));

  const getColor = (val: number) => {
    if (val === 0) return 'rgba(100, 100, 100, 0.1)'; // Subtle gray for no data

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
      <div className="grid gap-[2px]" style={{ gridTemplateColumns: `80px repeat(${columnsCount}, 1fr)` }}>
        {/* Header Row */}
        <div></div>
        {Array.from({ length: columnsCount }, (_, h) => (
          <div key={h} className="text-[10px] text-muted-foreground text-center overflow-hidden">
            {/* Show every 5th label if many columns, else all */}
            {columnsCount > 24 ? (h % 5 === 0 ? h : '') : h}
          </div>
        ))}

        {/* Data Rows */}
        {days.map((day, i) => (
          <>
            <div key={`label-${i}`} className="text-xs text-muted-foreground flex items-center truncate" title={day}>
              {day}
            </div>
            {Array.from({ length: columnsCount }, (_, h) => {
              const val = values[i][h];
              const bgColor = getColor(val);
              const intensity = val / maxVal;
              const shouldShowWhiteText = val > 0 && intensity > 0.5;
              // Show values for outliers (> 50% of max) or non-zero on hover
              const isOutlier = val > 0 && intensity > 0.5;

              return (
                <div
                  key={`cell-${i}-${h}`}
                  className="group aspect-square rounded flex items-center justify-center text-[8px] hover:scale-110 hover:z-10 transition-all cursor-pointer border border-border/50"
                  style={{ backgroundColor: bgColor }}
                  title={`${day} - ${columnsCount === 60 ? 'Minute ' + h : h + ':00'} - ${val} ${colorScheme === 'blue' ? 'Calls' : 'Errors'}`}
                >
                  {/* Only show text if there is space (not too many columns) or if it is an outlier */}
                  <span className={`font-semibold ${shouldShowWhiteText ? 'text-white' : 'text-foreground'} ${isOutlier ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'} transition-opacity`}>
                    {columnsCount <= 24 || isOutlier ? formatValue(val) : ''}
                  </span>
                </div>
              );
            })}
          </>
        ))}
      </div>

      {/* Time Labels */}
      <div className="flex justify-between text-xs text-muted-foreground px-20">
        <span>Start</span>
        <span>Middle</span>
        <span>End</span>
      </div>
    </div>
  );
}

