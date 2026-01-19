import { useTheme } from '@/components/ThemeProvider';

export function useChartTooltipStyles() {
  const { theme } = useTheme();
  
  const isDark = theme === 'dark';
  
  return {
    contentStyle: {
      backgroundColor: isDark ? '#111' : '#fff',
      border: isDark ? '1px solid #333' : '1px solid #e5e7eb',
      color: isDark ? '#fff' : '#111',
      borderRadius: '6px',
    },
    itemStyle: {
      color: isDark ? '#fff' : '#111',
    },
    cursor: {
      fill: isDark ? '#333' : '#e5e7eb',
    },
  };
}
