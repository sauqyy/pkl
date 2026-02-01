import { useEffect, useState, useRef } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { PanelRight, TrendingUp, TrendingDown, Activity, AlertTriangle, Clock, Zap } from "lucide-react"
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { useSidebar } from "@/components/SidebarContext"
import { useChartTooltipStyles } from "@/hooks/useChartTooltipStyles"
import InfoTooltip from "@/components/InfoTooltip"
import { GlobalSearch } from "@/components/GlobalSearch"

interface ComparisonPoint {
  timestamp: number;
  actual: number;
  predicted: number;
}

interface ForecastPoint {
  timestamp: number;
  value: number;
}

interface ForecastData {
  comparison: ComparisonPoint[];
  forecast: ForecastPoint[];
  model: string;
  accuracy: {
    mae: number;
    mape: number;
  };
}

interface MetricForecast {
  data: ForecastData | null;
  loading: boolean;
  error: string | null;
}

type MetricType = 'Load' | 'Response' | 'Error' | 'Slow';

const METRIC_CONFIG: Record<MetricType, {
  label: string;
  unit: string;
  color: string;
  predictedColor: string;
  icon: typeof Activity;
  description: string;
}> = {
  Load: {
    label: 'Load (Calls/min)',
    unit: 'calls',
    color: '#22c55e',
    predictedColor: '#86efac',
    icon: Zap,
    description: 'Predicted future traffic volume based on historical patterns.'
  },
  Response: {
    label: 'Response Time',
    unit: 'ms',
    color: '#38bdf8',
    predictedColor: '#7dd3fc',
    icon: Clock,
    description: 'Forecasted system latency trends.'
  },
  Error: {
    label: 'Errors',
    unit: 'errors',
    color: '#ef4444',
    predictedColor: '#fca5a5',
    icon: AlertTriangle,
    description: 'Predicted error rates to anticipate potential failures.'
  },
  Slow: {
    label: 'Slow Calls',
    unit: 'calls',
    color: '#f59e0b',
    predictedColor: '#fcd34d',
    icon: Activity,
    description: 'Forecasted volume of slow transactions.'
  }
};

const METRICS: MetricType[] = ['Load', 'Response', 'Error', 'Slow'];
const REFRESH_INTERVAL = 60000; // 60 seconds

import { useBusinessTransaction } from "@/components/BusinessTransactionContext"

// ... (existing code)

export default function Forecasting() {
  const [forecasts, setForecasts] = useState<Record<MetricType, MetricForecast>>({
    Load: { data: null, loading: true, error: null },
    Response: { data: null, loading: true, error: null },
    Error: { data: null, loading: true, error: null },
    Slow: { data: null, loading: true, error: null }
  })
  const { toggleSidebar } = useSidebar()
  const { selectedTier, selectedTransaction } = useBusinessTransaction()
  const tooltipStyles = useChartTooltipStyles()
  const intervalRef = useRef<NodeJS.Timeout | null>(null)
  const activeFetches = useRef<Set<string>>(new Set())
  const isMounted = useRef(true)


  useEffect(() => {
    isMounted.current = true
    return () => { isMounted.current = false }
  }, [])

  // Refetch when model or BT/Tier changes
  useEffect(() => {
    fetchAllForecasts(true)
  }, [selectedTier, selectedTransaction])

  const fetchForecast = async (metric: MetricType, showLoading = true) => {
    // Note: Removed activeFetches check for simplicity when switching models or re-add careful logic
    if (activeFetches.current.has(metric)) return
    activeFetches.current.add(metric)
    
    try {
      if (showLoading) {
        setForecasts(prev => ({
          ...prev,
          [metric]: { ...prev[metric], loading: true, error: null }
        }))
      }
      
      const params = new URLSearchParams({ 
        metric, 
        model: 'Prophet',
        tier: selectedTier,
        bt: selectedTransaction
      });
      
      const response = await fetch(`/api/forecast?${params}`)
      const result = await response.json()

      if (result.status === 'training') {
        if (isMounted.current) {
          setForecasts(prev => ({
            ...prev,
            [metric]: { data: null, loading: true, error: result.message || 'Training model...' }
          }))
          // Poll this specific metric again in 5 seconds
          setTimeout(() => {
            if (isMounted.current) fetchForecast(metric, false)
          }, 5000)
        }
        return
      }

      if (isMounted.current) {
        if (result.error) {
          setForecasts(prev => ({
            ...prev,
            [metric]: { data: null, loading: false, error: result.error }
          }))
        } else {
          setForecasts(prev => ({
            ...prev,
            [metric]: { data: result, loading: false, error: null }
          }))
        }
      }
    } catch (e) {
      console.error(e)
      if (isMounted.current) {
        setForecasts(prev => ({
          ...prev,
          [metric]: { data: null, loading: false, error: 'Failed to fetch forecast' }
        }))
      }
    } finally {
      activeFetches.current.delete(metric)
    }
  }

  const fetchAllForecasts = async (showLoading = true) => {
    await Promise.all(METRICS.map(metric => fetchForecast(metric, showLoading)))
  }

  useEffect(() => {
    // Initial fetch depends on dateRange
    fetchAllForecasts(true)

    // Set up auto-refresh
    intervalRef.current = setInterval(() => {
      fetchAllForecasts(false) // Don't show loading spinner on auto-refresh
    }, REFRESH_INTERVAL)

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [])

  const getChartData = (data: ForecastData | null) => {
    if (!data) return []

    // Comparison data (last 24h: actual vs predicted side by side)
    const comparisonData = data.comparison.map(d => ({
      time: new Date(d.timestamp).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }),
      actual: d.actual,
      predicted: d.predicted,
      type: 'comparison'
    }))

    // Future forecast (next 24h)
    const forecastData = []
    
    // Only attempt to create a connection point if we have historical data
    if (data.comparison.length > 0) {
        const lastActual = data.comparison[data.comparison.length - 1]
        forecastData.push({
            time: new Date(lastActual.timestamp).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }),
            actual: lastActual.actual,
            predicted: lastActual.predicted,
            forecast: lastActual.actual,
            type: 'connection'
        })
    }

    forecastData.push(...data.forecast.map(d => ({
        time: new Date(d.timestamp).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }),
        actual: null as number | null,
        predicted: null as number | null,
        forecast: d.value,
        type: 'forecast'
    })))


    return [...comparisonData, ...forecastData.slice(1)]
  }

  const getAccuracy = (data: ForecastData | null): string => {
    if (!data || !data.accuracy) return '--'
    const accuracy = 100 - data.accuracy.mape
    return `${accuracy.toFixed(1)}%`
  }

  const getTrend = (data: ForecastData | null): 'Rising' | 'Falling' | 'Stable' => {
    if (!data || data.forecast.length < 2) return 'Stable'
    const first = data.forecast[0].value
    const last = data.forecast[data.forecast.length - 1].value
    const diff = ((last - first) / (first || 1)) * 100
    if (diff > 5) return 'Rising'
    if (diff < -5) return 'Falling'
    return 'Stable'
  }

  return (
    <div className="space-y-6">
      {/* Top Bar */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button
            onClick={(e) => {
              e.stopPropagation();
              toggleSidebar();
            }}
            className="p-2 hover:bg-accent rounded-md transition-colors cursor-pointer"
          >
            <PanelRight className="h-5 w-5" />
          </button>
          <div className="h-6 w-px bg-border"></div>
          <div>
            <h1 className="text-lg font-semibold">Performance Forecast</h1>
          </div>
          
          {/* Model Selector REMOVED */}
          <div className="ml-4">
            {/* Empty placeholder */}
          </div>
        </div>


        <div className="flex items-center gap-2">
            <GlobalSearch />
        </div>
      </div>

      {/* Forecast Charts Grid */}
      <div className="grid gap-6 grid-cols-1 lg:grid-cols-2">
        {METRICS.map(metric => {
          const config = METRIC_CONFIG[metric]
          const forecast = forecasts[metric]
          const chartData = getChartData(forecast.data)
          const trend = getTrend(forecast.data)

          const Icon = config.icon

          return (
            <Card key={metric} className="bg-card">
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <div className="flex items-center gap-2">
                  <div
                    className="p-2 rounded-lg"
                    style={{ backgroundColor: `${config.color}20` }}
                  >
                    <Icon className="h-4 w-4" style={{ color: config.color }} />
                  </div>
                  <div>
                    <CardTitle className="text-base font-semibold flex items-center">
                      {config.label}
                      <InfoTooltip content={config.description} />
                    </CardTitle>
                    <p className="text-xs text-muted-foreground">
                      {/* Accuracy & Model Info Removed */}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {trend === 'Rising' && (
                    <span className="flex items-center gap-1 text-xs px-2 py-1 rounded-full bg-red-500/10 text-red-500 font-medium">
                      <TrendingUp className="h-3 w-3" /> Rising
                    </span>
                  )}
                  {trend === 'Falling' && (
                    <span className="flex items-center gap-1 text-xs px-2 py-1 rounded-full bg-green-500/10 text-green-500 font-medium">
                      <TrendingDown className="h-3 w-3" /> Falling
                    </span>
                  )}
                  {trend === 'Stable' && (
                    <span className="text-xs px-2 py-1 rounded-full bg-blue-500/10 text-blue-500 font-medium">
                      Stable
                    </span>
                  )}
                </div>
              </CardHeader>
              <CardContent>
                {forecast.loading ? (
                  <div className="h-[220px] flex items-center justify-center">
                    <div className="animate-pulse flex flex-col items-center gap-2">
                      <div className="w-8 h-8 rounded-full border-2 border-t-transparent animate-spin" style={{ borderColor: config.color, borderTopColor: 'transparent' }}></div>
                      <span className="text-xs text-muted-foreground">{forecast.error || "Loading..."}</span>
                    </div>
                  </div>
                ) : forecast.error ? (
                  <div className="h-[220px] flex items-center justify-center text-muted-foreground text-sm">
                    {forecast.error}
                  </div>
                ) : (
                  <div className="h-[220px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                        <XAxis
                          dataKey="time"
                          stroke="#666"
                          fontSize={10}
                          tickLine={false}
                          axisLine={false}
                          interval="preserveStartEnd"
                        />
                        <YAxis
                          stroke="#666"
                          fontSize={10}
                          tickLine={false}
                          axisLine={false}
                          width={45}
                        />
                        <Tooltip
                          contentStyle={tooltipStyles.contentStyle}
                          formatter={(value: number | undefined, name?: string) => {
                            if (value === undefined || value === null) return null
                            return [`${value.toFixed(1)} ${config.unit}`, name]
                          }}
                        />
                        {/* Legend Simplified */}
                        <Legend
                          verticalAlign="top"
                          height={36}
                          content={({ payload }) => (
                             <div className="flex items-center justify-center gap-4 text-xs mb-2">
                               {payload?.map((entry, index) => {
                                 if (entry.value === 'Predicted') return null; // Hide Predicted
                                 const isActual = entry.value === 'Actual';
                                 
                                 return (
                                   <div key={`legend-${index}`} className="flex items-center gap-1.5">
                                      <div className="flex items-center" style={{ width: 24, height: 12 }}>
                                        <div
                                            className="w-full h-0.5"
                                            style={{
                                              backgroundColor: isActual ? entry.color : 'transparent',
                                              borderTop: isActual ? undefined : `2px dashed ${entry.color}`,
                                              height: isActual ? 2.5 : 0
                                            }}
                                        />
                                      </div>
                                      <span style={{ color: entry.color }}>{entry.value}</span>
                                   </div>
                                 )
                               })}
                             </div>
                          )}
                        />
                        {/* Actual - Solid thick line */}
                        <Line
                          type="monotone"
                          dataKey="actual"
                          stroke={config.color}
                          strokeWidth={2.5}
                          dot={false}
                          connectNulls={false}
                          name="Actual"
                          isAnimationActive={false}
                        />
                        {/* Predicted (backtested) - REMOVED per user request */}
                        {/* Future Forecast - Dotted line in purple */}
                        <Line
                          type="monotone"
                          dataKey="forecast"
                          stroke="#a855f7"
                          strokeWidth={2}
                          strokeDasharray="2 2"
                          dot={false}
                          connectNulls={false}
                          name="Forecast"
                          isAnimationActive={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>

                )}
              </CardContent>
            </Card>
          )
        })}
      </div>


    </div>
  )
}
