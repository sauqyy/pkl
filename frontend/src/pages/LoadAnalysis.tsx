import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { PanelRight } from "lucide-react"
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { Heatmap } from "@/components/Heatmap"
import { useSidebar } from "@/components/SidebarContext"
import { useChartTooltipStyles } from "@/hooks/useChartTooltipStyles"
import { useBusinessTransaction } from "@/components/BusinessTransactionContext"
import { DateRangePicker } from "@/components/DateRangePicker"
import { useDateRange } from "@/components/DateRangeContext"
import { GlobalSearch } from "@/components/GlobalSearch"
import InfoTooltip from "@/components/InfoTooltip"

interface LoadAnalysisData {
  total: number;
  peak_hour: string;
  peak_day: string;
  hourly: number[];
  daily: number[];
  min: number;
  max: number;
  heatmap: {
    index: string[];
    data: number[][];
  };
}

export default function LoadAnalysis() {
  const [data, setData] = useState<LoadAnalysisData | null>(null)
  const { toggleSidebar } = useSidebar()
  const tooltipStyles = useChartTooltipStyles()
  const { selectedTier, selectedTransaction } = useBusinessTransaction()
  const { dateRange } = useDateRange()

  const fetchAnalysis = async () => {
    try {
      const params = new URLSearchParams({
        timeframe: dateRange.timeframe,
        tier: selectedTier,
        bt: selectedTransaction
      })
      if (dateRange.from) params.append('start_date', dateRange.from.toISOString())
      if (dateRange.to) params.append('end_date', dateRange.to.toISOString())

      const response = await fetch(`/api/load-analysis?${params}`)
      const result = await response.json()
      if (!result.error) {
        setData(result)
      }
    } catch (e) {
      console.error(e)
    }
  }

  useEffect(() => {
    setData(null) // Reset data to trigger loading state
    fetchAnalysis()
  }, [selectedTier, selectedTransaction, dateRange])

  const hourlyData = data?.hourly.map((count, i) => ({ hour: `${i}:00`, count })) || []
  const dailyData = data?.daily.map((count, i) => ({ day: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][i], count })) || []

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
            <h1 className="text-lg font-semibold">Business Transaction - Load</h1>
            <p className="text-xs text-muted-foreground">{dateRange.from && dateRange.to ? "Selected Period" : "Historical Analysis (Default View)"}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <GlobalSearch />
          <DateRangePicker />
        </div>
      </div>



      {
        data ? (
          <>
            {/* Stats Grid */}
            <div className="grid gap-4 grid-cols-5">
              <Card className="bg-card">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-muted-foreground flex items-center">
                    Total Calls
                    <InfoTooltip content="Total number of HTTP requests processed." />
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{data.total.toLocaleString()}</div>
                </CardContent>
              </Card>
              <Card className="bg-card">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-muted-foreground flex items-center">
                    Min Load
                    <InfoTooltip content="Lowest recorded calls per minute/hour." />
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-gray-400">{data.min.toLocaleString()}</div>
                </CardContent>
              </Card>
              <Card className="bg-card">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-muted-foreground flex items-center">
                    Max Load
                    <InfoTooltip content="Highest recorded calls per minute/hour." />
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-blue-400">{data.max.toLocaleString()}</div>
                </CardContent>
              </Card>
              <Card className="bg-card">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-muted-foreground flex items-center">
                    Peak Time
                    <InfoTooltip content="Time with the highest traffic volume." />
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-xl font-bold text-orange-400 truncate" title={data.peak_hour}>{data.peak_hour}</div>
                </CardContent>
              </Card>
              <Card className="bg-card">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-muted-foreground flex items-center">
                    Peak Day
                    <InfoTooltip content="Day of the week with the most traffic." />
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-xl font-bold text-green-400 truncate" title={data.peak_day}>{data.peak_day}</div>
                </CardContent>
              </Card>
            </div>

            {/* Heatmap */}
            <Card className="bg-card">
              <CardHeader>
                <CardTitle className="text-base font-semibold">Weekly Heatmap: When is the server busiest?</CardTitle>
              </CardHeader>
              <CardContent>
                <Heatmap data={data.heatmap} colorScheme="blue" />

                {/* Visual Explanation of Axes - Heatmap */}
                <div className="mt-4 pt-4 border-t border-border/50">
                  <div className="flex flex-wrap items-center justify-center gap-x-8 gap-y-2 text-sm text-muted-foreground">
                    <div className="flex items-center gap-2">
                      <span className="flex items-center justify-center w-5 h-5 rounded-md bg-muted font-mono text-xs font-bold text-foreground">X</span>
                      <span>{['5m', '15m'].includes(dateRange.timeframe) ? 'Minute' : (dateRange.timeframe === '1h' ? 'Minute Index' : 'Time of Day (Hour)')}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="flex items-center justify-center w-5 h-5 rounded-md bg-muted font-mono text-xs font-bold text-foreground">Y</span>
                      <span>{dateRange.timeframe === '24h' ? 'Date' : (['5m', '15m', '1h'].includes(dateRange.timeframe) ? 'Current' : 'Day of the Week')}</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Charts */}
            <div className="grid gap-4 grid-cols-2">
              <Card className="bg-card">
                <CardHeader>
                  <CardTitle className="text-base font-semibold">Hourly Load Distribution</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[350px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={hourlyData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                        <XAxis dataKey="hour" stroke="#666" fontSize={12} tickLine={false} axisLine={false} />
                        <YAxis stroke="#666" fontSize={12} tickLine={false} axisLine={false} />
                        <Tooltip contentStyle={tooltipStyles.contentStyle} />
                        <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Visual Explanation of Axes - Hourly */}
                  <div className="mt-4 pt-4 border-t border-border/50">
                    <div className="flex flex-wrap items-center justify-center gap-x-8 gap-y-2 text-sm text-muted-foreground">
                      <div className="flex items-center gap-2">
                        <span className="flex items-center justify-center w-5 h-5 rounded-md bg-muted font-mono text-xs font-bold text-foreground">X</span>
                        <span>Time (24-Hour Format)</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="flex items-center justify-center w-5 h-5 rounded-md bg-muted font-mono text-xs font-bold text-foreground">Y</span>
                        <span>Total Calls Processed</span>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-card">
                <CardHeader>
                  <CardTitle className="text-base font-semibold">Daily Load Distribution</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[350px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={dailyData} layout="vertical">
                        <CartesianGrid strokeDasharray="3 3" stroke="#333" horizontal={false} />
                        <XAxis type="number" stroke="#666" fontSize={12} tickLine={false} axisLine={false} />
                        <YAxis type="category" dataKey="day" stroke="#666" fontSize={12} tickLine={false} axisLine={false} />
                        <Tooltip contentStyle={tooltipStyles.contentStyle} />
                        <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Visual Explanation of Axes - Daily */}
                  <div className="mt-4 pt-4 border-t border-border/50">
                    <div className="flex flex-wrap items-center justify-center gap-x-8 gap-y-2 text-sm text-muted-foreground">
                      <div className="flex items-center gap-2">
                        <span className="flex items-center justify-center w-5 h-5 rounded-md bg-muted font-mono text-xs font-bold text-foreground">X</span>
                        <span>Total Calls Processed</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="flex items-center justify-center w-5 h-5 rounded-md bg-muted font-mono text-xs font-bold text-foreground">Y</span>
                        <span>Day of the Week</span>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </>
        ) : (
          <div className="flex h-[50vh] w-full items-center justify-center">
            <div className="flex flex-col items-center gap-2">
              <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
              <p className="text-muted-foreground">Loading dashboard data...</p>
            </div>
          </div>
        )
      }
    </div >
  )
}
