import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { PanelRight } from "lucide-react"
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, Cell, Pie, PieChart, Legend } from "recharts"
import { Heatmap } from "@/components/Heatmap"
import { useSidebar } from "@/components/SidebarContext"
import { useChartTooltipStyles } from "@/hooks/useChartTooltipStyles"
import { useBusinessTransaction } from "@/components/BusinessTransactionContext"
import { DateRangePicker } from "@/components/DateRangePicker"
import { useDateRange } from "@/components/DateRangeContext"
import { GlobalSearch } from "@/components/GlobalSearch"
import InfoTooltip from "@/components/InfoTooltip"

interface SlowCallsData {
  total: number;
  peak_hour: string;
  peak_day: string;
  hourly: number[];
  daily: number[];
  heatmap: {
    index: string[];
    data: number[][];
  };
  trend: {
    labels: string[];
    values: number[];
  };
  impact: {
    'Business Hours (8-18)': number;
    'Off-Hours': number;
  };
}

export default function SlowCallsAnalysis() {
  const [data, setData] = useState<SlowCallsData | null>(null)
  const [metricType, setMetricType] = useState("slow")
  const { toggleSidebar } = useSidebar()
  const tooltipStyles = useChartTooltipStyles()
  const { selectedTier, selectedTransaction } = useBusinessTransaction()
  const { dateRange } = useDateRange()

  const fetchAnalysis = async () => {
    try {
      const params = new URLSearchParams({
        timeframe: dateRange.timeframe,
        type: metricType,
        tier: selectedTier,
        bt: selectedTransaction
      })
      if (dateRange.from) params.append('start_date', dateRange.from.toISOString())
      if (dateRange.to) params.append('end_date', dateRange.to.toISOString())

      const response = await fetch(`/api/slow-calls-analysis?${params}`)
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
  }, [metricType, selectedTier, selectedTransaction, dateRange])

  const hourlyData = data?.hourly.map((count, i) => ({ hour: `${i}:00`, count })) || []

  const trendData = data?.trend.labels.map((label, i) => ({
    date: label,
    count: data.trend.values[i]
  })) || []

  const impactData = data ? [
    { name: 'Business Hours (8-18)', value: data.impact['Business Hours (8-18)'], color: '#ef4444' },
    { name: 'Off-Hours', value: data.impact['Off-Hours'], color: '#94a3b8' }
  ] : []

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
            <h1 className="text-lg font-semibold">Business Transaction - Slow</h1>
            <p className="text-xs text-muted-foreground">{dateRange.from && dateRange.to ? "Selected Period" : `Historical Analysis (Default View)`}</p>
          </div>
        </div>
        <div className="flex gap-3">
          <GlobalSearch />
          <DateRangePicker />
          <Select value={metricType} onValueChange={setMetricType}>
            <SelectTrigger className="w-[200px] bg-card">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="slow">Slow Calls</SelectItem>
              <SelectItem value="veryslow">Very Slow Calls</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>



      {
        data ? (
          <>
            {/* Stats Grid */}
            <div className="grid gap-4 grid-cols-3">
              <Card className="bg-card">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-muted-foreground flex items-center">
                    Total Slow Calls
                    <InfoTooltip content="Total number of requests exceeding the slow threshold." />
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold">{data.total.toLocaleString()}</div>
                </CardContent>
              </Card>
              <Card className="bg-card">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-muted-foreground flex items-center">
                    Peak Time (Jam Paling Lambat)
                    <InfoTooltip content="Hour of the day when most slow calls occur." />
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-red-500">{data.peak_hour}</div>
                </CardContent>
              </Card>
              <Card className="bg-card">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-muted-foreground flex items-center">
                    Peak Day (Hari Paling Lambat)
                    <InfoTooltip content="Day of the week with the highest frequency of slow calls." />
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-orange-500">{data.peak_day}</div>
                </CardContent>
              </Card>
            </div>

            {/* Trend Line Chart */}
            <Card className="bg-card">
              <CardHeader>
                <CardTitle className="text-base font-semibold">Daily Trend: Are things getting better?</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-[300px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={trendData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                      <XAxis
                        dataKey="date"
                        stroke="#666"
                        fontSize={12}
                        tickLine={false}
                        axisLine={false}
                        tickFormatter={(value) => {
                          const date = new Date(value)
                          return `${date.getMonth() + 1}/${date.getDate()}`
                        }}
                      />
                      <YAxis stroke="#666" fontSize={12} tickLine={false} axisLine={false} />
                      <Tooltip
                        contentStyle={tooltipStyles.contentStyle}
                        labelFormatter={(value) => new Date(value).toLocaleDateString()}
                      />
                      <Line
                        type="monotone"
                        dataKey="count"
                        stroke="#0ea5e9"
                        strokeWidth={2}
                        dot={false}
                        fill="url(#colorGradient)"
                      />
                      <defs>
                        <linearGradient id="colorGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                {/* Visual Explanation of Axes - Trend */}
                <div className="mt-4 pt-4 border-t border-border/50">
                  <div className="flex flex-wrap items-center justify-center gap-x-8 gap-y-2 text-sm text-muted-foreground">
                    <div className="flex items-center gap-2">
                      <span className="flex items-center justify-center w-5 h-5 rounded-md bg-muted font-mono text-xs font-bold text-foreground">X</span>
                      <span>Date (Month/Day)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="flex items-center justify-center w-5 h-5 rounded-md bg-muted font-mono text-xs font-bold text-foreground">Y</span>
                      <span>Daily Slow Calls Count</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Heatmap */}
            <Card className="bg-card">
              <CardHeader>
                <CardTitle className="text-base font-semibold">Intensity Heatmap: When do slow calls happen?</CardTitle>
              </CardHeader>
              <CardContent>
                <Heatmap data={data.heatmap} colorScheme="red" />

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

            {/* Bottom Row: Hourly Distribution + Business Impact */}
            <div className="grid gap-4 grid-cols-[2fr_1fr]">
              <Card className="bg-card">
                <CardHeader>
                  <CardTitle className="text-base font-semibold">Hourly Distribution</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[350px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={hourlyData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                        <XAxis dataKey="hour" stroke="#666" fontSize={12} tickLine={false} axisLine={false} />
                        <YAxis stroke="#666" fontSize={12} tickLine={false} axisLine={false} />
                        <Tooltip contentStyle={tooltipStyles.contentStyle} />
                        <Bar dataKey="count" fill="#ef4444" radius={[4, 4, 0, 0]} />
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
                        <span>Total Slow Calls</span>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-card">
                <CardHeader>
                  <CardTitle className="text-base font-semibold">Business Impact (Hours)</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[350px] w-full flex items-center justify-center">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={impactData}
                          cx="50%"
                          cy="50%"
                          innerRadius={60}
                          outerRadius={100}
                          paddingAngle={2}
                          dataKey="value"
                        >
                          {impactData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip contentStyle={tooltipStyles.contentStyle} />
                        <Legend
                          verticalAlign="bottom"
                          height={36}
                          iconType="circle"
                          formatter={(value) => <span className="text-sm">{value}</span>}
                        />
                      </PieChart>
                    </ResponsiveContainer>
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
