import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { PanelRight } from "lucide-react"
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { Heatmap } from "@/components/Heatmap"
import { useSidebar } from "@/components/SidebarContext"
import { useChartTooltipStyles } from "@/hooks/useChartTooltipStyles"
import { useBusinessTransaction } from "@/components/BusinessTransactionContext"
import { DateRangePicker } from "@/components/DateRangePicker"
import { useDateRange } from "@/components/DateRangeContext"

interface ErrorAnalysisData {
  total: number;
  peak_hour: string;
  peak_day: string;
  hourly: number[];
  daily: number[];
  heatmap: {
    index: string[];
    data: number[][];
  };
}

export default function ErrorAnalysis() {
  const [data, setData] = useState<ErrorAnalysisData | null>(null)
  const [timeframe, setTimeframe] = useState("all")
  const { toggleSidebar } = useSidebar()
  const tooltipStyles = useChartTooltipStyles()
  const { selectedTier, selectedTransaction } = useBusinessTransaction()
  const { dateRange } = useDateRange()

  const fetchAnalysis = async () => {
    try {
      const params = new URLSearchParams({
        timeframe,
        tier: selectedTier,
        bt: selectedTransaction
      })
      if (dateRange.from) params.append('start_date', dateRange.from.toISOString())
      if (dateRange.to) params.append('end_date', dateRange.to.toISOString())
      
      const response = await fetch(`/api/error-analysis?${params}`)
      const result = await response.json()
      if (!result.error) {
        setData(result)
      }
    } catch (e) {
      console.error(e)
    }
  }

  useEffect(() => {
    fetchAnalysis()
  }, [timeframe, selectedTier, selectedTransaction, dateRange])

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
            <h1 className="text-lg font-semibold">Business Transaction - Error</h1>
            <p className="text-xs text-muted-foreground">Historical Analysis of "Errors per Minute" (Last 3 Years)</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <DateRangePicker />
          <Select value={timeframe} onValueChange={setTimeframe}>
          <SelectTrigger className="w-[200px] bg-card">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Time (Lifetime)</SelectItem>
            <SelectItem value="1y">Last 1 Year</SelectItem>
            <SelectItem value="6m">Last 6 Months</SelectItem>
            <SelectItem value="30d">Last 30 Days</SelectItem>
            <SelectItem value="7d">Last 7 Days</SelectItem>
          </SelectContent>
        </Select>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid gap-4 grid-cols-3">
        <Card className="bg-card">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Total Errors Detected</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{data?.total.toLocaleString() || 'Loading...'}</div>
          </CardContent>
        </Card>
        <Card className="bg-card">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Peak Error Time</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-red-400">{data?.peak_hour || '--:--'}</div>
          </CardContent>
        </Card>
        <Card className="bg-card">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Highest Error Day</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-orange-400">{data?.peak_day || '--'}</div>
          </CardContent>
        </Card>
      </div>

      {/* Heatmap */}
      <Card className="bg-card">
        <CardHeader>
          <CardTitle className="text-base font-semibold">Weekly Heatmap: When do errors happen?</CardTitle>
        </CardHeader>
        <CardContent>
          {data?.heatmap ? (
            <Heatmap data={data.heatmap} colorScheme="red" />
          ) : (
            <div className="text-sm text-muted-foreground text-center py-8">Loading...</div>
          )}
        </CardContent>
      </Card>

      {/* Charts */}
      <div className="grid gap-4 grid-cols-2">
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
                  <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card">
          <CardHeader>
            <CardTitle className="text-base font-semibold">Daily Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[350px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={dailyData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" horizontal={false} />
                  <XAxis type="number" stroke="#666" fontSize={12} tickLine={false} axisLine={false} />
                  <YAxis type="category" dataKey="day" stroke="#666" fontSize={12} tickLine={false} axisLine={false} />
                  <Tooltip contentStyle={tooltipStyles.contentStyle} />
                  <Bar dataKey="count" fill="#ef4444" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
