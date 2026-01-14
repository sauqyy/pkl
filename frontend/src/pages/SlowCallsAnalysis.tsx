import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { PanelRight } from "lucide-react"
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, Cell, Pie, PieChart, Legend } from "recharts"
import { Heatmap } from "@/components/Heatmap"
import { useSidebar } from "@/components/SidebarContext"

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
  const [timeframe, setTimeframe] = useState("30d")
  const [metricType, setMetricType] = useState("slow")
  const { toggleSidebar } = useSidebar()

  const fetchAnalysis = async () => {
    try {
      const response = await fetch(`/api/slow-calls-analysis?timeframe=${timeframe}&type=${metricType}`)
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
  }, [timeframe, metricType])

  const hourlyData = data?.hourly.map((count, i) => ({ hour: `${i}:00`, count })) || []
  
  const trendData = data?.trend.labels.map((label, i) => ({
    date: label,
    count: data.trend.values[i]
  })) || []

  const impactData = data ? [
    { name: 'Business Hours (8-18)', value: data.impact['Business Hours (8-18)'], color: '#ef4444' },
    { name: 'Off-Hours', value: data.impact['Off-Hours'], color: '#94a3b8' }
  ] : []

  const subtitle = metricType === 'veryslow' 
    ? 'Historical Analysis of "Very Slow Calls"'
    : 'Historical Analysis of "Slow Calls"'

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
            <h1 className="text-lg font-semibold">Slow Calls Analysis</h1>
            <p className="text-xs text-muted-foreground">{subtitle}</p>
          </div>
        </div>
        <div className="flex gap-3">
          <Select value={metricType} onValueChange={setMetricType}>
            <SelectTrigger className="w-[200px] bg-card">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="slow">Slow Calls (&gt;X ms)</SelectItem>
              <SelectItem value="veryslow">Very Slow Calls (&gt;Y ms)</SelectItem>
            </SelectContent>
          </Select>
          <Select value={timeframe} onValueChange={setTimeframe}>
            <SelectTrigger className="w-[200px] bg-card">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Time</SelectItem>
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
            <CardTitle className="text-sm font-medium text-muted-foreground">Total Slow Calls</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{data?.total.toLocaleString() || '-'}</div>
          </CardContent>
        </Card>
        <Card className="bg-card">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Peak Time (Jam Paling Lambat)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-red-500">{data?.peak_hour || '--:--'}</div>
          </CardContent>
        </Card>
        <Card className="bg-card">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Peak Day (Hari Paling Lambat)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-orange-500">{data?.peak_day || '--'}</div>
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
                  contentStyle={{ backgroundColor: '#111', border: '1px solid #333' }}
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
                    <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0}/>
                  </linearGradient>
                </defs>
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Heatmap */}
      <Card className="bg-card">
        <CardHeader>
          <CardTitle className="text-base font-semibold">Intensity Heatmap: When do slow calls happen?</CardTitle>
        </CardHeader>
        <CardContent>
          {data?.heatmap ? (
            <Heatmap data={data.heatmap} colorScheme="red" />
          ) : (
            <div className="text-sm text-muted-foreground text-center py-8">Loading...</div>
          )}
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
                  <Tooltip contentStyle={{ backgroundColor: '#111', border: '1px solid #333' }} />
                  <Bar dataKey="count" fill="#ef4444" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
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
                  <Tooltip contentStyle={{ backgroundColor: '#111', border: '1px solid #333' }} />
                  <Legend 
                    verticalAlign="bottom" 
                    height={36}
                    formatter={(value) => <span className="text-sm">{value}</span>}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
