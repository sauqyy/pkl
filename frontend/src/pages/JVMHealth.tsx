import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { RefreshCw, PanelRight, AlertCircle, AlertTriangle, CheckCircle, Info } from "lucide-react"
import { Line, Bar } from 'react-chartjs-2'
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend, ChartOptions, TimeScale } from 'chart.js'
import 'chartjs-adapter-date-fns'
import { useSidebar } from "@/components/SidebarContext"
import { useBusinessTransaction } from "@/components/BusinessTransactionContext"
import { DateRangePicker } from "@/components/DateRangePicker"
import { useDateRange } from "@/components/DateRangeContext"

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend, TimeScale)

interface JVMData {
  data: {
    availability: Array<{ t: number; v: number }>
    heap_used: Array<{ t: number; v: number }>
    gc_time: Array<{ t: number; v: number }>
    gc_count: Array<{ t: number; v: number }>
    cpu_busy: Array<{ t: number; v: number }>
    threads_live: Array<{ t: number; v: number }>
  }
  insights: Array<{
    type: 'critical' | 'warning' | 'success' | 'info'
    msg: string
  }>
  tier?: string
}

export default function JVMHealth() {
  const [data, setData] = useState<JVMData | null>(null)
  const [loading, setLoading] = useState(true)
  const [duration, setDuration] = useState("60")
  const { toggleSidebar } = useSidebar()
  const { selectedTier } = useBusinessTransaction()
  const { dateRange } = useDateRange()

  const loadData = async (mins: string = duration) => {
    try {
      setLoading(true)
      const params = new URLSearchParams({
        duration: mins,
        tier: selectedTier
      })
      if (dateRange.from) params.append('start_date', dateRange.from.toISOString())
      if (dateRange.to) params.append('end_date', dateRange.to.toISOString())
      
      const response = await fetch(`/api/jvm-data?${params}`)
      if (!response.ok) throw new Error('Failed to fetch')
      const result = await response.json()
      setData(result)
    } catch (error) {
      console.error('Failed to fetch JVM data:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
  }, [duration, selectedTier, dateRange])

  if (loading || !data) {
    return <div className="p-8 text-muted-foreground">Loading JVM health data...</div>
  }

  const getAvailability = () => {
    if (!data.data.availability || data.data.availability.length === 0) return { percent: 'N/A', isHealthy: false }
    const avg = data.data.availability.reduce((sum, d) => sum + d.v, 0) / data.data.availability.length
    const pct = (avg * 100).toFixed(2)
    return { percent: `${pct}%`, isHealthy: avg >= 0.99 }
  }

  const availability = getAvailability()

  const lineOptions: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: { type: 'time', time: { unit: 'minute' }, ticks: { display: false } },
      y: { beginAtZero: true }
    }
  }

  const heapOptions: ChartOptions<'line'> = {
    ...lineOptions,
    scales: {
      ...lineOptions.scales,
      y: { beginAtZero: true, max: 100 }
    }
  }

  const createLineChartData = (dataPoints: Array<{ t: number; v: number}>, label: string, color: string) => ({
    labels: dataPoints.map(d => new Date(d.t)),
    datasets: [{
      label,
      data: dataPoints.map(d => d.v),
      borderColor: color,
      backgroundColor: color.replace(')', ', 0.1)').replace('rgb', 'rgba'),
      fill: true,
      tension: 0.4,
      pointRadius: 0
    }]
  })

  const createBarChartData = (dataPoints: Array<{ t: number; v: number}>, label: string, color: string) => ({
    labels: dataPoints.map(d => new Date(d.t)),
    datasets: [{
      label,
      data: dataPoints.map(d => d.v),
      backgroundColor: color
    }]
  })

  const getInsightIcon = (type: string) => {
    switch(type) {
      case 'critical': return <AlertCircle className="h-5 w-5 text-red-500" />
      case 'warning': return <AlertTriangle className="h-5 w-5 text-yellow-500" />
      case 'success': return <CheckCircle className="h-5 w-5 text-green-500" />
      default: return <Info className="h-5 w-5 text-blue-500" />
    }
  }

  const getInsightBgClass = (type: string) => {
    switch(type) {
      case 'critical': return 'bg-red-50 border-red-200 text-red-800 dark:bg-red-900/20 dark:border-red-800 dark:text-red-400'
      case 'warning': return 'bg-yellow-50 border-yellow-200 text-yellow-800 dark:bg-yellow-900/20 dark:border-yellow-800 dark:text-yellow-400'
      case 'success': return 'bg-green-50 border-green-200 text-green-800 dark:bg-green-900/20 dark:border-green-800 dark:text-green-400'
      default: return 'bg-blue-50 border-blue-200 text-blue-800 dark:bg-blue-900/20 dark:border-blue-800 dark:text-blue-400'
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button type="button" onClick={(e) => { e.stopPropagation(); toggleSidebar() }} className="p-2 hover:bg-accent rounded-md transition-colors" aria-label="Toggle sidebar">
            <PanelRight className="h-5 w-5" />
          </button>
          <div className="h-6 w-px bg-border"></div>
          <div>
            <h1 className="text-lg font-semibold">Tier - {selectedTier}</h1>
            <p className="text-xs text-muted-foreground">Availability, Memory & Garbage Collection</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <DateRangePicker />
          <Select value={duration} onValueChange={setDuration}>
            <SelectTrigger className="w-[180px] bg-card"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="60">Last 1 Hour</SelectItem>
              <SelectItem value="360">Last 6 Hours</SelectItem>
              <SelectItem value="720">Last 12 Hours</SelectItem>
              <SelectItem value="1440">Last 1 Day</SelectItem>
              <SelectItem value="4320">Last 3 Days</SelectItem>
              <SelectItem value="10080">Last 1 Week</SelectItem>
            </SelectContent>
          </Select>
          <button onClick={() => loadData()} className="flex items-center gap-2 px-3 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors">
            <RefreshCw className="h-4 w-4" /> Refresh
          </button>
        </div>
      </div>

      <div className="grid gap-4 grid-cols-2">
        <Card className="bg-card">
          <CardHeader>
            <CardTitle className="text-base font-semibold flex items-center gap-2">
              <Info className="h-5 w-5" /> Smart Insights
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 max-h-[200px] overflow-y-auto">
              {data.insights.length === 0 ? (
                <div className="text-muted-foreground text-sm italic">No critical issues detected. System appears stable.</div>
              ) : (
                data.insights.map((insight, idx) => (
                  <div key={idx} className={`flex items-start gap-3 p-3 rounded-lg border ${getInsightBgClass(insight.type)}`}>
                    {getInsightIcon(insight.type)}
                    <p className="text-sm">{insight.msg}</p>
                  </div>
                ))
              )}
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card">
          <CardHeader><CardTitle className="text-base font-semibold">System Availability</CardTitle></CardHeader>
          <CardContent>
            <div className="flex flex-col items-center justify-center h-[200px]">
              <div className={`text-5xl font-bold ${availability.isHealthy ? 'text-green-500' : 'text-red-500'}`}>{availability.percent}</div>
              <p className="text-sm text-muted-foreground mt-2">Uptime (Selected Period)</p>
              <p className={`text-sm font-semibold mt-1 ${availability.isHealthy ? 'text-green-500' : 'text-red-500'}`}>
                {availability.isHealthy ? 'System Healthy' : 'Downtime Detected'}
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 grid-cols-2">
        <Card className="bg-card">
          <CardHeader><CardTitle className="text-base font-semibold">Heap Memory (%)</CardTitle></CardHeader>
          <CardContent>
            <div className="h-[300px]">
              {data.data.heap_used && <Line data={createLineChartData(data.data.heap_used, 'Heap Used (%)', 'rgb(99, 102, 241)')} options={heapOptions} />}
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card">
          <CardHeader><CardTitle className="text-base font-semibold">CPU Usage (%)</CardTitle></CardHeader>
          <CardContent>
            <div className="h-[300px]">
              {data.data.cpu_busy && <Line data={createLineChartData(data.data.cpu_busy, 'CPU Busy %', 'rgb(6, 182, 212)')} options={heapOptions} />}
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 grid-cols-2">
        <Card className="bg-card">
          <CardHeader><CardTitle className="text-base font-semibold">Live Threads</CardTitle></CardHeader>
          <CardContent>
            <div className="h-[300px]">
              {data.data.threads_live && <Line data={createLineChartData(data.data.threads_live, 'Active Threads', 'rgb(139, 92, 246)')} options={lineOptions} />}
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card">
          <CardHeader><CardTitle className="text-base font-semibold">CPU Usage (%)</CardTitle></CardHeader>
          <CardContent>
            <div className="h-[300px]">
              {data.data.cpu_busy && <Line data={createLineChartData(data.data.cpu_busy, 'CPU Busy %', 'rgb(6, 182, 212)')} options={heapOptions} />}
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 grid-cols-2">
        <Card className="bg-card">
          <CardHeader><CardTitle className="text-base font-semibold">Major GC Time (ms/min)</CardTitle></CardHeader>
          <CardContent>
            <div className="h-[300px]">
              {data.data.gc_time && <Bar data={createBarChartData(data.data.gc_time, 'Major GC Time (ms/min)', 'rgb(245, 158, 11)')} options={lineOptions} />}
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card">
          <CardHeader><CardTitle className="text-base font-semibold">Number of Major Collections (/min)</CardTitle></CardHeader>
          <CardContent>
            <div className="h-[300px]">
              {data.data.gc_count && <Bar data={createBarChartData(data.data.gc_count, 'Major Collections/min', 'rgb(236, 72, 153)')} options={lineOptions} />}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
