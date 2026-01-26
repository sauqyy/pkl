import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { RefreshCw, PanelRight } from "lucide-react"
import { Bar, Doughnut, Scatter } from 'react-chartjs-2'
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, BarElement, ArcElement, Title, Tooltip, Legend, ChartOptions } from 'chart.js'
import { useSidebar } from "@/components/SidebarContext"
import { DateRangePicker } from "@/components/DateRangePicker"
import { useDateRange } from "@/components/DateRangeContext"
import InfoTooltip from "@/components/InfoTooltip"

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, ArcElement, Title, Tooltip, Legend)

interface BusinessTransaction {
  Name: string
  Health: 'Normal' | 'Warning' | 'Critical'
  'Response Time (ms)': number
  Calls: number
  '% Errors': number
  Tier?: string
}


interface BusinessTransactionsData {
  health_counts: Record<string, number>
  top_slowest: { labels: string[]; values: number[] }
  top_volume: { labels: string[]; values: number[] }
  top_errors: { labels: string[]; values: number[] }
  scatter: BusinessTransaction[]
  table: BusinessTransaction[]
}

export default function BusinessTransactions() {
  const [data, setData] = useState<BusinessTransactionsData | null>(null)
  const [loading, setLoading] = useState(true)
  const { toggleSidebar } = useSidebar()
  const { dateRange } = useDateRange()

  const loadData = async () => {
    try {
      setLoading(true)
      const params = new URLSearchParams()
      if (dateRange.from) params.append('start_date', dateRange.from.toISOString())
      if (dateRange.to) params.append('end_date', dateRange.to.toISOString())

      const response = await fetch(`/api/business-transactions?${params}`)
      if (!response.ok) throw new Error('Failed to fetch')
      const result = await response.json()
      setData(result)
    } catch (error) {
      console.error('Failed to fetch business transactions:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
  }, [dateRange])

  if (loading || !data) {
    return <div className="p-8 text-muted-foreground">Loading business transactions...</div>
  }

  const healthChartData = {
    labels: ['Normal', 'Warning', 'Critical'],
    datasets: [{
      data: [data.health_counts['Normal'] || 0, data.health_counts['Warning'] || 0, data.health_counts['Critical'] || 0],
      backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
      borderWidth: 0
    }]
  }

  const doughnutOptions: ChartOptions<'doughnut'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { position: 'bottom' } }
  }

  const slowestChartData = {
    labels: data.top_slowest.labels,
    datasets: [{
      label: 'Response Time (ms)',
      data: data.top_slowest.values,
      backgroundColor: '#6366f1',
      borderRadius: 4
    }]
  }

  const horizontalBarOptions: ChartOptions<'bar'> = {
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    scales: { x: { beginAtZero: true } }
  }

  const volumeChartData = {
    labels: data.top_volume.labels,
    datasets: [{
      label: 'Calls',
      data: data.top_volume.values,
      backgroundColor: '#3b82f6',
      borderRadius: 4
    }]
  }

  const errorChartData = {
    labels: data.top_errors.labels,
    datasets: [{
      label: 'Error Rate (%)',
      data: data.top_errors.values,
      backgroundColor: '#ef4444',
      borderRadius: 4
    }]
  }

  const barOptions: ChartOptions<'bar'> = {
    responsive: true,
    maintainAspectRatio: false,
    scales: { y: { beginAtZero: true } }
  }

  const scatterChartData = {
    datasets: [{
      label: 'Transactions',
      data: data.scatter.map(item => ({ x: item.Calls, y: item['Response Time (ms)'], name: item.Name })),
      backgroundColor: data.scatter.map(item => {
        if (item.Health === 'Critical') return '#ef4444'
        if (item.Health === 'Warning') return '#f59e0b'
        return '#10b981'
      })
    }]
  }

  const scatterOptions: ChartOptions<'scatter'> = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: { title: { display: true, text: 'Calls (Volume)' }, beginAtZero: true },
      y: { title: { display: true, text: 'Response Time (ms)' }, beginAtZero: true }
    },
    plugins: {
      tooltip: {
        callbacks: {
          label: function (context: any) {
            const point = context.raw
            return `${point.name}: ${point.y}ms (${point.x} calls)`
          }
        }
      }
    }
  }

  const getHealthBadgeClass = (health: string) => {
    switch (health) {
      case 'Normal': return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400'
      case 'Warning': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400'
      case 'Critical': return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400'
      default: return 'bg-gray-100 text-gray-800'
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
            <h1 className="text-lg font-semibold">Business Transaction - Analysis</h1>
            <p className="text-xs text-muted-foreground">Load, Response Time, Errors & Slow Calls</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <DateRangePicker />
          <button onClick={loadData} className="flex items-center gap-2 px-3 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors">
            <RefreshCw className="h-4 w-4" /> Refresh Data
          </button>
        </div>
      </div>

      <div className="grid gap-4 grid-cols-4">
        <StatCard label="Total Transactions" value={data.table.length.toString()} description="Total number of transactions monitored." />
        <StatCard label="Healthy (Normal)" value={(data.health_counts['Normal'] || 0).toString()} valueColor="text-green-500" description="Transactions performing within acceptable limits." />
        <StatCard label="Warning" value={(data.health_counts['Warning'] || 0).toString()} valueColor="text-yellow-500" description="Transactions showing signs of performance degradation." />
        <StatCard label="Critical" value={(data.health_counts['Critical'] || 0).toString()} valueColor="text-red-500" description="Transactions failing or exceeding critical thresholds." />
      </div>

      <div className="grid gap-4 grid-cols-3">
        <Card className="bg-card">
          <CardHeader>
            <CardTitle className="text-base font-semibold flex items-center">
              Health Distribution
              <InfoTooltip content="Breakdown of transactions by health status." />
            </CardTitle>
            <p className="text-xs text-muted-foreground mt-1">Transaction health categories</p>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]"><Doughnut data={healthChartData} options={doughnutOptions} /></div>
          </CardContent>
        </Card>

        <Card className="col-span-2 bg-card">
          <CardHeader>
            <CardTitle className="text-base font-semibold flex items-center">
              Top 10 Slowest Transactions
              <InfoTooltip content="Transactions with the highest average response times." />
            </CardTitle>
            <p className="text-xs text-muted-foreground mt-1">Highest response times</p>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]"><Bar data={slowestChartData} options={horizontalBarOptions} /></div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 grid-cols-2">
        <Card className="bg-card">
          <CardHeader>
            <CardTitle className="text-base font-semibold flex items-center">
              Top 10 High Volume Transactions
              <InfoTooltip content="Transactions with the most execution counts." />
            </CardTitle>
          </CardHeader>
          <CardContent><div className="h-[350px]"><Bar data={volumeChartData} options={barOptions} /></div></CardContent>
        </Card>
        <Card className="bg-card">
          <CardHeader>
            <CardTitle className="text-base font-semibold flex items-center">
              Transactions with Highest Error Rates
              <InfoTooltip content="Transactions encountering the most errors." />
            </CardTitle>
          </CardHeader>
          <CardContent><div className="h-[350px]"><Bar data={errorChartData} options={barOptions} /></div></CardContent>
        </Card>
      </div>

      <Card className="bg-card">
        <CardHeader>
          <CardTitle className="text-base font-semibold flex items-center">
            Calls vs Response Time (Correlation)
            <InfoTooltip content="Scatter plot showing relationship between load and latency." />
          </CardTitle>
          <p className="text-xs text-muted-foreground mt-1">Color indicates health status</p>
        </CardHeader>
        <CardContent><div className="h-[350px]"><Scatter data={scatterChartData} options={scatterOptions} /></div></CardContent>
      </Card>

      <Card className="bg-card">
        <CardHeader><CardTitle className="text-base font-semibold">Detailed Transaction List</CardTitle></CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="text-left p-3 font-medium text-muted-foreground">Name</th>
                  <th className="text-left p-3 font-medium text-muted-foreground">Health</th>
                  <th className="text-left p-3 font-medium text-muted-foreground">Response Time (ms)</th>
                  <th className="text-left p-3 font-medium text-muted-foreground">Calls</th>
                  <th className="text-left p-3 font-medium text-muted-foreground">% Errors</th>
                  {data.table.some(t => t.Tier) && <th className="text-left p-3 font-medium text-muted-foreground">Tier</th>}
                </tr>
              </thead>
              <tbody>
                {data.table.map((txn, index) => (
                  <tr key={index} className="border-b hover:bg-accent/50 transition-colors">
                    <td className="p-3 font-medium">{txn.Name}</td>
                    <td className="p-3">
                      <span className={`px-2 py-1 rounded-full text-xs font-semibold ${getHealthBadgeClass(txn.Health)}`}>
                        {txn.Health}
                      </span>
                    </td>
                    <td className="p-3">{txn['Response Time (ms)'].toLocaleString()}</td>
                    <td className="p-3">{txn.Calls.toLocaleString()}</td>
                    <td className="p-3">{txn['% Errors']}%</td>
                    {data.table.some(t => t.Tier) && <td className="p-3">{txn.Tier || '-'}</td>}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

        </CardContent>
      </Card>
    </div>
  )
}

function StatCard({ label, value, valueColor = "text-foreground", description }: { label: string; value: string; valueColor?: string; description?: string }) {
  return (
    <Card className="bg-card">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground flex items-center">
          {label}
          {description && <InfoTooltip content={description} />}
        </CardTitle>
      </CardHeader>
      <CardContent><div className={`text-3xl font-bold ${valueColor}`}>{value}</div></CardContent>
    </Card>
  )
}
