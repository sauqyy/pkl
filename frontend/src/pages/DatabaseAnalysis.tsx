import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { RefreshCw, PanelRight } from "lucide-react"
import { Line, Doughnut } from 'react-chartjs-2'
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend, ChartOptions, TimeScale, ArcElement } from 'chart.js'
import 'chartjs-adapter-date-fns'
import { useSidebar } from "@/components/SidebarContext"
import { DateRangePicker } from "@/components/DateRangePicker"
import { useDateRange } from "@/components/DateRangeContext"
import InfoTooltip from "@/components/InfoTooltip"

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend, TimeScale, ArcElement)

interface ChartDataDetails {
    timestamps: number[]
    time_spent: number[]
    load: number[]
    cpu: number[]
}

interface QueryData {
    id: string
    query: string
    elapsed_time: string
    executions: string
    avg_response: string
    weight: number
}

interface DatabaseData {
    chart_data: ChartDataDetails
    inefficiency_count: number
    total_points: number
    queries: QueryData[]
}

export default function DatabaseAnalysis() {
    const [data, setData] = useState<DatabaseData | null>(null)
    const [loading, setLoading] = useState(true)
    const [duration, setDuration] = useState("60")
    const { toggleSidebar } = useSidebar()
    const { dateRange } = useDateRange()

    const loadData = async (mins: string = duration) => {
        try {
            setLoading(true)
            const params = new URLSearchParams({ duration: mins })
            if (dateRange.from) params.append('start_date', dateRange.from.toISOString())
            if (dateRange.to) params.append('end_date', dateRange.to.toISOString())

            const response = await fetch(`/api/database-analysis?${params}`)
            if (!response.ok) throw new Error('Failed to fetch')
            const result = await response.json()
            setData(result)
        } catch (error) {
            console.error('Failed to fetch Database data:', error)
        } finally {
            setLoading(false)
        }
    }

    useEffect(() => {
        loadData()
    }, [duration, dateRange])

    if (loading || !data) {
        return <div className="p-8 text-muted-foreground">Loading specific database analysis data...</div>
    }

    // Calculate Metrics
    const timeVals = data.chart_data.time_spent || []
    const loadVals = data.chart_data.load || []
    const cpuVals = data.chart_data.cpu || []

    const avgTime = timeVals.length ? (timeVals.reduce((a, b) => a + b, 0) / timeVals.length).toFixed(1) : "0"
    const avgLoad = loadVals.length ? (loadVals.reduce((a, b) => a + b, 0) / loadVals.length).toFixed(0) : "0"
    const avgCpu = cpuVals.length ? (cpuVals.reduce((a, b) => a + b, 0) / cpuVals.length).toFixed(1) : "0"
    const maxTime = timeVals.length ? Math.max(...timeVals).toFixed(1) : "0"

    // -- Charts --
    const timestamps = data.chart_data.timestamps.map(ts => new Date(ts))

    // Main Chart: Time vs Load (Line + Bar combo)
    const mainChartData = {
        labels: timestamps,
        datasets: [
            {
                type: 'line' as const,
                label: 'Time Spent (s)',
                data: timeVals,
                borderColor: '#f59e0b', // Amber
                backgroundColor: 'rgba(245, 158, 11, 0.05)',
                borderWidth: 2,
                tension: 0.4,
                pointRadius: 0,
                yAxisID: 'y1',
                fill: false
            },
            {
                type: 'bar' as const,
                label: 'Load (CPM)',
                data: loadVals,
                backgroundColor: 'rgba(59, 130, 246, 0.5)', // Blue
                borderColor: '#3b82f6',
                borderWidth: 1,
                yAxisID: 'y',
                barPercentage: 0.6,
            }
        ]
    }

    const mainChartOptions: ChartOptions<'line' | 'bar'> = {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        scales: {
            x: { type: 'time', time: { unit: 'minute' }, grid: { display: false } },
            y: {
                type: 'linear', position: 'left',
                title: { display: true, text: 'Load (CPM)', color: '#3b82f6' },
                grid: { color: 'rgba(200,200,200,0.1)' }, beginAtZero: true
            },
            y1: {
                type: 'linear', position: 'right',
                title: { display: true, text: 'Time Spent (s)', color: '#f59e0b' },
                grid: { drawOnChartArea: false }, beginAtZero: true
            }
        }
    }

    // CPU Chart
    const cpuChartData = {
        labels: timestamps,
        datasets: [{
            label: 'CPU Busy (%)',
            data: cpuVals,
            borderColor: '#0ea5e9',
            backgroundColor: 'rgba(14, 165, 233, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.4,
            pointRadius: 0
        }]
    }

    const cpuChartOptions: ChartOptions<'line'> = {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: { legend: { display: false } },
        scales: {
            x: { type: 'time', time: { unit: 'minute' }, grid: { display: false } },
            y: { beginAtZero: true, max: 100, title: { display: true, text: 'Percent (%)' } }
        }
    }

    // Efficiency Donut
    const inefficient = data.inefficiency_count
    const total = data.total_points || 1
    const efficient = Math.max(0, total - inefficient)

    const donutData = {
        labels: ['Efficient (Load ≥ Time)', 'Inefficient (Time > Load)'],
        datasets: [{
            data: [efficient, inefficient],
            backgroundColor: ['#22c55e', '#ef4444'],
            borderWidth: 0
        }]
    }

    // Query Distribution Donut
    const topQueries = data.queries.slice(0, 10)
    const queryDonutData = {
        labels: topQueries.map((q, i) => {
            const text = q.query || `Query #${i + 1}`
            return text.length > 40 ? text.substring(0, 40) + '...' : text
        }),
        datasets: [{
            data: topQueries.map(q => q.weight),
            backgroundColor: [
                '#ef4444', '#f97316', '#f59e0b', '#84cc16', '#10b981',
                '#06b6d4', '#3b82f6', '#6366f1', '#8b5cf6', '#d946ef'
            ],
            borderWidth: 0
        }]
    }

    const donutOptions: ChartOptions<'doughnut'> = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { position: 'right', labels: { boxWidth: 12, usePointStyle: true } }
        },
        cutout: '65%'
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <button type="button" onClick={(e) => { e.stopPropagation(); toggleSidebar() }} className="p-2 hover:bg-accent rounded-md transition-colors" aria-label="Toggle sidebar">
                        <PanelRight className="h-5 w-5" />
                    </button>
                    <div className="h-6 w-px bg-border"></div>
                    <div>
                        <h1 className="text-lg font-semibold">Database Analysis</h1>
                        <p className="text-xs text-muted-foreground">Real-time Overview & Query Performance</p>
                    </div>
                </div>
                <div className="flex items-center gap-2">
                    <DateRangePicker />
                    <Select value={duration} onValueChange={setDuration}>
                        <SelectTrigger className="w-[180px] bg-card"><SelectValue /></SelectTrigger>
                        <SelectContent>
                            <SelectItem value="60">Last 1 Hour</SelectItem>
                            <SelectItem value="180">Last 3 Hours</SelectItem>
                            <SelectItem value="360">Last 6 Hours</SelectItem>
                            <SelectItem value="1440">Last 24 Hours</SelectItem>
                        </SelectContent>
                    </Select>
                    <button onClick={() => loadData()} className="flex items-center gap-2 px-3 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors">
                        <RefreshCw className="h-4 w-4" /> Refresh
                    </button>
                </div>
            </div>

            {/* Metrics Grid */}
            <div className="grid gap-4 grid-cols-5">
                <Card className="bg-card">
                    <CardContent className="p-4">
                        <div className="text-xs font-medium text-muted-foreground mb-1">Avg Time Spent</div>
                        <div className="flex items-baseline gap-1">
                            <div className="text-2xl font-bold text-blue-500">{avgTime}</div>
                            <span className="text-xs text-muted-foreground">s</span>
                        </div>
                    </CardContent>
                </Card>
                <Card className="bg-card">
                    <CardContent className="p-4">
                        <div className="text-xs font-medium text-muted-foreground mb-1">Avg Load</div>
                        <div className="flex items-baseline gap-1">
                            <div className="text-2xl font-bold">{Number(avgLoad).toLocaleString()}</div>
                            <span className="text-xs text-muted-foreground">cpm</span>
                        </div>
                    </CardContent>
                </Card>
                <Card className="bg-card">
                    <CardContent className="p-4">
                        <div className="text-xs font-medium text-muted-foreground mb-1">Max Latency</div>
                        <div className="flex items-baseline gap-1">
                            <div className="text-2xl font-bold text-red-500">{maxTime}</div>
                            <span className="text-xs text-muted-foreground">s</span>
                        </div>
                    </CardContent>
                </Card>
                <Card className="bg-card">
                    <CardContent className="p-4">
                        <div className="text-xs font-medium text-muted-foreground mb-1">Inefficient Events</div>
                        <div className="flex items-baseline gap-1">
                            <div className="text-2xl font-bold text-orange-500">{data.inefficiency_count}</div>
                            <span className="text-xs text-muted-foreground">times</span>
                        </div>
                    </CardContent>
                </Card>
                <Card className="bg-card">
                    <CardContent className="p-4">
                        <div className="text-xs font-medium text-muted-foreground mb-1">Avg CPU</div>
                        <div className="flex items-baseline gap-1">
                            <div className="text-2xl font-bold">{avgCpu}</div>
                            <span className="text-xs text-muted-foreground">%</span>
                        </div>
                    </CardContent>
                </Card>
            </div>

            {/* Main Charts Area */}
            <div className="grid gap-4 grid-cols-3">
                {/* Main Trend */}
                <Card className="bg-card col-span-2">
                    <CardHeader>
                        <CardTitle className="text-base font-semibold flex items-center">
                            Time Spent vs Load Trend
                            <InfoTooltip content="Correlation between query execution time and system load over time." />
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="h-[300px]">
                            <Line data={mainChartData as any} options={mainChartOptions as any} />
                        </div>
                    </CardContent>
                </Card>

                {/* Efficiency Donut */}
                <Card className="bg-card">
                    <CardHeader>
                        <CardTitle className="text-base font-semibold flex items-center">
                            Efficiency Distribution
                            <InfoTooltip content="Proportion of queries executing efficiently versus inefficiently." />
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="h-[250px] flex justify-center">
                            <Doughnut data={donutData} options={{ ...donutOptions, plugins: { legend: { position: 'bottom' } } }} />
                        </div>
                        <div className="text-center text-xs text-muted-foreground mt-4">
                            Time &gt; Load vs Time ≤ Load
                        </div>
                    </CardContent>
                </Card>
            </div>

            {/* CPU Chart */}
            <Card className="bg-card">
                <CardHeader>
                    <CardTitle className="text-base font-semibold flex items-center">
                        CPU Resource Usage
                        <InfoTooltip content="Trend of CPU consumption over the selected period." />
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="h-[250px]">
                        <Line data={cpuChartData} options={cpuChartOptions} />
                    </div>
                </CardContent>
            </Card>

            {/* Top 5 Queries Table */}
            <Card className="bg-card">
                <CardHeader>
                    <CardTitle className="text-base font-semibold flex items-center">
                        Top 5 Heavy Queries (by Weight)
                        <InfoTooltip content="Queries consuming the most database resources based on execution time and frequency." />
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="overflow-x-auto">
                        <table className="w-full text-left text-sm">
                            <thead className="border-b text-muted-foreground">
                                <tr>
                                    <th className="p-3 font-medium">Rank</th>
                                    <th className="p-3 font-medium w-[40%]">Query</th>
                                    <th className="p-3 font-medium">Weight (%)</th>
                                    <th className="p-3 font-medium">Executions</th>
                                    <th className="p-3 font-medium">Avg Response</th>
                                    <th className="p-3 font-medium">Total Time</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-border/50">
                                {data.queries.length === 0 ? (
                                    <tr><td colSpan={6} className="p-4 text-center text-muted-foreground">No query data available</td></tr>
                                ) : (
                                    data.queries.slice(0, 5).map((q, idx) => (
                                        <tr key={idx} className="hover:bg-accent/50 transition-colors">
                                            <td className="p-3 font-bold text-foreground">#{idx + 1}</td>
                                            <td className="p-3">
                                                <div className="max-h-[80px] overflow-y-auto font-mono text-xs text-muted-foreground bg-muted p-2 rounded border">
                                                    {q.query}
                                                </div>
                                            </td>
                                            <td className="p-3 font-bold text-red-500">{q.weight}%</td>
                                            <td className="p-3">{Number(q.executions).toLocaleString()}</td>
                                            <td className="p-3">{q.avg_response}</td>
                                            <td className="p-3">{q.elapsed_time}</td>
                                        </tr>
                                    ))
                                )}
                            </tbody>
                        </table>
                    </div>
                </CardContent>
            </Card>

            {/* Query Weight Distribution Donut */}
            <Card className="bg-card">
                <CardHeader>
                    <CardTitle className="text-base font-semibold flex items-center">
                        Top 10 Query Weight Distribution
                        <InfoTooltip content="Visual breakdown of resource consumption by top queries." />
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="h-[300px]">
                        <Doughnut data={queryDonutData} options={donutOptions} />
                    </div>
                </CardContent>
            </Card>
        </div>
    )
}
