import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { PanelRight } from "lucide-react"
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { fetchDashboardData, DashboardData } from "@/lib/api"
import { useSidebar } from "@/components/SidebarContext"
import { useChartTooltipStyles } from "@/hooks/useChartTooltipStyles"
import { useBusinessTransaction } from "@/components/BusinessTransactionContext"
import { DateRangePicker } from "@/components/DateRangePicker"
import { useDateRange } from "@/components/DateRangeContext"
import { GlobalSearch } from "@/components/GlobalSearch"
import InfoTooltip from "@/components/InfoTooltip"

export default function ResponseTime() {
    const [data, setData] = useState<DashboardData | null>(null)
    const { toggleSidebar } = useSidebar()
    const tooltipStyles = useChartTooltipStyles()
    const { selectedTier, selectedTransaction } = useBusinessTransaction()
    const { dateRange } = useDateRange()

    useEffect(() => {
        const load = async () => {
            try {
                const d = await fetchDashboardData(10080, selectedTier, selectedTransaction, dateRange.from, dateRange.to)
                setData(d)
            } catch (e) {
                console.error(e)
            }
        }
        load()
        const interval = setInterval(load, 30000)
        return () => clearInterval(interval)
    }, [selectedTier, selectedTransaction, dateRange])

    // Prepare Chart Data
    const lineData = data?.timeline.map(t => {
        // t.timestamp is already the timestamp in milliseconds
        const date = new Date(t.timestamp);
        return {
            time: date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            value: t.value
        };
    }) || [];

    const freqData = data ? Object.entries(data.frequency).map(([ms, count]) => ({
        ms: Number(ms),
        count
    })).sort((a, b) => a.ms - b.ms) : [];

    // Process buckets to match UI expectations (Aggregating Fast + Normal)
    const normalCount = (data?.buckets['Fast (<500ms)'] || 0) + (data?.buckets['Normal (500-1000ms)'] || 0) + (data?.buckets['Normal'] || 0);
    const slowCount = (data?.buckets['Slow (>1000ms)'] || 0) + (data?.buckets['Slow'] || 0);

    // Create pieData ensuring explicit order/keys for coloring
    const pieData = [
        { name: 'Normal', value: normalCount },
        { name: 'Slow', value: slowCount }
    ].filter(item => item.value > 0);

    const COLOR_MAP: Record<string, string> = {
        'Normal': '#22c55e',
        'Slow': '#ef4444'
    };

    // Calculate percentage of Normal transactions
    const totalCount = normalCount + slowCount;
    const normalPercentage = totalCount > 0
        ? ((normalCount / totalCount) * 100).toFixed(0)
        : 0;

    // Calculate stats
    const avg = (data?.raw_values?.length || 0) > 0 ? (data!.raw_values.reduce((a, b) => a + b, 0) / data!.raw_values.length).toFixed(1) : 0;

    // Calculate P95
    const p95 = (data?.raw_values?.length || 0) > 0
        ? data!.raw_values.sort((a, b) => a - b)[Math.floor(data!.raw_values.length * 0.95)]
        : 0;


    // Dynamic subtitle based on period

    return (
        <div className="space-y-6">
            {/* Top Bar with Sidebar Toggle and Title */}
            <div className="flex items-center justify-between relative z-50">
                <div className="flex items-center gap-3">
                    <button
                        type="button"
                        onClick={(e) => {
                            e.stopPropagation();
                            console.error('Toggle sidebar clicked (ERROR LEVEL)');
                            console.log('Toggle sidebar clicked');
                            toggleSidebar();
                        }}
                        className="p-2 hover:bg-accent rounded-md transition-colors cursor-pointer relative z-50 border border-transparent hover:border-border"
                        aria-label="Toggle sidebar"
                    >
                        <PanelRight className="h-5 w-5 pointer-events-none" />
                    </button>
                    <div className="h-6 w-px bg-border"></div>
                    <div>
                        <h1 className="text-lg font-semibold">Business Transaction - Response</h1>
                        <p className="text-xs text-muted-foreground">{dateRange.from && dateRange.to ? "Analysis for Selected Period" : "Real-time performance trend"}</p>
                    </div>
                </div>
                <div className="flex items-center gap-2">
                    <GlobalSearch />
                    <DateRangePicker />
                </div>
            </div>

            {data ? (
                <>
                    <div className="grid gap-4 grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
                        <Card className="bg-card">
                            <CardHeader className="pb-2">
                                <CardTitle className="text-sm font-medium text-muted-foreground flex items-center">
                                    Total Requests
                                    <InfoTooltip content="Total number of transactions processed in the selected period." />
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="text-2xl font-bold">{data.raw_values.length}</div>
                            </CardContent>
                        </Card>
                        <Card className="bg-card">
                            <CardHeader className="pb-2">
                                <CardTitle className="text-sm font-medium text-muted-foreground flex items-center">
                                    Min Response
                                    <InfoTooltip content="Fastest response time recorded." />
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="text-2xl font-bold text-green-500">{data.min} ms</div>
                            </CardContent>
                        </Card>
                        <Card className="bg-card">
                            <CardHeader className="pb-2">
                                <CardTitle className="text-sm font-medium text-muted-foreground flex items-center">
                                    Max Response
                                    <InfoTooltip content="Slowest response time recorded." />
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="text-2xl font-bold text-red-500">{data.max} ms</div>
                            </CardContent>
                        </Card>
                        <Card className="bg-card">
                            <CardHeader className="pb-2">
                                <CardTitle className="text-sm font-medium text-muted-foreground flex items-center">
                                    Avg Response Time
                                    <InfoTooltip content="Average time taken for transactions to complete." />
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="text-2xl font-bold">{avg} ms</div>
                            </CardContent>
                        </Card>
                        <Card className="bg-card">
                            <CardHeader className="pb-2">
                                <CardTitle className="text-sm font-medium text-muted-foreground flex items-center">
                                    P95 Response Time
                                    <InfoTooltip content="95th percentile response time (95% of requests are faster than this)." />
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="text-2xl font-bold">{p95} ms</div>
                            </CardContent>
                        </Card>
                        <Card className="bg-card">
                            <CardHeader className="pb-2">
                                <CardTitle className="text-sm font-medium text-muted-foreground flex items-center">
                                    Error Rate
                                    <InfoTooltip content="Percentage of transactions that resulted in an error or were slow." />
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="text-2xl font-bold text-red-400">
                                    {slowCount > 0 && data?.raw_values.length ? ((slowCount / data.raw_values.length) * 100).toFixed(1) : 0}%
                                </div>
                            </CardContent>
                        </Card>
                    </div>

                    {/* Charts Row 1 */}
                    <div className="grid gap-4 md:grid-cols-1 lg:grid-cols-3">
                        <Card className="col-span-2 bg-card">
                            <CardHeader>
                                <CardTitle className="text-base font-semibold">Response Time Trend</CardTitle>
                                <p className="text-xs text-muted-foreground mt-1">Timeline of average responsiveness</p>
                            </CardHeader>
                            <CardContent>
                                <div className="h-[300px] w-full">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <AreaChart data={lineData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                                            <defs>
                                                <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                                                    <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.3} />
                                                    <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0} />
                                                </linearGradient>
                                            </defs>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                                            <XAxis
                                                dataKey="time"
                                                stroke="#888"
                                                fontSize={12}
                                                tickLine={false}
                                                axisLine={false}
                                                minTickGap={30}
                                            />
                                            <YAxis
                                                stroke="#888"
                                                fontSize={12}
                                                tickLine={false}
                                                axisLine={false}
                                                tickFormatter={(value) => `${value}ms`}
                                            />
                                            <Tooltip
                                                contentStyle={tooltipStyles.contentStyle}
                                                itemStyle={tooltipStyles.itemStyle}
                                                formatter={(value: any) => [`${value} ms`, "Response Time"]}
                                            />
                                            <Area
                                                type="monotone"
                                                dataKey="value"
                                                stroke="#0ea5e9"
                                                strokeWidth={2}
                                                activeDot={{ r: 6, strokeWidth: 0 }}
                                                fillOpacity={1}
                                                fill="url(#colorValue)"
                                            />
                                        </AreaChart>
                                    </ResponsiveContainer>
                                </div>

                                {/* Visual Explanation of Axes */}
                                <div className="mt-4 pt-4 border-t border-border/50">
                                    <div className="flex flex-wrap items-center justify-center gap-x-8 gap-y-2 text-sm text-muted-foreground">
                                        <div className="flex items-center gap-2">
                                            <span className="flex items-center justify-center w-5 h-5 rounded-md bg-muted font-mono text-xs font-bold text-foreground">X</span>
                                            <span>Time (Hour:Minute)</span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <span className="flex items-center justify-center w-5 h-5 rounded-md bg-muted font-mono text-xs font-bold text-foreground">Y</span>
                                            <span>Response Time (ms)</span>
                                        </div>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>

                        <Card className="col-span-1 bg-card flex flex-col justify-center">
                            <CardHeader>
                                <CardTitle className="text-base font-semibold">Health Distribution</CardTitle>
                                <p className="text-xs text-muted-foreground mt-1">Overview of system health status</p>
                            </CardHeader>
                            <CardContent>
                                <div className="h-[250px] w-full relative">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <PieChart>
                                            <Pie
                                                data={pieData}
                                                innerRadius={60}
                                                outerRadius={80}
                                                paddingAngle={5}
                                                dataKey="value"
                                                stroke="none"
                                            >
                                                {pieData.map((entry, index) => (
                                                    <Cell key={`cell-${entry.name}-${index}`} fill={COLOR_MAP[entry.name]} />
                                                ))}
                                            </Pie>
                                            <Tooltip contentStyle={tooltipStyles.contentStyle} />
                                        </PieChart>
                                    </ResponsiveContainer>
                                    <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none pb-4">
                                        <span className="text-4xl font-bold">{normalPercentage}%</span>
                                        <span className="text-xs text-muted-foreground mt-1">Normal</span>
                                    </div>
                                </div>

                                {/* Legend for Health Distribution */}
                                <div className="pt-2 border-t border-border/50">
                                    <div className="flex flex-wrap items-center justify-center gap-x-6 gap-y-2 text-sm text-muted-foreground">
                                        <div className="flex items-center gap-2">
                                            <div className="w-3 h-3 rounded-full bg-[#22c55e]" />
                                            <span>Normal</span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <div className="w-3 h-3 rounded-full bg-[#ef4444]" />
                                            <span>Slow / Error</span>
                                        </div>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    </div>

                    {/* Charts Row 2 */}
                    <Card className="bg-card">
                        <CardHeader>
                            <CardTitle className="text-base font-semibold">Frequency Distribution</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="h-[250px] w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={freqData}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                                        <XAxis dataKey="ms" stroke="#666" fontSize={12} tickLine={false} axisLine={false} />
                                        <YAxis stroke="#666" fontSize={12} tickLine={false} axisLine={false} />
                                        <Tooltip cursor={tooltipStyles.cursor} contentStyle={tooltipStyles.contentStyle} />
                                        <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>

                            {/* Visual Explanation of Axes - Frequency */}
                            <div className="mt-4 pt-4 border-t border-border/50">
                                <div className="flex flex-wrap items-center justify-center gap-x-8 gap-y-2 text-sm text-muted-foreground">
                                    <div className="flex items-center gap-2">
                                        <span className="flex items-center justify-center w-5 h-5 rounded-md bg-muted font-mono text-xs font-bold text-foreground">X</span>
                                        <span>Response Time (ms)</span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <span className="flex items-center justify-center w-5 h-5 rounded-md bg-muted font-mono text-xs font-bold text-foreground">Y</span>
                                        <span>Frequency (Count)</span>
                                    </div>
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                </>
            ) : (
                <div className="flex h-[50vh] w-full items-center justify-center">
                    <div className="flex flex-col items-center gap-2">
                        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
                        <p className="text-muted-foreground">Loading dashboard data...</p>
                    </div>
                </div>
            )}
        </div>
    )
}


