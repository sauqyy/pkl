import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Input } from "@/components/ui/input"
import { Search, PanelRight } from "lucide-react"
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { fetchDashboardData, DashboardData } from "@/lib/api"
import { useSidebar } from "@/components/SidebarContext"
import { useChartTooltipStyles } from "@/hooks/useChartTooltipStyles"
import { DateRangePicker } from "@/components/DateRangePicker"
import { useDateRange } from "@/components/DateRangeContext"
import InfoTooltip from "@/components/InfoTooltip"

export default function Dashboard() {
    const [data, setData] = useState<DashboardData | null>(null)
    const [period, setPeriod] = useState("60")
    const { toggleSidebar } = useSidebar()
    const tooltipStyles = useChartTooltipStyles()
    const { dateRange } = useDateRange()

    useEffect(() => {
        const load = async () => {
            try {
                const d = await fetchDashboardData(Number(period), undefined, undefined, dateRange.from, dateRange.to)
                setData(d)
            } catch (e) {
                console.error(e)
            }
        }
        load()
        const interval = setInterval(load, 30000)
        return () => clearInterval(interval)
    }, [period, dateRange])

    // Prepare Chart Data
    const lineData = data?.timeline.map(t => {
        // t.timestamp is the timestamp in milliseconds
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

    const pieData = data ? Object.entries(data.buckets).map(([name, value]) => ({ name, value })) : [];
    const COLORS = ['#22c55e', '#f59e0b', '#ef4444'];

    // Calculate stats
    const avg = data && data.raw_values.length > 0 ? (data.raw_values.reduce((a, b) => a + b, 0) / data.raw_values.length).toFixed(1) : 0;
    const max = data && data.raw_values.length > 0 ? Math.max(...data.raw_values) : 0;
    const min = data && data.raw_values.length > 0 ? Math.min(...data.raw_values) : 0;

    // Dynamic subtitle based on period
    const getPeriodLabel = (minutes: string) => {
        switch (minutes) {
            case "15": return "15 minutes";
            case "60": return "1 hour";
            case "360": return "6 hours";
            case "1440": return "24 hours";
            default: return `${minutes} minutes`;
        }
    };

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
                        <h1 className="text-lg font-semibold">Dashboard</h1>
                        <p className="text-xs text-muted-foreground">Trend for the last {getPeriodLabel(period)}</p>
                    </div>
                </div>
                <div className="flex items-center gap-2">
                    <div className="relative">
                        <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                        <Input type="search" placeholder="Search for something..." className="pl-8 w-[250px] bg-card" />
                    </div>
                    <DateRangePicker />
                    <Select value={period} onValueChange={setPeriod}>
                        <SelectTrigger className="w-[180px] bg-card">
                            <SelectValue placeholder="Select period" />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectItem value="15">Last 15 Minutes</SelectItem>
                            <SelectItem value="60">Last 1 Hour</SelectItem>
                            <SelectItem value="360">Last 6 Hours</SelectItem>
                            <SelectItem value="1440">Last 24 Hours</SelectItem>
                        </SelectContent>
                    </Select>
                </div>
            </div>

            {data ? (
                <>
                    {/* Metrics Grid */}
                    <div className="grid gap-4 grid-cols-2">
                        <MetricCard
                            title="Total Requests"
                            value={data.raw_values.length.toLocaleString()}
                            subtitle="All processed events in the selected timeframe"
                            description="Total count of all HTTP requests processed by the backend services."
                        />
                        <MetricCard
                            title="Average Response Time"
                            value={`${avg} ms`}
                            subtitle="Mean system response across all requests"
                            description="The average time (in milliseconds) taken to process requests."
                        />
                        <MetricCard
                            title="Maximum Latency"
                            value={`${max} ms`}
                            subtitle="Peak delay recorded during high load"
                            description="The highest response time recorded for a single request."
                        />
                        <MetricCard
                            title="Minimum Latency"
                            value={`${min} ms`}
                            subtitle="Fastest recorded response"
                            description="The lowest response time recorded for a single request."
                        />
                    </div>

                    {/* Charts Row 1 */}
                    <div className="grid gap-4 md:grid-cols-1 lg:grid-cols-3">
                        <Card className="col-span-2 bg-card">
                            <CardHeader>
                                <CardTitle className="text-base font-semibold">Response Time Trend</CardTitle>
                                <p className="text-xs text-muted-foreground mt-1">Trend for the last {getPeriodLabel(period)}</p>
                            </CardHeader>
                            <CardContent>
                                <div className="h-[300px] w-full">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <AreaChart data={lineData}>
                                            <defs>
                                                <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                                                    <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.3} />
                                                    <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0} />
                                                </linearGradient>
                                            </defs>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                                            <XAxis dataKey="time" stroke="#666" fontSize={12} tickLine={false} axisLine={false} />
                                            <Tooltip
                                                contentStyle={tooltipStyles.contentStyle}
                                                itemStyle={tooltipStyles.itemStyle}
                                            />
                                            <Area type="monotone" dataKey="value" stroke="#0ea5e9" strokeWidth={2} fillOpacity={1} fill="url(#colorValue)" />
                                        </AreaChart>
                                    </ResponsiveContainer>
                                </div>
                            </CardContent>
                        </Card>

                        <Card className="col-span-1 bg-card flex flex-col justify-center">
                            <CardHeader>
                                <CardTitle className="text-base font-semibold">Health Distribution</CardTitle>
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
                                                    <Cell key={`cell-${entry.name}-${index}`} fill={COLORS[index % COLORS.length]} />
                                                ))}
                                            </Pie>
                                            <Tooltip contentStyle={tooltipStyles.contentStyle} />
                                        </PieChart>
                                    </ResponsiveContainer>
                                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                                        <span className="text-3xl font-bold">{data.raw_values.length}</span>
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

function MetricCard({ title, value, subtitle, description }: { title: string; value: string | number; subtitle: string; description?: string }) {
    return (
        <Card className="bg-card">
            <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground flex items-center">
                    {title}
                    {description && <InfoTooltip content={description} />}
                </CardTitle>
            </CardHeader>
            <CardContent>
                <div className="text-3xl font-bold text-foreground mb-1">{value}</div>
                <p className="text-xs text-muted-foreground">{subtitle}</p>
            </CardContent>
        </Card>
    )
}
