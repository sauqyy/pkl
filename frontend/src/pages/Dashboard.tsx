import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Input } from "@/components/ui/input"
import { Search, Calendar as CalendarIcon, PanelRight } from "lucide-react"
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { fetchDashboardData, DashboardData } from "@/lib/api"
import { useSidebar } from "@/components/SidebarContext"

export default function Dashboard() {
  const [data, setData] = useState<DashboardData | null>(null)
  const [loading, setLoading] = useState(true)
  const [period, setPeriod] = useState("60")
  const { toggleSidebar } = useSidebar()

  useEffect(() => {
    const load = async () => {
        try {
            const d = await fetchDashboardData(Number(period))
            setData(d)
        } catch (e) {
            console.error(e)
        } finally {
            setLoading(false)
        }
    }
    load()
    const interval = setInterval(load, 30000)
    return () => clearInterval(interval)
  }, [period])

  if (!data) return <div className="p-8 text-muted-foreground">Loading dashboard...</div>

  // Prepare Chart Data
  const lineData = data.timeline.map(t => {
      // t.time is already the timestamp in milliseconds
      const date = new Date(t.time);
      return {
          time: date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          value: t.value
      };
  });

  const freqData = Object.entries(data.frequency).map(([ms, count]) => ({
      ms: Number(ms),
      count
  })).sort((a,b) => a.ms - b.ms);

  const pieData = Object.entries(data.buckets).map(([name, value]) => ({ name, value }));
  const COLORS = ['#22c55e', '#f59e0b', '#ef4444'];

  // Calculate stats
  const avg = data.raw_values.length > 0 ? (data.raw_values.reduce((a, b) => a + b, 0) / data.raw_values.length).toFixed(1) : 0;
  const max = data.raw_values.length > 0 ? Math.max(...data.raw_values) : 0;
  const min = data.raw_values.length > 0 ? Math.min(...data.raw_values) : 0;
  
  // Dynamic subtitle based on period
  const getPeriodLabel = (minutes: string) => {
    switch(minutes) {
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
            <div className="flex items-center gap-2 bg-card border rounded-md px-3 py-2 text-sm text-muted-foreground">
                <CalendarIcon className="h-4 w-4" />
                <span>Select period</span>
            </div>
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

      {/* Metrics Grid */}
      <div className="grid gap-4 grid-cols-2">
        <MetricCard 
          title="Total Requests" 
          value={data.raw_values.length.toLocaleString()} 
          subtitle="All processed events in the selected timeframe" 
        />
        <MetricCard 
          title="Average Response Time" 
          value={`${avg} ms`} 
          subtitle="Mean system response across all requests" 
        />
        <MetricCard 
          title="Maximum Latency" 
          value={`${max} ms`} 
          subtitle="Peak delay recorded during high load" 
        />
        <MetricCard 
          title="Minimum Latency" 
          value={`${min} ms`} 
          subtitle="Fastest recorded response" 
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
                                    <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.3}/>
                                    <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0}/>
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                            <XAxis dataKey="time" stroke="#666" fontSize={12} tickLine={false} axisLine={false} />
                            <Tooltip 
                                contentStyle={{ backgroundColor: '#111', border: '1px solid #333' }}
                                itemStyle={{ color: '#fff' }}
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
                            <Tooltip contentStyle={{ backgroundColor: '#111', border: '1px solid #333' }} />
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
                            <Tooltip cursor={{fill: '#333'}} contentStyle={{ backgroundColor: '#111', border: '1px solid #333' }} />
                            <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </CardContent>
       </Card>
    </div>
  )
}

function MetricCard({ title, value, subtitle }: { title: string; value: string | number; subtitle: string }) {
    return (
        <Card className="bg-card">
            <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">{title}</CardTitle>
            </CardHeader>
            <CardContent>
                <div className="text-3xl font-bold text-foreground mb-1">{value}</div>
                <p className="text-xs text-muted-foreground">{subtitle}</p>
            </CardContent>
        </Card>
    )
}
