import { useEffect, useState } from "react"
import { useNavigate } from "react-router-dom"
import { Card, CardContent } from "@/components/ui/card"
import { Clock, AlertTriangle, Server, Hourglass, PanelRight, ListChecks, Activity, Database } from "lucide-react"
import { useSidebar } from "@/components/SidebarContext"
import { useBusinessTransaction } from "@/components/BusinessTransactionContext"

interface SummaryData {
  response_time: number;
  errors: number;
  load: number;
  slow_calls: number;
}

export default function ExecutiveDashboard() {
  const [data, setData] = useState<SummaryData | null>(null)
  const navigate = useNavigate()
  const { toggleSidebar } = useSidebar()
  const { selectedTier, selectedTransaction } = useBusinessTransaction()

  const fetchSummary = async () => {
    try {
      const response = await fetch(`/api/summary?tier=${encodeURIComponent(selectedTier)}&bt=${encodeURIComponent(selectedTransaction)}`)
      const result = await response.json()
      setData(result)
    } catch (e) {
      console.error("Error fetching summary:", e)
    }
  }

  useEffect(() => {
    fetchSummary()
    // Auto-refresh every 60s
    const interval = setInterval(fetchSummary, 60000)
    return () => clearInterval(interval)
  }, [selectedTier, selectedTransaction])

  const formatNumber = (num: number) => {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M'
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K'
    return num.toLocaleString()
  }

  return (
    <div className="space-y-6">
      {/* Header with Panel Toggle */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button 
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              toggleSidebar();
            }} 
            className="p-2 hover:bg-accent rounded-md transition-colors cursor-pointer relative z-50 border border-transparent hover:border-border"
            aria-label="Toggle sidebar"
          >
            <PanelRight className="h-5 w-5 pointer-events-none" />
          </button>
          <div className="h-6 w-px bg-border"></div>
          <div>
            <h1 className="text-lg font-semibold">System Overview</h1>
            <p className="text-xs text-muted-foreground">Real-time dashboard of system health metrics</p>
          </div>
        </div>
      </div>

      {/* 3x2 Grid Layout */}
      <div className="grid gap-4 grid-cols-3">
        {/* Response Time Card */}
        <MetricCard
          title="Response Time"
          value={data ? `${data.response_time}` : '--'}
          unit="ms"
          subtext="Current Average"
          icon={<Clock className="h-4 w-4" />}
          iconColor="text-blue-500"
          onClick={() => navigate('/response-time')}
        />

        {/* Errors Card */}
        <MetricCard
          title="Errors"
          value={data ? data.errors.toLocaleString() : '--'}
          subtext="Total (Last 30 Days)"
          icon={<AlertTriangle className="h-4 w-4" />}
          iconColor="text-red-500"
          onClick={() => navigate('/error-analysis')}
        />

        {/* System Load Card */}
        <MetricCard
          title="System Load"
          value={data ? formatNumber(data.load) : '--'}
          subtext="Total Calls (Last 30 Days)"
          icon={<Server className="h-4 w-4" />}
          iconColor="text-teal-500"
          onClick={() => navigate('/load-analysis')}
        />

        {/* Slow Calls Card */}
        <MetricCard
          title="Slow Calls"
          value={data ? data.slow_calls.toLocaleString() : '--'}
          subtext="Total Slow (Last 30 Days)"
          icon={<Hourglass className="h-4 w-4" />}
          iconColor="text-orange-500"
          onClick={() => navigate('/slow-calls-analysis')}
        />

        {/* Business Transactions Card */}
        <MetricCard
          title="Business Transactions"
          value="Analysis"
          subtext="Health, Volume & Latency"
          icon={<ListChecks className="h-4 w-4" />}
          iconColor="text-purple-500"
          onClick={() => navigate('/business-transactions')}
        />

        {/* JVM Health Card */}
        <MetricCard
          title="Infrastructure Health"
          value="JVM"
          subtext="Memory, GC & Availability"
          icon={<Activity className="h-4 w-4" />}
          iconColor="text-rose-500"
          onClick={() => navigate('/jvm-health')}
        />

        {/* Database Analysis Card */}
        <MetricCard
          title="Database Analysis"
          value="Query"
          subtext="Performance & Spikes"
          icon={<Database className="h-4 w-4" />}
          iconColor="text-indigo-500"
          onClick={() => navigate('/database-analysis')}
        />
      </div>
    </div>
  )
}

interface MetricCardProps {
  title: string;
  value: string;
  unit?: string;
  subtext: string;
  icon: React.ReactNode;
  iconColor: string;
  onClick: () => void;
}

function MetricCard({ title, value, unit, subtext, icon, iconColor, onClick }: MetricCardProps) {
  return (
    <Card 
      className="bg-card cursor-pointer transition-all duration-200 hover:shadow-md border-border/50 hover:border-border"
      onClick={onClick}
    >
      <CardContent className="p-6">
        <div className="flex items-center gap-2 mb-2">
          <span className={iconColor}>{icon}</span>
          <div className="text-sm font-medium text-muted-foreground">{title}</div>
        </div>
        <div className="flex items-baseline gap-2 mb-1">
          <div className="text-3xl font-bold text-foreground">{value}</div>
          {unit && <span className="text-lg font-medium text-muted-foreground">{unit}</span>}
        </div>
        <p className="text-xs text-muted-foreground mb-3">{subtext}</p>
        <div className="text-sm font-medium text-primary">View Details →</div>
      </CardContent>
    </Card>
  )
}
