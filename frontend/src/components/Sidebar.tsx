import { LayoutDashboard, LineChart, Clock, Loader, MessageSquare } from "lucide-react";
import { cn } from "@/lib/utils";
import HandalinLogo from "@/assets/handalin.svg";
import { Link, useLocation } from "react-router-dom";

interface SidebarProps extends React.HTMLAttributes<HTMLDivElement> {}

export function Sidebar({ className }: SidebarProps) {
  return (
    <div className={cn("pb-12 min-h-screen w-64 border-r", className)} style={{ backgroundColor: '#171717' }}>
      <div className="space-y-4 py-4">
        <div className="px-3 py-2">
          <div className="flex items-center px-4 mb-6">
             {/* Handalin Logo */}
             <img src={HandalinLogo} alt="Handalin" className="h-7 mr-2" />
             <h2 className="text-xl font-bold tracking-tight">Handalin</h2>
          </div>
          <div className="space-y-1">
            <SidebarItem icon={<LayoutDashboard className="h-4 w-4" />} label="Dashboard" to="/" />
            <SidebarItem icon={<LineChart className="h-4 w-4" />} label="Forecasting" to="/forecasting" />
            <SidebarItem icon={<Clock className="h-4 w-4" />} label="Response Time" to="/response-time" />
            <SidebarItem icon={<Loader className="h-4 w-4" />} label="Load Analysis" to="/load-analysis" />
            <SidebarItem icon={<MessageSquare className="h-4 w-4" />} label="Error Analysis" to="/error-analysis" />
          </div>
        </div>
      </div>
    </div>
  );
}

function SidebarItem({ icon, label, to }: { icon: React.ReactNode; label: string; to: string }) {
  const location = useLocation();
  const active = location.pathname === to;
  
  return (
    <Link
      to={to}
      className={cn(
        "w-full flex items-center rounded-md px-4 py-2 text-sm font-medium hover:bg-accent hover:text-accent-foreground transition-colors justify-start",
        active ? "text-primary" : "text-muted-foreground"
      )}
    >
      <span className="mr-2">{icon}</span>
      {label}
    </Link>
  );
}
