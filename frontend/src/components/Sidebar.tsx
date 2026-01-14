import { LineChart, Clock, Server, AlertTriangle, Hourglass, Sun, Moon, Home } from "lucide-react";
import { cn } from "@/lib/utils";
import HandalinLogo from "@/assets/handalin.svg";
import { Link, useLocation } from "react-router-dom";
import { useTheme } from "./ThemeProvider";

interface SidebarProps extends React.HTMLAttributes<HTMLDivElement> {
  collapsed?: boolean;
  onToggle?: () => void;
}

export function Sidebar({ className, collapsed = false }: SidebarProps) {
  const { theme, setTheme } = useTheme();
  
  return (
    <div 
      className={cn(
        "min-h-screen border-r flex flex-col transition-all duration-300",
        collapsed ? "w-16" : "w-64",
        className
      )} 
      style={{ backgroundColor: theme === 'dark' ? '#171717' : '#f8f9fa' }}
    >
      <div className="space-y-4 py-4 flex-1">
        <div className="px-3 py-2">
          {!collapsed && (
            <div className="flex items-center px-4 mb-6">
              <img src={HandalinLogo} alt="Handalin" className="h-7 mr-2" />
              <h2 className="text-xl font-bold tracking-tight">Handalin</h2>
            </div>
          )}
          {collapsed && (
            <div className="flex items-center justify-center mb-6">
              <img src={HandalinLogo} alt="Handalin" className="h-7" />
            </div>
          )}
          <div className="space-y-1">
            <SidebarItem icon={<Home className="h-4 w-4" />} label="Executive Dashboard" to="/" collapsed={collapsed} />
            <SidebarItem icon={<LineChart className="h-4 w-4" />} label="Forecasting" to="/forecasting" collapsed={collapsed} />
            <SidebarItem icon={<Clock className="h-4 w-4" />} label="Response Time" to="/response-time" collapsed={collapsed} />
            <SidebarItem icon={<Server className="h-4 w-4" />} label="Load Analysis" to="/load-analysis" collapsed={collapsed} />
            <SidebarItem icon={<AlertTriangle className="h-4 w-4" />} label="Error Analysis" to="/error-analysis" collapsed={collapsed} />
            <SidebarItem icon={<Hourglass className="h-4 w-4" />} label="Slow Calls" to="/slow-calls-analysis" collapsed={collapsed} />
          </div>
        </div>
      </div>
      
      {/* Theme Toggle - Icon Only */}
      <div className="px-3 py-4">
        <button
          onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          className={cn(
            "flex items-center justify-center p-2 rounded-md hover:bg-accent transition-colors",
            collapsed ? "w-full" : "w-10 h-10"
          )}
          title={theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
        >
          {theme === 'dark' ? (
            <Sun className="h-5 w-5" />
          ) : (
            <Moon className="h-5 w-5" />
          )}
        </button>
      </div>
    </div>
  );
}

function SidebarItem({ icon, label, to, collapsed = false }: { icon: React.ReactNode; label: string; to: string; collapsed?: boolean }) {
  const location = useLocation();
  const active = location.pathname === to;
  
  return (
    <Link
      to={to}
      className={cn(
        "flex items-center rounded-md px-4 py-2 text-sm font-medium hover:bg-accent hover:text-accent-foreground transition-colors",
        active ? "text-primary" : "text-muted-foreground",
        collapsed ? "justify-center" : "justify-start"
      )}
      title={collapsed ? label : undefined}
    >
      <span className={collapsed ? "" : "mr-2"}>{icon}</span>
      {!collapsed && label}
    </Link>
  );
}
