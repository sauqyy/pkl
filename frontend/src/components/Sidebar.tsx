import { LineChart, Sun, Moon, Home, Layers, Clock, AlertTriangle, Server, Hourglass, ChevronsUpDown, ListChecks, Database } from "lucide-react";
import { cn } from "@/lib/utils";
import HandalinLogo from "@/assets/handalin.svg";
import { Link, useLocation } from "react-router-dom";
import { useTheme } from "./ThemeProvider";
import { useBusinessTransaction, TIERS, TIER_BT_MAP } from "./BusinessTransactionContext";
import { useState, useRef, useEffect } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";

interface SidebarProps extends React.HTMLAttributes<HTMLDivElement> {
  collapsed?: boolean;
  onToggle?: () => void;
}

export function Sidebar({ className, collapsed = false }: SidebarProps) {
  const { theme, setTheme } = useTheme();
  const { selectedTier, setSelectedTier, selectedTransaction, setSelectedTransaction } = useBusinessTransaction();
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setDropdownOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const dropdownButtonClass = cn(
    "w-full flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors border",
    theme === 'dark' 
      ? "bg-neutral-800 hover:bg-neutral-700 text-white border-neutral-700" 
      : "bg-white hover:bg-neutral-100 text-neutral-900 border-neutral-300"
  );

  const dropdownMenuClass = cn(
    "absolute z-50 mt-1 w-56 rounded-md shadow-lg border",
    theme === 'dark' 
      ? "bg-neutral-800 border-neutral-700" 
      : "bg-white border-neutral-200"
  );

  const tierHeaderClass = cn(
    "px-3 py-2 text-xs font-semibold uppercase tracking-wider sticky top-0",
    theme === 'dark' 
      ? "bg-neutral-900 text-neutral-400 border-b border-neutral-700" 
      : "bg-neutral-100 text-neutral-500 border-b border-neutral-200"
  );

  const dropdownItemClass = (isSelected: boolean) => cn(
    "w-full text-left px-3 py-2 text-sm transition-colors",
    theme === 'dark'
      ? isSelected ? "bg-neutral-700 text-white" : "text-neutral-300 hover:bg-neutral-700"
      : isSelected ? "bg-neutral-100 text-neutral-900" : "text-neutral-700 hover:bg-neutral-100"
  );

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
          {/* Logo */}
          {!collapsed && (
            <div className="flex items-center px-4 mb-4">
              <img src={HandalinLogo} alt="Handalin" className="h-7 mr-2" />
              <h2 className="text-xl font-bold tracking-tight">Handalin</h2>
            </div>
          )}
          {collapsed && (
            <div className="flex items-center justify-center mb-4">
              <img src={HandalinLogo} alt="Handalin" className="h-7" />
            </div>
          )}

          {/* Combined Tier + BT Selector - Only when not collapsed */}
          {!collapsed && (
            <div className="mb-4 px-2">
              <div ref={dropdownRef} className="relative">
                <button
                  onClick={() => setDropdownOpen(!dropdownOpen)}
                  className={dropdownButtonClass}
                >
                  <div className="flex-1 text-left min-w-0">
                    <div className="text-[10px] text-muted-foreground truncate">{selectedTier}</div>
                    <div className="text-xs truncate">{selectedTransaction}</div>
                  </div>
                  <ChevronsUpDown className="h-4 w-4 opacity-60 flex-shrink-0" />
                </button>
                
                {dropdownOpen && (
                  <div className={dropdownMenuClass}>
                    <ScrollArea className="h-80">
                      {TIERS.map((tier) => (
                        <div key={tier}>
                          <div className={tierHeaderClass}>{tier}</div>
                          {TIER_BT_MAP[tier].map((bt) => (
                            <button
                              key={bt}
                              onClick={() => {
                                setSelectedTier(tier);
                                setSelectedTransaction(bt);
                                setDropdownOpen(false);
                              }}
                              className={dropdownItemClass(selectedTransaction === bt && selectedTier === tier)}
                            >
                              {bt}
                            </button>
                          ))}
                        </div>
                      ))}
                    </ScrollArea>
                  </div>
                )}
              </div>
            </div>
          )}
          
          {/* Main Group */}
          {!collapsed && (
            <div className="px-4 py-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
              Main
            </div>
          )}
          <div className="space-y-1">
            <SidebarItem icon={<Home className="h-4 w-4" />} label="Executive Dashboard" to="/" collapsed={collapsed} />
            <SidebarItem icon={<LineChart className="h-4 w-4" />} label="Forecasting" to="/forecasting" collapsed={collapsed} />
            <SidebarItem icon={<ListChecks className="h-4 w-4" />} label="BT Report" to="/business-transactions" collapsed={collapsed} />
            <SidebarItem icon={<Database className="h-4 w-4" />} label="Database" to="/database-analysis" collapsed={collapsed} />
            <SidebarItem icon={<Layers className="h-4 w-4" />} label="Tier" to="/jvm-health" collapsed={collapsed} />
          </div>

          {/* Business Transaction Group */}
          <div className={cn("pt-2", collapsed && "border-t border-border mt-2")}>
            {!collapsed && (
              <div className="px-4 py-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                Business Transaction
              </div>
            )}
            
            <div className="space-y-1">
              <SidebarItem icon={<Server className="h-4 w-4" />} label="Load" to="/load-analysis" collapsed={collapsed} />
              <SidebarItem icon={<Clock className="h-4 w-4" />} label="Response" to="/response-time" collapsed={collapsed} />
              <SidebarItem icon={<AlertTriangle className="h-4 w-4" />} label="Error" to="/error-analysis" collapsed={collapsed} />
              <SidebarItem icon={<Hourglass className="h-4 w-4" />} label="Slow" to="/slow-calls-analysis" collapsed={collapsed} />
            </div>
          </div>

        </div>
      </div>
      
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

function SidebarItem({ icon, label, to, collapsed = false }: { 
  icon?: React.ReactNode; 
  label: string; 
  to: string; 
  collapsed?: boolean;
}) {
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
      {icon && <span className={collapsed ? "" : "mr-2"}>{icon}</span>}
      {!collapsed && <span>{label}</span>}
    </Link>
  );
}
