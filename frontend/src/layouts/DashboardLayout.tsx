import { Sidebar } from "@/components/Sidebar";
import { SidebarProvider, useSidebar } from "@/components/SidebarContext";

function LayoutContent({ children }: { children: React.ReactNode }) {
  const { collapsed, toggleSidebar } = useSidebar();
  
  return (
    <div className="flex min-h-screen bg-background text-foreground font-sans antialiased">
      <Sidebar collapsed={collapsed} onToggle={toggleSidebar} />
      <main className="flex-1 p-8 overflow-y-auto h-screen bg-background">
        {children}
      </main>
    </div>
  );
}

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <SidebarProvider>
      <LayoutContent>{children}</LayoutContent>
    </SidebarProvider>
  );
}
