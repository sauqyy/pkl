import { Sidebar } from "@/components/Sidebar";
import { SidebarProvider, useSidebar } from "@/components/SidebarContext";
import { usePageTitle } from "@/hooks/usePageTitle";
import { Chatbot } from "@/components/Chatbot";

function LayoutContent({ children }: { children: React.ReactNode }) {
  const { collapsed, toggleSidebar } = useSidebar();
  usePageTitle();
  
  return (
    <div className="flex min-h-screen bg-background text-foreground font-sans antialiased">
      <Sidebar collapsed={collapsed} onToggle={toggleSidebar} />
      <main className="flex-1 p-8 overflow-y-auto h-screen bg-background">
        {children}
      </main>
      <Chatbot />
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
