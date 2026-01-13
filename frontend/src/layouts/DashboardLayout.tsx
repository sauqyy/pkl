import { Sidebar } from "@/components/Sidebar";

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-screen bg-background text-foreground font-sans antialiased dark">
      <Sidebar />
      <main className="flex-1 p-8 overflow-y-auto h-screen bg-background">
        {children}
      </main>
    </div>
  );
}
