import { useState, useRef, useEffect } from "react";
import { MessageCircle, X, Send, Bot } from "lucide-react";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Card } from "./ui/card";
import { ScrollArea } from "./ui/scroll-area";
import { cn } from "@/lib/utils";

interface Message {
  id: string;
  role: "user" | "bot";
  content: string;
}

export function Chatbot() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "bot",
      content: "Halo! Saya Hana. Ada yang bisa saya bantu terkait data di halaman ini?",
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (messagesEndRef.current) {
        messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isOpen]);

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: inputValue,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    setIsLoading(true);

    try {
      const mainElement = document.querySelector('main') || document.body;
      const pageContent = mainElement.innerText.slice(0, 50000); 

      const response = await fetch("http://localhost:5000/api/ask-bot", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: userMessage.content,
          html_content: pageContent, 
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to get response");
      }

      const data = await response.json();
      
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "bot",
        content: data.answer || "Maaf, saya tidak mengerti.",
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error("Chat error:", error);
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: "bot",
          content: "Maaf, terjadi kesalahan. Silakan coba lagi nanti.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const primaryColor = "bg-[#499ED7]";
  const headerGradient = "bg-gradient-to-r from-[#499ED7] to-[#2D82B5]";

  return (
    <div className="fixed bottom-6 right-6 z-50 flex flex-col items-end">
      {isOpen && (
        <Card className="mb-4 w-[400px] h-[600px] border-0 shadow-2xl flex flex-col overflow-hidden animate-in slide-in-from-bottom-5 fade-in duration-200 rounded-2xl">
          {/* Header */}
          <div className={cn("p-4 flex items-center justify-between text-white shrink-0", headerGradient)}>
            <div className="flex items-center gap-3">
              <div className="bg-white/20 p-2 rounded-full">
                <Bot className="w-6 h-6 text-white" />
              </div>
              <div className="flex flex-col">
                <span className="font-semibold text-base">Hana</span>
                <span className="text-xs text-white/80">Handalin AI</span>
              </div>
            </div>
            <Button
              variant="ghost"
              size="icon"
              className="text-white hover:bg-white/20 rounded-full h-8 w-8"
              onClick={() => setIsOpen(false)}
            >
              <X className="w-5 h-5" />
            </Button>
          </div>

          {/* Messages Area */}
          <ScrollArea className="flex-1 p-4 bg-secondary/30">
            <div className="space-y-4 flex flex-col pb-4">
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={cn(
                    "flex w-full",
                    msg.role === "user" ? "justify-end" : "justify-start"
                  )}
                >
                  <div className={cn(
                    "max-w-[85%] rounded-2xl px-4 py-3 text-sm shadow-sm",
                    msg.role === "user" 
                      ? "bg-[#499ED7] text-white rounded-tr-none" 
                      : "bg-white dark:bg-card border dark:border-border text-foreground rounded-tl-none"
                  )}>
                    {msg.role === "bot" ? (
                        <div className="prose prose-sm dark:prose-invert max-w-none">
                            <ReactMarkdown 
                              remarkPlugins={[remarkGfm]}
                              components={{
                                p: ({node, ...props}) => <p className="mb-4 last:mb-0 leading-relaxed" {...props} />,
                                ul: ({node, ...props}) => <ul className="my-4 list-disc pl-4 space-y-2" {...props} />,
                                ol: ({node, ...props}) => <ol className="my-4 list-decimal pl-4 space-y-2" {...props} />,
                                li: ({node, ...props}) => <li className="mb-1" {...props} />,
                                strong: ({node, ...props}) => <strong className="font-bold text-[#499ED7]" {...props} />
                              }}
                            >
                                {msg.content}
                            </ReactMarkdown>
                        </div>
                    ) : (
                        msg.content
                    )}
                  </div>
                </div>
              ))}
              {isLoading && (
                 <div className="flex w-full justify-start">
                  <div className="bg-white dark:bg-card border dark:border-border rounded-2xl rounded-tl-none px-4 py-3 shadow-sm">
                    <div className="flex gap-1">
                      <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:-0.3s]"></span>
                      <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:-0.15s]"></span>
                      <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce"></span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </ScrollArea>

          {/* Input Area */}
          <div className="p-4 bg-background border-t">
            <div className="relative">
              <Input
                placeholder="Tanya tentang halaman ini..."
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                className="pr-12 py-6 rounded-full border-slate-200 dark:border-slate-700 focus-visible:ring-offset-0 focus-visible:ring-1 focus-visible:ring-primary shadow-sm"
              />
              <Button
                size="icon"
                className={cn("absolute right-1 top-1 h-10 w-10 rounded-full", primaryColor, "hover:bg-opacity-90 transition-opacity")}
                onClick={handleSendMessage}
                disabled={isLoading || !inputValue.trim()}
              >
                <Send className="w-4 h-4 text-white" />
              </Button>
            </div>
          </div>
        </Card>
      )}

      {/* Launcher Button */}
      <Button
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          "h-14 w-14 rounded-full shadow-lg transition-all duration-300 hover:scale-105",
          primaryColor
        )}
      >
        {isOpen ? (
          <X className="w-7 h-7 text-white" />
        ) : (
          <MessageCircle className="w-7 h-7 text-white" />
        )}
      </Button>
    </div>
  );
}
