import { useState, useEffect } from "react"
import { Search, X } from "lucide-react"
import { Input } from "@/components/ui/input"
import { cn } from "@/lib/utils"

interface GlobalSearchProps {
  className?: string
  placeholder?: string
}

export function GlobalSearch({ className, placeholder = "Find in page..." }: GlobalSearchProps) {
  const [query, setQuery] = useState("")
  const [currentIndex, setCurrentIndex] = useState(-1)
  // Store all highlighted elements reference to cycle through them
  // We don't store DOM elements in state to avoid re-renders, just track index
  
  // Custom highlight function
  const highlightText = (searchText: string) => {
    // 1. Clear existing highlights
    const existingHighlights = document.querySelectorAll('.search-highlight')
    existingHighlights.forEach(mark => {
      const parent = mark.parentNode
      if (parent) {
        parent.replaceChild(document.createTextNode(mark.textContent || ''), mark)
        // Normalize to merge adjacent text nodes
        parent.normalize()
      }
    })

    if (!searchText) {
      setCurrentIndex(-1)
      return
    }

    // 2. Find new matches
    // Create a TreeWalker to find all text nodes in the body
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
        acceptNode: (node) => {
            // Skip scripts, styles, and the search input itself to avoid weirdness
            if (node.parentElement?.tagName === 'SCRIPT' || 
                node.parentElement?.tagName === 'STYLE' ||
                node.parentElement?.tagName === 'INPUT' ||
                node.parentElement?.tagName === 'TEXTAREA' ||
                (node.parentElement?.closest('.global-search-container')) // Skip self
            ) {
                return NodeFilter.FILTER_REJECT
            }
            return NodeFilter.FILTER_ACCEPT
        }
    })

    const textNodes: Text[] = []
    let currentNode = walker.nextNode()
    while (currentNode) {
        textNodes.push(currentNode as Text)
        currentNode = walker.nextNode()
    }



    // We iterate backwards or carefully to avoid messing up traversal if modifying?
    // Actually, safest is to collect nodes, then modifying them might invalidate potential next matches in same node.
    // Simple approach: Iterate nodes, if match found, split matching text, wrap in span.

    // Note: Replacing nodes invalidates the list.
    // We process nodes one by one.
    
    // Simplification: We only highlight the *first* several matches or all? 
    // Highlights ALL matches.

    // Regex for case insensitive global search
    const regex = new RegExp(`(${searchText.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi')

    // Track matches to scroll to
    const matches: HTMLElement[] = []

    textNodes.forEach(node => {
        const text = node.nodeValue || ''
        if (!text.match(regex)) return

        const fragment = document.createDocumentFragment()
        let lastIndex = 0
        let match

        // We need to re-run regex because match() isn't enough for indices or multiple matches
        while ((match = regex.exec(text)) !== null) {
            // Append text before match
            fragment.appendChild(document.createTextNode(text.slice(lastIndex, match.index)))
            
            // Create highlight span
            const span = document.createElement('span')
            span.className = 'search-highlight'
            span.textContent = match[0]
            fragment.appendChild(span)
            matches.push(span)

            lastIndex = regex.lastIndex
        }
        
        // Append remaining text
        fragment.appendChild(document.createTextNode(text.slice(lastIndex)))

        // Replace text node with fragment
        node.parentNode?.replaceChild(fragment, node)
    })

    if (matches.length > 0) {
        // Scroll to first match
        matches[0].scrollIntoView({ behavior: 'smooth', block: 'center' })
        setCurrentIndex(0)
    } else {
        setCurrentIndex(-1)
    }
  }

  // Navigate function (Next match)
  const navigateNext = () => {
      const allMarks = document.querySelectorAll('.search-highlight')
      if (allMarks.length === 0) return

      let nextIndex = currentIndex + 1
      if (nextIndex >= allMarks.length) nextIndex = 0 // Loop back

      allMarks[nextIndex].scrollIntoView({ behavior: 'smooth', block: 'center' })
      setCurrentIndex(nextIndex)
  }

  // Run highlight when query changes
  // Using a small debounce could be nice, but "as typing" requests immediate feedback.
  // We'll put it in useEffect.
  useEffect(() => {
    // Debounce slightly to avoid heavy DOM thrashing on every keystroke
    const timeoutId = setTimeout(() => {
        highlightText(query)
    }, 100)
    return () => clearTimeout(timeoutId)
  }, [query])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
        e.preventDefault()
        navigateNext()
    }
  }

  return (
    <div className={cn("relative global-search-container", className)}>
      <div className="relative">
        <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
        <Input
          type="search"
          placeholder={placeholder}
          className="pl-8 w-[250px] bg-card"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        {query && (
          <button 
            onClick={() => {
              setQuery("") 
              // Cleanup highlights immediately on clear
              highlightText("")
            }}
            className="absolute right-2.5 top-2.5 text-muted-foreground hover:text-foreground"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>
    </div>
  )
}
