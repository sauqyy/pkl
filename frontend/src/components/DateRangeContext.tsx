import { createContext, useContext, useState, ReactNode } from "react"
import { subDays, startOfDay, endOfDay } from "date-fns"

export interface DateRange {
  from: Date | undefined
  to: Date | undefined
  timeframe: string // Added to carry context of the selection (e.g., '5m', '1h', 'custom')
}

interface DateRangeContextType {
  dateRange: DateRange
  setDateRange: (range: DateRange) => void
  resetToDefault: () => void
}

const DateRangeContext = createContext<DateRangeContextType | undefined>(undefined)

// Default: last 7 days (updated from 30 to be more aligned with common usage)
const getDefaultRange = (): DateRange => ({
  from: startOfDay(subDays(new Date(), 7)),
  to: endOfDay(new Date()),
  timeframe: '7d'
})

export function DateRangeProvider({ children }: { children: ReactNode }) {
  const [dateRange, setDateRange] = useState<DateRange>(getDefaultRange)

  const resetToDefault = () => {
    setDateRange(getDefaultRange())
  }

  return (
    <DateRangeContext.Provider value={{ dateRange, setDateRange, resetToDefault }}>
      {children}
    </DateRangeContext.Provider>
  )
}

export function useDateRange() {
  const context = useContext(DateRangeContext)
  if (context === undefined) {
    throw new Error("useDateRange must be used within a DateRangeProvider")
  }
  return context
}
