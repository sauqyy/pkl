import { createContext, useContext, useState, ReactNode } from "react"
import { subDays, startOfDay, endOfDay } from "date-fns"

export interface DateRange {
  from: Date | undefined
  to: Date | undefined
}

interface DateRangeContextType {
  dateRange: DateRange
  setDateRange: (range: DateRange) => void
  resetToDefault: () => void
}

const DateRangeContext = createContext<DateRangeContextType | undefined>(undefined)

// Default: last 30 days
const getDefaultRange = (): DateRange => ({
  from: startOfDay(subDays(new Date(), 30)),
  to: endOfDay(new Date())
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
