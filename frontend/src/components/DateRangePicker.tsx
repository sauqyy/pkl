import { useState, useEffect } from "react"
import { format, subDays, subMonths, subYears, subHours, subMinutes } from "date-fns"
import { Calendar as CalendarIcon } from "lucide-react"
import { DateRange as DayPickerRange } from "react-day-picker"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Calendar } from "@/components/ui/calendar"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { useDateRange } from "@/components/DateRangeContext"

interface DateRangePickerProps {
  className?: string
}

const PRESETS = [
  { key: "5m", label: "Last 5 Minutes", getValue: () => ({ from: subMinutes(new Date(), 5), to: new Date() }) },
  { key: "15m", label: "Last 15 Minutes", getValue: () => ({ from: subMinutes(new Date(), 15), to: new Date() }) },
  { key: "1h", label: "Last 1 Hour", getValue: () => ({ from: subHours(new Date(), 1), to: new Date() }) },
  { key: "6h", label: "Last 6 Hours", getValue: () => ({ from: subHours(new Date(), 6), to: new Date() }) },
  { key: "24h", label: "Last 24 Hours", getValue: () => ({ from: subHours(new Date(), 24), to: new Date() }) },
  { key: "7d", label: "Last 7 Days", getValue: () => ({ from: subDays(new Date(), 7), to: new Date() }) },
  { key: "30d", label: "Last 30 Days", getValue: () => ({ from: subDays(new Date(), 30), to: new Date() }) },
  { key: "6m", label: "Last 6 Months", getValue: () => ({ from: subMonths(new Date(), 6), to: new Date() }) },
  { key: "1y", label: "Last 1 Year", getValue: () => ({ from: subYears(new Date(), 1), to: new Date() }) },
  { key: "all", label: "All Time (Lifetime)", getValue: () => ({ from: new Date(2020, 0, 1), to: new Date() }) },
]

export function DateRangePicker({ className }: DateRangePickerProps) {
  const { dateRange, setDateRange } = useDateRange()
  const [isOpen, setIsOpen] = useState(false)
  const [tempRange, setTempRange] = useState<DayPickerRange | undefined>({
    from: dateRange.from,
    to: dateRange.to
  })

  // Sync temp range when opening
  useEffect(() => {
    if (isOpen) {
      setTempRange({
        from: dateRange.from,
        to: dateRange.to
      })
    }
  }, [isOpen, dateRange])

  const handleSelect = (range: DayPickerRange | undefined) => {
    setTempRange(range)
  }

  const handlePresetSelect = (preset: typeof PRESETS[0]) => {
    const range = preset.getValue()
    setTempRange(range)
    setDateRange({
      ...range,
      timeframe: preset.key
    })
    setIsOpen(false)
  }

  const handleApply = () => {
    if (tempRange?.from) {
      setDateRange({
        from: tempRange.from,
        to: tempRange.to || tempRange.from,
        timeframe: 'custom'
      })
    }
    setIsOpen(false)
  }

  const handleClear = () => {
    // Default to Last 7 Days as a "Reset" state
    const defaultPreset = PRESETS[5]
    const range = defaultPreset.getValue()
    setTempRange(range)
    setDateRange({
      ...range,
      timeframe: defaultPreset.key
    })
    setIsOpen(false)
  }

  return (
    <div className={cn("grid gap-2", className)}>
      <Popover open={isOpen} onOpenChange={setIsOpen}>
        <PopoverTrigger asChild>
          <Button
            id="date-range-picker"
            variant="outline"
            className={cn(
              "w-auto justify-start text-left font-normal bg-card",
              !dateRange && "text-muted-foreground"
            )}
          >
            <CalendarIcon className="mr-2 h-4 w-4" />
            {dateRange?.from ? (
              dateRange.to ? (
                <>
                  {format(dateRange.from, "LLL dd, y")} -{" "}
                  {format(dateRange.to, "LLL dd, y")}
                </>
              ) : (
                format(dateRange.from, "LLL dd, y")
              )
            ) : (
              <span>Pick a date range</span>
            )}
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-auto p-0 bg-card border-border" align="end">
          <div className="flex">
            {/* Presets Sidebar */}
            <div className="flex flex-col border-r border-border p-2 w-[160px] gap-1">
               <div className="text-xs font-semibold text-muted-foreground mb-2 px-2 py-1">Presets</div>
               {PRESETS.map((preset) => (
                 <button
                    key={preset.label}
                    onClick={() => handlePresetSelect(preset)}
                    className={cn(
                      "text-sm text-left px-2 py-1.5 rounded-sm hover:bg-accent transition-colors",
                      // Highlight if roughly matches tempRange
                      dateRange.timeframe === preset.key
                        ? "bg-accent/50 text-accent-foreground font-medium" 
                        : "text-foreground"
                    )}
                 >
                   {preset.label}
                 </button>
               ))}
            </div>

            {/* Calendar Area */}
            <div className="p-3">
              <Calendar
                mode="range"
                defaultMonth={tempRange?.from}
                selected={tempRange}
                onSelect={handleSelect}
                numberOfMonths={1}
              />
              
              {/* Actions */}
              <div className="flex justify-end gap-2 pt-3 border-t border-border">
                <Button variant="outline" size="sm" onClick={handleClear}>
                  Reset
                </Button>
                <Button size="sm" onClick={handleApply}>
                  Apply
                </Button>
              </div>
            </div>
          </div>
        </PopoverContent>
      </Popover>
    </div>
  )
}
