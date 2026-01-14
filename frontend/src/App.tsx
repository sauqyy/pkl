import { BrowserRouter, Routes, Route } from 'react-router-dom'
import DashboardLayout from '@/layouts/DashboardLayout'
import ExecutiveDashboard from '@/pages/ExecutiveDashboard'
import Dashboard from '@/pages/Dashboard'
import Forecasting from '@/pages/Forecasting'
import ResponseTime from '@/pages/ResponseTime'
import LoadAnalysis from '@/pages/LoadAnalysis'
import ErrorAnalysis from '@/pages/ErrorAnalysis'
import SlowCallsAnalysis from '@/pages/SlowCallsAnalysis'
import { ThemeProvider } from '@/components/ThemeProvider'

function App() {
  return (
    <ThemeProvider defaultTheme="dark" storageKey="handalin-theme">
      <BrowserRouter>
        <DashboardLayout>
          <Routes>
            <Route path="/" element={<ExecutiveDashboard />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/forecasting" element={<Forecasting />} />
            <Route path="/response-time" element={<ResponseTime />} />
            <Route path="/load-analysis" element={<LoadAnalysis />} />
            <Route path="/error-analysis" element={<ErrorAnalysis />} />
            <Route path="/slow-calls-analysis" element={<SlowCallsAnalysis />} />
          </Routes>
        </DashboardLayout>
      </BrowserRouter>
    </ThemeProvider>
  )
}

export default App
