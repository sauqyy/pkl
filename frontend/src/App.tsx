import { BrowserRouter, Routes, Route } from 'react-router-dom'
import DashboardLayout from '@/layouts/DashboardLayout'
import Dashboard from '@/pages/Dashboard'
import Forecasting from '@/pages/Forecasting'
import ResponseTime from '@/pages/ResponseTime'
import LoadAnalysis from '@/pages/LoadAnalysis'
import ErrorAnalysis from '@/pages/ErrorAnalysis'
import { ThemeProvider } from '@/components/ThemeProvider'

function App() {
  return (
    <ThemeProvider defaultTheme="dark" storageKey="handalin-theme">
      <BrowserRouter>
        <DashboardLayout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/forecasting" element={<Forecasting />} />
            <Route path="/response-time" element={<ResponseTime />} />
            <Route path="/load-analysis" element={<LoadAnalysis />} />
            <Route path="/error-analysis" element={<ErrorAnalysis />} />
          </Routes>
        </DashboardLayout>
      </BrowserRouter>
    </ThemeProvider>
  )
}

export default App
