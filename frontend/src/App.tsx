import { BrowserRouter, Routes, Route } from 'react-router-dom'
import DashboardLayout from '@/layouts/DashboardLayout'
import ExecutiveDashboard from '@/pages/ExecutiveDashboard'
import Dashboard from '@/pages/Dashboard'
import Forecasting from '@/pages/Forecasting'
import ResponseTime from '@/pages/ResponseTime'
import LoadAnalysis from '@/pages/LoadAnalysis'
import ErrorAnalysis from '@/pages/ErrorAnalysis'
import SlowCallsAnalysis from '@/pages/SlowCallsAnalysis'
import DatabaseAnalysis from '@/pages/DatabaseAnalysis'
import BusinessTransactions from '@/pages/BusinessTransactions'
import JVMHealth from '@/pages/JVMHealth'
import { ThemeProvider } from '@/components/ThemeProvider'
import { BusinessTransactionProvider } from '@/components/BusinessTransactionContext'

function App() {
  return (
    <ThemeProvider defaultTheme="dark" storageKey="handalin-theme">
      <BusinessTransactionProvider>
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
              <Route path="/business-transactions" element={<BusinessTransactions />} />
              <Route path="/database-analysis" element={<DatabaseAnalysis />} />
              <Route path="/jvm-health" element={<JVMHealth />} />
            </Routes>
          </DashboardLayout>
        </BrowserRouter>
      </BusinessTransactionProvider>
    </ThemeProvider>
  )
}

export default App
