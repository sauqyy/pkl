import { useEffect } from 'react';
import { useLocation } from 'react-router-dom';

const routeTitles: Record<string, string> = {
  '/': 'Executive Dashboard | Handalin',
  '/dashboard': 'Dashboard | Handalin',
  '/forecasting': 'Forecasting | Handalin',
  '/response-time': 'Response Time | Handalin',
  '/load-analysis': 'Load Analysis | Handalin',
  '/error-analysis': 'Error Analysis | Handalin',
  '/slow-calls-analysis': 'Slow Calls Analysis | Handalin',
  '/business-transactions': 'Business Transactions | Handalin',
  '/jvm-health': 'JVM Health | Handalin',
};

export function usePageTitle() {
  const location = useLocation();

  useEffect(() => {
    const title = routeTitles[location.pathname] || 'Handalin';
    document.title = title;
  }, [location]);
}
