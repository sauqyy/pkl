import { createContext, useContext, useState, ReactNode, useMemo } from 'react';

// Tier to Business Transactions mapping extracted from business_transactions.csv
export const TIER_BT_MAP: Record<string, string[]> = {
  'integration-service': [
    '/smart-integration/users',
    '/smart-integration/si',
    '/smart-integration/smart2',
    '/smart-integration/email',
    '/smart-integration/spaj',
    '/smart-integration/aws',
    '/smart-integration/application',
    '/smart-integration/payment',
    '/smart-integration/agent',
    '/smart-integration/dukcapli',
    '/smart-integration/blacklist',
    '/smart-integration/doc',
    '/smart-integration/phone',
    '/smart-integration/otp',
    '/smart-integration/ocr',
    '/smart-integration/seojk',
    '/smart-integration/crossSelling',
    '/smart-integration/error',
    '/smart-integration/RemoteSellingVideo',
    '/smart-integration/tts',
  ],
  'remote-selling-service-deployment': [
    '/remote-selling-service/v1',
    '/remote-selling-service/error',
  ],
  'dynamic-letter-deployment': [
    '/dynamic-letter/api',
  ],
  'payment-service': [
    '/smart-payment/pay',
    '/smart-payment/doku',
    '/smart-payment/payment',
    '/smart-payment/error',
  ],
  'validation-ph-service-deployment': [
    '/validation-ph-service/api',
  ],
  'otp-smart2-deployment': [
    '/otp-smart2/otp',
    '/otp-smart2/apps',
  ],
};

export const TIERS = Object.keys(TIER_BT_MAP);

interface BusinessTransactionContextType {
  selectedTier: string;
  setSelectedTier: (tier: string) => void;
  selectedTransaction: string;
  setSelectedTransaction: (transaction: string) => void;
  availableTransactions: string[];
}

const BusinessTransactionContext = createContext<BusinessTransactionContextType | undefined>(undefined);

export function BusinessTransactionProvider({ children }: { children: ReactNode }) {
  const [selectedTier, setSelectedTier] = useState(TIERS[0]);
  const [selectedTransaction, setSelectedTransaction] = useState(TIER_BT_MAP[TIERS[0]][0]);

  const availableTransactions = useMemo(() => {
    return TIER_BT_MAP[selectedTier] || [];
  }, [selectedTier]);

  return (
    <BusinessTransactionContext.Provider value={{ 
      selectedTier, 
      setSelectedTier, 
      selectedTransaction, 
      setSelectedTransaction,
      availableTransactions 
    }}>
      {children}
    </BusinessTransactionContext.Provider>
  );
}

export function useBusinessTransaction() {
  const context = useContext(BusinessTransactionContext);
  if (!context) {
    throw new Error('useBusinessTransaction must be used within a BusinessTransactionProvider');
  }
  return context;
}
