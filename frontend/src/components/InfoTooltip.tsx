
import { Info } from 'lucide-react';
import React from 'react';

interface InfoTooltipProps {
  content: string;
}

const InfoTooltip: React.FC<InfoTooltipProps> = ({ content }) => {
  return (
    <div className="group relative inline-flex ml-2 items-center">
      <Info size={14} className="text-gray-400 hover:text-gray-600 cursor-help transition-colors" />
      {/* Tooltip Container */}
      <div className="pointer-events-none absolute bottom-full left-1/2 mb-2 hidden w-48 -translate-x-1/2 rounded bg-gray-900 p-2 text-xs text-center text-white opacity-0 shadow-lg transition-opacity group-hover:block group-hover:opacity-100 z-50">
        {content}
        {/* Arrow */}
        <div className="absolute top-full left-1/2 -ml-1 h-0 w-0 border-x-4 border-t-4 border-x-transparent border-t-gray-900"></div>
      </div>
    </div>
  );
};

export default InfoTooltip;
