import { useState } from 'react';
import ScenarioTab from './ScenarioTab';
import AttackEvolutionTab from './AttackEvolutionTab';
import ExtractorDebuggerTab from './ExtractorDebuggerTab';
import VerificationTraceTab from './VerificationTraceTab';

const tabs = [
  { id: 'scenario', label: 'Scenario' },
  { id: 'evolution', label: 'Attack Evolution' },
  { id: 'extractor', label: 'Extractor Debugger' },
  { id: 'verification', label: 'Verification' },
];

export default function InvestigationTabs() {
  const [activeTab, setActiveTab] = useState('scenario');

  return (
    <div className="border-t border-slate-200 bg-white flex-shrink-0">
      <div className="flex items-center gap-1 px-4 border-b border-slate-200">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === tab.id
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-slate-500 hover:text-slate-700'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div className="p-4 max-h-96 overflow-y-auto bg-slate-50">
        {activeTab === 'scenario' && <ScenarioTab />}
        {activeTab === 'evolution' && <AttackEvolutionTab />}
        {activeTab === 'extractor' && <ExtractorDebuggerTab />}
        {activeTab === 'verification' && <VerificationTraceTab />}
      </div>
    </div>
  );
}
