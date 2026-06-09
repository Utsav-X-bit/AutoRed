import { useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useRunStore } from '../store/runStore';
import TimelineSidebar from '../components/TimelineSidebar';
import GeneratorCard from '../components/GeneratorCard';
import VictimCard from '../components/VictimCard';
import ExtractorCard from '../components/ExtractorCard';
import VerifierCard from '../components/VerifierCard';
import AnalyticsPanel from '../components/AnalyticsPanel';
import InvestigationTabs from '../components/InvestigationTabs';

export default function InvestigationPage() {
  const { runId } = useParams<{ runId: string }>();
  const navigate = useNavigate();
  const { selectedRun, selectedAttemptIndex, setSelectedRun } = useRunStore();

  useEffect(() => {
    if (!runId) return;
    fetch(`/api/run/${runId}`)
      .then((res) => res.json())
      .then((data) => setSelectedRun(data))
      .catch((err) => {
        console.error('Failed to load run:', err);
        navigate('/runs');
      });
  }, [runId]);

  if (!selectedRun) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50">
        <p className="text-slate-500">Loading run...</p>
      </div>
    );
  }

  const attempt = selectedRun.attempts[selectedAttemptIndex];
  if (!attempt) return null;

  return (
    <div className="h-screen flex flex-col">
      {/* Top Bar */}
      <header className="bg-white border-b border-slate-200 px-4 py-2 flex items-center justify-between flex-shrink-0">
        <div className="flex items-center gap-3">
          <button onClick={() => navigate('/runs')} className="text-sm text-slate-500 hover:text-slate-900 transition-colors">
            ← Runs
          </button>
          <span className="text-slate-300">|</span>
          <h1 className="font-mono text-sm font-bold text-slate-900">{selectedRun.experiment.run_id}</h1>
          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${selectedRun.result.ground_truth_success ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
            {selectedRun.result.ground_truth_success ? 'SUCCESS' : 'FAILED'}
          </span>
        </div>
        <div className="flex items-center gap-2 text-sm text-slate-500">
          <span>Attempt {attempt.attempt_number}/{selectedRun.result.total_attempts}</span>
        </div>
      </header>

      {/* 3-Panel Layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left: Timeline */}
        <TimelineSidebar />

        {/* Center: Investigation */}
        <div className="flex-1 overflow-y-auto bg-slate-50 p-6">
          <div className="max-w-4xl mx-auto space-y-4">
            {/* Attempt Header */}
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-bold text-slate-900">Attempt {attempt.attempt_number}</h2>
              <div className="flex items-center gap-2">
                <span className="px-2 py-0.5 bg-purple-100 text-purple-700 text-xs font-medium rounded-full">
                  {attempt.generator.strategy}
                </span>
                <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${attempt.judge.decision === 'ATTACK' ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'}`}>
                  {attempt.judge.decision} ({attempt.judge.confidence.toFixed(2)})
                </span>
              </div>
            </div>

            {/* Pipeline */}
            <GeneratorCard attempt={attempt} />
            <div className="flex justify-center text-slate-400 text-lg">↓</div>
            <VictimCard attempt={attempt} accessCode={selectedRun.scenario.access_code} />
            <div className="flex justify-center text-slate-400 text-lg">↓</div>
            <ExtractorCard attempt={attempt} />
            <div className="flex justify-center text-slate-400 text-lg">↓</div>
            <VerifierCard attempt={attempt} />
          </div>
        </div>

        {/* Right: Analytics */}
        <AnalyticsPanel />
      </div>

      {/* Bottom: Investigation Tabs */}
      <InvestigationTabs />
    </div>
  );
}
