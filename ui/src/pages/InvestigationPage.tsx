import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useRunStore } from '../store/runStore';
import { useWebSocket } from '../hooks/useWebSocket';
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
  const { connect, isConnected } = useWebSocket(runId);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    if (!runId) return;
    setLoadError(null);
    console.log('[InvestigationPage] Loading run:', runId);

    fetch(`/api/run/${runId}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        return res.json();
      })
      .then((data) => {
        console.log('[InvestigationPage] Run loaded:', {
          run_id: data.experiment?.run_id,
          attempts_count: data.attempts?.length,
          result: data.result,
        });
        if (!data.attempts || !Array.isArray(data.attempts)) {
          console.error('[InvestigationPage] Run data missing attempts array:', data);
          setLoadError('Invalid run data: missing attempts. The run file may be corrupted.');
          return;
        }
        setSelectedRun(data);
      })
      .catch((err) => {
        console.error('[InvestigationPage] Failed to load run:', err);
        setLoadError(`Failed to load run: ${err.message}`);
        navigate('/runs');
      });
  }, [runId]);

  if (loadError) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50">
        <div className="text-center">
          <p className="text-red-600 font-medium mb-2">Error Loading Run</p>
          <p className="text-sm text-slate-500 mb-4">{loadError}</p>
          <button
            onClick={() => navigate('/runs')}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700"
          >
            ← Back to Runs
          </button>
        </div>
      </div>
    );
  }

  if (!selectedRun) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50">
        <p className="text-slate-500">Loading run...</p>
      </div>
    );
  }

  const attempt = selectedRun.attempts?.[selectedAttemptIndex];
  if (!attempt) {
    console.error('[InvestigationPage] Attempt not found:', {
      index: selectedAttemptIndex,
      total_attempts: selectedRun.attempts?.length,
    });
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50">
        <div className="text-center">
          <p className="text-yellow-600 font-medium mb-2">No Attempt Data</p>
          <p className="text-sm text-slate-500 mb-4">
            Attempt {selectedAttemptIndex + 1} not found (total: {selectedRun.attempts?.length || 0})
          </p>
          <button
            onClick={() => navigate('/runs')}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700"
          >
            ← Back to Runs
          </button>
        </div>
      </div>
    );
  }

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
          {isConnected && (
            <span className="flex items-center gap-1.5 text-xs text-green-600">
              <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
              Live
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <a
            href={`/api/export/${selectedRun.experiment.run_id}/json`}
            download={`${selectedRun.experiment.run_id}.json`}
            className="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 rounded-lg text-xs font-medium transition-colors"
          >
            Export JSON
          </a>
          <a
            href={`/api/export/${selectedRun.experiment.run_id}/csv`}
            download={`${selectedRun.experiment.run_id}.csv`}
            className="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 rounded-lg text-xs font-medium transition-colors"
          >
            Export CSV
          </a>
          <a
            href={`/api/export/${selectedRun.experiment.run_id}/html`}
            download={`${selectedRun.experiment.run_id}.html`}
            className="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 rounded-lg text-xs font-medium transition-colors"
          >
            Export HTML
          </a>
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
