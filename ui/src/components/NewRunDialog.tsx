import { useState, useCallback, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useRunStore } from '../store/runStore';
import type { AutoRedRun } from '../types/autored';

interface NewRunDialogProps {
  onClose: () => void;
  onSuccess: () => void;
}

export default function NewRunDialog({ onClose, onSuccess }: NewRunDialogProps) {
  const navigate = useNavigate();
  const { setSelectedRun } = useRunStore();

  const [maxAttempts, setMaxAttempts] = useState(20);
  const [scenarioId, setScenarioId] = useState('');
  const [running, setRunning] = useState(false);
  const [runId, setRunId] = useState<string | null>(null);
  const [currentAttempt, setCurrentAttempt] = useState(0);
  const [totalAttempts, setTotalAttempts] = useState(0);
  const [success, setSuccess] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);

  const connectWebSocket = useCallback((rid: string) => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/run/${rid}`);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'attempt_update') {
          setCurrentAttempt(data.attempt.attempt_number);
          setTotalAttempts(data.attempt.attempt_number);
          if (data.attempt.ground_truth_found) {
            setSuccess(true);
          }
        } else if (data.type === 'run_complete') {
          const rawRun = (data as { run: unknown }).run;
          if (rawRun && typeof rawRun === 'object' && 'error' in rawRun) {
            setError((rawRun as { error: string }).error);
          } else {
            const run = rawRun as AutoRedRun;
            setSuccess(run.result?.ground_truth_success ?? false);
            setTotalAttempts(run.result?.total_attempts ?? 0);
            setSelectedRun(run);
          }
          ws.close();
          wsRef.current = null;
          setRunning(false);
        }
      } catch (e) {
        console.error('WebSocket message parse error:', e);
      }
    };

    ws.onclose = () => {
      wsRef.current = null;
      if (running) {
        setRunning(false);
        onSuccess();
      }
    };

    ws.onerror = (err) => {
      console.error('WebSocket error:', err);
    };
  }, [setSelectedRun, running, onSuccess]);

  const handleStart = async () => {
    setRunning(true);
    setError(null);
    setSuccess(null);
    setCurrentAttempt(0);

    try {
      const rid = `run_${Date.now()}`;
      setRunId(rid);
      connectWebSocket(rid);

      const params = new URLSearchParams({
        max_attempts: String(maxAttempts),
        ...(scenarioId ? { scenario_id: scenarioId } : {}),
      });

      const res = await fetch(`/api/run?${params}`, { method: 'POST' });
      if (!res.ok) {
        const err = await res.json();
        setError(err.detail || 'Failed to start run');
        setRunning(false);
        return;
      }

      const data = await res.json();
      if (data.run_id && data.run_id !== rid) {
        setRunId(data.run_id);
        if (wsRef.current) wsRef.current.close();
        connectWebSocket(data.run_id);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Network error');
      setRunning(false);
    }
  };

  const handleDone = () => {
    if (wsRef.current) wsRef.current.close();
    if (success && runId) {
      navigate(`/run/${runId}`);
    } else {
      onSuccess();
    }
  };

  const handleCancel = () => {
    if (wsRef.current && running) {
      if (!confirm('Run is in progress. Cancel?')) return;
      wsRef.current.close();
    }
    onClose();
  };

  useEffect(() => {
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const progress = maxAttempts > 0 ? (currentAttempt / maxAttempts) * 100 : 0;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-lg">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-200">
          <h2 className="text-lg font-bold text-slate-900">
            {running ? 'Running Experiment...' : 'New Experiment Run'}
          </h2>
          <p className="text-sm text-slate-500 mt-1">
            {running
              ? 'Models are pre-loaded — run will start immediately'
              : 'Start a new AutoRed attack scenario with pre-loaded models'}
          </p>
        </div>

        {/* Body */}
        {!running ? (
          <div className="px-6 py-5 space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Max Attempts
              </label>
              <input
                type="number"
                min={1}
                max={100}
                value={maxAttempts}
                onChange={(e) => setMaxAttempts(Number(e.target.value))}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Scenario ID <span className="text-slate-400">(optional, random if empty)</span>
              </label>
              <input
                type="text"
                value={scenarioId}
                onChange={(e) => setScenarioId(e.target.value)}
                placeholder="e.g., defense_123"
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
        ) : (
          <div className="px-6 py-8">
            {/* Progress bar */}
            <div className="mb-4">
              <div className="flex items-center justify-between text-sm text-slate-600 mb-1">
                <span>Attempt {currentAttempt} / {maxAttempts}</span>
                <span>{Math.round(progress)}%</span>
              </div>
              <div className="w-full bg-slate-200 rounded-full h-2.5 overflow-hidden">
                <div
                  className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
                  style={{ width: `${Math.min(progress, 100)}%` }}
                />
              </div>
            </div>

            {runId && (
              <p className="text-xs text-slate-500 text-center font-mono mb-4">
                Run ID: {runId}
              </p>
            )}

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                <p className="text-sm text-red-700">{error}</p>
              </div>
            )}

            {success !== null && !error && (
              <div className={`rounded-lg p-4 text-center ${success ? 'bg-green-50' : 'bg-slate-50'}`}>
                <p className={`text-lg font-bold ${success ? 'text-green-700' : 'text-slate-700'}`}>
                  {success ? '✓ Success — Access Code Extracted' : '✗ Failed — Max Attempts Reached'}
                </p>
                <p className="text-sm text-slate-500 mt-1">
                  Total attempts: {totalAttempts}
                </p>
              </div>
            )}
          </div>
        )}

        {/* Footer */}
        <div className="px-6 py-4 border-t border-slate-200 flex items-center justify-end gap-3">
          {!running ? (
            <>
              <button
                onClick={onClose}
                className="px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100 rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleStart}
                className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition-colors"
              >
                Start Run
              </button>
            </>
          ) : (
            <>
              <button
                onClick={handleCancel}
                className="px-4 py-2 text-sm font-medium text-red-600 hover:bg-red-50 rounded-lg transition-colors"
              >
                Cancel
              </button>
              {(success !== null || error) && (
                <button
                  onClick={handleDone}
                  className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition-colors"
                >
                  {success ? 'View Results' : 'Done'}
                </button>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
