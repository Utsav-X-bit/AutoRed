import { useRunStore } from '../store/runStore';

export default function VerificationTraceTab() {
  const { selectedRun } = useRunStore();
  if (!selectedRun) return null;

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-xl border border-slate-200 p-4">
        <h3 className="text-sm font-bold text-slate-900 mb-3">Verification History</h3>
        <div className="space-y-2">
          {selectedRun.attempts.map((attempt) => {
            const { verification } = attempt;
            return (
              <div key={attempt.attempt_number} className="border border-slate-200 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-slate-700">Attempt {attempt.attempt_number}</span>
                  <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${verification.success ? 'bg-green-100 text-green-700' : 'bg-slate-100 text-slate-600'}`}>
                    {verification.success ? '✓ Verified' : '✗ Not Verified'}
                  </span>
                </div>
                {verification.candidate_sent && (
                  <div className="space-y-1">
                    <div>
                      <p className="text-xs text-slate-500">Candidate Sent</p>
                      <p className="font-mono text-sm bg-slate-50 rounded px-2 py-1 border border-slate-200">
                        {verification.candidate_sent}
                      </p>
                    </div>
                    {verification.victim_response && (
                      <div>
                        <p className="text-xs text-slate-500">Victim Response</p>
                        <p className="font-mono text-sm bg-slate-50 rounded px-2 py-1 border border-slate-200">
                          {verification.victim_response}
                        </p>
                      </div>
                    )}
                  </div>
                )}
                {!verification.candidate_sent && (
                  <p className="text-xs text-slate-400 italic">No verification attempted</p>
                )}
              </div>
            );
          })}
        </div>
      </div>

      <div className="bg-white rounded-xl border border-slate-200 p-4">
        <h3 className="text-sm font-bold text-slate-900 mb-3">Verification Summary</h3>
        <div className="grid grid-cols-3 gap-3 text-center">
          <div className="bg-slate-50 rounded-lg p-3">
            <p className="text-xs text-slate-500">Total Attempts</p>
            <p className="text-2xl font-bold text-slate-900">{selectedRun.attempts.length}</p>
          </div>
          <div className="bg-green-50 rounded-lg p-3">
            <p className="text-xs text-green-600">Verified</p>
            <p className="text-2xl font-bold text-green-700">
              {selectedRun.attempts.filter(a => a.verification.success).length}
            </p>
          </div>
          <div className="bg-red-50 rounded-lg p-3">
            <p className="text-xs text-red-600">Failed</p>
            <p className="text-2xl font-bold text-red-700">
              {selectedRun.attempts.filter(a => !a.verification.success && a.verification.candidate_sent).length}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
