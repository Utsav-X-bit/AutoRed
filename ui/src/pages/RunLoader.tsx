import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useRunStore } from '../store/runStore';
import FilterBar from '../components/FilterBar';
import NewRunDialog from '../components/NewRunDialog';
import type { RunListItem } from '../types/autored';

export default function RunLoader() {
  const navigate = useNavigate();
  const { runs, setRuns } = useRunStore();
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [selectedRuns, setSelectedRuns] = useState<string[]>([]);
  const [filteredRuns, setFilteredRuns] = useState<RunListItem[]>([]);
  const [showNewRun, setShowNewRun] = useState(false);

  useEffect(() => {
    fetchRuns();
  }, []);

  const fetchRuns = async () => {
    try {
      const res = await fetch('/api/runs');
      const data = await res.json();
      setRuns(data);
      setFilteredRuns(data);
    } catch (e) {
      console.error('Failed to load runs:', e);
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);
    try {
      await fetch('/api/runs/upload', { method: 'POST', body: formData });
      fetchRuns();
    } catch (err) {
      console.error('Upload failed:', err);
    } finally {
      setUploading(false);
    }
  };

  const handleNewRunSuccess = () => {
    setShowNewRun(false);
    fetchRuns();
  };

  if (loading) return <div className="p-8 text-center text-slate-500">Loading runs...</div>;

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <h1 className="text-xl font-bold text-slate-900">AutoRed — Run History</h1>
          <div className="flex gap-3 items-center">
            <button
              onClick={() => setShowNewRun(true)}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors"
            >
              + New Run
            </button>
            {selectedRuns.length === 2 && (
              <button
                onClick={() => navigate(`/compare/${selectedRuns[0]}/${selectedRuns[1]}`)}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors"
              >
                Compare
              </button>
            )}
            <label className="px-4 py-2 bg-slate-100 hover:bg-slate-200 rounded-lg cursor-pointer text-sm font-medium transition-colors">
              Upload JSON
              <input type="file" accept=".json" onChange={handleUpload} className="hidden" />
            </label>
            <span className="text-sm text-slate-500 self-center">{runs.length} runs</span>
          </div>
        </div>
      </header>

      {/* Filter Bar */}
      <div className="max-w-7xl mx-auto px-6 py-4">
        <FilterBar runs={runs} onFilter={setFilteredRuns} />
      </div>

      {/* Run List */}
      <main className="max-w-7xl mx-auto p-6">
        {runs.length === 0 ? (
          <div className="text-center py-20 text-slate-400">
            <p className="text-lg">No runs yet</p>
            <p className="text-sm mt-2">Upload a JSON file or start a new experiment</p>
          </div>
        ) : (
          <div className="space-y-3">
            {filteredRuns.map((run: RunListItem) => (
              <button
                key={run.run_id}
                onClick={() => navigate(`/run/${run.run_id}`)}
                className="w-full text-left bg-white rounded-xl border border-slate-200 p-4 hover:border-blue-400 hover:shadow-md transition-all"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <input
                      type="checkbox"
                      checked={selectedRuns.includes(run.run_id)}
                      onChange={(e) => {
                        e.stopPropagation();
                        if (e.target.checked) {
                          if (selectedRuns.length < 2) setSelectedRuns([...selectedRuns, run.run_id]);
                        } else {
                          setSelectedRuns(selectedRuns.filter(id => id !== run.run_id));
                        }
                      }}
                      onClick={(e) => e.stopPropagation()}
                      className="mr-1"
                      disabled={selectedRuns.length >= 2 && !selectedRuns.includes(run.run_id)}
                    />
                    <span className={`inline-flex items-center justify-center w-8 h-8 rounded-full text-sm font-bold ${run.success ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                      {run.success ? '✓' : '✗'}
                    </span>
                    <div>
                      <p className="font-mono text-sm font-semibold text-slate-900">{run.run_id}</p>
                      <p className="text-xs text-slate-500 mt-0.5">{run.timestamp}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-6 text-sm">
                    <div className="text-right">
                      <p className="text-slate-500 text-xs">Generator</p>
                      <p className="font-medium text-slate-700">{run.generator.split('/').pop() || run.generator}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-slate-500 text-xs">Victim</p>
                      <p className="font-medium text-slate-700">{run.victim.split('/').pop() || run.victim}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-slate-500 text-xs">Attempts</p>
                      <p className="font-medium text-slate-700">{run.total_attempts}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-slate-500 text-xs">Access Code</p>
                      <p className="font-mono font-medium text-amber-600">{run.access_code}</p>
                    </div>
                  </div>
                </div>
              </button>
            ))}
          </div>
        )}
      </main>

      {/* New Run Dialog */}
      {showNewRun && (
        <NewRunDialog onClose={() => setShowNewRun(false)} onSuccess={handleNewRunSuccess} />
      )}
    </div>
  );
}
