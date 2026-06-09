import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useRunStore } from '../store/runStore';
import type { RunListItem, AutoRedRun } from '../types/autored';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell,
} from 'recharts';

export default function BenchmarkDashboard() {
  const navigate = useNavigate();
  const { runs } = useRunStore();
  const [benchmarkRuns, setBenchmarkRuns] = useState<AutoRedRun[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Load all runs, filter benchmark ones
    const load = async () => {
      try {
        const res = await fetch('/api/runs');
        const allRuns: RunListItem[] = await res.json();
        // Fetch full data for benchmark runs
        const benchmarkIds = allRuns.filter(r => r.benchmark_mode);
        const fullRuns = await Promise.all(
          benchmarkIds.map(r => fetch(`/api/run/${r.run_id}`).then(res => res.json()))
        );
        setBenchmarkRuns(fullRuns);
      } catch (e) {
        console.error('Failed to load benchmark data:', e);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  if (loading) return <div className="p-8 text-center text-slate-500">Loading benchmark data...</div>;
  if (benchmarkRuns.length === 0) {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center">
        <div className="text-center">
          <p className="text-lg text-slate-500">No benchmark runs found</p>
          <p className="text-sm text-slate-400 mt-2">Run experiments with --rounds flag to generate benchmark data</p>
          <button onClick={() => navigate('/runs')} className="mt-4 text-sm text-blue-600 hover:text-blue-700">
            ← Back to Runs
          </button>
        </div>
      </div>
    );
  }

  // Compute stats
  const totalRuns = benchmarkRuns.length;
  const successes = benchmarkRuns.filter(r => r.result.ground_truth_success).length;
  const successRate = (successes / totalRuns * 100).toFixed(1);
  const avgAttempts = (benchmarkRuns.reduce((sum, r) => sum + r.result.total_attempts, 0) / totalRuns).toFixed(1);

  // Strategy stats
  const strategyStats: Record<string, { total: number; successes: number }> = {};
  benchmarkRuns.forEach(run => {
    Object.entries(run.strategy_stats).forEach(([strategy, stat]) => {
      if (!strategyStats[strategy]) strategyStats[strategy] = { total: 0, successes: 0 };
      strategyStats[strategy].total += stat.failures + stat.successes;
      strategyStats[strategy].successes += stat.successes;
    });
  });

  const strategyData = Object.entries(strategyStats).map(([name, s]) => ({
    name,
    winRate: s.total > 0 ? (s.successes / s.total * 100).toFixed(1) : '0',
    total: s.total,
  }));

  // Attempts distribution
  const attemptsDist: Record<string, number> = {};
  benchmarkRuns.forEach(r => {
    const key = r.result.total_attempts <= 10 ? `${r.result.total_attempts}` : '>10';
    attemptsDist[key] = (attemptsDist[key] || 0) + 1;
  });
  const attemptsData = Object.entries(attemptsDist).map(([attempts, count]) => ({ attempts, count }));

  // Success/failure data
  const successData = [
    { name: 'Success', value: successes, fill: '#22c55e' },
    { name: 'Failure', value: totalRuns - successes, fill: '#ef4444' },
  ];

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button onClick={() => navigate('/runs')} className="text-sm text-slate-500 hover:text-slate-900">
              ← Runs
            </button>
            <span className="text-slate-300">|</span>
            <h1 className="text-xl font-bold text-slate-900">Benchmark Dashboard</h1>
          </div>
          <span className="text-sm text-slate-500">{totalRuns} benchmark runs</span>
        </div>
      </header>

      <main className="max-w-7xl mx-auto p-6 space-y-6">
        {/* Top Cards */}
        <div className="grid grid-cols-4 gap-4">
          <div className="bg-white rounded-xl border border-slate-200 p-4">
            <p className="text-xs text-slate-500 mb-1">Success Rate</p>
            <p className="text-3xl font-bold text-green-600">{successRate}%</p>
            <p className="text-xs text-slate-400 mt-1">{successes}/{totalRuns} runs</p>
          </div>
          <div className="bg-white rounded-xl border border-slate-200 p-4">
            <p className="text-xs text-slate-500 mb-1">Avg Attempts</p>
            <p className="text-3xl font-bold text-slate-900">{avgAttempts}</p>
          </div>
          <div className="bg-white rounded-xl border border-slate-200 p-4">
            <p className="text-xs text-slate-500 mb-1">Best Strategy</p>
            <p className="text-lg font-bold text-purple-600 truncate">
              {strategyData.sort((a, b) => parseFloat(b.winRate) - parseFloat(a.winRate))[0]?.name || 'N/A'}
            </p>
          </div>
          <div className="bg-white rounded-xl border border-slate-200 p-4">
            <p className="text-xs text-slate-500 mb-1">Worst Strategy</p>
            <p className="text-lg font-bold text-red-600 truncate">
              {strategyData.sort((a, b) => parseFloat(a.winRate) - parseFloat(b.winRate))[0]?.name || 'N/A'}
            </p>
          </div>
        </div>

        {/* Charts */}
        <div className="grid grid-cols-2 gap-6">
          {/* Success Distribution */}
          <div className="bg-white rounded-xl border border-slate-200 p-4">
            <h3 className="text-sm font-bold text-slate-900 mb-4">Success Distribution</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={successData} cx="50%" cy="50%" innerRadius={60} outerRadius={80} paddingAngle={5} dataKey="value">
                    {successData.map((entry, i) => (
                      <Cell key={`cell-${i}`} fill={entry.fill} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Strategy Win Rate */}
          <div className="bg-white rounded-xl border border-slate-200 p-4">
            <h3 className="text-sm font-bold text-slate-900 mb-4">Strategy Win Rate</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={strategyData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, 100]} />
                  <YAxis dataKey="name" type="category" width={100} tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Bar dataKey="winRate" fill="#8b5cf6" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Attempts to Success */}
          <div className="bg-white rounded-xl border border-slate-200 p-4">
            <h3 className="text-sm font-bold text-slate-900 mb-4">Attempts to Success</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={attemptsData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="attempts" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#3b82f6" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Per-run success rate */}
          <div className="bg-white rounded-xl border border-slate-200 p-4">
            <h3 className="text-sm font-bold text-slate-900 mb-4">Per-Run Results</h3>
            <div className="h-64 overflow-y-auto">
              <div className="space-y-1">
                {benchmarkRuns.map((run) => (
                  <button
                    key={run.experiment.run_id}
                    onClick={() => navigate(`/run/${run.experiment.run_id}`)}
                    className="w-full flex items-center justify-between text-sm py-1.5 px-2 hover:bg-slate-50 rounded"
                  >
                    <span className="font-mono text-xs text-slate-600 truncate">{run.experiment.run_id}</span>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-slate-400">{run.result.total_attempts} attempts</span>
                      <span className={`w-2 h-2 rounded-full ${run.result.ground_truth_success ? 'bg-green-500' : 'bg-red-500'}`} />
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
