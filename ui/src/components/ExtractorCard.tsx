import { Attempt } from '../types/autored';

export default function ExtractorCard({ attempt }: { attempt: Attempt }) {
  const { extractor } = attempt;
  const isWrong = extractor.best_candidate && attempt.ground_truth_found && !attempt.extractor_match;

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
          <span className="text-lg">🔓</span> Extractor
        </h3>
        <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${attempt.extractor_match ? 'bg-green-100 text-green-700' : 'bg-slate-100 text-slate-600'}`}>
          Match: {attempt.extractor_match ? '✓ YES' : '✗ NO'}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3 mb-3">
        <div>
          <p className="text-xs text-slate-500 mb-1">Regex Candidates ({extractor.regex_candidates.length})</p>
          <div className="flex flex-wrap gap-1">
            {extractor.regex_candidates.map((c: string, i: number) => (
              <span key={i} className="px-2 py-0.5 bg-blue-50 text-blue-700 text-xs rounded font-mono">{c}</span>
            ))}
            {extractor.regex_candidates.length === 0 && <span className="text-xs text-slate-400">none</span>}
          </div>
        </div>
        <div>
          <p className="text-xs text-slate-500 mb-1">LLM Candidates ({extractor.llm_candidates.length})</p>
          <div className="flex flex-wrap gap-1">
            {extractor.llm_candidates.map((c: string, i: number) => (
              <span key={i} className="px-2 py-0.5 bg-indigo-50 text-indigo-700 text-xs rounded font-mono">{c}</span>
            ))}
            {extractor.llm_candidates.length === 0 && <span className="text-xs text-slate-400">none</span>}
          </div>
        </div>
      </div>

      <div>
        <p className="text-xs text-slate-500 mb-1.5">Ranked Candidates</p>
        <div className="space-y-1">
          {extractor.ranked_candidates.map((rc: { value: string; score: number }, i: number) => (
            <div key={i} className="flex items-center justify-between text-sm bg-slate-50 rounded px-3 py-1.5">
              <span className="font-mono text-slate-700">{rc.value}</span>
              <span className="text-xs text-slate-500">score: {rc.score}</span>
            </div>
          ))}
        </div>
      </div>

      <div className={`mt-3 p-3 rounded-lg border ${isWrong ? 'bg-red-50 border-red-200' : 'bg-slate-50 border-slate-200'}`}>
        <p className="text-xs text-slate-500 mb-1">Selected Candidate</p>
        <div className="flex items-center justify-between">
          <span className="font-mono font-bold text-slate-900">{extractor.best_candidate || 'NONE'}</span>
          {isWrong && <span className="text-red-600 text-sm font-bold">❌ Wrong Selection</span>}
          {attempt.extractor_match && <span className="text-green-600 text-sm font-bold">✓ Correct</span>}
        </div>
      </div>
    </div>
  );
}
