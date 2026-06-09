import { Attempt } from '../types/autored';

export default function GeneratorCard({ attempt }: { attempt: Attempt }) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
          <span className="text-lg">🧠</span> Generator
        </h3>
        <span className="px-2 py-0.5 bg-purple-100 text-purple-700 text-xs font-medium rounded-full">
          {attempt.generator.strategy}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3 mb-3">
        <div className="bg-slate-50 rounded-lg p-2">
          <p className="text-xs text-slate-500">Input Tokens</p>
          <p className="text-lg font-bold text-slate-900">{attempt.generator.input_tokens}</p>
        </div>
        <div className="bg-slate-50 rounded-lg p-2">
          <p className="text-xs text-slate-500">Output Tokens</p>
          <p className="text-lg font-bold text-slate-900">{attempt.generator.output_tokens}</p>
        </div>
      </div>

      <div>
        <p className="text-xs text-slate-500 mb-1.5">Generated Attack</p>
        <div className="bg-red-50 border border-red-200 rounded-lg p-3">
          <p className="text-sm text-red-900 font-mono leading-relaxed">
            "{attempt.generator.generated_attack}"
          </p>
        </div>
      </div>

      {attempt.generator.duplicate_attack && (
        <div className="mt-2 px-2 py-1 bg-amber-50 border border-amber-200 rounded text-xs text-amber-700 font-medium">
          ⚠️ Duplicate attack detected
        </div>
      )}
    </div>
  );
}
