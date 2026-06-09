import { create } from 'zustand';
import type { AutoRedRun, Attempt, RunListItem } from '../types/autored';

interface RunStore {
  runs: RunListItem[];
  selectedRun: AutoRedRun | null;
  selectedAttemptIndex: number;

  setRuns: (runs: RunListItem[]) => void;
  setSelectedRun: (run: AutoRedRun | null) => void;
  setSelectedAttempt: (index: number) => void;
  addAttempt: (attempt: Attempt) => void;
  clearSelectedRun: () => void;
}

export const useRunStore = create<RunStore>((set) => ({
  runs: [],
  selectedRun: null,
  selectedAttemptIndex: 0,

  setRuns: (runs) => set({ runs }),
  setSelectedRun: (run) => set({ selectedRun: run, selectedAttemptIndex: run ? run.attempts.length - 1 : 0 }),
  setSelectedAttempt: (index) => set({ selectedAttemptIndex: index }),
  addAttempt: (attempt) => set((state) => {
    if (!state.selectedRun) return state;
    const newAttempts = [...state.selectedRun.attempts, attempt];
    return {
      selectedRun: { ...state.selectedRun, attempts: newAttempts },
      selectedAttemptIndex: newAttempts.length - 1,
    };
  }),
  clearSelectedRun: () => set({ selectedRun: null, selectedAttemptIndex: 0 }),
}));
