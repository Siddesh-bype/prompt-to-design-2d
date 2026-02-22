import { create } from 'zustand';

/**
 * Zustand store for layout state management.
 *
 * Features:
 * - Layout graph state (from backend)
 * - SVG string state (for canvas rendering)
 * - Vastu compliance data
 * - Undo/redo with history stack (max 20)
 */

const MAX_HISTORY = 20;

const useLayoutStore = create((set, get) => ({
    // ─── State ────────────────────────────────────────────────
    layout: null,
    svgString: null,
    vastu: null,

    // History
    history: [],
    historyIndex: -1,

    // ─── Actions ──────────────────────────────────────────────
    setLayout: (layout) => {
        const state = get();
        const newHistory = [
            ...state.history.slice(0, state.historyIndex + 1),
            { layout, svgString: state.svgString, vastu: state.vastu },
        ].slice(-MAX_HISTORY);

        set({
            layout,
            history: newHistory,
            historyIndex: newHistory.length - 1,
        });
    },

    setSvg: (svgString) => set({ svgString }),

    setVastu: (vastu) => set({ vastu }),

    clearLayout: () =>
        set({
            layout: null,
            svgString: null,
            vastu: null,
            history: [],
            historyIndex: -1,
        }),

    // ─── Undo / Redo ─────────────────────────────────────────
    undo: () => {
        const { historyIndex, history } = get();
        if (historyIndex > 0) {
            const prev = history[historyIndex - 1];
            set({
                layout: prev.layout,
                svgString: prev.svgString,
                vastu: prev.vastu,
                historyIndex: historyIndex - 1,
            });
        }
    },

    redo: () => {
        const { historyIndex, history } = get();
        if (historyIndex < history.length - 1) {
            const next = history[historyIndex + 1];
            set({
                layout: next.layout,
                svgString: next.svgString,
                vastu: next.vastu,
                historyIndex: historyIndex + 1,
            });
        }
    },

    canUndo: () => get().historyIndex > 0,
    canRedo: () => get().historyIndex < get().history.length - 1,
}));

export default useLayoutStore;
