import { useState, useCallback, useEffect } from 'react';
import useLayoutStore from '../store/layoutStore';

/**
 * ExportToolbar — top toolbar with export actions, scale selector,
 * undo/redo, and view mode toggle.
 */
export default function ExportToolbar() {
    const [scale, setScale] = useState('1:100');
    const [exporting, setExporting] = useState(null);

    const layout = useLayoutStore((s) => s.layout);
    const { undo, redo, canUndo, canRedo } = useLayoutStore();

    const hasLayout = !!layout;

    // Export DXF
    const handleExportDXF = useCallback(async () => {
        if (!layout) return;
        setExporting('dxf');

        try {
            const res = await fetch('/api/v1/export/dxf', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ layout_graph: layout, scale }),
            });

            if (!res.ok) throw new Error(`HTTP ${res.status}`);

            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'floorplan.dxf';
            a.click();
            URL.revokeObjectURL(url);
        } catch (err) {
            console.error('DXF export failed:', err);
        } finally {
            setExporting(null);
        }
    }, [layout, scale]);

    // Export SVG
    const handleExportSVG = useCallback(async () => {
        if (!layout) return;
        setExporting('svg');

        try {
            const res = await fetch('/api/v1/export/svg', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ layout_graph: layout, scale }),
            });

            if (!res.ok) throw new Error(`HTTP ${res.status}`);

            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'floorplan.svg';
            a.click();
            URL.revokeObjectURL(url);
        } catch (err) {
            console.error('SVG export failed:', err);
        } finally {
            setExporting(null);
        }
    }, [layout, scale]);

    // Keyboard shortcuts
    useEffect(() => {
        const handleKeyDown = (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
                e.preventDefault();
                if (e.shiftKey) {
                    redo();
                } else {
                    undo();
                }
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [undo, redo]);

    return (
        <div className="flex items-center justify-between px-4 py-2 bg-surface-800 border-b border-surface-700">
            {/* Left: Logo + Title */}
            <div className="flex items-center gap-3">
                <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-blueprint-500 to-blueprint-600 flex items-center justify-center">
                    <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
                    </svg>
                </div>
                <h1 className="text-sm font-bold text-surface-100 tracking-tight">
                    Prompt-to-Blueprint
                </h1>
            </div>

            {/* Center: Undo/Redo + Scale */}
            <div className="flex items-center gap-2">
                <button
                    onClick={undo}
                    disabled={!canUndo()}
                    className={`p-1.5 rounded transition-colors ${canUndo()
                            ? 'text-surface-300 hover:bg-surface-700'
                            : 'text-surface-600 cursor-not-allowed'
                        }`}
                    title="Undo (Ctrl+Z)"
                >
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h10a8 8 0 018 8v2M3 10l6 6m-6-6l6-6" />
                    </svg>
                </button>
                <button
                    onClick={redo}
                    disabled={!canRedo()}
                    className={`p-1.5 rounded transition-colors ${canRedo()
                            ? 'text-surface-300 hover:bg-surface-700'
                            : 'text-surface-600 cursor-not-allowed'
                        }`}
                    title="Redo (Ctrl+Shift+Z)"
                >
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 10H11a8 8 0 00-8 8v2m18-10l-6 6m6-6l-6-6" />
                    </svg>
                </button>

                <div className="w-px h-5 bg-surface-600 mx-1" />

                <select
                    id="scale-selector"
                    value={scale}
                    onChange={(e) => setScale(e.target.value)}
                    className="bg-surface-700 border border-surface-600 rounded px-2 py-1 text-xs text-surface-200 focus:outline-none focus:ring-1 focus:ring-blueprint-500"
                >
                    <option value="1:50">1:50</option>
                    <option value="1:100">1:100</option>
                    <option value="1:200">1:200</option>
                </select>
            </div>

            {/* Right: Export buttons */}
            <div className="flex items-center gap-2">
                <button
                    id="export-svg-btn"
                    onClick={handleExportSVG}
                    disabled={!hasLayout || exporting === 'svg'}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${hasLayout
                            ? 'bg-surface-700 text-surface-200 hover:bg-surface-600 border border-surface-600'
                            : 'bg-surface-800 text-surface-600 cursor-not-allowed border border-surface-700'
                        }`}
                >
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                    </svg>
                    {exporting === 'svg' ? 'Saving...' : 'SVG'}
                </button>

                <button
                    id="export-dxf-btn"
                    onClick={handleExportDXF}
                    disabled={!hasLayout || exporting === 'dxf'}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${hasLayout
                            ? 'bg-blueprint-600 text-white hover:bg-blueprint-500 shadow-sm'
                            : 'bg-surface-800 text-surface-600 cursor-not-allowed border border-surface-700'
                        }`}
                >
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                    </svg>
                    {exporting === 'dxf' ? 'Saving...' : 'DXF'}
                </button>
            </div>
        </div>
    );
}
