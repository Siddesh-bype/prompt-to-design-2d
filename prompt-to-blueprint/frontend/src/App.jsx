import { useState } from 'react';
import PromptPanel from './components/PromptPanel';
import CanvasEditor from './components/CanvasEditor';
import VastuPanel from './components/VastuPanel';
import ExportToolbar from './components/ExportToolbar';

/**
 * App — Root application component.
 * 
 * Layout: 3-panel design
 * ┌──────────────┬─────────────────────────┬──────────────┐
 * │              │      ExportToolbar       │              │
 * │  PromptPanel ├─────────────────────────┤  VastuPanel  │
 * │   (320px)    │     CanvasEditor         │   (280px)    │
 * │              │     (flex-1)             │              │
 * └──────────────┴─────────────────────────┴──────────────┘
 */
export default function App() {
    const [showVastu, setShowVastu] = useState(true);

    return (
        <div className="h-screen w-screen flex bg-surface-900 text-surface-100 overflow-hidden">
            {/* Left Panel — Prompt */}
            <aside className="w-80 shrink-0 flex flex-col">
                <PromptPanel />
            </aside>

            {/* Center — Toolbar + Canvas */}
            <main className="flex-1 flex flex-col min-w-0">
                <ExportToolbar />
                <CanvasEditor />
            </main>

            {/* Right Panel — Vastu (collapsible) */}
            {showVastu && (
                <aside className="w-72 shrink-0 flex flex-col">
                    <VastuPanel />
                </aside>
            )}

            {/* Vastu toggle button */}
            <button
                onClick={() => setShowVastu(!showVastu)}
                className="fixed bottom-4 left-1/2 -translate-x-1/2 z-50 px-3 py-1.5 bg-surface-700/80 backdrop-blur-sm border border-surface-600 rounded-full text-xs text-surface-300 hover:bg-surface-600 transition-colors"
                title={showVastu ? 'Hide Vastu panel' : 'Show Vastu panel'}
            >
                {showVastu ? 'Hide Vastu ✕' : 'Show Vastu ◈'}
            </button>
        </div>
    );
}
