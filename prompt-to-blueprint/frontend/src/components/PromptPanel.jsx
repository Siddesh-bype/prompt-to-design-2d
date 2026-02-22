import { useState, useCallback } from 'react';
import useLayoutStore from '../store/layoutStore';
import useJobStream from '../hooks/useJobStream';

/**
 * PromptPanel — left sidebar with prompt input, generation controls,
 * and real-time progress display.
 */
export default function PromptPanel() {
    const [prompt, setPrompt] = useState('');
    const [plotSqm, setPlotSqm] = useState(100);
    const [facing, setFacing] = useState('NORTH');
    const [vastuEnabled, setVastuEnabled] = useState(false);

    const { setLayout, setSvg, setVastu } = useLayoutStore();
    const { status, progress, connect } = useJobStream();

    const isGenerating = ['queued', 'parsing', 'layout_generating', 'solving_constraints', 'rendering'].includes(status);

    const handleGenerate = useCallback(async () => {
        if (!prompt.trim() || prompt.trim().length < 10) return;

        try {
            const res = await fetch('/api/v1/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: prompt.trim(),
                    plot_sqm: plotSqm,
                    facing,
                    vastu_enabled: vastuEnabled,
                }),
            });

            if (!res.ok) throw new Error(`HTTP ${res.status}`);

            const data = await res.json();
            connect(data.job_id, (update) => {
                if (update.status === 'complete') {
                    if (update.result) setLayout(update.result);
                    if (update.svg_string) setSvg(update.svg_string);
                    if (update.result?.vastu) setVastu(update.result.vastu);
                }
            });
        } catch (err) {
            console.error('Generation failed:', err);
        }
    }, [prompt, plotSqm, facing, vastuEnabled, connect, setLayout, setSvg, setVastu]);

    const stageLabel = {
        queued: 'Queued...',
        parsing: 'Parsing prompt...',
        layout_generating: 'Generating layout...',
        solving_constraints: 'Solving constraints...',
        rendering: 'Rendering SVG...',
        complete: 'Complete!',
        error: 'Error',
    }[status] || '';

    // Quick presets
    const presets = [
        { label: '2 BHK', text: '2BHK apartment with open kitchen, 2 bathrooms, south facing, modern style' },
        { label: '3 BHK', text: '3BHK flat with master bedroom attached bathroom, modular kitchen, study room, north facing' },
        { label: 'Vastu', text: '3BHK vastu-compliant house, pooja room in NE, kitchen in SE, master bedroom in SW, 1200 sqft' },
    ];

    return (
        <div className="flex flex-col h-full bg-surface-800 border-r border-surface-700">
            {/* Header */}
            <div className="p-4 border-b border-surface-700">
                <h2 className="text-sm font-semibold text-surface-200 uppercase tracking-wider">
                    Prompt
                </h2>
            </div>

            {/* Prompt input */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
                <div>
                    <label className="block text-xs font-medium text-surface-400 mb-1.5">
                        Describe your floor plan
                    </label>
                    <textarea
                        id="prompt-input"
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        placeholder="e.g. 3BHK apartment, north facing, with attached bathrooms and open kitchen..."
                        className="w-full h-32 bg-surface-900 border border-surface-600 rounded-lg px-3 py-2.5 text-sm text-surface-100 placeholder:text-surface-500 focus:outline-none focus:ring-2 focus:ring-blueprint-500 focus:border-transparent resize-none transition-all"
                        disabled={isGenerating}
                    />
                    <div className="flex justify-between mt-1">
                        <span className="text-xs text-surface-500">{prompt.length}/800</span>
                        {prompt.length > 0 && prompt.length < 10 && (
                            <span className="text-xs text-red-400">Min 10 characters</span>
                        )}
                    </div>
                </div>

                {/* Quick presets */}
                <div>
                    <label className="block text-xs font-medium text-surface-400 mb-1.5">Quick Presets</label>
                    <div className="flex flex-wrap gap-1.5">
                        {presets.map((p) => (
                            <button
                                key={p.label}
                                onClick={() => setPrompt(p.text)}
                                className="px-2.5 py-1 text-xs rounded-md bg-surface-700 text-surface-300 hover:bg-blueprint-600 hover:text-white transition-colors"
                                disabled={isGenerating}
                            >
                                {p.label}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Controls */}
                <div className="space-y-3">
                    <div>
                        <label className="block text-xs font-medium text-surface-400 mb-1.5">
                            Plot Area (m²)
                        </label>
                        <input
                            id="plot-area-input"
                            type="range"
                            min={30}
                            max={500}
                            value={plotSqm}
                            onChange={(e) => setPlotSqm(Number(e.target.value))}
                            className="w-full accent-blueprint-500"
                            disabled={isGenerating}
                        />
                        <div className="flex justify-between text-xs text-surface-500 mt-0.5">
                            <span>30</span>
                            <span className="text-blueprint-400 font-medium">{plotSqm} m²</span>
                            <span>500</span>
                        </div>
                    </div>

                    <div>
                        <label className="block text-xs font-medium text-surface-400 mb-1.5">
                            Facing
                        </label>
                        <div className="grid grid-cols-4 gap-1.5">
                            {['NORTH', 'SOUTH', 'EAST', 'WEST'].map((dir) => (
                                <button
                                    key={dir}
                                    onClick={() => setFacing(dir)}
                                    className={`px-2 py-1.5 text-xs rounded-md border transition-all ${facing === dir
                                            ? 'bg-blueprint-600 border-blueprint-500 text-white'
                                            : 'bg-surface-700 border-surface-600 text-surface-300 hover:border-surface-500'
                                        }`}
                                    disabled={isGenerating}
                                >
                                    {dir[0]}{dir.slice(1).toLowerCase()}
                                </button>
                            ))}
                        </div>
                    </div>

                    <div className="flex items-center justify-between">
                        <label className="text-xs font-medium text-surface-400">
                            Vastu Compliance
                        </label>
                        <button
                            id="vastu-toggle"
                            onClick={() => setVastuEnabled(!vastuEnabled)}
                            className={`relative w-10 h-5 rounded-full transition-colors ${vastuEnabled ? 'bg-blueprint-500' : 'bg-surface-600'
                                }`}
                            disabled={isGenerating}
                        >
                            <div
                                className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${vastuEnabled ? 'translate-x-5' : 'translate-x-0'
                                    }`}
                            />
                        </button>
                    </div>
                </div>
            </div>

            {/* Generate button + progress */}
            <div className="p-4 border-t border-surface-700 space-y-3">
                {isGenerating && (
                    <div>
                        <div className="flex justify-between text-xs text-surface-400 mb-1">
                            <span>{stageLabel}</span>
                            <span>{progress}%</span>
                        </div>
                        <div className="w-full h-1.5 bg-surface-700 rounded-full overflow-hidden">
                            <div
                                className="h-full bg-gradient-to-r from-blueprint-500 to-blueprint-400 rounded-full transition-all duration-500"
                                style={{ width: `${progress}%` }}
                            />
                        </div>
                    </div>
                )}

                <button
                    id="generate-btn"
                    onClick={handleGenerate}
                    disabled={isGenerating || prompt.trim().length < 10}
                    className={`w-full py-2.5 rounded-lg font-medium text-sm transition-all ${isGenerating || prompt.trim().length < 10
                            ? 'bg-surface-600 text-surface-400 cursor-not-allowed'
                            : 'bg-gradient-to-r from-blueprint-600 to-blueprint-500 text-white hover:from-blueprint-500 hover:to-blueprint-400 shadow-lg shadow-blueprint-500/25 active:scale-[0.98]'
                        }`}
                >
                    {isGenerating ? 'Generating...' : '⚡ Generate Floor Plan'}
                </button>
            </div>
        </div>
    );
}
