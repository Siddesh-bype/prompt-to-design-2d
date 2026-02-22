import { useRef, useEffect, useState, useCallback } from 'react';
import useLayoutStore from '../store/layoutStore';

/**
 * CanvasEditor — central panel displaying the generated floor plan SVG.
 * 
 * Features:
 * - Inline SVG rendering via dangerouslySetInnerHTML
 * - Zoom (scroll wheel) and pan (drag)
 * - Room hover highlighting
 * - Grid background
 */
export default function CanvasEditor() {
    const containerRef = useRef(null);
    const svgRef = useRef(null);

    const svgString = useLayoutStore((s) => s.svgString);
    const layout = useLayoutStore((s) => s.layout);

    const [zoom, setZoom] = useState(1);
    const [pan, setPan] = useState({ x: 0, y: 0 });
    const [isPanning, setIsPanning] = useState(false);
    const [panStart, setPanStart] = useState({ x: 0, y: 0 });
    const [hoveredRoom, setHoveredRoom] = useState(null);

    // Zoom with scroll wheel
    const handleWheel = useCallback((e) => {
        e.preventDefault();
        const delta = e.deltaY > 0 ? 0.9 : 1.1;
        setZoom((z) => Math.min(Math.max(z * delta, 0.3), 5));
    }, []);

    // Pan with mouse drag
    const handleMouseDown = useCallback((e) => {
        if (e.button === 0) {
            setIsPanning(true);
            setPanStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
        }
    }, [pan]);

    const handleMouseMove = useCallback((e) => {
        if (isPanning) {
            setPan({
                x: e.clientX - panStart.x,
                y: e.clientY - panStart.y,
            });
        }

        // Room hover detection
        const target = e.target;
        if (target && target.getAttribute) {
            const roomId = target.getAttribute('data-room-id');
            setHoveredRoom(roomId);
        } else {
            setHoveredRoom(null);
        }
    }, [isPanning, panStart]);

    const handleMouseUp = useCallback(() => {
        setIsPanning(false);
    }, []);

    // Reset view
    const resetView = useCallback(() => {
        setZoom(1);
        setPan({ x: 0, y: 0 });
    }, []);

    // Attach wheel listener (passive: false for preventDefault)
    useEffect(() => {
        const el = containerRef.current;
        if (el) {
            el.addEventListener('wheel', handleWheel, { passive: false });
            return () => el.removeEventListener('wheel', handleWheel);
        }
    }, [handleWheel]);

    return (
        <div
            ref={containerRef}
            className="relative flex-1 h-full overflow-hidden bg-surface-900 select-none"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            style={{ cursor: isPanning ? 'grabbing' : 'grab' }}
        >
            {/* Grid background */}
            <div
                className="absolute inset-0 opacity-[0.05]"
                style={{
                    backgroundImage: `
            linear-gradient(rgba(100,150,255,0.3) 1px, transparent 1px),
            linear-gradient(90deg, rgba(100,150,255,0.3) 1px, transparent 1px)
          `,
                    backgroundSize: `${20 * zoom}px ${20 * zoom}px`,
                    backgroundPosition: `${pan.x}px ${pan.y}px`,
                }}
            />

            {/* SVG container */}
            <div
                className="absolute inset-0 flex items-center justify-center"
                style={{
                    transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                    transformOrigin: 'center center',
                }}
            >
                {svgString ? (
                    <div
                        ref={svgRef}
                        className="transition-opacity duration-300"
                        dangerouslySetInnerHTML={{ __html: svgString }}
                    />
                ) : (
                    <div className="flex flex-col items-center justify-center text-surface-500 space-y-4">
                        <div className="w-24 h-24 rounded-2xl bg-surface-800 border-2 border-dashed border-surface-600 flex items-center justify-center">
                            <svg className="w-10 h-10 text-surface-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                            </svg>
                        </div>
                        <div className="text-center">
                            <p className="text-sm font-medium text-surface-400">No floor plan generated yet</p>
                            <p className="text-xs text-surface-600 mt-1">Write a prompt and click Generate</p>
                        </div>
                    </div>
                )}
            </div>

            {/* Zoom controls */}
            <div className="absolute bottom-4 right-4 flex items-center gap-1.5 bg-surface-800/90 backdrop-blur-sm border border-surface-700 rounded-lg p-1">
                <button
                    onClick={() => setZoom((z) => Math.min(z * 1.2, 5))}
                    className="w-7 h-7 flex items-center justify-center rounded text-surface-300 hover:bg-surface-700 transition-colors text-sm"
                    title="Zoom in"
                >
                    +
                </button>
                <span className="text-xs text-surface-400 w-12 text-center font-mono">
                    {Math.round(zoom * 100)}%
                </span>
                <button
                    onClick={() => setZoom((z) => Math.max(z * 0.8, 0.3))}
                    className="w-7 h-7 flex items-center justify-center rounded text-surface-300 hover:bg-surface-700 transition-colors text-sm"
                    title="Zoom out"
                >
                    −
                </button>
                <div className="w-px h-4 bg-surface-600" />
                <button
                    onClick={resetView}
                    className="w-7 h-7 flex items-center justify-center rounded text-surface-300 hover:bg-surface-700 transition-colors text-xs"
                    title="Reset view"
                >
                    ⟲
                </button>
            </div>

            {/* Hovered room tooltip */}
            {hoveredRoom && layout && (
                <div className="absolute top-4 left-4 bg-surface-800/90 backdrop-blur-sm border border-surface-700 rounded-lg px-3 py-2">
                    <p className="text-xs text-blueprint-400 font-medium">{hoveredRoom}</p>
                </div>
            )}
        </div>
    );
}
