import React, { useRef, useEffect, useState, useCallback } from 'react';
import useLayoutStore from '../store/layoutStore';
import { Stage, Layer, Rect, Transformer, Text as KonvaText } from 'react-konva';

// Render mapping matches backend renderer.py
const CANVAS_SIZE = 800;
const PAD = 50;
const DRAW_W = CANVAS_SIZE - 2 * PAD;
const DRAW_H = CANVAS_SIZE - 2 * PAD;

/**
 * InteractiveRoom — A single Konva Rect that can be dragged and resized.
 */
const InteractiveRoom = ({ room, isSelected, onSelect, onChange, otherRooms }) => {
    const shapeRef = useRef();
    const trRef = useRef();
    const [liveBox, setLiveBox] = useState(room.bbox);
    const [isOverlapping, setIsOverlapping] = useState(false);

    // Map [0,1] bbox to [0,DRAW] canvas coords
    const x = PAD + liveBox.x_min * DRAW_W;
    const y = PAD + liveBox.y_min * DRAW_H;
    const width = (liveBox.x_max - liveBox.x_min) * DRAW_W;
    const height = (liveBox.y_max - liveBox.y_min) * DRAW_H;

    // Dimensions formatter (assume 10m roughly)
    const w_m = (liveBox.x_max - liveBox.x_min) * 10;
    const h_m = (liveBox.y_max - liveBox.y_min) * 10;
    const formatDisplay = (w, h) => `${w.toFixed(1)}m × ${h.toFixed(1)}m`;

    useEffect(() => {
        if (isSelected && trRef.current) {
            trRef.current.nodes([shapeRef.current]);
            trRef.current.getLayer().batchDraw();
        }
    }, [isSelected]);

    // Update liveBox when room prop changes (e.g. from backend regenerations)
    useEffect(() => {
        setLiveBox(room.bbox);
    }, [room.bbox]);

    const checkOverlap = (box, others) => {
        for (const other of others) {
            const ob = other.bbox;
            // AABB Overlap check with small epsilon
            const eps = 0.02;
            if (
                box.x_min < ob.x_max - eps &&
                box.x_max > ob.x_min + eps &&
                box.y_min < ob.y_max - eps &&
                box.y_max > ob.y_min + eps
            ) {
                return true;
            }
        }
        return false;
    };

    const handleUpdate = (node, isFinal = false) => {
        // Apply 10px snap grid (approx 0.15m)
        const snap = 10;
        const snappedX = Math.round(node.x() / snap) * snap;
        const snappedY = Math.round(node.y() / snap) * snap;
        const snappedW = Math.round(node.width() * node.scaleX() / snap) * snap;
        const snappedH = Math.round(node.height() * node.scaleY() / snap) * snap;

        // Don't apply snapping to the node itself during drag to avoid stutter,
        // but snap the model coordinates.
        const newXMin = Math.max(0, (snappedX - PAD) / DRAW_W);
        const newYMin = Math.max(0, (snappedY - PAD) / DRAW_H);
        const newXMax = Math.min(1, newXMin + snappedW / DRAW_W);
        const newYMax = Math.min(1, newYMin + snappedH / DRAW_H);

        const newBox = { x_min: newXMin, y_min: newYMin, x_max: newXMax, y_max: newYMax, area: (newXMax - newXMin) * (newYMax - newYMin) };

        setLiveBox(newBox);
        const overlap = checkOverlap(newBox, otherRooms);
        setIsOverlapping(overlap);

        if (isFinal) {
            // Reset node to model position to enforce snap
            node.setAttrs({
                x: PAD + newXMin * DRAW_W,
                y: PAD + newYMin * DRAW_H,
                width: (newXMax - newXMin) * DRAW_W,
                height: (newYMax - newYMin) * DRAW_H,
                scaleX: 1,
                scaleY: 1
            });
            onChange({ ...room, bbox: newBox });
        }
    };

    const label = room.room_spec?.label || room.room_spec?.room_type?.replace(/_/g, ' ') || 'Room';

    // Styling based on state
    const baseColor = isOverlapping ? '239, 68, 68' : '59, 130, 246'; // Red if overlap, Blue if OK

    return (
        <React.Fragment>
            <Rect
                ref={shapeRef}
                x={x}
                y={y}
                width={width}
                height={height}
                fill={`rgba(${baseColor}, ${isOverlapping ? 0.6 : 0.4})`}
                stroke={`rgba(${baseColor}, 0.8)`}
                strokeWidth={isOverlapping ? 3 : 2}
                draggable
                onClick={onSelect}
                onTap={onSelect}
                onDragStart={onSelect}
                onDragMove={(e) => handleUpdate(shapeRef.current, false)}
                onDragEnd={(e) => handleUpdate(shapeRef.current, true)}
                onTransform={(e) => handleUpdate(shapeRef.current, false)}
                onTransformEnd={(e) => handleUpdate(shapeRef.current, true)}
            />
            {/* Overlay label and live dimensions */}
            {(isSelected || isOverlapping) && (
                <KonvaText
                    x={x + 5}
                    y={y + 5}
                    text={`${label}\n${formatDisplay(w_m, h_m)}${isOverlapping ? '\n❌ OVERLAP' : ''}`}
                    fontSize={11}
                    lineHeight={1.4}
                    fill={isOverlapping ? "#7F1D1D" : "#1E293B"} // Dark red if overlapping
                    fontStyle="bold"
                    listening={false}
                />
            )}
            {isSelected && (
                <Transformer
                    ref={trRef}
                    boundBoxFunc={(oldBox, newBox) => {
                        if (newBox.width < 30 || newBox.height < 30) return oldBox;
                        return newBox;
                    }}
                    rotateEnabled={false}
                    borderStroke={`rgba(${baseColor}, 1)`}
                    anchorStroke={`rgba(${baseColor}, 1)`}
                    anchorFill="#FFFFFF"
                    anchorSize={8}
                />
            )}
        </React.Fragment>
    );
};

/**
 * CanvasEditor — central panel displaying the generated floor plan SVG.
 * 
 * Features:
 * - Inline SVG rendering
 * - Zoom (scroll wheel) and pan (drag)
 * - Interactive Edit Mode using Konva for drag/resize
 */
export default function CanvasEditor() {
    const containerRef = useRef(null);
    const svgRef = useRef(null);

    const { layout, svgString, setLayout, setSvg } = useLayoutStore();

    const [zoom, setZoom] = useState(1);
    const [pan, setPan] = useState({ x: 0, y: 0 });
    const [isPanning, setIsPanning] = useState(false);
    const [panStart, setPanStart] = useState({ x: 0, y: 0 });
    const [hoveredRoom, setHoveredRoom] = useState(null);

    // Edit mode state
    const [isEditMode, setIsEditMode] = useState(false);
    const [selectedRoomId, setSelectedRoomId] = useState(null);
    const [isRegenerating, setIsRegenerating] = useState(false);

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
            const roomType = target.getAttribute('data-room-type');
            if (roomId && roomType) {
                setHoveredRoom({ id: roomId, type: roomType });
            } else {
                setHoveredRoom(null);
            }
        } else {
            setHoveredRoom(null);
        }
    }, [isPanning, panStart]);

    const handleMouseUp = useCallback(() => {
        setIsPanning(false);
    }, []);

    // Deselect background click in Konva
    const checkDeselect = (e) => {
        // clicked on empty area
        const clickedOnEmpty = e.target === e.target.getStage();
        if (clickedOnEmpty) {
            setSelectedRoomId(null);
        }
    };

    // Callback when a room is resized/moved
    const handleRoomChange = async (index, newRoomAttrs) => {
        if (!layout) return;
        const newRooms = layout.rooms.slice();
        newRooms[index] = newRoomAttrs;

        const newLayout = { ...layout, rooms: newRooms };
        setLayout(newLayout); // Optimistic UI update for Konva

        // Fetch regenerated SVG
        setIsRegenerating(true);
        try {
            const res = await fetch('/api/v1/export/svg', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    layout_graph: newLayout,
                    scale: '1:100' // Default scale
                })
            });
            if (res.ok) {
                const text = await res.text();
                setSvg(text);
            } else {
                console.error("Failed to regenerate SVG", await res.text());
            }
        } catch (err) {
            console.error("Failed to fetch new SVG", err);
        } finally {
            setIsRegenerating(false);
        }
    };

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
            >
                {/* Wrap both SVG and Konva in the same transforming container */}
                <div
                    style={{
                        transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                        transformOrigin: 'center center',
                        width: `${CANVAS_SIZE}px`,
                        height: `${CANVAS_SIZE}px`,
                        position: 'relative',
                    }}
                >
                    {svgString ? (
                        <div
                            ref={svgRef}
                            className={`transition-opacity duration-300 w-full h-full absolute inset-0 ${isEditMode ? 'opacity-50 pointer-events-none' : 'opacity-100'}`}
                            dangerouslySetInnerHTML={{ __html: svgString }}
                        />
                    ) : (
                        <div className="w-full h-full flex flex-col items-center justify-center text-surface-500 space-y-4">
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

                    {/* Konva Edit Overlay */}
                    {isEditMode && layout && svgString && (
                        <div className="absolute inset-0 z-10">
                            <Stage
                                width={CANVAS_SIZE}
                                height={CANVAS_SIZE}
                                onMouseDown={checkDeselect}
                                onTouchStart={checkDeselect}
                            >
                                <Layer>
                                    {layout.rooms.map((room, i) => (
                                        <InteractiveRoom
                                            key={room.room_spec?.room_id || i}
                                            room={room}
                                            isSelected={room.room_spec?.room_id === selectedRoomId}
                                            onSelect={() => setSelectedRoomId(room.room_spec?.room_id)}
                                            onChange={(newAttrs) => handleRoomChange(i, newAttrs)}
                                            otherRooms={layout.rooms.filter((_, idx) => idx !== i)}
                                        />
                                    ))}
                                </Layer>
                            </Stage>
                        </div>
                    )}
                </div>
            </div>

            {/* Edit Mode Toggle & Status */}
            {svgString && (
                <div className="absolute top-4 right-4 flex items-center gap-3">
                    {isRegenerating && (
                        <span className="text-xs text-surface-400 animate-pulse bg-surface-800/80 px-3 py-1.5 rounded-full border border-surface-700">
                            Regenerating blueprint...
                        </span>
                    )}
                    <button
                        onClick={() => {
                            setIsEditMode(!isEditMode);
                            setSelectedRoomId(null);
                        }}
                        className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-sm transition-all shadow-lg backdrop-blur-sm
                            ${isEditMode
                                ? 'bg-blueprint-500 hover:bg-blueprint-600 text-white shadow-blueprint-500/25 border-transparent'
                                : 'bg-surface-800/90 text-surface-300 hover:text-white border border-surface-700 hover:border-surface-500'}`}
                    >
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                        </svg>
                        {isEditMode ? 'Finish Editing' : 'Edit Layout'}
                    </button>
                </div>
            )}

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

            {/* Hovered room tooltip (Disabled in Edit mode) */}
            {hoveredRoom && layout && !isEditMode && (
                <div className="absolute top-4 left-4 bg-surface-800/90 backdrop-blur-sm border border-surface-700 rounded-lg px-3 py-2">
                    <p className="text-xs text-blueprint-400 font-medium">
                        {hoveredRoom.type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                    </p>
                    <p className="text-[10px] text-surface-500 mt-0.5">{hoveredRoom.id}</p>
                </div>
            )}
        </div>
    );
}
