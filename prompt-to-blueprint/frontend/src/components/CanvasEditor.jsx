import { useRef, useEffect, useState, useCallback } from 'react';
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
const InteractiveRoom = ({ room, isSelected, onSelect, onChange }) => {
    const shapeRef = useRef();
    const trRef = useRef();

    useEffect(() => {
        if (isSelected && trRef.current) {
            trRef.current.nodes([shapeRef.current]);
            trRef.current.getLayer().batchDraw();
        }
    }, [isSelected]);

    // Map [0,1] bbox to [0,800] canvas coords
    const x = PAD + room.bbox.x_min * DRAW_W;
    const y = PAD + room.bbox.y_min * DRAW_H;
    const width = (room.bbox.x_max - room.bbox.x_min) * DRAW_W;
    const height = (room.bbox.y_max - room.bbox.y_min) * DRAW_H;

    // Label
    const label = room.room_spec?.label || room.room_spec?.room_type?.replace(/_/g, ' ') || 'Room';

    return (
        <React.Fragment>
            <Rect
                ref={shapeRef}
                x={x}
                y={y}
                width={width}
                height={height}
                fill="rgba(59, 130, 246, 0.4)" // blueprint-500 with opacity
                stroke="rgba(37, 99, 235, 0.8)"
                strokeWidth={2}
                draggable
                onClick={onSelect}
                onTap={onSelect}
                onDragStart={onSelect}
                onDragEnd={(e) => {
                    const node = shapeRef.current;
                    const newXMin = (node.x() - PAD) / DRAW_W;
                    const newYMin = (node.y() - PAD) / DRAW_H;
                    const newXMax = newXMin + node.width() / DRAW_W;
                    const newYMax = newYMin + node.height() / DRAW_H;
                    onChange({
                        ...room,
                        bbox: { x_min: newXMin, y_min: newYMin, x_max: newXMax, y_max: newYMax, area: (newXMax - newXMin) * (newYMax - newYMin) }
                    });
                }}
                onTransformEnd={(e) => {
                    const node = shapeRef.current;
                    const scaleX = node.scaleX();
                    const scaleY = node.scaleY();

                    // Reset scale to 1 after transform to keep border clean, update width/height
                    node.scaleX(1);
                    node.scaleY(1);

                    const newWidth = Math.max(5, node.width() * scaleX);
                    const newHeight = Math.max(5, node.height() * scaleY);

                    const newXMin = (node.x() - PAD) / DRAW_W;
                    const newYMin = (node.y() - PAD) / DRAW_H;
                    const newXMax = newXMin + newWidth / DRAW_W;
                    const newYMax = newYMin + newHeight / DRAW_H;

                    onChange({
                        ...room,
                        bbox: { x_min: newXMin, y_min: newYMin, x_max: newXMax, y_max: newYMax, area: (newXMax - newXMin) * (newYMax - newYMin) }
                    });
                }}
            />
            {/* Dimensions overlay label */}
            {isSelected && (
                <KonvaText
                    x={x + 5}
                    y={y + 5}
                    text={label}
                    fontSize={12}
                    fill="#1E293B"
                    fontStyle="bold"
                    listening={false}
                />
            )}
            {isSelected && (
                <Transformer
                    ref={trRef}
                    boundBoxFunc={(oldBox, newBox) => {
                        // Limit minimum size
                        if (newBox.width < 20 || newBox.height < 20) {
                            return oldBox;
                        }
                        return newBox;
                    }}
                    rotateEnabled={false}
                    borderStroke="#2563EB"
                    anchorStroke="#2563EB"
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
                className="absolute inset-0 flex items-center justify-center pointer-events-none"
            >
                {/* Wrap both SVG and Konva in the same transforming container */}
                <div
                    style={{
                        transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                        transformOrigin: 'center center',
                        width: `${CANVAS_SIZE}px`,
                        height: `${CANVAS_SIZE}px`,
                        position: 'relative',
                        pointerEvents: isEditMode ? 'auto' : 'none', // Konva intercepts clicks in edit mode
                    }}
                >
                    {svgString ? (
                        <div
                            ref={svgRef}
                            className={`transition-opacity duration-300 w-full h-full absolute inset-0 ${isEditMode ? 'opacity-50' : 'opacity-100'} pointer-events-auto`}
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
