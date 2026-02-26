import { useRef, useMemo, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Environment, ContactShadows } from '@react-three/drei';
import * as THREE from 'three';
import useLayoutStore from '../store/layoutStore';

/**
 * ThreePreview — 3D floor plan viewer using Three.js / React Three Fiber.
 *
 * Renders rooms as extruded 3D boxes with:
 * - Colored walls with edge outlines
 * - Room labels floating above
 * - Ground plane with grid
 * - Orbit controls for rotation/zoom
 * - Room hover highlighting
 */

// ─── Room Colors ──────────────────────────────────────────────
const ROOM_COLORS = {
    LIVING_ROOM: '#60A5FA',
    KITCHEN: '#FBBF24',
    MASTER_BEDROOM: '#818CF8',
    BEDROOM: '#A78BFA',
    BATHROOM: '#34D399',
    TOILET: '#6EE7B7',
    CORRIDOR: '#9CA3AF',
    BALCONY: '#4ADE80',
    STUDY: '#FB923C',
    DINING: '#F87171',
    UTILITY: '#D1D5DB',
    GARAGE: '#9CA3AF',
};

const WALL_HEIGHT_MAP = {
    BATHROOM: 2.6,
    TOILET: 2.6,
    BALCONY: 1.2,
    CORRIDOR: 2.8,
};
const DEFAULT_WALL_HEIGHT = 3.0;

// ─── Single Room ──────────────────────────────────────────────
function Room({ bbox, roomType, label, plotWidth = 10, plotHeight = 10, isHovered, onHover }) {
    const meshRef = useRef();
    const wallHeight = WALL_HEIGHT_MAP[roomType] || DEFAULT_WALL_HEIGHT;

    // Convert normalised [0,1] bbox to world coordinates centred at origin
    const x = ((bbox.x_min + bbox.x_max) / 2 - 0.5) * plotWidth;
    const z = ((bbox.y_min + bbox.y_max) / 2 - 0.5) * plotHeight;
    const w = (bbox.x_max - bbox.x_min) * plotWidth;
    const d = (bbox.y_max - bbox.y_min) * plotHeight;

    const baseColor = ROOM_COLORS[roomType] || '#E5E7EB';
    const displayName = (label || roomType || '').replace(/_/g, ' ');
    const area = w * d;

    // Animate hover
    useFrame(() => {
        if (meshRef.current) {
            const target = isHovered ? wallHeight / 2 + 0.15 : wallHeight / 2;
            meshRef.current.position.y += (target - meshRef.current.position.y) * 0.15;
        }
    });

    return (
        <group position={[x, 0, z]}>
            {/* Room box */}
            <mesh
                ref={meshRef}
                position={[0, wallHeight / 2, 0]}
                onPointerEnter={(e) => { e.stopPropagation(); onHover(true); }}
                onPointerLeave={(e) => { e.stopPropagation(); onHover(false); }}
                castShadow
                receiveShadow
            >
                <boxGeometry args={[w - 0.05, wallHeight, d - 0.05]} />
                <meshPhysicalMaterial
                    color={baseColor}
                    transparent
                    opacity={isHovered ? 0.85 : 0.65}
                    roughness={0.4}
                    metalness={0.05}
                    clearcoat={0.3}
                    side={THREE.DoubleSide}
                />
            </mesh>

            {/* Wireframe edges */}
            <mesh position={[0, wallHeight / 2, 0]}>
                <boxGeometry args={[w - 0.05, wallHeight, d - 0.05]} />
                <meshBasicMaterial color="#334155" wireframe transparent opacity={0.3} />
            </mesh>

            {/* Room floor */}
            <mesh position={[0, 0.01, 0]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
                <planeGeometry args={[w - 0.05, d - 0.05]} />
                <meshStandardMaterial color={baseColor} opacity={0.35} transparent />
            </mesh>

            {/* Room label */}
            <Text
                position={[0, wallHeight + 0.4, 0]}
                fontSize={Math.min(0.45, w * 0.12, d * 0.12)}
                color="#E2E8F0"
                anchorX="center"
                anchorY="bottom"
                outlineWidth={0.02}
                outlineColor="#0F172A"
            >
                {displayName}
            </Text>

            {/* Area label */}
            <Text
                position={[0, wallHeight + 0.1, 0]}
                fontSize={Math.min(0.3, w * 0.09, d * 0.09)}
                color="#94A3B8"
                anchorX="center"
                anchorY="bottom"
                outlineWidth={0.015}
                outlineColor="#0F172A"
            >
                {area.toFixed(1)} m²
            </Text>
        </group>
    );
}

// ─── Ground plane with grid ───────────────────────────────────
function Ground({ plotWidth = 10, plotHeight = 10 }) {
    return (
        <group>
            {/* Main ground */}
            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]} receiveShadow>
                <planeGeometry args={[plotWidth + 2, plotHeight + 2]} />
                <meshStandardMaterial color="#1E293B" />
            </mesh>

            {/* Grid lines */}
            <gridHelper
                args={[Math.max(plotWidth, plotHeight) + 2, Math.max(plotWidth, plotHeight) + 2, '#334155', '#1E293B']}
                position={[0, 0, 0]}
            />

            {/* Plot boundary outline */}
            <lineSegments position={[0, 0.02, 0]}>
                <edgesGeometry
                    args={[new THREE.PlaneGeometry(plotWidth, plotHeight)]}
                />
                <lineBasicMaterial color="#3B82F6" linewidth={2} />
            </lineSegments>
        </group>
    );
}

// ─── Compass Arrow ────────────────────────────────────────────
function CompassArrow({ facing, plotWidth = 10, plotHeight = 10 }) {
    const pos = useMemo(() => {
        const offset = Math.max(plotWidth, plotHeight) / 2 + 1.5;
        return [0, 0.1, -offset];
    }, [plotWidth, plotHeight]);

    return (
        <group position={pos}>
            <mesh rotation={[-Math.PI / 2, 0, 0]}>
                <coneGeometry args={[0.3, 0.8, 4]} />
                <meshStandardMaterial color="#EF4444" />
            </mesh>
            <Text
                position={[0, 0.6, 0]}
                fontSize={0.4}
                color="#EF4444"
                anchorX="center"
                anchorY="bottom"
                outlineWidth={0.02}
                outlineColor="#0F172A"
            >
                N
            </Text>
        </group>
    );
}

// ─── Main Component ───────────────────────────────────────────
export default function ThreePreview() {
    const layout = useLayoutStore((s) => s.layout);
    const [hoveredIdx, setHoveredIdx] = useState(null);

    if (!layout?.rooms?.length) {
        return (
            <div className="w-full h-full flex items-center justify-center bg-surface-900 text-surface-500 text-sm">
                <div className="flex flex-col items-center gap-3">
                    <svg className="w-12 h-12 text-surface-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1}
                            d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
                    </svg>
                    <p className="text-surface-400 text-xs">Generate a layout to see 3D preview</p>
                </div>
            </div>
        );
    }

    const plotArea = layout.plot_area_sqm || 100;
    const plotWidth = Math.sqrt(plotArea);
    const plotHeight = plotArea / plotWidth;

    return (
        <div className="w-full h-full bg-surface-900 relative">
            <Canvas
                shadows
                camera={{ position: [plotWidth * 0.8, plotWidth * 0.6, plotWidth * 0.8], fov: 50 }}
                gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 1.2 }}
            >
                {/* Lighting */}
                <ambientLight intensity={0.4} />
                <directionalLight
                    position={[plotWidth, plotWidth * 1.5, plotWidth]}
                    intensity={0.9}
                    castShadow
                    shadow-mapSize-width={2048}
                    shadow-mapSize-height={2048}
                    shadow-camera-far={50}
                    shadow-camera-left={-15}
                    shadow-camera-right={15}
                    shadow-camera-top={15}
                    shadow-camera-bottom={-15}
                />
                <directionalLight position={[-5, 8, -5]} intensity={0.3} />

                {/* Environment */}
                <fog attach="fog" args={['#0F172A', 20, 60]} />

                {/* Ground */}
                <Ground plotWidth={plotWidth} plotHeight={plotHeight} />

                {/* Compass */}
                <CompassArrow facing={layout.facing} plotWidth={plotWidth} plotHeight={plotHeight} />

                {/* Contact shadows for depth */}
                <ContactShadows
                    position={[0, 0, 0]}
                    opacity={0.4}
                    scale={plotWidth * 2}
                    blur={2}
                    far={10}
                />

                {/* Room boxes */}
                {layout.rooms.map((room, i) => (
                    <Room
                        key={room.room_spec?.room_id || `room-${i}`}
                        bbox={room.bbox}
                        roomType={room.room_spec?.room_type}
                        label={room.room_spec?.label || room.room_spec?.room_type}
                        plotWidth={plotWidth}
                        plotHeight={plotHeight}
                        isHovered={hoveredIdx === i}
                        onHover={(h) => setHoveredIdx(h ? i : null)}
                    />
                ))}

                {/* Controls */}
                <OrbitControls
                    enableDamping
                    dampingFactor={0.08}
                    maxPolarAngle={Math.PI / 2.1}
                    minDistance={3}
                    maxDistance={plotWidth * 4}
                    target={[0, 1, 0]}
                />
            </Canvas>

            {/* Hovered room info overlay */}
            {hoveredIdx !== null && layout.rooms[hoveredIdx] && (
                <div className="absolute top-3 left-3 bg-surface-800/90 backdrop-blur-sm border border-surface-700 rounded-lg px-3 py-2 pointer-events-none">
                    <p className="text-xs text-blueprint-400 font-semibold">
                        {(layout.rooms[hoveredIdx].room_spec?.room_type || '').replace(/_/g, ' ')}
                    </p>
                    <p className="text-[10px] text-surface-400 mt-0.5">
                        {layout.rooms[hoveredIdx].room_spec?.target_area_sqm?.toFixed(1)} m² target
                    </p>
                </div>
            )}

            {/* 3D mode badge */}
            <div className="absolute bottom-3 left-3 flex items-center gap-1.5 bg-surface-800/80 backdrop-blur-sm border border-surface-700 rounded-full px-2.5 py-1">
                <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                <span className="text-[10px] text-surface-400 font-medium">3D View</span>
            </div>
        </div>
    );
}
