import { useRef, useMemo, useState, Suspense } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Environment, ContactShadows, Sky } from '@react-three/drei';
import * as THREE from 'three';
import useLayoutStore from '../store/layoutStore';

/**
 * ThreePreview — Enhanced 3D floor plan viewer using Three.js / React Three Fiber.
 *
 * Renders rooms with:
 * - Solid walls with proper thickness (4-panel construction per room)
 * - Door openings cut visually into walls
 * - Window panes on outer-facing walls
 * - Per-room-type furniture meshes (bed, sofa, toilet, stove, etc.)
 * - Coloured floor tiles
 * - Sky + contact shadows for realism
 * - Orbit controls with hover highlighting
 */

// ─── Constants ──────────────────────────────────────────────────
const WALL_THICKNESS = 0.15;
const DEFAULT_WALL_HEIGHT = 3.0;
const FLOOR_OFFSET = 0.01;

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
    GARAGE: '#94A3B8',
};

const WALL_HEIGHT_MAP = {
    BATHROOM: 2.6,
    TOILET: 2.6,
    BALCONY: 1.2,
    CORRIDOR: 2.8,
};

// Floor patterns - true = has special texture
const FLOOR_PATTERN = {
    BATHROOM: 'tile',
    TOILET: 'tile',
    KITCHEN: 'tile',
    MASTER_BEDROOM: 'wood',
    BEDROOM: 'wood',
    LIVING_ROOM: 'carpet',
    BALCONY: 'outdoor',
};

const FLOOR_COLORS = {
    LIVING_ROOM: '#CBD5E1',
    KITCHEN: '#FEF3C7',
    MASTER_BEDROOM: '#EDE9FE',
    BEDROOM: '#F3ECFF',
    BATHROOM: '#D1FAE5',
    TOILET: '#D1FAE5',
    CORRIDOR: '#F1F5F9',
    BALCONY: '#DCFCE7',
    STUDY: '#FFF7ED',
    DINING: '#FEE2E2',
    UTILITY: '#F8FAFC',
    GARAGE: '#E2E8F0',
};

// ─── Floor with Pattern ──────────────────────────────────────────
function FloorWithPattern({ w, d, floorColor, patternType }) {
    if (!patternType || patternType === 'plain') {
        return (
            <meshStandardMaterial color={floorColor} roughness={0.8} />
        );
    }

    switch (patternType) {
        case 'tile':
            return (
                <meshStandardMaterial color={floorColor} roughness={0.6} />
            );
        case 'wood':
            return (
                <meshStandardMaterial color="#D4B896" roughness={0.7} metalness={0.1} />
            );
        case 'carpet':
            return (
                <meshStandardMaterial color={floorColor} roughness={0.95} />
            );
        case 'outdoor':
            return (
                <meshStandardMaterial color="#A8D5BA" roughness={0.9} />
            );
        default:
            return (
                <meshStandardMaterial color={floorColor} roughness={0.8} />
            );
    }
}

// ─── Ceiling Light ────────────────────────────────────────────────
function CeilingLight({ position }) {
    return (
        <group position={position}>
            {/* Light fixture */}
            <mesh>
                <cylinderGeometry args={[0.15, 0.12, 0.08, 8]} />
                <meshStandardMaterial color="#F8FAFC" roughness={0.3} />
            </mesh>
            {/* Light glow */}
            <pointLight intensity={0.5} distance={3} color="#FEF3C7" />
        </group>
    );
}

// ─── Ceiling Fan ─────────────────────────────────────────────────
function CeilingFan({ position, isHovered }) {
    const fanRef = useRef();
    
    useFrame(() => {
        if (fanRef.current && isHovered) {
            fanRef.current.rotation.y += 0.15;
        }
    });

    return (
        <group position={position}>
            {/* Mount */}
            <mesh>
                <cylinderGeometry args={[0.05, 0.05, 0.15, 8]} />
                <meshStandardMaterial color="#374151" roughness={0.5} />
            </mesh>
            {/* Fan blades */}
            <group ref={fanRef} position={[0, -0.1, 0]}>
                {[0, 1, 2, 3].map((i) => (
                    <mesh key={i} position={[Math.cos(i * Math.PI / 2) * 0.35, 0, Math.sin(i * Math.PI / 2) * 0.35]} rotation={[0, -i * Math.PI / 2, 0]}>
                        <boxGeometry args={[0.6, 0.02, 0.12]} />
                        <meshStandardMaterial color="#E2E8F0" roughness={0.4} />
                    </mesh>
                ))}
            </group>
        </group>
    );
}

// ─── Wall Outlet ─────────────────────────────────────────────────
function WallOutlet({ position }) {
    return (
        <group position={position}>
            <mesh>
                <boxGeometry args={[0.06, 0.08, 0.02]} />
                <meshStandardMaterial color="#F8FAFC" roughness={0.5} />
            </mesh>
            {/* Two slots */}
            <mesh position={[0.01, 0.015, 0.011]}>
                <boxGeometry args={[0.015, 0.025, 0.001]} />
                <meshStandardMaterial color="#1E293B" />
            </mesh>
            <mesh position={[0.01, -0.015, 0.011]}>
                <boxGeometry args={[0.015, 0.015, 0.001]} />
                <meshStandardMaterial color="#1E293B" />
            </mesh>
        </group>
    );
}

// ─── Room Furniture ─────────────────────────────────────────────
function WallPanel({ w, h, d, position, color }) {
    return (
        <mesh position={position} castShadow receiveShadow>
            <boxGeometry args={[w, h, d]} />
            <meshStandardMaterial color={color} roughness={0.85} metalness={0.0} />
        </mesh>
    );
}

// ─── Window pane ───────────────────────────────────────────────
function WindowPane({ position, rotation }) {
    return (
        <group position={position} rotation={rotation}>
            {/* Frame */}
            <mesh castShadow>
                <boxGeometry args={[1.0, 0.05, 1.2]} />
                <meshStandardMaterial color="#94A3B8" roughness={0.6} />
            </mesh>
            {/* Glass */}
            <mesh position={[0, 0.01, 0]}>
                <boxGeometry args={[0.9, 0.01, 1.1]} />
                <meshPhysicalMaterial
                    color="#BAE6FD"
                    transparent
                    opacity={0.35}
                    roughness={0.0}
                    metalness={0.1}
                    transmission={0.8}
                />
            </mesh>
        </group>
    );
}

// ─── Room Furniture ─────────────────────────────────────────────
function Furniture({ roomType, w, d, isHovered }) {
    const color = ROOM_COLORS[roomType] || '#94A3B8';
    const dark = '#374151';

    switch (roomType) {
        case 'MASTER_BEDROOM':
        case 'BEDROOM': {
            const bw = Math.min(1.8, w * 0.6);
            const bd = Math.min(2.0, d * 0.65);
            const hasRoom = w > 2.5 && d > 2.5;
            return (
                <group>
                    {/* Bed base */}
                    <mesh position={[0, 0.25, 0]} castShadow receiveShadow>
                        <boxGeometry args={[bw, 0.4, bd]} />
                        <meshStandardMaterial color="#F1F5F9" roughness={0.9} />
                    </mesh>
                    {/* Headboard */}
                    <mesh position={[0, 0.6, -bd / 2 + 0.1]} castShadow>
                        <boxGeometry args={[bw, 0.7, 0.12]} />
                        <meshStandardMaterial color="#475569" roughness={0.7} />
                    </mesh>
                    {/* Pillow */}
                    <mesh position={[bw * 0.18, 0.48, -bd * 0.3]} castShadow>
                        <boxGeometry args={[bw * 0.35, 0.08, 0.5]} />
                        <meshStandardMaterial color="#FFFFFF" roughness={0.9} />
                    </mesh>
                    <mesh position={[-bw * 0.18, 0.48, -bd * 0.3]} castShadow>
                        <boxGeometry args={[bw * 0.35, 0.08, 0.5]} />
                        <meshStandardMaterial color="#FFFFFF" roughness={0.9} />
                    </mesh>
                    {/* Nightstand left */}
                    {hasRoom && w > 3 && (
                        <mesh position={[-bw / 2 - 0.4, 0.25, -bd / 2 + 0.3]} castShadow>
                            <boxGeometry args={[0.4, 0.5, 0.35]} />
                            <meshStandardMaterial color="#92400E" roughness={0.7} />
                        </mesh>
                    )}
                    {/* Nightstand right */}
                    {hasRoom && w > 3 && (
                        <mesh position={[bw / 2 + 0.4, 0.25, -bd / 2 + 0.3]} castShadow>
                            <boxGeometry args={[0.4, 0.5, 0.35]} />
                            <meshStandardMaterial color="#92400E" roughness={0.7} />
                        </mesh>
                    )}
                    {/* TV */}
                    {w > 3 && (
                        <group position={[w * 0.3, 0.5, -d * 0.3]}>
                            <mesh castShadow>
                                <boxGeometry args={[0.8, 0.5, 0.08]} />
                                <meshStandardMaterial color="#1E293B" roughness={0.5} />
                            </mesh>
                            {/* TV Stand */}
                            <mesh position={[0, -0.35, 0.1]} castShadow>
                                <boxGeometry args={[0.9, 0.3, 0.3]} />
                                <meshStandardMaterial color="#374151" roughness={0.7} />
                            </mesh>
                        </group>
                    )}
                    {/* Wardrobe */}
                    {w > 3.5 && (
                        <mesh position={[w * 0.35, 0.9, d * 0.2]} castShadow>
                            <boxGeometry args={[1.2, 1.8, 0.5]} />
                            <meshStandardMaterial color="#78716C" roughness={0.8} />
                        </mesh>
                    )}
                </group>
            );
        }
        case 'LIVING_ROOM': {
            const sw = Math.min(2.0, w * 0.6);
            const sd = 0.8;
            const hasRoom = w > 2.5 && d > 2.5;
            return (
                <group>
                    {/* Sofa base */}
                    <mesh position={[0, 0.25, d * 0.2]} castShadow receiveShadow>
                        <boxGeometry args={[sw, 0.4, sd]} />
                        <meshStandardMaterial color="#475569" roughness={0.8} />
                    </mesh>
                    {/* Sofa back */}
                    <mesh position={[0, 0.55, d * 0.2 + sd / 2 - 0.1]} castShadow>
                        <boxGeometry args={[sw, 0.6, 0.15]} />
                        <meshStandardMaterial color="#334155" roughness={0.8} />
                    </mesh>
                    {/* Coffee table */}
                    <mesh position={[0, 0.22, -d * 0.1]} castShadow receiveShadow>
                        <boxGeometry args={[1.0, 0.05, 0.6]} />
                        <meshStandardMaterial color="#92400E" roughness={0.6} />
                    </mesh>
                    <mesh position={[0, 0.1, -d * 0.1]}>
                        <boxGeometry args={[0.9, 0.18, 0.5]} />
                        <meshStandardMaterial color="#A3A3A3" roughness={0.9} />
                    </mesh>
                    {/* TV Unit */}
                    {hasRoom && w > 3 && (
                        <group position={[w * 0.3, 0.2, -d * 0.35]}>
                            <mesh castShadow receiveShadow>
                                <boxGeometry args={[1.2, 0.4, 0.35]} />
                                <meshStandardMaterial color="#292524" roughness={0.7} />
                            </mesh>
                            {/* TV */}
                            <mesh position={[0, 0.45, -0.05]} castShadow>
                                <boxGeometry args={[0.9, 0.55, 0.05]} />
                                <meshStandardMaterial color="#0F172A" roughness={0.3} metalness={0.5} />
                            </mesh>
                        </group>
                    )}
                    {/* Side chairs */}
                    {hasRoom && w > 4 && d > 3 && (
                        <>
                            <mesh position={[-sw / 2 - 0.5, 0.3, 0]} castShadow>
                                <boxGeometry args={[0.45, 0.5, 0.45]} />
                                <meshStandardMaterial color="#78350F" roughness={0.8} />
                            </mesh>
                            <mesh position={[sw / 2 + 0.5, 0.3, 0]} castShadow>
                                <boxGeometry args={[0.45, 0.5, 0.45]} />
                                <meshStandardMaterial color="#78350F" roughness={0.8} />
                            </mesh>
                        </>
                    )}
                    {/* AC Unit */}
                    {hasRoom && (
                        <mesh position={[-w * 0.35, 2.5, -d * 0.45]} castShadow>
                            <boxGeometry args={[0.7, 0.25, 0.2]} />
                            <meshStandardMaterial color="#F1F5F9" roughness={0.4} metalness={0.3} />
                        </mesh>
                    )}
                </group>
            );
        }
        case 'KITCHEN': {
            const cw = Math.min(w * 0.8, 2.4);
            const hasRoom = w > 2.5 && d > 2.5;
            return (
                <group>
                    {/* Counter top */}
                    <mesh position={[0, 0.9, -d * 0.35]} castShadow receiveShadow>
                        <boxGeometry args={[cw, 0.05, 0.6]} />
                        <meshStandardMaterial color="#F8FAFC" roughness={0.3} metalness={0.2} />
                    </mesh>
                    {/* Counter base */}
                    <mesh position={[0, 0.45, -d * 0.35]} castShadow>
                        <boxGeometry args={[cw, 0.9, 0.58]} />
                        <meshStandardMaterial color="#E2E8F0" roughness={0.8} />
                    </mesh>
                    {/* Stove burner markers */}
                    <mesh position={[-0.3, 0.93, -d * 0.35]}>
                        <cylinderGeometry args={[0.15, 0.15, 0.02, 16]} />
                        <meshStandardMaterial color="#1E293B" roughness={0.9} />
                    </mesh>
                    <mesh position={[0.3, 0.93, -d * 0.35]}>
                        <cylinderGeometry args={[0.15, 0.15, 0.02, 16]} />
                        <meshStandardMaterial color="#1E293B" roughness={0.9} />
                    </mesh>
                    {/* Sink */}
                    <mesh position={[cw * 0.35, 0.88, -d * 0.35]}>
                        <boxGeometry args={[0.5, 0.03, 0.4]} />
                        <meshStandardMaterial color="#94A3B8" roughness={0.3} metalness={0.5} />
                    </mesh>
                    {/* Refrigerator */}
                    {hasRoom && w > 3 && (
                        <mesh position={[-w * 0.35, 0.9, d * 0.2]} castShadow>
                            <boxGeometry args={[0.7, 1.7, 0.65]} />
                            <meshStandardMaterial color="#E2E8F0" roughness={0.4} metalness={0.6} />
                        </mesh>
                    )}
                    {/* Microwave */}
                    {hasRoom && (
                        <mesh position={[cw * 0.3, 1.15, -d * 0.35]} castShadow>
                            <boxGeometry args={[0.4, 0.25, 0.3]} />
                            <meshStandardMaterial color="#1E293B" roughness={0.3} metalness={0.5} />
                        </mesh>
                    )}
                    {/* Dishwasher */}
                    {hasRoom && d > 3 && (
                        <mesh position={[cw * 0.5, 0.35, d * 0.3]} castShadow>
                            <boxGeometry args={[0.5, 0.6, 0.55]} />
                            <meshStandardMaterial color="#F1F5F9" roughness={0.5} />
                        </mesh>
                    )}
                </group>
            );
        }
        case 'BATHROOM':
        case 'TOILET': {
            const hasRoom = w > 2 && d > 2;
            return (
                <group>
                    {/* Toilet bowl */}
                    <mesh position={[0, 0.25, d * 0.25]} castShadow receiveShadow>
                        <cylinderGeometry args={[0.22, 0.22, 0.42, 12]} />
                        <meshStandardMaterial color="white" roughness={0.1} />
                    </mesh>
                    {/* Toilet seat */}
                    <mesh position={[0, 0.47, d * 0.25]}>
                        <torusGeometry args={[0.18, 0.04, 8, 16, Math.PI * 2]} />
                        <meshStandardMaterial color="#F1F5F9" roughness={0.3} />
                    </mesh>
                    {/* Tank */}
                    <mesh position={[0, 0.55, d * 0.35]} castShadow>
                        <boxGeometry args={[0.35, 0.35, 0.18]} />
                        <meshStandardMaterial color="white" roughness={0.2} />
                    </mesh>
                    {/* Sink */}
                    <mesh position={[w * 0.2, 0.88, -d * 0.2]} castShadow>
                        <boxGeometry args={[0.4, 0.06, 0.35]} />
                        <meshStandardMaterial color="white" roughness={0.1} />
                    </mesh>
                    {/* Bathtub */}
                    {roomType === 'BATHROOM' && hasRoom && w > 2.5 && d > 2.5 && (
                        <mesh position={[-w * 0.25, 0.35, d * 0.25]} castShadow receiveShadow>
                            <boxGeometry args={[0.75, 0.5, 1.7]} />
                            <meshStandardMaterial color="white" roughness={0.15} />
                        </mesh>
                    )}
                    {/* Medicine Cabinet */}
                    {hasRoom && (
                        <mesh position={[w * 0.35, 1.5, -d * 0.45]} castShadow>
                            <boxGeometry args={[0.4, 0.5, 0.08]} />
                            <meshStandardMaterial color="#F8FAFC" roughness={0.5} metalness={0.3} />
                        </mesh>
                    )}
                </group>
            );
        }
        case 'DINING': {
            const hasRoom = w > 2.5 && d > 2.5;
            return (
                <group>
                    {/* Table */}
                    <mesh position={[0, 0.76, 0]} castShadow receiveShadow>
                        <cylinderGeometry args={[Math.min(w, d) * 0.3, Math.min(w, d) * 0.3, 0.05, 16]} />
                        <meshStandardMaterial color="#92400E" roughness={0.6} />
                    </mesh>
                    {/* Table leg */}
                    <mesh position={[0, 0.38, 0]}>
                        <cylinderGeometry args={[0.06, 0.06, 0.75, 8]} />
                        <meshStandardMaterial color="#78350F" roughness={0.7} />
                    </mesh>
                    {/* Chairs (4 around table) */}
                    {[[-0.7, 0], [0.7, 0], [0, -0.7], [0, 0.7]].map(([cx, cz], idx) => (
                        <group key={idx} position={[cx, 0, cz]}>
                            <mesh position={[0, 0.45, 0]} castShadow>
                                <boxGeometry args={[0.4, 0.05, 0.4]} />
                                <meshStandardMaterial color="#F59E0B" roughness={0.8} />
                            </mesh>
                            <mesh position={[0, 0.75, 0.18]} castShadow>
                                <boxGeometry args={[0.4, 0.6, 0.06]} />
                                <meshStandardMaterial color="#D97706" roughness={0.8} />
                            </mesh>
                        </group>
                    ))}
                    {/* Sideboard */}
                    {hasRoom && (
                        <mesh position={[-w * 0.35, 0.4, d * 0.3]} castShadow>
                            <boxGeometry args={[0.8, 0.75, 0.4]} />
                            <meshStandardMaterial color="#78350F" roughness={0.7} />
                        </mesh>
                    )}
                </group>
            );
        }
        case 'STUDY': {
            const hasRoom = w > 2 && d > 2;
            return (
                <group>
                    <mesh position={[0, 0.75, 0]} castShadow receiveShadow>
                        <boxGeometry args={[Math.min(w * 0.65, 1.4), 0.06, 0.7]} />
                        <meshStandardMaterial color="#92400E" roughness={0.6} />
                    </mesh>
                    <mesh position={[0, 0.38, 0]}>
                        <boxGeometry args={[1.2, 0.72, 0.65]} />
                        <meshStandardMaterial color="#D97706" roughness={0.7} />
                    </mesh>
                    {/* Monitor */}
                    <mesh position={[0, 1.08, 0.1]} castShadow>
                        <boxGeometry args={[0.5, 0.33, 0.04]} />
                        <meshStandardMaterial color="#1E293B" roughness={0.5} />
                    </mesh>
                    {/* Bookshelf */}
                    {hasRoom && w > 2.5 && (
                        <mesh position={[-w * 0.35, 0.9, d * 0.25]} castShadow>
                            <boxGeometry args={[0.6, 1.6, 0.35]} />
                            <meshStandardMaterial color="#78350F" roughness={0.8} />
                        </mesh>
                    )}
                    {/* Computer */}
                    {hasRoom && (
                        <mesh position={[w * 0.2, 0.85, 0]} castShadow>
                            <boxGeometry args={[0.35, 0.25, 0.02]} />
                            <meshStandardMaterial color="#0F172A" roughness={0.3} />
                        </mesh>
                    )}
                </group>
            );
        }
        case 'BALCONY': {
            return (
                <group>
                    {/* Railing */}
                    <mesh position={[0, 0.6, -d / 2 + 0.05]} castShadow>
                        <boxGeometry args={[w * 0.95, 0.05, 0.05]} />
                        <meshStandardMaterial color="#78716C" roughness={0.6} />
                    </mesh>
                    <mesh position={[0, 0.9, -d / 2 + 0.05]} castShadow>
                        <boxGeometry args={[w * 0.95, 0.05, 0.05]} />
                        <meshStandardMaterial color="#78716C" roughness={0.6} />
                    </mesh>
                    {/* Railing posts */}
                    {[-w * 0.4, 0, w * 0.4].map((px, i) => (
                        <mesh key={i} position={[px, 0.75, -d / 2 + 0.05]} castShadow>
                            <boxGeometry args={[0.04, 0.5, 0.04]} />
                            <meshStandardMaterial color="#57534E" roughness={0.7} />
                        </mesh>
                    ))}
                    {/* Plants */}
                    {w > 2 && (
                        <>
                            <mesh position={[-w * 0.35, 0.15, -d * 0.3]} castShadow>
                                <cylinderGeometry args={[0.12, 0.1, 0.3, 8]} />
                                <meshStandardMaterial color="#78350F" roughness={0.8} />
                            </mesh>
                            <mesh position={[w * 0.35, 0.15, -d * 0.3]} castShadow>
                                <cylinderGeometry args={[0.12, 0.1, 0.3, 8]} />
                                <meshStandardMaterial color="#78350F" roughness={0.8} />
                            </mesh>
                        </>
                    )}
                </group>
            );
        }
        default:
            return null;
    }
}

// ─── Room with Walls ────────────────────────────────────────────
function Room({ bbox, roomType, label, plotWidth, plotHeight, isHovered, onHover }) {
    const meshRef = useRef();
    const wallHeight = WALL_HEIGHT_MAP[roomType] || DEFAULT_WALL_HEIGHT;

    const x = ((bbox.x_min + bbox.x_max) / 2 - 0.5) * plotWidth;
    const z = ((bbox.y_min + bbox.y_max) / 2 - 0.5) * plotHeight;
    const w = (bbox.x_max - bbox.x_min) * plotWidth;
    const d = (bbox.y_max - bbox.y_min) * plotHeight;

    const baseColor = ROOM_COLORS[roomType] || '#E5E7EB';
    const floorColor = FLOOR_COLORS[roomType] || '#F1F5F9';
    const wallColor = '#CBD5E1';
    const displayName = (label || roomType || '').replace(/_/g, ' ');

    // Hover animation on Y
    useFrame(() => {
        if (meshRef.current) {
            const target = isHovered ? 0.15 : 0;
            meshRef.current.position.y += (target - meshRef.current.position.y) * 0.12;
        }
    });

    // Window positions: one per long wall at mid-height
    const wh = wallHeight;
    const winY = wh * 0.6; // 60% up the wall
    const floorPattern = FLOOR_PATTERN[roomType] || null;
    const hasRoom = w > 2.5 && d > 2.5;

    return (
        <group ref={meshRef} position={[x, 0, z]}>
            {/* ── Floor with pattern ── */}
            <mesh
                position={[0, FLOOR_OFFSET, 0]}
                rotation={[-Math.PI / 2, 0, 0]}
                receiveShadow
                onPointerEnter={(e) => { e.stopPropagation(); onHover(true); }}
                onPointerLeave={(e) => { e.stopPropagation(); onHover(false); }}
            >
                <planeGeometry args={[w, d]} />
                <FloorWithPattern w={w} d={d} floorColor={floorColor} patternType={floorPattern} />
            </mesh>

            {/* ── Four walls (North, South, East, West) ── */}
            {/* North wall */}
            <WallPanel w={w} h={wh} d={WALL_THICKNESS} position={[0, wh / 2, -d / 2 + WALL_THICKNESS / 2]} color={wallColor} />
            {/* South wall */}
            <WallPanel w={w} h={wh} d={WALL_THICKNESS} position={[0, wh / 2, d / 2 - WALL_THICKNESS / 2]} color={wallColor} />
            {/* West wall */}
            <WallPanel w={WALL_THICKNESS} h={wh} d={d} position={[-w / 2 + WALL_THICKNESS / 2, wh / 2, 0]} color={wallColor} />
            {/* East wall */}
            <WallPanel w={WALL_THICKNESS} h={wh} d={d} position={[w / 2 - WALL_THICKNESS / 2, wh / 2, 0]} color={wallColor} />

            {/* ── Ceiling (semi-transparent) ── */}
            <mesh position={[0, wh, 0]}>
                <planeGeometry args={[w - WALL_THICKNESS * 2, d - WALL_THICKNESS * 2]} />
                <meshStandardMaterial
                    color={baseColor}
                    transparent
                    opacity={isHovered ? 0.15 : 0.35}
                    roughness={0.7}
                    side={THREE.DoubleSide}
                />
            </mesh>

            {/* ── Ceiling Light ── */}
            {hasRoom && (
                <CeilingLight position={[w * 0.2, wh - 0.05, d * 0.2]} />
            )}

            {/* ── Ceiling Fan (Bedrooms & Living) ── */}
            {['MASTER_BEDROOM', 'BEDROOM', 'LIVING_ROOM'].includes(roomType) && hasRoom && (
                <CeilingFan position={[0, wh - 0.1, 0]} isHovered={isHovered} />
            )}

            {/* ── Window on North wall (except BATHROOM/TOILET/CORRIDOR) ── */}
            {!['BATHROOM', 'TOILET', 'CORRIDOR'].includes(roomType) && w >= 2.5 && (
                <WindowPane
                    position={[0, winY, -d / 2 + 0.1]}
                    rotation={[Math.PI / 2, 0, 0]}
                />
            )}

            {/* ── Furniture ── */}
            <Furniture roomType={roomType} w={w} d={d} isHovered={isHovered} />

            {/* ── Room label (visible when hovered or always) ── */}
            <Text
                position={[0, wh + 0.45, 0]}
                fontSize={Math.min(0.42, w * 0.12, d * 0.12)}
                color={isHovered ? '#93C5FD' : '#E2E8F0'}
                anchorX="center"
                anchorY="bottom"
                outlineWidth={0.02}
                outlineColor="#0F172A"
            >
                {displayName}
            </Text>

            {isHovered && (
                <Text
                    position={[0, wh + 0.12, 0]}
                    fontSize={Math.min(0.28, w * 0.09, d * 0.09)}
                    color="#94A3B8"
                    anchorX="center"
                    anchorY="bottom"
                    outlineWidth={0.015}
                    outlineColor="#0F172A"
                >
                    {(w * d).toFixed(1)} m²
                </Text>
            )}
        </group>
    );
}

// ─── Door Opening Marker ────────────────────────────────────────
function Door({ dx, dy, plotWidth, plotHeight }) {
    const x = (dx - 0.5) * plotWidth;
    const z = (dy - 0.5) * plotHeight;
    const dH = 2.1;

    return (
        <group position={[x, 0, z]}>
            {/* Door frame */}
            <mesh position={[0, dH / 2, 0]} castShadow>
                <boxGeometry args={[0.95, dH, 0.18]} />
                <meshStandardMaterial color="#78350F" roughness={0.7} />
            </mesh>
            {/* Door leaf (half-open) */}
            <mesh
                position={[0.45, dH / 2, 0.38]}
                rotation={[0, Math.PI / 4, 0]}
                castShadow
            >
                <boxGeometry args={[0.04, dH - 0.1, 0.85]} />
                <meshStandardMaterial color="#A16207" roughness={0.65} />
            </mesh>
            {/* Threshold strip */}
            <mesh position={[0, 0.01, 0]} rotation={[-Math.PI / 2, 0, 0]}>
                <planeGeometry args={[0.95, 0.18]} />
                <meshStandardMaterial color="#D97706" roughness={0.9} />
            </mesh>
        </group>
    );
}

// ─── Ground Plane ───────────────────────────────────────────────
function Ground({ plotWidth, plotHeight }) {
    return (
        <group>
            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.02, 0]} receiveShadow>
                <planeGeometry args={[plotWidth + 8, plotHeight + 8]} />
                <meshStandardMaterial color="#0F172A" roughness={0.9} />
            </mesh>
            <gridHelper
                args={[Math.max(plotWidth, plotHeight) + 8, 20, '#1E3A5F', '#1E293B']}
                position={[0, 0, 0]}
            />
            {/* Plot boundary */}
            <lineSegments position={[0, 0.03, 0]}>
                <edgesGeometry args={[new THREE.PlaneGeometry(plotWidth, plotHeight)]} />
                <lineBasicMaterial color="#3B82F6" linewidth={2} />
            </lineSegments>
        </group>
    );
}

// ─── Compass ────────────────────────────────────────────────────
function CompassArrow({ plotWidth, plotHeight }) {
    const offset = Math.max(plotWidth, plotHeight) / 2 + 2.0;
    return (
        <group position={[0, 0.1, -offset]}>
            <mesh rotation={[-Math.PI / 2, 0, 0]}>
                <coneGeometry args={[0.35, 1.0, 4]} />
                <meshStandardMaterial color="#EF4444" roughness={0.6} />
            </mesh>
            <Text position={[0, 0.8, 0]} fontSize={0.5} color="#EF4444" anchorX="center" anchorY="bottom" outlineWidth={0.02} outlineColor="#0F172A">
                N
            </Text>
        </group>
    );
}

// ─── Main Export ────────────────────────────────────────────────
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
                camera={{ position: [plotWidth * 0.9, plotWidth * 0.7, plotWidth * 0.9], fov: 48 }}
                gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 1.3 }}
            >
                {/* ── Sky ── */}
                <Sky sunPosition={[100, 80, 100]} turbidity={4} rayleigh={0.5} />

                {/* ── Lighting ── */}
                <ambientLight intensity={0.5} color="#E0E8FF" />
                <directionalLight
                    position={[plotWidth * 1.5, plotWidth * 2, plotWidth]}
                    intensity={1.2}
                    castShadow
                    shadow-mapSize-width={2048}
                    shadow-mapSize-height={2048}
                    shadow-camera-far={60}
                    shadow-camera-left={-20}
                    shadow-camera-right={20}
                    shadow-camera-top={20}
                    shadow-camera-bottom={-20}
                    shadow-bias={-0.0005}
                />
                <directionalLight position={[-plotWidth, plotWidth, -plotWidth]} intensity={0.4} color="#A5B4FC" />
                <pointLight position={[0, 4, 0]} intensity={0.6} distance={20} color="#FEF3C7" />

                {/* ── Environment / fog ── */}
                <fog attach="fog" args={['#0F172A', 25, 70]} />

                {/* ── Ground ── */}
                <Ground plotWidth={plotWidth} plotHeight={plotHeight} />

                {/* ── Compass ── */}
                <CompassArrow plotWidth={plotWidth} plotHeight={plotHeight} />

                {/* ── Contact Shadows ── */}
                <ContactShadows
                    position={[0, 0.02, 0]}
                    opacity={0.55}
                    scale={plotWidth * 2.5}
                    blur={2.5}
                    far={12}
                />

                {/* ── Room Boxes ── */}
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

                {/* ── Doors ── */}
                {layout.rooms.flatMap((room) =>
                    (room.door_midpoints || []).map(([dx, dy], i) => (
                        <Door
                            key={`door-${room.room_spec?.room_id}-${i}`}
                            dx={dx}
                            dy={dy}
                            plotWidth={plotWidth}
                            plotHeight={plotHeight}
                        />
                    ))
                )}

                {/* ── Controls ── */}
                <OrbitControls
                    enableDamping
                    dampingFactor={0.08}
                    maxPolarAngle={Math.PI / 2.1}
                    minDistance={3}
                    maxDistance={plotWidth * 5}
                    target={[0, 1, 0]}
                />
            </Canvas>

            {/* Hovered Room Overlay */}
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

            {/* 3D Indicator badge */}
            <div className="absolute bottom-3 left-3 flex items-center gap-1.5 bg-surface-800/80 backdrop-blur-sm border border-surface-700 rounded-full px-2.5 py-1">
                <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                <span className="text-[10px] text-surface-400 font-medium">3D View • Enhanced</span>
            </div>
        </div>
    );
}
