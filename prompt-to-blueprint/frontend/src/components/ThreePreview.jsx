import { Suspense, lazy } from 'react';
import useLayoutStore from '../store/layoutStore';

/**
 * ThreePreview — Optional 3D floor plan preview using Three.js.
 * 
 * Lazy-loaded to avoid bundling Three.js unless needed.
 * Renders extruded room boxes with orbit controls.
 */

// Lazy import Three.js components
let Canvas, OrbitControls;
try {
    const fiber = await import('@react-three/fiber');
    const drei = await import('@react-three/drei');
    Canvas = fiber.Canvas;
    OrbitControls = drei.OrbitControls;
} catch {
    Canvas = null;
    OrbitControls = null;
}

function RoomBox({ bbox, roomType, plotWidth = 10, plotHeight = 10 }) {
    const wallHeight = roomType === 'BATHROOM' || roomType === 'TOILET' ? 2.5 : 3.0;

    const x = ((bbox.x_min + bbox.x_max) / 2 - 0.5) * plotWidth;
    const z = ((bbox.y_min + bbox.y_max) / 2 - 0.5) * plotHeight;
    const w = (bbox.x_max - bbox.x_min) * plotWidth;
    const d = (bbox.y_max - bbox.y_min) * plotHeight;

    const colorMap = {
        LIVING_ROOM: '#93C5FD',
        KITCHEN: '#FCD34D',
        MASTER_BEDROOM: '#A5B4FC',
        BEDROOM: '#C4B5FD',
        BATHROOM: '#6EE7B7',
        TOILET: '#6EE7B7',
        CORRIDOR: '#D1D5DB',
        BALCONY: '#86EFAC',
        STUDY: '#FDBA74',
        DINING: '#FCA5A5',
        UTILITY: '#E5E7EB',
        GARAGE: '#D1D5DB',
    };

    return (
        <mesh position={[x, wallHeight / 2, z]}>
            <boxGeometry args={[w, wallHeight, d]} />
            <meshStandardMaterial
                color={colorMap[roomType] || '#E5E7EB'}
                transparent
                opacity={0.7}
            />
        </mesh>
    );
}

export default function ThreePreview() {
    const layout = useLayoutStore((s) => s.layout);

    if (!Canvas) {
        return (
            <div className="w-full h-full flex items-center justify-center bg-surface-900 text-surface-500 text-sm">
                3D preview requires @react-three/fiber
            </div>
        );
    }

    if (!layout?.rooms?.length) {
        return (
            <div className="w-full h-full flex items-center justify-center bg-surface-900 text-surface-500 text-sm">
                Generate a layout to see 3D preview
            </div>
        );
    }

    return (
        <div className="w-full h-full bg-surface-900">
            <Canvas camera={{ position: [12, 10, 12], fov: 50 }}>
                <ambientLight intensity={0.6} />
                <directionalLight position={[10, 15, 10]} intensity={0.8} />

                {/* Ground plane */}
                <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
                    <planeGeometry args={[12, 12]} />
                    <meshStandardMaterial color="#1E293B" />
                </mesh>

                {/* Room boxes */}
                {layout.rooms.map((room, i) => (
                    <RoomBox
                        key={room.room_spec?.room_id || i}
                        bbox={room.bbox}
                        roomType={room.room_spec?.room_type}
                    />
                ))}

                <OrbitControls
                    enableDamping
                    dampingFactor={0.05}
                    maxPolarAngle={Math.PI / 2.2}
                />
            </Canvas>
        </div>
    );
}
