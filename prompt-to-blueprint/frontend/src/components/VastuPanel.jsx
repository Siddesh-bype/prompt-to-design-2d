import useLayoutStore from '../store/layoutStore';

/**
 * VastuPanel — right sidebar displaying Vastu Shastra compliance results.
 * 
 * Shows:
 * - Circular score gauge (0–100)
 * - 3×3 zone map with room placements
 * - Improvement suggestions
 */
export default function VastuPanel() {
    const vastu = useLayoutStore((s) => s.vastu);
    const layout = useLayoutStore((s) => s.layout);

    if (!vastu && !layout) {
        return (
            <div className="flex flex-col h-full bg-surface-800 border-l border-surface-700">
                <div className="p-4 border-b border-surface-700">
                    <h2 className="text-sm font-semibold text-surface-200 uppercase tracking-wider">
                        Vastu
                    </h2>
                </div>
                <div className="flex-1 flex items-center justify-center p-4">
                    <p className="text-xs text-surface-500 text-center">
                        Enable Vastu compliance in the prompt panel to see results here
                    </p>
                </div>
            </div>
        );
    }

    const score = vastu?.score ?? 0;
    const suggestions = vastu?.suggestions ?? [];
    const zoneMap = vastu?.zone_map ?? {};

    // Score colour
    const scoreColor =
        score >= 80 ? '#22C55E' :
            score >= 60 ? '#EAB308' :
                score >= 40 ? '#F97316' : '#EF4444';

    // SVG gauge parameters
    const circumference = 2 * Math.PI * 40;
    const offset = circumference - (score / 100) * circumference;

    // Zone grid
    const zones = [
        ['NW', 'N', 'NE'],
        ['W', 'CENTER', 'E'],
        ['SW', 'S', 'SE'],
    ];

    return (
        <div className="flex flex-col h-full bg-surface-800 border-l border-surface-700">
            {/* Header */}
            <div className="p-4 border-b border-surface-700">
                <h2 className="text-sm font-semibold text-surface-200 uppercase tracking-wider">
                    Vastu Shastra
                </h2>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-5 custom-scrollbar">
                {/* Score gauge */}
                <div className="flex flex-col items-center">
                    <div className="relative w-24 h-24">
                        <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
                            {/* Background circle */}
                            <circle
                                cx="50" cy="50" r="40"
                                fill="none" stroke="currentColor"
                                className="text-surface-700"
                                strokeWidth="8"
                            />
                            {/* Score arc */}
                            <circle
                                cx="50" cy="50" r="40"
                                fill="none"
                                stroke={scoreColor}
                                strokeWidth="8"
                                strokeLinecap="round"
                                strokeDasharray={circumference}
                                strokeDashoffset={offset}
                                className="transition-all duration-1000"
                            />
                        </svg>
                        <div className="absolute inset-0 flex items-center justify-center">
                            <span className="text-xl font-bold" style={{ color: scoreColor }}>
                                {score}
                            </span>
                        </div>
                    </div>
                    <p className="text-xs text-surface-400 mt-2">
                        {score >= 80 ? 'Excellent' : score >= 60 ? 'Good' : score >= 40 ? 'Fair' : 'Needs work'}
                    </p>
                </div>

                {/* Zone map */}
                <div>
                    <h3 className="text-xs font-medium text-surface-400 mb-2 uppercase tracking-wider">
                        Zone Map
                    </h3>
                    <div className="grid grid-cols-3 gap-1">
                        {zones.map((row, ri) =>
                            row.map((zone) => {
                                const roomName = zoneMap[zone];
                                const isOccupied = !!roomName;
                                return (
                                    <div
                                        key={zone}
                                        className={`aspect-square rounded flex flex-col items-center justify-center p-1 text-center border transition-colors ${isOccupied
                                                ? 'bg-blueprint-900/30 border-blueprint-700/50'
                                                : 'bg-surface-750 border-surface-700'
                                            }`}
                                        title={roomName || zone}
                                    >
                                        <span className="text-[9px] font-bold text-surface-500">
                                            {zone}
                                        </span>
                                        {roomName && (
                                            <span className="text-[8px] text-blueprint-400 leading-tight mt-0.5 truncate w-full">
                                                {roomName}
                                            </span>
                                        )}
                                    </div>
                                );
                            })
                        )}
                    </div>
                </div>

                {/* Suggestions */}
                {suggestions.length > 0 && (
                    <div>
                        <h3 className="text-xs font-medium text-surface-400 mb-2 uppercase tracking-wider">
                            Suggestions
                        </h3>
                        <div className="space-y-2">
                            {suggestions.map((s, i) => (
                                <div
                                    key={i}
                                    className="flex gap-2 p-2.5 bg-surface-750 rounded-lg border border-surface-700"
                                >
                                    <span className="text-amber-400 text-sm mt-0.5 shrink-0">⚠</span>
                                    <p className="text-xs text-surface-300 leading-relaxed">{s}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Room list */}
                {layout?.rooms && (
                    <div>
                        <h3 className="text-xs font-medium text-surface-400 mb-2 uppercase tracking-wider">
                            Rooms ({layout.rooms.length})
                        </h3>
                        <div className="space-y-1">
                            {layout.rooms.map((room, i) => (
                                <div
                                    key={room.room_spec?.room_id || i}
                                    className="flex justify-between items-center px-2.5 py-1.5 bg-surface-750 rounded text-xs"
                                >
                                    <span className="text-surface-200">
                                        {(room.room_spec?.room_type || '').replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, c => c.toUpperCase())}
                                    </span>
                                    <span className="text-surface-500 font-mono">
                                        {room.room_spec?.target_area_sqm?.toFixed(1)} m²
                                    </span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
