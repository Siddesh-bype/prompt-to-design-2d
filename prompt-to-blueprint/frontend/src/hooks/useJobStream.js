import { useState, useCallback, useRef, useEffect } from 'react';

/**
 * useJobStream — Custom hook for WebSocket-based job status streaming.
 *
 * Connects to the backend WebSocket endpoint for a given job_id
 * and provides real-time status/progress updates.
 *
 * Features:
 * - Auto-reconnect with exponential backoff (max 3 attempts)
 * - Cleanup on unmount
 * - Callback on each status update
 */
export default function useJobStream() {
    const [status, setStatus] = useState('idle');
    const [progress, setProgress] = useState(0);
    const [error, setError] = useState(null);

    const wsRef = useRef(null);
    const retriesRef = useRef(0);
    const statusRef = useRef('idle');
    const maxRetries = 3;

    // Helper to sync ref and state
    const updateStatus = useCallback((newStatus) => {
        setStatus(newStatus);
        statusRef.current = newStatus;
    }, []);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (wsRef.current) {
                wsRef.current.close();
                wsRef.current = null;
            }
        };
    }, []);

    const connect = useCallback((jobId, onUpdate) => {
        // Close existing connection
        if (wsRef.current) {
            wsRef.current.close();
        }

        retriesRef.current = 0;
        updateStatus('queued');
        setProgress(0);
        setError(null);

        const wsUrl = `ws://${window.location.hostname}:8000/api/v1/ws/${jobId}`;

        const createWs = () => {
            const ws = new WebSocket(wsUrl);
            wsRef.current = ws;

            ws.onopen = () => {
                retriesRef.current = 0;
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);

                    if (data.status) updateStatus(data.status);
                    if (data.progress_pct !== undefined) setProgress(data.progress_pct);

                    if (data.status === 'error') {
                        setError(data.error || { message: 'Unknown error' });
                    }

                    // Forward to callback
                    if (onUpdate) onUpdate(data);

                    // Close on terminal states
                    if (data.status === 'complete' || data.status === 'error') {
                        ws.close();
                    }
                } catch (err) {
                    console.error('Failed to parse WS message:', err);
                }
            };

            ws.onerror = (event) => {
                console.error('WebSocket error:', event);
            };

            ws.onclose = (event) => {
                // Reconnect if not a clean close and not terminal
                if (
                    !event.wasClean &&
                    retriesRef.current < maxRetries &&
                    !['complete', 'error'].includes(statusRef.current)
                ) {
                    retriesRef.current += 1;
                    const delay = Math.min(1000 * Math.pow(2, retriesRef.current), 8000);
                    setTimeout(createWs, delay);
                }
            };
        };

        createWs();
    }, []);

    const disconnect = useCallback(() => {
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
        updateStatus('idle');
        setProgress(0);
        setError(null);
    }, [updateStatus]);

    return {
        status,
        progress,
        error,
        connect,
        disconnect,
    };
}
