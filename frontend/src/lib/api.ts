export interface MetricValue {
    timestamp: number;
    value: number;
    time?: number; // For backwards compatibility
}

export interface DashboardData {
    frequency: Record<string, number>;
    timeline: MetricValue[];
    buckets: Record<string, number>;
    raw_values: number[];
    min: number;
    max: number;
}

export const fetchDashboardData = async (
    duration: number = 60,
    tier?: string,
    bt?: string,
    startDate?: Date,
    endDate?: Date
): Promise<DashboardData> => {
    const params = new URLSearchParams({ duration: duration.toString() });
    if (tier) params.append('tier', tier);
    if (bt) params.append('bt', bt);
    if (startDate) params.append('start_date', startDate.toISOString());
    if (endDate) params.append('end_date', endDate.toISOString());
    const response = await fetch(`/api/data?${params}`);
    if (!response.ok) {
        throw new Error('Failed to fetch data');
    }
    return response.json();
};
