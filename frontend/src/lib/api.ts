export interface MetricValue {
    timestamp: number;
    value: number;
}

export interface DashboardData {
    frequency: Record<string, number>;
    timeline: MetricValue[];
    buckets: Record<string, number>;
    raw_values: number[];
}

export const fetchDashboardData = async (duration: number = 60, tier?: string, bt?: string): Promise<DashboardData> => {
    const params = new URLSearchParams({ duration: duration.toString() });
    if (tier) params.append('tier', tier);
    if (bt) params.append('bt', bt);
    const response = await fetch(`/api/data?${params}`);
    if (!response.ok) {
        throw new Error('Failed to fetch data');
    }
    return response.json();
};
