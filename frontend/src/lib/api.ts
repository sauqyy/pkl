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

export const fetchDashboardData = async (duration: number = 60): Promise<DashboardData> => {
    const response = await fetch(`/api/data?duration=${duration}`);
    if (!response.ok) {
        throw new Error('Failed to fetch data');
    }
    return response.json();
};
