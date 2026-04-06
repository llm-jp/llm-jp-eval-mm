export interface GpuInfo {
  id: number;
  name: string;
  /** 0-100 */
  utilization: number;
  /** MiB used */
  memoryUsed: number;
  /** MiB total */
  memoryTotal: number;
  /** Celsius */
  temperature: number;
  /** "idle" | "active" | "high" */
  status: "idle" | "active" | "high";
}

export const MOCK_GPUS: GpuInfo[] = [
  {
    id: 0,
    name: "NVIDIA A100 80GB",
    utilization: 45,
    memoryUsed: 34_200,
    memoryTotal: 81_920,
    temperature: 62,
    status: "active",
  },
  {
    id: 1,
    name: "NVIDIA A100 80GB",
    utilization: 0,
    memoryUsed: 512,
    memoryTotal: 81_920,
    temperature: 34,
    status: "idle",
  },
  {
    id: 2,
    name: "NVIDIA A100 80GB",
    utilization: 92,
    memoryUsed: 76_800,
    memoryTotal: 81_920,
    temperature: 81,
    status: "high",
  },
  {
    id: 3,
    name: "NVIDIA A100 80GB",
    utilization: 28,
    memoryUsed: 22_400,
    memoryTotal: 81_920,
    temperature: 51,
    status: "active",
  },
];
