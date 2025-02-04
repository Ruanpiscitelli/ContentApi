from prometheus_client import start_http_server, Gauge
from typing import Dict

GPU_UTILIZATION = Gauge('gpu_utilization', 'Current GPU utilization', ['gpu_id'])
GPU_MEMORY = Gauge('gpu_memory_used', 'VRAM used in MB', ['gpu_id'])

class MetricsExporter:
    def __init__(self, port=8000):
        self.port = port
        
    async def start(self):
        start_http_server(self.port)
        
    def update_metrics(self, gpu_data: Dict[str, Dict]):
        for gpu_id, metrics in gpu_data.items():
            if metrics['status'] == 'OK':
                GPU_UTILIZATION.labels(gpu_id).set(metrics['utilization'])
                GPU_MEMORY.labels(gpu_id).set(metrics['memory']['used']) 