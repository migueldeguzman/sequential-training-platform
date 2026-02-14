import torch
import numpy as np
import transformers
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import uuid
import json
import time

from backend.profiling.pipeline_profiler import InferencePipelineProfiler
from backend.profiling.power_monitor import PowerMonitor
from backend.profiling.layer_profiler import LayerProfiler
from backend.profiling.database import ProfileDatabase

@dataclass
class BatchSizeExperimentResult:
    run_id: str
    model_name: str
    batch_size: int
    total_energy_mj: float
    avg_power_mw: float
    energy_per_token_mj: float
    latency_ms: float
    throughput_tokens_per_sec: float

class BatchSizeEnergyExperiment:
    def __init__(self, model_name: str, batch_sizes: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256]):
        self.model_name = model_name
        self.batch_sizes = batch_sizes
        self.results: List[BatchSizeExperimentResult] = []
        
        # Load model and tokenizer
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        
        # Database for storing results
        self.database = ProfileDatabase()
        
        # Ensure model is on MPS device
        self.device = torch.device('mps')
        self.model.to(self.device)
        
    def _generate_standard_prompt(self, length: int = 128) -> str:
        """Generate a standard prompt for consistent testing."""
        base_prompt = "The future of artificial intelligence is deeply intertwined with our understanding of human cognition. As machine learning models become more sophisticated, they begin to exhibit behaviors that challenge our traditional notions of intelligence."
        return base_prompt * (length // len(base_prompt) + 1)[:length]
    
    def run_batch_size_experiment(self, prompt: str, iterations: int = 5):
        """Run energy profiling for different batch sizes."""
        for batch_size in self.batch_sizes:
            batch_results = []
            
            # Generate batch of inputs
            inputs = self.tokenizer(
                [prompt] * batch_size, 
                return_tensors='pt', 
                padding=True, 
                truncation=True
            ).to(self.device)
            
            for _ in range(iterations):
                # Power monitor setup
                power_monitor = PowerMonitor(sample_interval_ms=10)
                layer_profiler = LayerProfiler(self.model)
                
                # Run profiled inference
                with InferencePipelineProfiler(
                    power_monitor=power_monitor, 
                    layer_profiler=layer_profiler
                ) as profiler:
                    with profiler.run(
                        model=self.model, 
                        inputs=inputs, 
                        model_name=self.model_name,
                        experiment_name='BatchSizeEnergyExperiment',
                        tags=[f'batch_size_{batch_size}']
                    ) as session:
                        start_time = time.time()
                        outputs = self.model.generate(
                            inputs.input_ids, 
                            max_new_tokens=32,  # Standard generation length
                            do_sample=False    # Greedy decoding for consistency
                        )
                        end_time = time.time()
                    
                    # Aggregate metrics
                    profiling_data = profiler.aggregate_profiling_data()
                    
                    # Compute additional metrics
                    latency_ms = (end_time - start_time) * 1000
                    total_tokens = batch_size * (inputs.input_ids.shape[1] + outputs.shape[1] - inputs.input_ids.shape[1])
                    throughput_tokens_per_sec = total_tokens / (end_time - start_time)
                    
                    # Create experiment result
                    result = BatchSizeExperimentResult(
                        run_id=str(uuid.uuid4()),
                        model_name=self.model_name,
                        batch_size=batch_size,
                        total_energy_mj=profiling_data.get('energy_summary', {}).get('total_energy_mj', 0),
                        avg_power_mw=profiling_data.get('energy_summary', {}).get('avg_power_mw', 0),
                        energy_per_token_mj=profiling_data.get('energy_summary', {}).get('total_energy_mj', 0) / total_tokens,
                        latency_ms=latency_ms,
                        throughput_tokens_per_sec=throughput_tokens_per_sec
                    )
                    
                    batch_results.append(result)
            
            # Compute and store aggregate results
            avg_result = self._aggregate_batch_results(batch_results)
            self.results.append(avg_result)
            self._save_result_to_database(avg_result)
            
            # Cooldown to stabilize thermal conditions
            time.sleep(30)
    
    def _aggregate_batch_results(self, results: List[BatchSizeExperimentResult]) -> BatchSizeExperimentResult:
        """Compute average metrics across experiment iterations."""
        return BatchSizeExperimentResult(
            run_id=results[0].run_id,  # Use first run_id
            model_name=results[0].model_name,
            batch_size=results[0].batch_size,
            total_energy_mj=np.mean([r.total_energy_mj for r in results]),
            avg_power_mw=np.mean([r.avg_power_mw for r in results]),
            energy_per_token_mj=np.mean([r.energy_per_token_mj for r in results]),
            latency_ms=np.mean([r.latency_ms for r in results]),
            throughput_tokens_per_sec=np.mean([r.throughput_tokens_per_sec for r in results])
        )
    
    def _save_result_to_database(self, result: BatchSizeExperimentResult):
        """Save experiment result to SQLite database."""
        with self.database:
            run_id = self.database.create_run(
                model=result.model_name,
                prompt="Batch Size Energy Experiment",
                tags=['batch_size_energy', f'batch_size_{result.batch_size}']
            )
            
            # Save detailed metrics
            metrics = {
                'batch_size': result.batch_size,
                'total_energy_mj': result.total_energy_mj,
                'avg_power_mw': result.avg_power_mw,
                'energy_per_token_mj': result.energy_per_token_mj,
                'latency_ms': result.latency_ms,
                'throughput_tokens_per_sec': result.throughput_tokens_per_sec
            }
            
            self.database.execute(
                "INSERT INTO batch_size_experiments (run_id, metrics) VALUES (?, ?)",
                (run_id, json.dumps(metrics))
            )
    
    def plot_results(self):
        """Generate plots of energy and performance metrics."""
        # Placeholder for visualization logic
        pass
    
    def export_results(self, filename: str = 'batch_size_energy_results.json'):
        """Export results to JSON for further analysis."""
        with open(filename, 'w') as f:
            json.dump([asdict(result) for result in self.results], f, indent=2)

def main():
    experiment = BatchSizeEnergyExperiment(
        model_name='meta-llama/Llama-2-7b-chat-hf',
        batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256]
    )
    
    prompt = experiment._generate_standard_prompt()
    experiment.run_batch_size_experiment(prompt)
    experiment.export_results()
    experiment.plot_results()

if __name__ == '__main__':
    main()