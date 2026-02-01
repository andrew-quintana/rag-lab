"""I/O utilities for loading/saving datasets and artifacts"""

import json
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DataLoader:
    """Utility class for loading and saving datasets and artifacts"""
    
    def __init__(self, base_path: str = "."):
        """
        Initialize DataLoader with base path for data operations
        
        Args:
            base_path: Base path for data files (default: current directory)
        """
        self.base_path = Path(base_path)
    
    def load_corpus(self, file_path: str) -> pd.DataFrame:
        """
        Load document corpus from parquet file
        
        Args:
            file_path: Path to corpus.parquet file
            
        Returns:
            DataFrame with document corpus
        """
        path = self.base_path / file_path
        logger.info(f"Loading corpus from {path}")
        return pd.read_parquet(path)
    
    def save_corpus(self, corpus: pd.DataFrame, file_path: str) -> None:
        """
        Save document corpus to parquet file
        
        Args:
            corpus: DataFrame with document corpus
            file_path: Path to save corpus.parquet file
        """
        path = self.base_path / file_path
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving corpus to {path}")
        corpus.to_parquet(path)
    
    def load_tasks(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load evaluation tasks from JSONL file
        
        Args:
            file_path: Path to tasks.jsonl file
            
        Returns:
            List of task dictionaries
        """
        path = self.base_path / file_path
        logger.info(f"Loading tasks from {path}")
        tasks = []
        with open(path, 'r') as f:
            for line in f:
                tasks.append(json.loads(line.strip()))
        return tasks
    
    def save_tasks(self, tasks: List[Dict[str, Any]], file_path: str) -> None:
        """
        Save evaluation tasks to JSONL file
        
        Args:
            tasks: List of task dictionaries
            file_path: Path to save tasks.jsonl file
        """
        path = self.base_path / file_path
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving {len(tasks)} tasks to {path}")
        with open(path, 'w') as f:
            for task in tasks:
                f.write(json.dumps(task) + '\n')
    
    def load_embeddings(self, file_path: str) -> np.ndarray:
        """
        Load embeddings from numpy file
        
        Args:
            file_path: Path to embeddings.npy file
            
        Returns:
            Numpy array with embeddings
        """
        path = self.base_path / file_path
        logger.info(f"Loading embeddings from {path}")
        return np.load(path)
    
    def save_embeddings(self, embeddings: np.ndarray, file_path: str) -> None:
        """
        Save embeddings to numpy file
        
        Args:
            embeddings: Numpy array with embeddings
            file_path: Path to save embeddings.npy file
        """
        path = self.base_path / file_path
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving embeddings shape {embeddings.shape} to {path}")
        np.save(path, embeddings)
    
    def load_docstore(self, file_path: str) -> pd.DataFrame:
        """
        Load docstore mapping chunk_id -> text/metadata from parquet
        
        Args:
            file_path: Path to docstore.parquet file
            
        Returns:
            DataFrame with chunk_id, text, and metadata columns
        """
        path = self.base_path / file_path
        logger.info(f"Loading docstore from {path}")
        return pd.read_parquet(path)
    
    def save_docstore(self, docstore: pd.DataFrame, file_path: str) -> None:
        """
        Save docstore mapping to parquet file
        
        Args:
            docstore: DataFrame with chunk_id, text, and metadata columns
            file_path: Path to save docstore.parquet file
        """
        path = self.base_path / file_path
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving docstore with {len(docstore)} chunks to {path}")
        docstore.to_parquet(path)
    
    def load_faiss_index(self, file_path: str):
        """
        Load FAISS index from file
        
        Args:
            file_path: Path to faiss.index file
            
        Returns:
            FAISS index object
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("FAISS is required. Install with: pip install faiss-cpu")
        
        path = self.base_path / file_path
        logger.info(f"Loading FAISS index from {path}")
        return faiss.read_index(str(path))
    
    def save_faiss_index(self, index, file_path: str) -> None:
        """
        Save FAISS index to file
        
        Args:
            index: FAISS index object
            file_path: Path to save faiss.index file
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("FAISS is required. Install with: pip install faiss-cpu")
        
        path = self.base_path / file_path
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving FAISS index to {path}")
        faiss.write_index(index, str(path))


class RunManager:
    """Manages evaluation run directories and artifacts"""
    
    def __init__(self, runs_base_path: str = "runs"):
        """
        Initialize RunManager
        
        Args:
            runs_base_path: Base path for runs directory
        """
        self.runs_base_path = Path(runs_base_path)
        self.runs_base_path.mkdir(parents=True, exist_ok=True)
    
    def create_run_dir(self, run_name: Optional[str] = None) -> str:
        """
        Create new run directory with timestamp
        
        Args:
            run_name: Optional custom run name (default: auto-generated with timestamp)
            
        Returns:
            Path to created run directory
        """
        if run_name is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
            run_name = f"{timestamp}_eval_run"
        
        run_dir = self.runs_base_path / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created run directory: {run_dir}")
        return str(run_dir)
    
    def save_config(self, config: Dict[str, Any], run_dir: str) -> None:
        """
        Save run configuration to YAML file
        
        Args:
            config: Configuration dictionary
            run_dir: Run directory path
        """
        config_path = Path(run_dir) / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Saved config to {config_path}")
    
    def load_config(self, run_dir: str) -> Dict[str, Any]:
        """
        Load run configuration from YAML file
        
        Args:
            run_dir: Run directory path
            
        Returns:
            Configuration dictionary
        """
        config_path = Path(run_dir) / "config.yaml"
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def save_outputs(self, outputs: List[Dict[str, Any]], run_dir: str) -> None:
        """
        Save evaluation outputs to JSONL file
        
        Args:
            outputs: List of output dictionaries
            run_dir: Run directory path
        """
        outputs_path = Path(run_dir) / "outputs.jsonl"
        with open(outputs_path, 'w') as f:
            for output in outputs:
                f.write(json.dumps(output, default=str) + '\n')
        logger.info(f"Saved {len(outputs)} outputs to {outputs_path}")
    
    def load_outputs(self, run_dir: str) -> List[Dict[str, Any]]:
        """
        Load evaluation outputs from JSONL file
        
        Args:
            run_dir: Run directory path
            
        Returns:
            List of output dictionaries
        """
        outputs_path = Path(run_dir) / "outputs.jsonl"
        outputs = []
        with open(outputs_path, 'r') as f:
            for line in f:
                outputs.append(json.loads(line.strip()))
        return outputs
    
    def save_metrics(self, metrics: Dict[str, Any], run_dir: str) -> None:
        """
        Save evaluation metrics to JSON file
        
        Args:
            metrics: Metrics dictionary
            run_dir: Run directory path
        """
        metrics_path = Path(run_dir) / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"Saved metrics to {metrics_path}")
    
    def load_metrics(self, run_dir: str) -> Dict[str, Any]:
        """
        Load evaluation metrics from JSON file
        
        Args:
            run_dir: Run directory path
            
        Returns:
            Metrics dictionary
        """
        metrics_path = Path(run_dir) / "metrics.json"
        with open(metrics_path, 'r') as f:
            return json.load(f)
    
    def save_traces(self, traces: List[Dict[str, Any]], run_dir: str) -> None:
        """
        Save execution traces to JSONL file
        
        Args:
            traces: List of trace dictionaries
            run_dir: Run directory path
        """
        traces_path = Path(run_dir) / "traces.jsonl"
        with open(traces_path, 'w') as f:
            for trace in traces:
                f.write(json.dumps(trace, default=str) + '\n')
        logger.info(f"Saved {len(traces)} traces to {traces_path}")
    
    def load_traces(self, run_dir: str) -> List[Dict[str, Any]]:
        """
        Load execution traces from JSONL file
        
        Args:
            run_dir: Run directory path
            
        Returns:
            List of trace dictionaries
        """
        traces_path = Path(run_dir) / "traces.jsonl"
        traces = []
        with open(traces_path, 'r') as f:
            for line in f:
                traces.append(json.loads(line.strip()))
        return traces
    
    def list_runs(self) -> List[str]:
        """
        List all available run directories
        
        Returns:
            List of run directory names
        """
        return [d.name for d in self.runs_base_path.iterdir() if d.is_dir()]


# Convenience functions for common operations
def load_evaluation_dataset(file_path: str, base_path: str = ".") -> List[Dict[str, Any]]:
    """Load evaluation dataset from JSONL file"""
    loader = DataLoader(base_path)
    return loader.load_tasks(file_path)


def save_evaluation_results(results: List[Dict[str, Any]], run_dir: str) -> None:
    """Save evaluation results to run directory"""
    run_manager = RunManager()
    run_manager.save_outputs(results, run_dir)


def create_new_run(run_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> str:
    """Create new evaluation run with optional config"""
    run_manager = RunManager()
    run_dir = run_manager.create_run_dir(run_name)
    
    if config:
        run_manager.save_config(config, run_dir)
    
    return run_dir