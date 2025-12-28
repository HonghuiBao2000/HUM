#!/usr/bin/env python3
"""
HUM Comprehensive Logger Framework
Unified logging system, integrating all types of output
"""

import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json
from pathlib import Path

# Try to import torch and metrics, skip if not installed
try:
    import torch
    from metrics import cal_recall, cal_ndcg
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class HUMLogger:
    """
    HUM Unified Logger
    Integrates training, evaluation, debugging, and all other types of log output
    """
    
    def __init__(self, 
                 rank: int = 0, 
                 log_dir: str = "logs",
                 log_level: str = "INFO",
                 enable_file_logging: bool = True,
                 enable_console_logging: bool = True):
        """
        Initialize HUM Logger
        
        Args:
            rank: Process rank, only rank=0 will output logs
            log_dir: Log file directory
            log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            enable_file_logging: Whether to enable file logging
            enable_console_logging: Whether to enable console logging
        """
        self.rank = rank
        self.log_dir = log_dir
        self.enable_file_logging = enable_file_logging
        self.enable_console_logging = enable_console_logging
        
        # Create log directory
        if self.enable_file_logging:
            os.makedirs(log_dir, exist_ok=True)
        
        # Set up logger
        self.logger = self._setup_logger(log_level)
        
        # Store different types of result data
        self.training_metrics = defaultdict(list)
        self.evaluation_results = defaultdict(dict)
        self.debug_info = defaultdict(list)
        
        # Define standard domain order
        self.standard_domain_order = [
            'Industrial_and_Scientific',
            'Automotive', 
            'Tools_and_Home_Improvement',
            'Office_Products',
            'Books',
            'CDs_and_Vinyl'
        ]
        
        # Record startup info during initialization
        self.info("HUM Logger initialized", extra={"category": "system"})
        
    def _setup_logger(self, log_level: str) -> logging.Logger:
        """Set up logger configuration"""
        logger = logging.getLogger(f'HUMLogger_Rank{self.rank}')
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # 创建formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(category)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台输出
        if self.enable_console_logging and self.rank == 0:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File output
        if self.enable_file_logging:
            # Main log file
            main_log_file = os.path.join(self.log_dir, f'hum_main_rank{self.rank}.log')
            file_handler = logging.FileHandler(main_log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # Training log file
            train_log_file = os.path.join(self.log_dir, f'hum_training_rank{self.rank}.log')
            train_handler = logging.FileHandler(train_log_file, encoding='utf-8')
            train_handler.setFormatter(formatter)
            train_handler.addFilter(lambda record: record.category == 'training')
            logger.addHandler(train_handler)
            
            # Evaluation log file
            eval_log_file = os.path.join(self.log_dir, f'hum_evaluation_rank{self.rank}.log')
            eval_handler = logging.FileHandler(eval_log_file, encoding='utf-8')
            eval_handler.setFormatter(formatter)
            eval_handler.addFilter(lambda record: record.category == 'evaluation')
            logger.addHandler(eval_handler)
        
        return logger
    
    def _log(self, level: str, message: str, category: str = "general", **kwargs):
        """Internal logging method"""
        if self.rank != 0:
            return
            
        extra = {"category": category}
        extra.update(kwargs)
        
        log_method = getattr(self.logger, level.lower())
        log_method(message, extra=extra)
    
    # ==================== Basic Logging Methods ====================
    
    def debug(self, message: str, category: str = "general", **kwargs):
        """调试信息"""
        self._log("DEBUG", message, category, **kwargs)
    
    def info(self, message: str, category: str = "general", **kwargs):
        """General information"""
        self._log("INFO", message, category, **kwargs)
    
    def warning(self, message: str, category: str = "general", **kwargs):
        """Warning information"""
        self._log("WARNING", message, category, **kwargs)
    
    def error(self, message: str, category: str = "general", **kwargs):
        """Error information"""
        self._log("ERROR", message, category, **kwargs)
    
    def critical(self, message: str, category: str = "general", **kwargs):
        """Critical error"""
        self._log("CRITICAL", message, category, **kwargs)
    
    # ==================== System Related Logs ====================
    
    def log_system_info(self, message: str):
        """System information"""
        self.info(f"🔧 {message}", category="system")
    
    def log_model_info(self, message: str):
        """Model information"""
        self.info(f"🤖 {message}", category="model")
    
    def log_gpu_info(self, message: str):
        """GPU information"""
        self.info(f"💻 {message}", category="gpu")
    
    def log_data_info(self, message: str):
        """Data information"""
        self.info(f"📊 {message}", category="data")
    
    # ==================== Training Related Logs ====================
    
    def log_training_start(self, epoch: int, total_epochs: int):
        """Training start"""
        self.info(f"🚀 Starting Training - Epoch {epoch}/{total_epochs}", 
                 category="training", epoch=epoch, total_epochs=total_epochs)
    
    def log_training_end(self, epoch: int, final_loss: float):
        """Training end"""
        self.info(f"✅ Training Complete - Epoch {epoch}, Final Loss: {final_loss:.6f}", 
                 category="training", epoch=epoch, final_loss=final_loss)
    
    def log_training_step(self, epoch: int, step: int, total_steps: int, 
                         loss: float, lr: float = None, **metrics):
        """Training step"""
        message = f"Epoch {epoch} Step {step}/{total_steps} - Loss: {loss:.6f}"
        if lr is not None:
            message += f", LR: {lr:.2e}"
        
        for metric_name, metric_value in metrics.items():
            message += f", {metric_name}: {metric_value:.6f}"
        
        self.info(message, category="training", 
                 epoch=epoch, step=step, total_steps=total_steps, 
                 loss=loss, lr=lr, **metrics)
        
        # Store training metrics
        self.training_metrics[epoch].append({
            "step": step,
            "loss": loss,
            "lr": lr,
            **metrics
        })
    
    def log_validation_start(self, epoch: int):
        """Validation start"""
        self.info(f"🔍 Starting Validation - Epoch {epoch}", 
                 category="evaluation", epoch=epoch)
    
    def log_validation_end(self, epoch: int, results: Dict[str, float]):
        """Validation end"""
        results_str = ", ".join([f"{k}: {v:.6f}" for k, v in results.items()])
        self.info(f"✅ Validation Complete - Epoch {epoch} - {results_str}", 
                 category="evaluation", epoch=epoch, **results)
    
    # ==================== Evaluation Related Logs ====================
    
    def log_section_header(self, section_name: str):
        """Record section header"""
        separator = "=" * 60
        self.info(f"\n{separator}", category="evaluation")
        self.info(f" {section_name.upper()} ", category="evaluation")
        self.info(f"{separator}", category="evaluation")
    
    def log_section_footer(self):
        """Record section footer"""
        separator = "-" * 60
        self.info(f"{separator}", category="evaluation")
    
    def log_domain_results(self, domain: str, sample_count: int, metrics: Dict[str, float]):
        """Record domain results"""
        domain_header = f"-----------------{domain}-----------------"
        self.info(domain_header, category="evaluation")
        self.info(f"There are {sample_count} samples in the domain of {domain}", 
                 category="evaluation")
        
        # Format metrics output
        metrics_str = " ---- ".join([f"{k}:{v:.6f}" for k, v in metrics.items()])
        self.info(metrics_str, category="evaluation")
        
        # Store results
        if domain not in self.evaluation_results:
            self.evaluation_results[domain] = {}
        self.evaluation_results[domain].update(metrics)
    
    def evaluate_by_domain(self, predict_score, label, id2domain_test, result_type="Main"):
        """Evaluate by domain and record results"""
        if not TORCH_AVAILABLE:
            self.warning("Torch not available, skipping evaluation", category="evaluation")
            return {}
        
        # Group by domain and calculate metrics
        domain_results = {}
        for domain in self.standard_domain_order:
            if domain not in id2domain_test:
                continue
                
            domain_indices = id2domain_test[domain]
            if len(domain_indices) == 0:
                continue
            
            # Extract predictions and labels for this domain
            domain_predict = predict_score[domain_indices]
            domain_label = label[domain_indices]
            
            # Calculate metrics
            recall_5 = cal_recall(domain_predict, domain_label, 5)
            recall_10 = cal_recall(domain_predict, domain_label, 10)
            ndcg_5 = cal_ndcg(domain_predict, domain_label, 5)
            ndcg_10 = cal_ndcg(domain_predict, domain_label, 10)
            
            metrics = {
                "Recall@5": recall_5,
                "Recall@10": recall_10,
                "NDCG@5": ndcg_5,
                "NDCG@10": ndcg_10
            }
            
            domain_results[domain] = metrics
            
            # Record results
            self.log_domain_results(domain, len(domain_indices), metrics)
        
        return domain_results
    
    def log_summary_table(self, result_type: str, results: Dict[str, Dict[str, float]]):
        """Record summary table"""
        self.info(f"\n============================================================", 
                 category="evaluation")
        self.info(f" SUMMARY TABLE - {result_type} ", 
                 category="evaluation")
        self.info(f"============================================================", 
                 category="evaluation")
        
        # Create table header
        header = f"{'Domain':<30} {'Recall@5':<10} {'Recall@10':<12} {'NDCG@5':<10} {'NDCG@10':<12}"
        self.info(header, category="evaluation")
        self.info("-" * 80, category="evaluation")
        
        # Record results for each domain
        for domain in self.standard_domain_order:
            if domain in results:
                metrics = results[domain]
                row = f"{domain:<30} {metrics.get('Recall@5', 0):<10.6f} {metrics.get('Recall@10', 0):<12.6f} {metrics.get('NDCG@5', 0):<10.6f} {metrics.get('NDCG@10', 0):<12.6f}"
                self.info(row, category="evaluation")
    
    # ==================== Debugging Related Logs ====================
    
    def log_debug_info(self, message: str, **kwargs):
        """调试信息"""
        self.debug(f"🔍 {message}", category="debug", **kwargs)
    
    def log_tensor_info(self, tensor_name: str, tensor_shape: tuple, tensor_dtype: str = None):
        """Tensor information"""
        shape_str = "x".join(map(str, tensor_shape))
        dtype_str = f", dtype: {tensor_dtype}" if tensor_dtype else ""
        self.debug(f"📦 {tensor_name}: shape={shape_str}{dtype_str}", 
                  category="tensor", tensor_name=tensor_name, 
                  tensor_shape=tensor_shape, tensor_dtype=tensor_dtype)
    
    def log_memory_usage(self, message: str):
        """Memory usage information"""
        self.debug(f"💾 {message}", category="memory")
    
    # ==================== Configuration and Parameter Logs ====================
    
    def log_config(self, config_dict: Dict[str, Any]):
        """Record configuration information"""
        self.info("📋 Configuration:", category="config")
        for key, value in config_dict.items():
            self.info(f"  {key}: {value}", category="config")
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """Record hyperparameters"""
        self.info("⚙️ Hyperparameters:", category="hyperparams")
        for key, value in hyperparams.items():
            self.info(f"  {key}: {value}", category="hyperparams")
    
    # ==================== Performance Monitoring ====================
    
    def log_performance_metrics(self, metrics: Dict[str, float]):
        """Record performance metrics"""
        self.info("📈 Performance Metrics:", category="performance")
        for metric_name, metric_value in metrics.items():
            self.info(f"  {metric_name}: {metric_value:.4f}", category="performance")
    
    def log_timing_info(self, operation: str, duration: float):
        """Record timing information"""
        self.info(f"⏱️ {operation}: {duration:.4f}s", category="timing", 
                 operation=operation, duration=duration)
    
    # ==================== File Operation Logs ====================
    
    def log_file_operation(self, operation: str, file_path: str, success: bool = True):
        """Record file operation"""
        status = "✅" if success else "❌"
        self.info(f"{status} {operation}: {file_path}", 
                 category="file", operation=operation, 
                 file_path=file_path, success=success)
    
    def log_model_save(self, model_path: str, epoch: int):
        """Record model save"""
        self.info(f"💾 Model saved to {model_path} (Epoch {epoch})", 
                 category="model", model_path=model_path, epoch=epoch)
    
    def log_model_load(self, model_path: str):
        """Record model load"""
        self.info(f"📂 Model loaded from {model_path}", 
                 category="model", model_path=model_path)
    
    # ==================== Data Export Functionality ====================
    
    def export_training_metrics(self, file_path: str):
        """Export training metrics to JSON file"""
        if self.rank != 0:
            return
            
        export_data = {
            "training_metrics": dict(self.training_metrics),
            "evaluation_results": dict(self.evaluation_results),
            "export_time": datetime.now().isoformat(),
            "rank": self.rank
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.log_file_operation("Export training metrics", file_path)
    
    def export_evaluation_results(self, file_path: str):
        """Export evaluation results to JSON file"""
        if self.rank != 0:
            return
            
        export_data = {
            "evaluation_results": dict(self.evaluation_results),
            "export_time": datetime.now().isoformat(),
            "rank": self.rank
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.log_file_operation("Export evaluation results", file_path)
    
    # ==================== Compatibility Methods ====================
    
    def print_rank0(self, message: str):
        """Compatibility with original print_rank0 function"""
        self.info(message, category="legacy")
    
    def log(self, message: str, category: str = "general"):
        """Simplified logging method"""
        self.info(message, category=category)
    
    # ==================== Cleanup and Close ====================
    
    def close(self):
        """Close logger"""
        if self.rank == 0:
            self.info("HUM Logger closing", category="system")
        
        # Close all handlers
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)


# ==================== Global Logger Instance ====================
_global_logger = None

def get_global_logger() -> Optional[HUMLogger]:
    """Get global logger instance"""
    return _global_logger

def set_global_logger(logger: HUMLogger):
    """Set global logger instance"""
    global _global_logger
    _global_logger = logger

def init_logger(rank: int = 0, **kwargs) -> HUMLogger:
    """Initialize global logger"""
    global _global_logger
    _global_logger = HUMLogger(rank=rank, **kwargs)
    return _global_logger


class EvaluationLogger:
    """Professional evaluation results logger (merged from evaluation_logger.py)"""
    
    def __init__(self, rank: int = 0, log_file: Optional[str] = None):
        """
        Initialize evaluation logger
        
        Args:
            rank: Process rank, only rank=0 will output logs
            log_file: Log file path, if None only output to console
        """
        self.rank = rank
        self.logger = self._setup_logger(log_file)
        
        # 定义标准的domain顺序
        self.standard_domain_order = [
            'Industrial_and_Scientific',
            'Automotive', 
            'Tools_and_Home_Improvement',
            'Office_Products',
            'Books',
            'CDs_and_Vinyl'
        ]
        
        # Store all result data
        self.results_data = defaultdict(dict)
        
    def _setup_logger(self, log_file: Optional[str] = None) -> logging.Logger:
        """Set up logger"""
        logger = logging.getLogger(f'EvaluationLogger_Rank{self.rank}')
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 创建formatter
        formatter = logging.Formatter('%(message)s')
        
        # 控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File output (if specified)
        if log_file and self.rank == 0:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Prevent duplicate logs
        logger.propagate = False
        
        return logger
    
    def log(self, message: str):
        """Only rank=0 process outputs logs"""
        if self.rank == 0:
            self.logger.info(message)
    
    def log_section_header(self, title: str):
        """Output section header"""
        self.log("")
        self.log("=" * 60)
        self.log(f" {title} ")
        self.log("=" * 60)
    
    def log_section_footer(self):
        """Output section footer"""
        self.log("=" * 60)
        self.log("")
    
    def log_subsection_header(self, title: str):
        """Output subsection header"""
        self.log("")
        self.log("-" * 40)
        self.log(f" {title} ")
        self.log("-" * 40)
    
    def evaluate_by_domain(self, predict_score, label, 
                          id2domain_test: Dict[str, List[int]], 
                          result_type: str = "", 
                          domain_order: Optional[List[str]] = None):
        """
        Evaluate by domain and record results
        
        Args:
            predict_score: Prediction scores
            label: True labels
            id2domain_test: Mapping from domain to sample indices
            result_type: Result type name
            domain_order: Domain order, if None use standard order
            
        Returns:
            Result data dictionary
        """
        if not TORCH_AVAILABLE:
            self.log("Warning: torch not available, skipping evaluation")
            return {}
            
        if domain_order is None:
            domain_order = self.standard_domain_order
            
        results = {}
        
        # Output domain results in specified order
        for domain_key in domain_order:
            if domain_key in id2domain_test:
                indices = id2domain_test[domain_key]
                domain_scores = predict_score[indices].cpu()
                domain_label = label[indices].cpu()
                
                recall = cal_recall(domain_label, domain_scores, [1, 5, 10, 20])
                ndcg = cal_ndcg(domain_label, domain_scores, [1, 5, 10, 20])
                
                self.log(f"-----------------{domain_key}-----------------")
                self.log(f"There are {len(domain_label)} samples in the domain of {domain_key}")
                self.log(f"Recall@5:{recall[1]} ---- Recall@10:{recall[2]}")
                self.log(f"NDCG@5:{ndcg[1]} ---- NDCG@10:{ndcg[2]}")
                
                # Save result data
                results[domain_key] = {
                    'recall_5': recall[1],
                    'recall_10': recall[2],
                    'ndcg_5': ndcg[1],
                    'ndcg_10': ndcg[2],
                    'sample_count': len(domain_label)
                }
        
        # Store in total results
        if result_type:
            self.results_data[result_type] = results
            
        return results
    
    def log_summary_table(self):
        """Output summary table"""
        self.log_section_header("SUMMARY TABLE - Key Metrics")
        
        # 表头
        header = f"{'Domain':<25} {'Main':<25} {'Long-tail':<25} {'Non-Long-tail':<25} {'Non-Hetero':<25} {'Hetero':<25}"
        self.log(header)
        self.log("-" * 150)
        
        # 数据行
        for domain in self.standard_domain_order:
            row = f"{domain:<25}"
            for result_type in ['Main', 'Long-tail', 'Non-Long-tail', 'Non-Hetero', 'Hetero']:
                if domain in self.results_data.get(result_type, {}):
                    data = self.results_data[result_type][domain]
                    metrics = f"R@5:{data['recall_5']:.3f} R@10:{data['recall_10']:.3f}"
                    row += f" {metrics:<24}"
                else:
                    row += f" {'N/A':<24}"
            self.log(row)
        
        self.log("-" * 150)
        self.log("Legend: R@5=Recall@5, R@10=Recall@10")
        self.log_section_footer()
    
    def log_detailed_summary_table(self):
        """Output detailed summary table (including NDCG)"""
        self.log_section_header("DETAILED SUMMARY TABLE - All Metrics")
        
        # 表头
        header = f"{'Domain':<25} {'Main':<35} {'Long-tail':<35} {'Non-Long-tail':<35} {'Non-Hetero':<35} {'Hetero':<35}"
        self.log(header)
        self.log("-" * 200)
        
        # 数据行
        for domain in self.standard_domain_order:
            row = f"{domain:<25}"
            for result_type in ['Main', 'Long-tail', 'Non-Long-tail', 'Non-Hetero', 'Hetero']:
                if domain in self.results_data.get(result_type, {}):
                    data = self.results_data[result_type][domain]
                    metrics = f"R@5:{data['recall_5']:.3f} R@10:{data['recall_10']:.3f} N@5:{data['ndcg_5']:.3f} N@10:{data['ndcg_10']:.3f}"
                    row += f" {metrics:<34}"
                else:
                    row += f" {'N/A':<34}"
            self.log(row)
        
        self.log("-" * 200)
        self.log("Legend: R@5=Recall@5, R@10=Recall@10, N@5=NDCG@5, N@10=NDCG@10")
        self.log_section_footer()
    
    def log_length_analysis(self, predict_score, label, 
                           dataloader, length_filter_fn, standard_domain_order: List[str]):
        """Record length analysis results"""
        self.log_subsection_header("Length Analysis")
        
        length_types = [
            ("Short (0-3)", 0),
            ("Medium (4-6)", 1), 
            ("Long (7-10)", 2)
        ]
        
        for length_name, keep_length in length_types:
            self.log(f"--- {length_name} Sequences ---")
            filter_fn = lambda idx, sample: length_filter_fn(idx, sample, keep_length)
            id2domain_length = self._build_id2domain_test(dataloader, filter_fn)
            self.evaluate_by_domain(predict_score, label, id2domain_length, length_name, standard_domain_order)
    
    def log_cold_start_analysis(self, predict_score, label,
                               dataloader, cold_warm_filter_fn, train_single_domain_iid, standard_domain_order: List[str]):
        """Record cold start analysis results"""
        self.log_subsection_header("Cold Start Analysis")
        
        # Cold Start
        filter_fn = lambda idx, sample: cold_warm_filter_fn(idx, sample, train_single_domain_iid, keep_cold=True)
        id2domain_cold = self._build_id2domain_test(dataloader, filter_fn)
        self.evaluate_by_domain(predict_score, label, id2domain_cold, "Cold Start", standard_domain_order)
        
        # Warm Start
        filter_fn = lambda idx, sample: cold_warm_filter_fn(idx, sample, train_single_domain_iid, keep_cold=False)
        id2domain_warm = self._build_id2domain_test(dataloader, filter_fn)
        self.evaluate_by_domain(predict_score, label, id2domain_warm, "Warm Start", standard_domain_order)
    
    def _build_id2domain_test(self, dataloader, filter_fn=None):
        """
        构建domain到样本索引的映射
        """
        import collections
        id2domain_test = collections.defaultdict(list)
        for idx, sample in enumerate(dataloader.dataset.data):
            if filter_fn is None or filter_fn(idx, sample):
                domain = sample[-1]
                id2domain_test[domain].append(idx)
        return id2domain_test
    
    def clear_results(self):
        """Clear stored result data"""
        self.results_data.clear()

# ==================== Compatibility Functions ====================

def print_rank0(msg: str, rank: int = 0):
    """兼容原有的print_rank0函数"""
    if rank == 0:
        if _global_logger:
            _global_logger.print_rank0(msg)
        else:
            print(msg)


if __name__ == "__main__":
    # Test logger functionality
    logger = RMDRLogger(rank=0, log_dir="test_logs")
    
    # Test various log types
    logger.log_system_info("System initialized")
    logger.log_model_info("Model loaded successfully")
    logger.log_training_start(1, 10)
    logger.log_training_step(1, 100, 1000, 0.5, lr=1e-4, accuracy=0.95)
    logger.log_validation_start(1)
    logger.log_validation_end(1, {"loss": 0.4, "accuracy": 0.92})
    
    # Test evaluation logs
    logger.log_section_header("Test Results")
    logger.log_domain_results("Test_Domain", 1000, {"Recall@5": 0.1, "NDCG@5": 0.05})
    logger.log_section_footer()
    
    # Export data
    logger.export_training_metrics("test_training_metrics.json")
    logger.export_evaluation_results("test_evaluation_results.json")
    
    logger.close()
    print("Logger test completed successfully!")
