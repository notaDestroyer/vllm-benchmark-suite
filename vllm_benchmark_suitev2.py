#!/usr/bin/env python3
"""
vLLM Performance Benchmark Suite - Enhanced Edition

Comprehensive benchmarking tool for evaluating vLLM inference performance with:
- Automatic vLLM backend and configuration detection
- Advanced performance metrics (P50/P90/P99, ITL, prefill/decode separation)
- Real-time GPU monitoring with efficiency calculations
- System information collection
- Configurable test parameters via CLI or interactive mode

Author: amit
License: MIT
Version: 2.0
"""

import requests
import time
import threading
from statistics import mean, stdev, median
import json
import matplotlib
import random
import string

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import subprocess
import re
import platform
import argparse
from typing import Dict, List, Optional, Tuple
import sys
import os
import nltk
from collections import Counter
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich import box
from rich.text import Text
from rich.prompt import Prompt, Confirm, IntPrompt
from transformers import AutoTokenizer

# Rich Console for beautiful terminal output
console = Console()

# Configuration Constants
API_BASE_URL = "http://localhost:8000"
API_ENDPOINT = f"{API_BASE_URL}/v1/chat/completions"
API_MODELS_ENDPOINT = f"{API_BASE_URL}/v1/models"
API_HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
API_VERSION_ENDPOINT = f"{API_BASE_URL}/version"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-FP8"
REQUEST_TIMEOUT = 900  # seconds
GPU_POLL_INTERVAL = 0.1  # seconds
TEST_PAUSE_DURATION = 5  # seconds between tests
OUTPUT_DIR = "./outputs"  # Output directory for results and visualizations
DASHBOARD_REFRESH_RATE = 2  # Hz

#establish global random wordlist for sampling
nltk.download('brown', quiet=True)
word_freq = Counter(w.lower() for w in nltk.corpus.brown.words() if w.isalpha())
COMMON_WORDS = [word for word, count in word_freq.most_common(2500)]
DEFAULT_TOKENIZER = "google/gemma3-4b-it"

class SystemInfo:
    """Collect and store system configuration information."""

    @staticmethod
    def get_cuda_version() -> Optional[str]:
        """Get CUDA version from nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                match = re.search(r"CUDA Version: ([\d.]+)", result.stdout)
                if match:
                    return match.group(1)
        except Exception:
            pass
        return None

    @staticmethod
    def get_driver_version() -> Optional[str]:
        """Get NVIDIA driver version."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    @staticmethod
    def get_gpu_name() -> Optional[str]:
        """Get GPU model name."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    @staticmethod
    def get_total_vram() -> Optional[float]:
        """Get total GPU VRAM in GB."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                return float(result.stdout.strip()) / 1024  # Convert MB to GB
        except Exception:
            pass
        return None

    @staticmethod
    def get_system_info() -> Dict:
        """Collect comprehensive system information."""
        return {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cuda_version": SystemInfo.get_cuda_version(),
            "driver_version": SystemInfo.get_driver_version(),
            "gpu_name": SystemInfo.get_gpu_name(),
            "total_vram_gb": SystemInfo.get_total_vram(),
            "timestamp": datetime.now().isoformat(),
        }


class VLLMServerInfo:
    """Query and store vLLM server configuration and capabilities."""

    @staticmethod
    def get_server_info() -> Dict:
        """Retrieve comprehensive vLLM server information."""
        info = {
            "model_name": None,
            "max_model_len": None,
            "backend": None,
            "version": None,
            "quantization": None,
            "tensor_parallel": None,
            "pipeline_parallel": None,
            "max_num_seqs": None,
            "gpu_memory_utilization": None,
            "kv_cache_usage": None,
            "prefix_caching": None,
            "additional_info": {},
        }

        # Try to get model information
        try:
            response = requests.get(API_MODELS_ENDPOINT, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if "data" in data and len(data["data"]) > 0:
                    model_data = data["data"][0]
                    info["model_name"] = model_data.get("id")
                    if "max_model_len" in model_data:
                        info["max_model_len"] = model_data["max_model_len"]
                    # Some vLLM versions expose root with more details
                    if "root" in model_data:
                        info["additional_info"]["root"] = model_data["root"]
                    print(f"[INFO] Model endpoint: {json.dumps(model_data, indent=2)}")
        except Exception as e:
            print(f"[WARNING] Failed to query models endpoint: {e}", file=sys.stderr)

        # Try version endpoint
        try:
            response = requests.get(API_VERSION_ENDPOINT, timeout=5)
            if response.status_code == 200:
                version_data = response.json()
                info["version"] = version_data.get("version")
                print(f"[INFO] vLLM Version: {info['version']}")
        except Exception:
            pass

        # Try metrics endpoint (Prometheus format)
        try:
            metrics_url = f"{API_BASE_URL}/metrics"
            response = requests.get(metrics_url, timeout=5)
            if response.status_code == 200:
                metrics_text = response.text
                print(f"[INFO] Metrics endpoint available")
                
                # Parse key metrics
                for line in metrics_text.split('\n'):
                    if line.startswith('#') or not line.strip():
                        continue
                    
                    # KV cache usage
                    if 'vllm:gpu_cache_usage_perc' in line:
                        try:
                            parts = line.split()
                            if len(parts) >= 2:
                                info["kv_cache_usage"] = float(parts[-1])
                        except:
                            pass
                    
                    # Number of running requests
                    if 'vllm:num_requests_running' in line:
                        try:
                            parts = line.split()
                            if len(parts) >= 2:
                                info["additional_info"]["running_requests"] = int(float(parts[-1]))
                        except:
                            pass
                            
        except Exception as e:
            print(f"[INFO] Metrics endpoint not available: {e}", file=sys.stderr)

        # Try to get server args from health endpoint (some versions expose this)
        try:
            response = requests.get(API_HEALTH_ENDPOINT, timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                
                # Some vLLM versions include server config in health response
                if "model_config" in health_data:
                    config = health_data["model_config"]
                    info["additional_info"]["model_config"] = config
                
                print(f"[INFO] Health: {json.dumps(health_data, indent=2)}")
        except Exception:
            pass

        # Try completions endpoint with special system prompt to get config (last resort)
        try:
            test_data = {
                "model": info["model_name"] or DEFAULT_MODEL_NAME,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
                "logprobs": True
            }
            response = requests.post(API_ENDPOINT, json=test_data, timeout=10)
            if response.status_code == 200:
                resp_data = response.json()
                # Check headers for server info
                if "x-request-id" in response.headers:
                    info["additional_info"]["supports_request_id"] = True
        except Exception:
            pass

        # Infer quantization from model name
        model_name = info["model_name"] or DEFAULT_MODEL_NAME
        if "FP8" in model_name or "fp8" in model_name:
            info["quantization"] = "FP8"
        elif "AWQ" in model_name:
            info["quantization"] = "AWQ"
        elif "GPTQ" in model_name:
            info["quantization"] = "GPTQ"
        elif "INT8" in model_name or "int8" in model_name:
            info["quantization"] = "INT8"
        elif "INT4" in model_name or "int4" in model_name:
            info["quantization"] = "INT4"
        else:
            info["quantization"] = "FP16/BF16"

        return info


class MetricsMonitor:
    """
    vLLM metrics monitoring system.
    
    Queries vLLM metrics endpoint to track:
    - Prefix cache hit rate (queries and hits)
    - Prefill time (input processing)
    - Decode time (output generation)
    """
    
    def __init__(self):
        """Initialize metrics monitor."""
        self.baseline_queries = 0
        self.baseline_hits = 0
        self.baseline_prefill_time = 0.0
        self.baseline_decode_time = 0.0
        self.available = False
    
    def get_metrics(self) -> Optional[Dict]:
        """
        Query vLLM metrics endpoint for current statistics.
        
        Returns:
            Dictionary containing metrics, or None if unavailable
        """
        try:
            metrics_url = f"{API_BASE_URL}/metrics"
            response = requests.get(metrics_url, timeout=2)
            
            if response.status_code != 200:
                return None
            
            metrics_text = response.text
            queries = 0.0
            hits = 0.0
            prefill_time = 0.0
            decode_time = 0.0
            
            for line in metrics_text.split('\n'):
                if line.startswith('#') or not line.strip():
                    continue
                
                # Extract prefix cache queries
                if 'vllm:prefix_cache_queries_total' in line:
                    try:
                        parts = line.split()
                        if len(parts) >= 2:
                            queries = float(parts[-1])
                    except:
                        pass
                
                # Extract prefix cache hits
                if 'vllm:prefix_cache_hits_total' in line:
                    try:
                        parts = line.split()
                        if len(parts) >= 2:
                            hits = float(parts[-1])
                    except:
                        pass
                
                # Extract prefill time sum
                if 'vllm:request_prefill_time_seconds_sum' in line:
                    try:
                        parts = line.split()
                        if len(parts) >= 2:
                            prefill_time = float(parts[-1])
                    except:
                        pass
                
                # Extract decode time sum
                if 'vllm:request_decode_time_seconds_sum' in line:
                    try:
                        parts = line.split()
                        if len(parts) >= 2:
                            decode_time = float(parts[-1])
                    except:
                        pass
            
            return {
                "cache_queries": queries,
                "cache_hits": hits,
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "timestamp": time.time()
            }
        
        except Exception:
            return None
    
    def start(self) -> bool:
        """
        Capture baseline metrics at start of test.
        
        Returns:
            True if metrics are available, False otherwise
        """
        stats = self.get_metrics()
        if stats:
            self.baseline_queries = stats["cache_queries"]
            self.baseline_hits = stats["cache_hits"]
            self.baseline_prefill_time = stats["prefill_time"]
            self.baseline_decode_time = stats["decode_time"]
            self.available = True
            return True
        else:
            self.available = False
            return False
    
    def stop(self) -> Optional[Dict]:
        """
        Calculate metrics deltas since start.
        
        Returns:
            Dictionary containing cache hit rate, prefill/decode times, or None if unavailable
        """
        if not self.available:
            return None
        
        stats = self.get_metrics()
        if not stats:
            return None
        
        # Calculate cache deltas
        delta_queries = stats["cache_queries"] - self.baseline_queries
        delta_hits = stats["cache_hits"] - self.baseline_hits
        hit_rate = (delta_hits / delta_queries * 100) if delta_queries > 0 else 0
        
        # Calculate time deltas
        delta_prefill = stats["prefill_time"] - self.baseline_prefill_time
        delta_decode = stats["decode_time"] - self.baseline_decode_time
        
        return {
            "cache_hit_rate": hit_rate,
            "cache_queries_delta": delta_queries,
            "cache_hits_delta": delta_hits,
            "total_cache_queries": stats["cache_queries"],
            "total_cache_hits": stats["cache_hits"],
            "actual_prefill_time": delta_prefill,
            "actual_decode_time": delta_decode
        }


class GPUMonitor:
    """
    Real-time GPU performance monitoring system.

    Polls nvidia-smi at regular intervals to collect GPU utilization, memory usage,
    temperature, power draw, and clock frequencies during benchmark execution.
    """

    def __init__(self, poll_interval: float = GPU_POLL_INTERVAL):
        """
        Initialize GPU monitor.

        Args:
            poll_interval: Polling interval in seconds (default: 1.0)
        """
        self.monitoring = False
        self.stats = []
        self.thread = None
        self.poll_interval = poll_interval

    def get_gpu_stats(self) -> Optional[Dict]:
        """
        Query nvidia-smi for current GPU statistics.

        Returns:
            Dictionary containing GPU metrics or None if query fails
        """
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.gr,clocks.mem",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )

            if result.returncode == 0:
                values = result.stdout.strip().split(", ")
                return {
                    "gpu_util": float(values[0]),
                    "mem_used": float(values[1]),
                    "mem_total": float(values[2]),
                    "temperature": float(values[3]),
                    "power_draw": float(values[4]),
                    "gpu_clock": float(values[5]),
                    "mem_clock": float(values[6]),
                    "timestamp": time.time(),
                }
        except Exception as e:
            print(f"[WARNING] GPU monitoring error: {e}", file=sys.stderr)
        return None

    def monitor_loop(self) -> None:
        """Background thread loop for continuous GPU monitoring."""
        while self.monitoring:
            stats = self.get_gpu_stats()
            if stats:
                self.stats.append(stats)
            time.sleep(self.poll_interval)

    def start(self) -> None:
        """Start monitoring in background thread."""
        self.monitoring = True
        self.stats = []
        self.thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.thread.start()

    def stop(self) -> Optional[Dict]:
        """
        Stop monitoring and return aggregated statistics.

        Returns:
            Dictionary containing averaged and peak GPU metrics
        """
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=2)

        if not self.stats:
            return None

        return {
            "avg_gpu_util": mean([s["gpu_util"] for s in self.stats]),
            "max_gpu_util": max([s["gpu_util"] for s in self.stats]),
            "avg_mem_used": mean([s["mem_used"] for s in self.stats]),
            "max_mem_used": max([s["mem_used"] for s in self.stats]),
            "avg_temperature": mean([s["temperature"] for s in self.stats]),
            "max_temperature": max([s["temperature"] for s in self.stats]),
            "avg_power": mean([s["power_draw"] for s in self.stats]),
            "max_power": max([s["power_draw"] for s in self.stats]),
            "avg_gpu_clock": mean([s["gpu_clock"] for s in self.stats]),
            "max_gpu_clock": max([s["gpu_clock"] for s in self.stats]),
            "avg_mem_clock": mean([s["mem_clock"] for s in self.stats]),
            "samples": len(self.stats),
        }


def get_model_name() -> str:
    """
    Query the vLLM server for the currently loaded model name.

    Returns:
        Model name string or default if query fails
    """
    try:
        response = requests.get(API_MODELS_ENDPOINT, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "data" in data and len(data["data"]) > 0:
                model_name = data["data"][0].get("id", DEFAULT_MODEL_NAME)
                print(f"[INFO] Detected model: {model_name}")
                return model_name
    except Exception as e:
        print(f"[WARNING] Failed to query model name: {e}", file=sys.stderr)

    print(f"[INFO] Using default model name: {DEFAULT_MODEL_NAME}")
    return DEFAULT_MODEL_NAME


def generate_prompt(target_tokens: int, model_name: str="") -> str:
    """
    Generate a synthetic prompt of approximately target_tokens length.

    Uses cybersecurity threat intelligence text as content to simulate
    realistic workload patterns for CTI use cases.

    Args:
        target_tokens: Approximate desired token count
        model_name: Unused argument, supplied to conform to prompt variety interface.

    Returns:
        Generated prompt string
    """
    base_text = (
        "Analyze the following cybersecurity threat intelligence data in detail. "
    )
    repeat_text = (
        "Advanced Persistent Threat (APT) groups continue to evolve their tactics, techniques, and procedures (TTPs) "
        "as documented in the MITRE ATT&CK framework. Nation-state actors leverage sophisticated malware campaigns "
        "targeting critical infrastructure including SCADA systems, industrial control systems, and OT networks. "
        "Recent ransomware operations demonstrate increased professionalization with affiliates using double extortion "
        "techniques, data exfiltration, and targeted attacks on backup systems. Dark web marketplaces facilitate "
        "the sale of exploits, credentials, and access to compromised networks. Vulnerability intelligence indicates "
        "zero-day exploits are being actively weaponized against enterprise systems. Threat actors employ "
        "living-off-the-land binaries (LOLBins), fileless malware, and memory-only payloads to evade detection. "
        "Network intrusion detection systems identify command and control (C2) infrastructure using domain generation "
        "algorithms (DGA) and fast-flux DNS techniques. Security operations centers analyze indicators of compromise "
        "(IOCs) including file hashes, IP addresses, and behavioral patterns to attribute attacks to specific threat actors. "
    )

    # Approximate token calculation (4 characters per token)
    base_tokens = len(base_text) // 4
    repeat_tokens = len(repeat_text) // 4
    repetitions = max(1, (target_tokens - base_tokens) // repeat_tokens)

    return base_text + (repeat_text * repetitions)

def make_random_text(ntoks: int) -> str:
    """Helper function which chooses `ntoks` random words from the NLTK Brown corpus, then stitches them together with whitespace."""
    return " ".join([random.choice(COMMON_WORDS) for _ in range(ntoks)])

def make_perturbed_story() -> str:
    RaR = make_random_text(3)
    SSS = make_random_text(3)
    CCC = make_random_text(3)
    BBB = make_random_text(3)
    WWW = make_random_text(3)

    story_text = f"The wheels on the bus go {RaR}. {RaR}. {RaR}. The wheels on the bus go {RaR}. All through the town.. The wipers on the bus go {SSS}. {SSS}. {SSS}. The wipers on the bus go {SSS}. All through the town.. The people on the bus go {CCC}. {CCC}. {CCC}. The people on the bus go {CCC}. All through the town.. The horn on the bus goes {BBB}. {BBB}. {BBB}. The horn on the bus goes {BBB}. All through the town.. The babies on the bus go {WWW}. {WWW}. {WWW}. The babies on the bus go {WWW}. All through the town."

    return story_text

def generate_deterministic_prompt(target_tokens: int, tokenizer_model: str) -> str:
    """Creates a purely deterministic prompt which tells a story with some repetition."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    except Exception as e:
        print(f"Error loading model tokenizer! Using `{DEFAULT_TOKENIZER}` instead.")
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER)

    query_text = "Provide a concise summary of this story. "
    story_text = "The wheels on the bus go round and round. Round and round. Round and round. The wheels on the bus go round and round. All through the town.. The wipers on the bus go swish, swish, swish. Swish, swish, swish. Swish, swish, swish. The wipers on the bus go swish, swish, swish. All through the town.. The people on the bus go chat, chat, chat. Chat, chat, chat. Chat, chat, chat. The people on the bus go chat, chat, chat. All through the town.. The horn on the bus goes beep, beep, beep. Beep, beep, beep. Beep, beep, beep. The horn on the bus goes beep, beep, beep. All through the town.. The babies on the bus go waa, waa, waa. Waa, waa, waa. Waa, waa, waa. The babies on the bus go waa, waa, waa. All through the town."

    base_tokens = tokenizer.encode(query_text) + tokenizer.encode(story_text) #core tokens are query + story 1x
    if len(base_tokens) > target_tokens: return "" #if not enough room for one iteration, return empty str

    #if enough room to repeat, calculate repetitions
    n_remaining_tokens = target_tokens - len(base_tokens)
    repetitions = n_remaining_tokens // len(tokenizer.encode(story_text)) #calculate how many times we can repeat the story, use floor division
    repeat_text = repetitions * (" " + story_text)

    return query_text + story_text + repeat_text

def generate_madlib_prompt(target_tokens: int, tokenizer_model: str) -> str:
    """Creates a prompt in a mad-lib style, injecting some randomness to promote a moderate amount of cache misses."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    except Exception as e:
        print(f"Error loading model tokenizer! Using `{DEFAULT_TOKENIZER}` instead.")
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER)
    
    query_text = "Provide a concise summary of this story. "
    story_text = make_perturbed_story()
    
    base_tokens = tokenizer.encode(query_text) + tokenizer.encode(story_text) #core tokens are query + story 1x #core tokens are query + story 1x
    if len(base_tokens) > target_tokens: return ""

    #if enough room to repeat, calculate repetitions
    buffer_tokens = int(target_tokens * 0.01) #generate 1% less tokens than context length because random words could tokenize nonuniformly. this should be very rare though.
    n_remaining_tokens = target_tokens - (len(base_tokens) + buffer_tokens)
    repetitions = n_remaining_tokens // len(tokenizer.encode(story_text)) #calculate how many times we can repeat the story, use floor division
    repeat_text = " ".join([make_perturbed_story() for _ in range(repetitions)]) #need to iteratively call story generation for fresh randoms
    
    return query_text + story_text + " " + repeat_text

def generate_random_prompt(target_tokens: int, tokenizer_model: str) -> str:
    """Creates a prompt with almost entirely random text to promote a high number of cache misses."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    except Exception as e:
        print(f"Error loading model tokenizer! Using `{DEFAULT_TOKENIZER}` instead.")
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER)
    
    query_text = "Find the pattern in the following string. "
    random_text = ""
    buffer_tokens = int(target_tokens * 0.05) #generate 5% less than the boundary in case tokenizer is aggressive!
    n_remaining_tokens = target_tokens - (len(tokenizer.encode(query_text)) + buffer_tokens)
    random_text = make_random_text(n_remaining_tokens)

    return query_text + random_text


def calculate_percentiles(values: List[float]) -> Dict[str, float]:
    """Calculate P50, P90, P95, P99 percentiles efficiently using numpy."""
    if not values:
        return {"p50": 0, "p90": 0, "p95": 0, "p99": 0}

    # Use numpy for efficient percentile calculation
    values_array = np.array(values)
    return {
        "p50": float(np.percentile(values_array, 50)),
        "p90": float(np.percentile(values_array, 90)),
        "p95": float(np.percentile(values_array, 95)),
        "p99": float(np.percentile(values_array, 99)),
    }


def make_request(
    prompt: str,
    request_id: int,
    results: List[Dict],
    max_tokens: int = 500,
    model_name: str = None,
) -> None:
    """
    Execute a single API request and record timing metrics.

    Args:
        prompt: Input prompt text
        request_id: Unique identifier for this request
        results: Shared list to append results
        max_tokens: Maximum output tokens
        model_name: Model identifier for API request
    """
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }

    try:
        start = time.time()
        response = requests.post(API_ENDPOINT, json=data, timeout=REQUEST_TIMEOUT)
        duration = time.time() - start

        if response.status_code == 200:
            result = response.json()
            usage = result.get("usage", {})

            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            # Calculate inter-token latency (ITL) - average time per output token
            itl = duration / completion_tokens if completion_tokens > 0 else 0

            # Estimate prefill vs decode time (rough estimate, streaming would be exact)
            prefill_estimate = duration * 0.15  # Typically 10-20% for large contexts
            decode_estimate = duration - prefill_estimate

            results.append(
                {
                    "request_id": request_id,
                    "duration": duration,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": usage.get("total_tokens", 0),
                    "inter_token_latency": itl,
                    "prefill_time_estimate": prefill_estimate,
                    "decode_time_estimate": decode_estimate,
                    "success": True,
                }
            )
        else:
            results.append(
                {
                    "request_id": request_id,
                    "duration": duration,
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                }
            )
    except Exception as e:
        results.append({"request_id": request_id, "success": False, "error": str(e)})


def run_benchmark(
    context_length: int,
    num_concurrent_users: int,
    output_tokens: int = 500,
    model_name: str = None,
    live_display=None,
    gpu_monitor: GPUMonitor = None,
    prompt_type: str = "classic",
) -> Optional[Dict]:
    """
    Execute benchmark for specific context length and concurrency level.

    Args:
        context_length: Input context size in tokens
        num_concurrent_users: Number of concurrent requests
        output_tokens: Target output token count
        model_name: Model identifier
        live_display: Rich Live display object for real-time updates
        gpu_monitor: GPU monitor instance for real-time stats
        prompt_type: Type of prompt to generate (classic, deterministic, madlib, random)

    Returns:
        Dictionary containing performance metrics or None on failure
    """
    if not live_display:
        print(f"\n{'=' * 100}")
        print(
            f"Testing: {context_length:,} token context | {num_concurrent_users} concurrent users | {prompt_type} prompt"
        )
        print(f"{'=' * 100}")

    # Map prompt type to generation function
    prompt_generators = {
        "classic": generate_prompt,
        "deterministic": generate_deterministic_prompt,
        "madlib": generate_madlib_prompt,
        "random": generate_random_prompt,
    }
    
    prompt_gen_func = prompt_generators.get(prompt_type, generate_prompt)
    prompt = prompt_gen_func(context_length, model_name)
    actual_prompt_tokens = len(prompt) // 4

    results = []
    threads = []

    # Use existing monitor or create new one
    local_monitor = gpu_monitor if gpu_monitor else GPUMonitor()
    if not gpu_monitor:
        local_monitor.start()

    # Initialize metrics monitor for this test
    metrics_monitor = MetricsMonitor()
    metrics_monitor.start()

    start_time = time.time()

    # Launch concurrent requests
    for i in range(num_concurrent_users):
        t = threading.Thread(
            target=make_request, args=(prompt, i, results, output_tokens, model_name)
        )
        threads.append(t)
        t.start()

    # Wait for completion with live updates
    while any(t.is_alive() for t in threads):
        time.sleep(0.5)
        # Live display is updated by main loop

    # Ensure all threads complete
    for t in threads:
        t.join()

    total_time = time.time() - start_time

    # Stop local monitor if we created it
    gpu_stats = None
    if not gpu_monitor:
        gpu_stats = local_monitor.stop()

    # Stop metrics monitor and get metrics
    metrics_stats = metrics_monitor.stop()

    # Calculate statistics
    successful = [r for r in results if r.get("success", False)]
    failed = len(results) - len(successful)

    if successful:
        durations = [r["duration"] for r in successful]
        completion_tokens = [r["completion_tokens"] for r in successful]
        prompt_tokens = [r["prompt_tokens"] for r in successful]

        avg_duration = mean(durations)
        std_duration = stdev(durations) if len(durations) > 1 else 0
        min_duration = min(durations)
        max_duration = max(durations)

        total_completion_tokens = sum(completion_tokens)
        avg_prompt_tokens = (
            mean(prompt_tokens) if prompt_tokens else actual_prompt_tokens
        )

        tokens_per_second = total_completion_tokens / total_time
        requests_per_second = len(successful) / total_time
        avg_tokens_per_request = mean(completion_tokens) if completion_tokens else 0

        ttft_estimate = avg_duration * 0.1
        throughput_per_user = (
            tokens_per_second / num_concurrent_users if num_concurrent_users > 0 else 0
        )

        # Print results if no live display
        if not live_display:
            print(f"\nResults:")
            print(f"  Total time:              {total_time:.2f}s")
            print(
                f"  Successful:              {len(successful)}/{num_concurrent_users}"
            )
            print(f"  Failed:                  {failed}")
            print(f"\nLatency Metrics:")
            print(f"  Average:                 {avg_duration:.2f}s")
            print(f"  Std Dev:                 {std_duration:.2f}s")
            print(f"  Min:                     {min_duration:.2f}s")
            print(f"  Max:                     {max_duration:.2f}s")
            print(f"  Est. TTFT:               {ttft_estimate:.3f}s")
            print(f"\nThroughput Metrics:")
            print(f"  Tokens/second:           {tokens_per_second:.1f}")
            print(f"  Requests/second:         {requests_per_second:.2f}")
            print(f"  Tokens/second/user:      {throughput_per_user:.1f}")
            print(f"\nToken Usage:")
            print(f"  Avg prompt tokens:       {avg_prompt_tokens:.0f}")
            print(f"  Avg completion tokens:   {avg_tokens_per_request:.0f}")

            if gpu_stats:
                print(f"\nGPU Metrics:")
                print(f"  Avg utilization:         {gpu_stats['avg_gpu_util']:.1f}%")
                print(f"  Max utilization:         {gpu_stats['max_gpu_util']:.1f}%")
                print(
                    f"  Avg memory used:         {gpu_stats['avg_mem_used']:.0f} MB ({gpu_stats['avg_mem_used'] / 1024:.1f} GB)"
                )
                print(
                    f"  Max memory used:         {gpu_stats['max_mem_used']:.0f} MB ({gpu_stats['max_mem_used'] / 1024:.1f} GB)"
                )
                print(f"  Avg temperature:         {gpu_stats['avg_temperature']:.1f}C")
                print(f"  Max temperature:         {gpu_stats['max_temperature']:.1f}C")
                print(f"  Avg power draw:          {gpu_stats['avg_power']:.1f} W")
                print(f"  Max power draw:          {gpu_stats['max_power']:.1f} W")
                print(
                    f"  Avg GPU clock:           {gpu_stats['avg_gpu_clock']:.0f} MHz"
                )
                print(
                    f"  Max GPU clock:           {gpu_stats['max_gpu_clock']:.0f} MHz"
                )
                print(
                    f"  Avg memory clock:        {gpu_stats['avg_mem_clock']:.0f} MHz"
                )

        result_dict = {
            "context_length": context_length,
            "concurrent_users": num_concurrent_users,
            "prompt_type": prompt_type,
            "total_time": total_time,
            "successful": len(successful),
            "failed": failed,
            "avg_latency": avg_duration,
            "std_latency": std_duration,
            "min_latency": min_duration,
            "max_latency": max_duration,
            "ttft_estimate": ttft_estimate,
            "tokens_per_second": tokens_per_second,
            "requests_per_second": requests_per_second,
            "throughput_per_user": throughput_per_user,
            "avg_prompt_tokens": avg_prompt_tokens,
            "avg_completion_tokens": avg_tokens_per_request,
        }

        # Add metrics statistics if available
        if metrics_stats:
            result_dict.update(metrics_stats)
            
            # Calculate prefill speed if we have actual prefill time
            if "actual_prefill_time" in metrics_stats and metrics_stats["actual_prefill_time"] > 0:
                # Prefill speed = average prompt tokens / actual prefill time
                result_dict["prefill_speed"] = avg_prompt_tokens / metrics_stats["actual_prefill_time"]
            else:
                # Fallback to estimate if metrics not available
                result_dict["prefill_speed"] = 0

        # Merge GPU statistics and calculate energy efficiency
        if gpu_stats:
            result_dict.update(gpu_stats)
            # Watts per token
            result_dict["watts_per_token"] = (
                gpu_stats["avg_power"] / tokens_per_second
                if tokens_per_second > 0
                else 0
            )
            # Watts per token per user (energy efficiency metric)
            result_dict["watts_per_token_per_user"] = (
                (gpu_stats["avg_power"] / tokens_per_second / num_concurrent_users)
                if (tokens_per_second > 0 and num_concurrent_users > 0)
                else 0
            )
            # Throughput per user per watt (efficiency per user normalized by power)
            result_dict["throughput_per_user_per_watt"] = (
                throughput_per_user / gpu_stats["avg_power"]
                if gpu_stats["avg_power"] > 0
                else 0
            )
            # Watts per token per user per 1K context (normalized energy efficiency)
            context_k = context_length / 1000
            result_dict["watts_per_token_per_user_per_1k_context"] = (
                result_dict["watts_per_token_per_user"] / context_k
                if context_k > 0
                else 0
            )
            # Energy efficiency score (tokens per watt)
            result_dict["tokens_per_watt"] = (
                tokens_per_second / gpu_stats["avg_power"]
                if gpu_stats["avg_power"] > 0
                else 0
            )
            # Total energy consumed (joules)
            result_dict["energy_joules"] = gpu_stats["avg_power"] * total_time
            # Energy per token (joules)
            result_dict["energy_per_token"] = (
                result_dict["energy_joules"] / total_completion_tokens
                if total_completion_tokens > 0
                else 0
            )
            # Watt-hours consumed
            result_dict["energy_watt_hours"] = result_dict["energy_joules"] / 3600
            # Energy efficiency per context (joules per token per 1K context)
            result_dict["energy_per_token_per_1k_context"] = (
                result_dict["energy_per_token"] / context_k if context_k > 0 else 0
            )

        return result_dict
    else:
        if not live_display:
            print(f"\n[ERROR] All requests failed!")
            for r in results[:3]:
                if not r.get("success"):
                    print(f"  Error: {r.get('error', 'Unknown')}")
        return None



def visualize_results(
    all_results: List[Dict],
    model_name: str,
    system_info: Dict = None,
    server_info: Dict = None,
    output_tokens: int = 500,
) -> str:
    """Generate 10-graph benchmark visualization."""
    df = pd.DataFrame(all_results)
    
    #this block downselects the results dataframe to utilize the average results of prompt types the core visualizations
    has_prompt_types = "prompt_type" in df.columns
    if has_prompt_types:
        # Aggregate across prompt types by averaging for each (context_length, concurrent_users) pair
        console.print(f"[dim]Averaging across {len(df['prompt_type'].unique())} prompt types for main visualizations[/dim]")
        
        # Group by context_length and concurrent_users, then average all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove columns we don't want to average (keep as-is from first row)
        preserve_cols = ['context_length', 'concurrent_users']
        avg_cols = [col for col in numeric_cols if col not in preserve_cols]
        
        # Aggregate: mean for numeric, first for non-numeric
        agg_dict = {col: 'mean' for col in avg_cols}
        df_main = df.groupby(['context_length', 'concurrent_users'], as_index=False).agg(agg_dict)
    else:
        df_main = df.copy()
    
    has_gpu_stats = "avg_gpu_util" in df_main.columns

    # Calculate prompt processing speed (tokens/second during prefill)
    if "prefill_time_estimate" in df_main.columns and "avg_prompt_tokens" in df_main.columns:
        df_main["prompt_processing_speed"] = df_main["avg_prompt_tokens"] / (df_main["prefill_time_estimate"] + 0.001)
    else:
        df_main["prompt_processing_speed"] = df_main["avg_prompt_tokens"] / (df_main["avg_latency"] * 0.15 + 0.001)

    # Calculate inter-token latency (milliseconds between tokens)
    if "decode_time_estimate" in df_main.columns and "avg_completion_tokens" in df_main.columns:
        df_main["inter_token_latency"] = (df_main["decode_time_estimate"] / (df_main["avg_completion_tokens"] + 0.001)) * 1000
    else:
        # Fallback: estimate decode time as 85% of total
        df_main["inter_token_latency"] = ((df_main["avg_latency"] * 0.85) / (df_main["avg_completion_tokens"] + 0.001)) * 1000

    # Calculate batch scaling efficiency
    baseline_throughput = {}
    for ctx in df_main["context_length"].unique():
        single_user = df_main[(df_main["context_length"] == ctx) & (df_main["concurrent_users"] == 1)]
        if len(single_user) > 0:
            baseline_throughput[ctx] = single_user["tokens_per_second"].values[0]
    
    def calc_efficiency(row):
        baseline = baseline_throughput.get(row["context_length"], 1)
        if row["concurrent_users"] == 1:
            return 100.0
        return (row["tokens_per_second"] / baseline / row["concurrent_users"]) * 100
    
    df_main["batch_efficiency"] = df_main.apply(calc_efficiency, axis=1)

    context_lengths = sorted(df_main["context_length"].unique())
    context_labels = [f"{int(c / 1000)}K" for c in context_lengths]

    # Styling
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 13
    })

    # Layout: Row 1 landscape, Rows 2-5 grid (11 graphs total)
    fig = plt.figure(figsize=(24, 24))
    gs = fig.add_gridspec(
        5, 3, hspace=0.40, wspace=0.28, left=0.06, right=0.98, top=0.96, bottom=0.04,
        height_ratios=[1, 1, 1, 1, 1]
    )

    # Layout: 13+ graphs total (12 existing + 1 prompt type comparison + cache heatmaps)
    # Row 1: Throughput landscape
    # Row 2: Two heatmaps only
    # Rows 3-5: Line plots (3×3 grid)
    # Row 6: Prompt type comparison landscape (only if prompt types available)
    # Row 7+: Cache heatmaps (one per prompt type, if available)
    
    # Check if we need cache rows
    has_cache_stats = "cache_hit_rate" in df.columns
    need_cache_rows = has_prompt_types and has_cache_stats
    
    if need_cache_rows:
        num_prompt_types = len(df["prompt_type"].unique())
        # Calculate how many rows needed for cache heatmaps (2 per row)
        cache_rows = (num_prompt_types + 1) // 2
        num_rows = 6 + cache_rows
        height_ratios = [1, 1, 1, 1, 1, 1] + [0.8] * cache_rows
        fig_height = 24 + (cache_rows * 5)
    elif has_prompt_types:
        num_rows = 6
        height_ratios = [1, 1, 1, 1, 1, 1]
        fig_height = 24
    else:
        num_rows = 5
        height_ratios = [1, 1, 1, 1, 1]
        fig_height = 24
    
    fig = plt.figure(figsize=(24, fig_height))
    gs = fig.add_gridspec(
        num_rows, 3, hspace=0.40, wspace=0.28, left=0.06, right=0.98, top=0.96, bottom=0.04,
        height_ratios=height_ratios
    )

    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E", "#BC4B51"]

    # GRAPH 1: Throughput vs Context Length (Landscape)
    ax1 = fig.add_subplot(gs[0, :])
    for idx, users in enumerate(sorted(df_main["concurrent_users"].unique())):
        data = df_main[df_main["concurrent_users"] == users].sort_values("context_length")
        ax1.plot(
            data["context_length"] / 1000, data["tokens_per_second"],
            marker="o", linewidth=3, markersize=10, label=f"{users} users",
            color=colors[idx % len(colors)], markeredgecolor="white", markeredgewidth=1.5
        )
    ax1.set_xlabel("Context Length (K tokens)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Throughput (tokens/sec)", fontsize=13, fontweight="bold")
    ax1.set_title("Throughput vs Context Length by Concurrency", fontsize=14, fontweight="bold", pad=15)
    ax1.set_xticks([c / 1000 for c in context_lengths])
    ax1.set_xticklabels(context_labels)
    ax1.legend(title="Concurrent Users", fontsize=10, title_fontsize=11, loc="best", frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_facecolor("#FAFAFA")

    # ROW 2: HEATMAPS ONLY (spanning full width)
    # GRAPH 2: Throughput Heatmap
    ax2 = fig.add_subplot(gs[1, :2])  # Span 2 columns
    pivot_throughput = df_main.pivot(index="context_length", columns="concurrent_users", values="tokens_per_second")
    sns.heatmap(pivot_throughput, annot=True, fmt=".0f", cmap="RdYlGn", ax=ax2,
                cbar_kws={"label": "Tokens/sec", "shrink": 0.9}, linewidths=1.5,
                linecolor="white", annot_kws={"fontsize": 10, "weight": "bold"})
    ax2.set_xlabel("Concurrent Users", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Context Length", fontsize=11, fontweight="bold")
    ax2.set_title("Throughput Heatmap", fontsize=12, fontweight="bold", pad=10)
    ax2.set_yticklabels([f"{int(y / 1000)}K" for y in pivot_throughput.index], rotation=0, fontsize=9)

    # GRAPH 3: Latency Heatmap
    ax3 = fig.add_subplot(gs[1, 2])  # Single column
    pivot_latency = df_main.pivot(index="context_length", columns="concurrent_users", values="avg_latency")
    sns.heatmap(pivot_latency, annot=True, fmt=".1f", cmap="RdYlGn_r", ax=ax3,
                cbar_kws={"label": "Latency (sec)", "shrink": 0.9}, linewidths=1.5,
                linecolor="white", annot_kws={"fontsize": 10, "weight": "bold"})
    ax3.set_xlabel("Concurrent Users", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Context Length", fontsize=11, fontweight="bold")
    ax3.set_title("Latency Heatmap", fontsize=12, fontweight="bold", pad=10)
    ax3.set_yticklabels([f"{int(y / 1000)}K" for y in pivot_latency.index], rotation=0, fontsize=9)

    # ROW 3: Line plots
    # GRAPH 4: Throughput per User
    ax4 = fig.add_subplot(gs[2, 0])
    for idx, users in enumerate(sorted(df_main["concurrent_users"].unique())):
        data = df_main[df_main["concurrent_users"] == users].sort_values("context_length")
        ax4.plot(
            data["context_length"] / 1000, data["throughput_per_user"],
            marker="s", linewidth=2.5, markersize=9, label=f"{users} users",
            color=colors[idx % len(colors)], markeredgecolor="white", markeredgewidth=1.5
        )
    ax4.set_xlabel("Context Length (K tokens)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Tokens/sec per User", fontsize=11, fontweight="bold")
    ax4.set_title("Throughput per User", fontsize=12, fontweight="bold", pad=10)
    ax4.set_xticks([c / 1000 for c in context_lengths])
    ax4.set_xticklabels(context_labels)
    ax4.legend(fontsize=9, loc="best")
    ax4.grid(True, alpha=0.3, linestyle="--")
    ax4.set_facecolor("#FAFAFA")

    # GRAPH 5: Time to First Token
    ax5 = fig.add_subplot(gs[2, 1])
    for idx, users in enumerate(sorted(df_main["concurrent_users"].unique())):
        data = df_main[df_main["concurrent_users"] == users].sort_values("context_length")
        ax5.plot(
            data["context_length"] / 1000, data["ttft_estimate"] * 1000,
            marker="*", linewidth=2.5, markersize=11, label=f"{users} users",
            color=colors[idx % len(colors)], markeredgecolor="white", markeredgewidth=1.5
        )
    
    # UX quality zones for TTFT
    ax5.axhspan(0, 200, alpha=0.08, color='green')     # <200ms: Instant
    ax5.axhspan(200, 1000, alpha=0.08, color='yellow') # 200-1000ms: Responsive
    ax5.axhspan(1000, 3000, alpha=0.08, color='orange') # 1-3s: Noticeable
    
    # Add UX labels
    ax5.text(0.98, 0.08, 'INSTANT (<200ms)', transform=ax5.transAxes,
             fontsize=8, weight='bold', color='darkgreen', alpha=0.7,
             ha='right', va='bottom', style='italic')
    ax5.text(0.98, 0.35, 'RESPONSIVE', transform=ax5.transAxes,
             fontsize=8, weight='bold', color='darkorange', alpha=0.7,
             ha='right', va='center', style='italic')
    ax5.text(0.98, 0.75, 'NOTICEABLE DELAY', transform=ax5.transAxes,
             fontsize=8, weight='bold', color='darkred', alpha=0.7,
             ha='right', va='center', style='italic')
    
    ax5.set_xlabel("Context Length (K tokens)", fontsize=11, fontweight="bold")
    ax5.set_ylabel("TTFT (milliseconds)", fontsize=11, fontweight="bold")
    ax5.set_title("Time to First Token (UX Quality)", fontsize=12, fontweight="bold", pad=10)
    ax5.set_xticks([c / 1000 for c in context_lengths])
    ax5.set_xticklabels(context_labels)
    ax5.legend(fontsize=9, loc="upper left")
    ax5.grid(True, alpha=0.3, linestyle="--")
    ax5.set_facecolor("#FAFAFA")

    # GRAPH 6: Average Latency vs Context Length
    ax6 = fig.add_subplot(gs[2, 2])
    for idx, users in enumerate(sorted(df_main["concurrent_users"].unique())):
        data = df_main[df_main["concurrent_users"] == users].sort_values("context_length")
        ax6.plot(
            data["context_length"] / 1000, data["avg_latency"],
            marker="D", linewidth=2.5, markersize=9, label=f"{users} users",
            color=colors[idx % len(colors)], markeredgecolor="white", markeredgewidth=1.5
        )
    
    # UX quality zones for total latency
    ax6.axhspan(0, 2, alpha=0.08, color='green')      # <2s: Good
    ax6.axhspan(2, 5, alpha=0.08, color='yellow')     # 2-5s: Acceptable
    ax6.axhspan(5, 10, alpha=0.08, color='orange')    # 5-10s: Slow
    
    # Add UX labels
    ax6.text(0.02, 0.15, 'GOOD (<2s)', transform=ax6.transAxes,
             fontsize=8, weight='bold', color='darkgreen', alpha=0.7,
             ha='left', va='center', style='italic')
    ax6.text(0.02, 0.40, 'ACCEPTABLE', transform=ax6.transAxes,
             fontsize=8, weight='bold', color='darkorange', alpha=0.7,
             ha='left', va='center', style='italic')
    ax6.text(0.02, 0.75, 'SLOW', transform=ax6.transAxes,
             fontsize=8, weight='bold', color='darkred', alpha=0.7,
             ha='left', va='center', style='italic')
    
    ax6.set_xlabel("Context Length (K tokens)", fontsize=11, fontweight="bold")
    ax6.set_ylabel("Latency (seconds)", fontsize=11, fontweight="bold")
    ax6.set_title("Average Latency (UX Quality)", fontsize=12, fontweight="bold", pad=10)
    ax6.set_xticks([c / 1000 for c in context_lengths])
    ax6.set_xticklabels(context_labels)
    ax6.legend(fontsize=9, loc="upper left")
    ax6.grid(True, alpha=0.3, linestyle="--")
    ax6.set_facecolor("#FAFAFA")

    # ROW 4
    # GRAPH 7: Prompt Processing Speed
    ax7 = fig.add_subplot(gs[3, 0])
    for idx, users in enumerate(sorted(df_main["concurrent_users"].unique())):
        data = df_main[df_main["concurrent_users"] == users].sort_values("context_length")
        ax7.plot(
            data["context_length"] / 1000, data["prompt_processing_speed"],
            marker="^", linewidth=2.5, markersize=9, label=f"{users} users",
            color=colors[idx % len(colors)], markeredgecolor="white", markeredgewidth=1.5
        )
    ax7.set_xlabel("Context Length (K tokens)", fontsize=11, fontweight="bold")
    ax7.set_ylabel("Prompt Processing (tokens/sec)", fontsize=11, fontweight="bold")
    ax7.set_title("Prompt Processing Speed", fontsize=12, fontweight="bold", pad=10)
    ax7.set_xticks([c / 1000 for c in context_lengths])
    ax7.set_xticklabels(context_labels)
    ax7.legend(fontsize=9, loc="best")
    ax7.grid(True, alpha=0.3, linestyle="--")
    ax7.set_facecolor("#FAFAFA")

    # GRAPH 8: Power Draw
    ax8 = fig.add_subplot(gs[3, 1])
    if has_gpu_stats:
        for idx, users in enumerate(sorted(df_main["concurrent_users"].unique())):
            data = df_main[df_main["concurrent_users"] == users].sort_values("context_length")
            color = colors[idx % len(colors)]
            ax8.plot(data["context_length"] / 1000, data["avg_power"],
                    marker="o", linewidth=2.5, markersize=9, label=f"{users} users",
                    color=color, markeredgecolor="white", markeredgewidth=1.5)
        
        ax8.set_xlabel("Context Length (K tokens)", fontsize=11, fontweight="bold")
        ax8.set_ylabel("Power Draw (W)", fontsize=11, fontweight="bold", color="#E63946")
        ax8.set_title("Average Power Draw", fontsize=12, fontweight="bold", pad=10)
        ax8.set_xticks([c / 1000 for c in context_lengths])
        ax8.set_xticklabels(context_labels)
        ax8.tick_params(axis="y", labelcolor="#E63946")
        ax8.grid(True, alpha=0.3, linestyle="--")
        ax8.set_facecolor("#FAFAFA")
        ax8.legend(fontsize=9, loc="best")
    else:
        ax8.text(0.5, 0.5, "GPU stats not available", ha="center", va="center",
                fontsize=12, transform=ax8.transAxes)
        ax8.axis("off")

    # GRAPH 9: GPU Clock Frequency
    ax9 = fig.add_subplot(gs[3, 2])
    if has_gpu_stats:
        for idx, users in enumerate(sorted(df_main["concurrent_users"].unique())):
            data = df_main[df_main["concurrent_users"] == users].sort_values("context_length")
            ax9.plot(
                data["context_length"] / 1000, data["avg_gpu_clock"],
                marker="H", linewidth=2.5, markersize=9, label=f"{users} users",
                color=colors[idx % len(colors)], markeredgecolor="white", markeredgewidth=1.5
            )
        ax9.set_xlabel("Context Length (K tokens)", fontsize=11, fontweight="bold")
        ax9.set_ylabel("GPU Clock (MHz)", fontsize=11, fontweight="bold")
        ax9.set_title("GPU Clock Frequency", fontsize=12, fontweight="bold", pad=10)
        ax9.set_xticks([c / 1000 for c in context_lengths])
        ax9.set_xticklabels(context_labels)
        ax9.legend(fontsize=9, loc="best")
        ax9.grid(True, alpha=0.3, linestyle="--")
        ax9.set_facecolor("#FAFAFA")
    else:
        ax9.text(0.5, 0.5, "GPU stats not available", ha="center", va="center",
                fontsize=12, transform=ax9.transAxes)
        ax9.axis("off")

    # ROW 5
    # GRAPH 10: Inter-Token Latency
    ax10 = fig.add_subplot(gs[4, 0])
    for idx, users in enumerate(sorted(df_main["concurrent_users"].unique())):
        data = df_main[df_main["concurrent_users"] == users].sort_values("context_length")
        ax10.plot(
            data["context_length"] / 1000, data["inter_token_latency"],
            marker="v", linewidth=2.5, markersize=9, label=f"{users} users",
            color=colors[idx % len(colors)], markeredgecolor="white", markeredgewidth=1.5
        )
    
    # UX quality zones with labels
    ax10.axhspan(0, 50, alpha=0.1, color='green')
    ax10.axhspan(50, 100, alpha=0.1, color='yellow')
    ax10.axhspan(100, 200, alpha=0.1, color='orange')
    
    # Add UX quality labels
    ax10.text(0.98, 0.15, 'INSTANT', transform=ax10.transAxes, 
             fontsize=9, weight='bold', color='darkgreen', alpha=0.7,
             ha='right', va='center', style='italic')
    ax10.text(0.98, 0.35, 'NATURAL', transform=ax10.transAxes,
             fontsize=9, weight='bold', color='darkorange', alpha=0.7,
             ha='right', va='center', style='italic')
    ax10.text(0.98, 0.65, 'DELAYED', transform=ax10.transAxes,
             fontsize=9, weight='bold', color='darkred', alpha=0.7,
             ha='right', va='center', style='italic')
    
    ax10.set_xlabel("Context Length (K tokens)", fontsize=11, fontweight="bold")
    ax10.set_ylabel("Inter-Token Latency (ms)", fontsize=11, fontweight="bold")
    ax10.set_title("Inter-Token Latency (UX Quality)", fontsize=12, fontweight="bold", pad=10)
    ax10.set_xticks([c / 1000 for c in context_lengths])
    ax10.set_xticklabels(context_labels)
    ax10.legend(fontsize=8, loc="upper left")
    ax10.grid(True, alpha=0.3, linestyle="--")
    ax10.set_facecolor("#FAFAFA")

    # GRAPH 11: Batch Scaling Efficiency
    ax11 = fig.add_subplot(gs[4, 1])
    for idx, users in enumerate(sorted(df_main["concurrent_users"].unique())):
        if users == 1:
            continue
        data = df_main[df_main["concurrent_users"] == users].sort_values("context_length")
        ax11.plot(
            data["context_length"] / 1000, data["batch_efficiency"],
            marker="p", linewidth=2.5, markersize=9, label=f"{users} users",
            color=colors[idx % len(colors)], markeredgecolor="white", markeredgewidth=1.5
        )
    ax11.axhline(y=100, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Perfect scaling')
    
    # Performance quality zones
    ax11.axhspan(80, 150, alpha=0.08, color='green')
    ax11.axhspan(50, 80, alpha=0.08, color='yellow')
    ax11.axhspan(0, 50, alpha=0.08, color='red')
    
    ax11.set_xlabel("Context Length (K tokens)", fontsize=11, fontweight="bold")
    ax11.set_ylabel("Scaling Efficiency (%)", fontsize=11, fontweight="bold")
    ax11.set_title("Batch Scaling Efficiency", fontsize=12, fontweight="bold", pad=10)
    ax11.set_xticks([c / 1000 for c in context_lengths])
    ax11.set_xticklabels(context_labels)
    
    # Dynamically set y-axis limits based on data
    if "batch_efficiency" in df_main.columns and len(df_main[df_main["concurrent_users"] > 1]) > 0:
        max_efficiency = df_main[df_main["concurrent_users"] > 1]["batch_efficiency"].max()
        min_efficiency = df_main[df_main["concurrent_users"] > 1]["batch_efficiency"].min()
        y_max = min(max(110, max_efficiency + 10), 200)  # Cap at 200% but ensure 110 minimum
        y_min = max(0, min_efficiency - 5)
        ax11.set_ylim([y_min, y_max])
        
        # Add labels based on visible range
        if y_max > 80:
            label_y_excellent = min(90, (80 + y_max) / 2)
            ax11.text(0.02, label_y_excellent / y_max, 'EXCELLENT', transform=ax11.transAxes,
                     fontsize=9, weight='bold', color='darkgreen', alpha=0.7,
                     ha='left', va='center', style='italic')
        
        if y_min < 80 and y_max > 50:
            ax11.text(0.02, 65 / y_max if y_max > 65 else 0.5, 'GOOD', transform=ax11.transAxes,
                     fontsize=9, weight='bold', color='darkorange', alpha=0.7,
                     ha='left', va='center', style='italic')
        
        if y_min < 50:
            ax11.text(0.02, 25 / y_max if y_max > 25 else 0.2, 'POOR', transform=ax11.transAxes,
                     fontsize=9, weight='bold', color='darkred', alpha=0.7,
                     ha='left', va='center', style='italic')
    else:
        ax11.set_ylim([0, 110])
        ax11.text(0.02, 0.85, 'EXCELLENT', transform=ax11.transAxes,
                 fontsize=9, weight='bold', color='darkgreen', alpha=0.7,
                 ha='left', va='center', style='italic')
        ax11.text(0.02, 0.55, 'GOOD', transform=ax11.transAxes,
                 fontsize=9, weight='bold', color='darkorange', alpha=0.7,
                 ha='left', va='center', style='italic')
    
    ax11.legend(fontsize=8, loc="best")
    ax11.grid(True, alpha=0.3, linestyle="--")
    ax11.set_facecolor("#FAFAFA")

    # GRAPH 12: Decode Speed (Generation)
    ax12 = fig.add_subplot(gs[4, 2])
    for idx, users in enumerate(sorted(df_main["concurrent_users"].unique())):
        data = df_main[df_main["concurrent_users"] == users].sort_values("context_length")
        # Calculate decode speed: completion_tokens / decode_time
        if "decode_time_estimate" in data.columns and "avg_completion_tokens" in data.columns:
            decode_speed = data["avg_completion_tokens"] / (data["decode_time_estimate"] + 0.001)
        else:
            # Fallback: 85% of time is decode
            decode_speed = data["avg_completion_tokens"] / (data["avg_latency"] * 0.85 + 0.001)
        
        ax12.plot(
            data["context_length"] / 1000, decode_speed,
            marker="o", linewidth=2.5, markersize=9, label=f"{users} users",
            color=colors[idx % len(colors)], markeredgecolor="white", markeredgewidth=1.5
        )
    
    ax12.set_xlabel("Context Length (K tokens)", fontsize=11, fontweight="bold")
    ax12.set_ylabel("Decode Speed (tokens/sec)", fontsize=11, fontweight="bold")
    ax12.set_title("Decode Speed (Generation)", fontsize=12, fontweight="bold", pad=10)
    ax12.set_xticks([c / 1000 for c in context_lengths])
    ax12.set_xticklabels(context_labels)
    ax12.legend(fontsize=8, loc="best")
    ax12.grid(True, alpha=0.3, linestyle="--")
    ax12.set_facecolor("#FAFAFA")
    
    # GRAPH 13: Prompt Type Comparison - Prefill Time (if available)
    if has_prompt_types and df["prompt_type"].nunique() >= 2:
        ax13 = fig.add_subplot(gs[5, :])  # Landscape in row 6
        
        # Find best user count to compare (prefer 20, or use max available)
        available_users = sorted(df["concurrent_users"].unique())
        compare_users = 20 if 20 in available_users else max(available_users)
        
        comparison_data = df[df["concurrent_users"] == compare_users]
        
        if len(comparison_data) > 0 and "actual_prefill_time" in comparison_data.columns:
            # Plot each prompt type showing actual prefill time
            prompt_colors = {
                "classic": "#2E86AB",
                "deterministic": "#6A994E", 
                "madlib": "#F18F01",
                "random": "#C73E1D"
            }
            
            for idx, ptype in enumerate(sorted(comparison_data["prompt_type"].unique())):
                data = comparison_data[comparison_data["prompt_type"] == ptype].sort_values("context_length")
                color = prompt_colors.get(ptype, colors[idx % len(colors)])
                
                # Use actual_prefill_time if available, otherwise fall back to estimate
                if "actual_prefill_time" in data.columns and data["actual_prefill_time"].sum() > 0:
                    y_data = data["actual_prefill_time"]
                else:
                    # Fallback to estimate (15% of total latency)
                    y_data = data["prefill_time_estimate"]
                
                ax13.plot(
                    data["context_length"] / 1000, y_data,
                    marker="o", linewidth=3, markersize=10, label=ptype.capitalize(),
                    color=color, markeredgecolor="white", markeredgewidth=1.5
                )
            
            ax13.set_xlabel("Context Length (K tokens)", fontsize=13, fontweight="bold")
            ax13.set_ylabel("Prefill Time (seconds)", fontsize=13, fontweight="bold")
            ax13.set_title(f"Prefill Time by Prompt Type ({compare_users} users)", fontsize=14, fontweight="bold", pad=15)
            ax13.set_xticks([c / 1000 for c in context_lengths])
            ax13.set_xticklabels(context_labels)
            ax13.legend(title="Prompt Type", fontsize=11, title_fontsize=12, loc="best", frameon=True, shadow=True)
            ax13.grid(True, alpha=0.3, linestyle="--")
            ax13.set_facecolor("#FAFAFA")
            
            # Add explanatory note
            ax13.text(0.02, 0.98, 'Lower prefill time indicates better cache performance', 
                     transform=ax13.transAxes, fontsize=9, style='italic',
                     verticalalignment='top', alpha=0.7)
        else:
            ax13.text(0.5, 0.5, f"No data available for {compare_users} users", ha="center", va="center",
                     fontsize=12, transform=ax13.transAxes)
            ax13.axis("off")
    
    # Title
    gpu_info = []
    if system_info:
        if system_info.get("gpu_name"):
            vram = system_info.get("total_vram_gb")
            gpu_info.append(f"{system_info['gpu_name']} ({vram:.0f}GB)" if vram else system_info["gpu_name"])
        if system_info.get("driver_version"):
            gpu_info.append(f"Driver {system_info['driver_version']}")
        if system_info.get("cuda_version"):
            gpu_info.append(f"CUDA {system_info['cuda_version']}")
    if server_info:
        if server_info.get("version"):
            gpu_info.append(f"vLLM {server_info['version']}")
        if server_info.get("backend"):
            gpu_info.append(server_info["backend"])
        if server_info.get("quantization"):
            gpu_info.append(server_info["quantization"])
        if server_info.get("tensor_parallel"):
            gpu_info.append(f"TP={server_info['tensor_parallel']}")
        if server_info.get("pipeline_parallel"):
            gpu_info.append(f"PP={server_info['pipeline_parallel']}")
        if server_info.get("max_num_seqs"):
            gpu_info.append(f"MaxSeqs={server_info['max_num_seqs']}")
        if server_info.get("prefix_caching"):
            gpu_info.append("PrefixCache")

    subtitle = " | ".join(gpu_info) if gpu_info else "Performance Benchmark"
    subtitle += f" | Output: {output_tokens} tokens"
    fig.suptitle(f"{model_name} - Performance Benchmark\n{subtitle}",
                fontsize=13, fontweight="bold", y=0.995,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.5))

    # CACHE HIT RATE HEATMAPS: Add to main figure if available
    if need_cache_rows:
        prompt_types_list = sorted(df["prompt_type"].unique())
        
        for idx, ptype in enumerate(prompt_types_list):
            # Calculate position: 2 heatmaps per row, starting at row 6
            row_offset = 6
            cache_row = row_offset + (idx // 2)
            cache_col = (idx % 2) * 2  # 0 or 2 (span 2 columns each)
            col_span = 2 if idx % 2 == 0 else 1  # Last one spans remaining space
            
            # For odd number of types, last one can span more columns
            if idx == len(prompt_types_list) - 1 and idx % 2 == 0:
                col_span = 3  # Span all 3 columns if it's alone
            
            if col_span == 2:
                ax_cache = fig.add_subplot(gs[cache_row, cache_col:cache_col+2])
            elif col_span == 3:
                ax_cache = fig.add_subplot(gs[cache_row, :])
            else:
                ax_cache = fig.add_subplot(gs[cache_row, 2])
            
            # Filter data for this prompt type
            ptype_data = df[df["prompt_type"] == ptype]
            
            # Create pivot table for heatmap
            pivot_cache = ptype_data.pivot(
                index="context_length",
                columns="concurrent_users",
                values="cache_hit_rate"
            )
            
            # Create heatmap
            sns.heatmap(
                pivot_cache,
                annot=True,
                fmt=".1f",
                cmap="RdYlGn",
                ax=ax_cache,
                cbar_kws={"label": "Hit Rate (%)", "shrink": 0.8},
                linewidths=1.5,
                linecolor="white",
                annot_kws={"fontsize": 9, "weight": "bold"},
                vmin=0,
                vmax=100
            )
            
            ax_cache.set_xlabel("Concurrent Users", fontsize=11, fontweight="bold")
            ax_cache.set_ylabel("Context Length", fontsize=11, fontweight="bold")
            ax_cache.set_title(f"Cache Hit Rate: {ptype.capitalize()}", fontsize=12, fontweight="bold", pad=10)
            ax_cache.set_yticklabels(
                [f"{int(y / 1000)}K" for y in pivot_cache.index],
                rotation=0,
                fontsize=9
            )
    
    # Save - generate timestamp and filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = sanitize_filename(model_name)
    output_path = ensure_output_directory()
    filename = output_path / f"benchmark_{safe_model_name}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n[INFO] Visualization saved: {filename}")
    
    return str(filename)
def print_summary_table(all_results: List[Dict]) -> None:
    """
    Print detailed performance summary tables to console.

    Args:
        all_results: List of benchmark result dictionaries
    """
    df = pd.DataFrame(all_results)
    has_gpu_stats = "avg_gpu_util" in df.columns
    has_energy_stats = "watts_per_token_per_user" in df.columns
    has_cache_stats = "cache_hit_rate" in df.columns

    print("\n" + "=" * 180)
    print("DETAILED PERFORMANCE SUMMARY")
    print("=" * 180)

    # Per-context results
    for context in sorted(df["context_length"].unique()):
        context_data = df[df["context_length"] == context].sort_values(
            "concurrent_users"
        )
        print(f"\nContext Length: {context:,} tokens ({context / 1000:.0f}K)")
        print("-" * 180)

        if has_gpu_stats and has_energy_stats and has_cache_stats:
            print(
                f"{'Users':<8} {'Latency(s)':<12} {'Tok/s':<10} {'Req/s':<10} {'TTFT(ms)':<10} "
                f"{'GPU%':<8} {'Temp(C)':<10} {'Power(W)':<10} {'W/tok/usr':<12} {'Cache%':<10} {'Success%':<10}"
            )
            print("-" * 180)

            for _, row in context_data.iterrows():
                success_rate = (
                    row["successful"] / (row["successful"] + row["failed"])
                ) * 100
                cache_hit = row.get("cache_hit_rate", 0)
                print(
                    f"{row['concurrent_users']:<8} "
                    f"{row['avg_latency']:<12.2f} "
                    f"{row['tokens_per_second']:<10.1f} "
                    f"{row['requests_per_second']:<10.2f} "
                    f"{row['ttft_estimate'] * 1000:<10.0f} "
                    f"{row['avg_gpu_util']:<8.1f} "
                    f"{row['avg_temperature']:<10.1f} "
                    f"{row['avg_power']:<10.1f} "
                    f"{row['watts_per_token_per_user']:<12.4f} "
                    f"{cache_hit:<10.1f} "
                    f"{success_rate:<10.1f}"
                )
        elif has_gpu_stats and has_energy_stats:
            print(
                f"{'Users':<8} {'Latency(s)':<12} {'Tok/s':<10} {'Req/s':<10} {'TTFT(ms)':<10} "
                f"{'GPU%':<8} {'Temp(C)':<10} {'Power(W)':<10} {'W/tok/usr':<12} {'Success%':<10}"
            )
            print("-" * 180)

            for _, row in context_data.iterrows():
                success_rate = (
                    row["successful"] / (row["successful"] + row["failed"])
                ) * 100
                print(
                    f"{row['concurrent_users']:<8} "
                    f"{row['avg_latency']:<12.2f} "
                    f"{row['tokens_per_second']:<10.1f} "
                    f"{row['requests_per_second']:<10.2f} "
                    f"{row['ttft_estimate'] * 1000:<10.0f} "
                    f"{row['avg_gpu_util']:<8.1f} "
                    f"{row['avg_temperature']:<10.1f} "
                    f"{row['avg_power']:<10.1f} "
                    f"{row['watts_per_token_per_user']:<12.4f} "
                    f"{success_rate:<10.1f}"
                )
        elif has_gpu_stats:
            print(
                f"{'Users':<8} {'Latency(s)':<12} {'Tok/s':<10} {'Req/s':<10} {'TTFT(ms)':<10} "
                f"{'GPU%':<8} {'Temp(C)':<10} {'Power(W)':<10} {'GPU MHz':<10} {'Success%':<10}"
            )
            print("-" * 160)

            for _, row in context_data.iterrows():
                success_rate = (
                    row["successful"] / (row["successful"] + row["failed"])
                ) * 100
                print(
                    f"{row['concurrent_users']:<8} "
                    f"{row['avg_latency']:<12.2f} "
                    f"{row['tokens_per_second']:<10.1f} "
                    f"{row['requests_per_second']:<10.2f} "
                    f"{row['ttft_estimate'] * 1000:<10.0f} "
                    f"{row['avg_gpu_util']:<8.1f} "
                    f"{row['avg_temperature']:<10.1f} "
                    f"{row['avg_power']:<10.1f} "
                    f"{row['avg_gpu_clock']:<10.0f} "
                    f"{success_rate:<10.1f}"
                )
        else:
            print(
                f"{'Users':<8} {'Latency(s)':<12} {'Tokens/s':<12} {'Req/s':<10} "
                f"{'TTFT(ms)':<12} {'Tok/s/User':<15} {'Success%':<10}"
            )
            print("-" * 160)

            for _, row in context_data.iterrows():
                success_rate = (
                    row["successful"] / (row["successful"] + row["failed"])
                ) * 100
                print(
                    f"{row['concurrent_users']:<8} "
                    f"{row['avg_latency']:<12.2f} "
                    f"{row['tokens_per_second']:<12.1f} "
                    f"{row['requests_per_second']:<10.2f} "
                    f"{row['ttft_estimate'] * 1000:<12.0f} "
                    f"{row['throughput_per_user']:<15.1f} "
                    f"{success_rate:<10.1f}"
                )

    # Optimal configurations
    print("\n" + "=" * 160)
    print("OPTIMAL CONFIGURATIONS")
    print("=" * 160)

    max_throughput = df.loc[df["tokens_per_second"].idxmax()]
    print(f"\nMaximum Throughput:")
    print(
        f"  {max_throughput['tokens_per_second']:.1f} tokens/s at {max_throughput['concurrent_users']} users "
        f"with {max_throughput['context_length'] / 1000:.0f}K context"
    )
    if has_gpu_stats:
        print(
            f"  GPU: {max_throughput['avg_gpu_util']:.1f}% util, "
            f"{max_throughput['avg_temperature']:.1f}C, {max_throughput['avg_power']:.1f}W"
        )
    if has_energy_stats:
        print(f"  Energy: {max_throughput['watts_per_token_per_user']:.4f} W/tok/user")

    best_efficiency = df.loc[df["throughput_per_user"].idxmax()]
    print(f"\nBest Efficiency (tokens/s per user):")
    print(
        f"  {best_efficiency['throughput_per_user']:.1f} tokens/s/user at {best_efficiency['concurrent_users']} users "
        f"with {best_efficiency['context_length'] / 1000:.0f}K context"
    )

    min_latency = df.loc[df["avg_latency"].idxmin()]
    print(f"\nLowest Latency:")
    print(
        f"  {min_latency['avg_latency']:.2f}s at {min_latency['concurrent_users']} users "
        f"with {min_latency['context_length'] / 1000:.0f}K context"
    )

    best_req_throughput = df.loc[df["requests_per_second"].idxmax()]
    print(f"\nHighest Request Throughput:")
    print(
        f"  {best_req_throughput['requests_per_second']:.2f} req/s at {best_req_throughput['concurrent_users']} users "
        f"with {best_req_throughput['context_length'] / 1000:.0f}K context"
    )

    # Energy analysis
    if has_energy_stats:
        print(f"\nEnergy Analysis:")
        print(f"  Best: {df['watts_per_token_per_user'].min():.4f} W/tok/user")
        print(f"  Worst: {df['watts_per_token_per_user'].max():.4f} W/tok/user")
        print(f"  Average: {df['watts_per_token_per_user'].mean():.4f} W/tok/user")

        if "tokens_per_watt" in df.columns:
            print(f"\nEnergy Efficiency (tokens per watt):")
            print(f"  Best: {df['tokens_per_watt'].max():.2f} tok/W")
            print(f"  Worst: {df['tokens_per_watt'].min():.2f} tok/W")
            print(f"  Average: {df['tokens_per_watt'].mean():.2f} tok/W")

        if "energy_watt_hours" in df.columns:
            total_wh = df["energy_watt_hours"].sum()
            print(
                f"\n  Total energy consumed: {df['energy_joules'].sum():.0f} J ({total_wh:.4f} Wh / {total_wh * 1000:.2f} mWh)"
            )

        if "watts_per_token_per_user_per_1k_context" in df.columns:
            print(f"\nNormalized Energy Efficiency (per 1K context):")
            print(
                f"  Best: {df['watts_per_token_per_user_per_1k_context'].min():.6f} W/tok/usr/1K"
            )
            print(
                f"  Worst: {df['watts_per_token_per_user_per_1k_context'].max():.6f} W/tok/usr/1K"
            )
            print(
                f"  Average: {df['watts_per_token_per_user_per_1k_context'].mean():.6f} W/tok/usr/1K"
            )

            # Energy efficiency by context size
            print(f"\n  Energy efficiency by context size:")
            for ctx in sorted(df["context_length"].unique()):
                ctx_data = df[df["context_length"] == ctx]
                avg_eff = ctx_data["watts_per_token_per_user_per_1k_context"].mean()
                print(
                    f"    {ctx // 1000}K context: {avg_eff:.6f} W/tok/usr/1K (avg across all user counts)"
                )

    # Scaling analysis
    print(f"\nContext Scaling Analysis:")
    single_user = df[df["concurrent_users"] == 1].sort_values("context_length")
    if len(single_user) > 1:
        baseline_throughput = single_user.iloc[0]["tokens_per_second"]
        max_context_throughput = single_user.iloc[-1]["tokens_per_second"]
        degradation = (
            (baseline_throughput - max_context_throughput) / baseline_throughput
        ) * 100
        print(
            f"  Throughput degradation from 1K to {single_user.iloc[-1]['context_length'] / 1000:.0f}K: {degradation:.1f}%"
        )

    # Cache efficiency
    if has_cache_stats:
        print(f"\nCache Hit Rate Analysis:")
        print(f"  Best: {df['cache_hit_rate'].max():.1f}%")
        print(f"  Worst: {df['cache_hit_rate'].min():.1f}%")
        print(f"  Average: {df['cache_hit_rate'].mean():.1f}%")
        
        # If prompt types available, show by prompt type
        if "prompt_type" in df.columns:
            print(f"\n  Cache hit rate by prompt type:")
            for ptype in sorted(df["prompt_type"].unique()):
                ptype_data = df[df["prompt_type"] == ptype]
                avg_cache = ptype_data["cache_hit_rate"].mean()
                print(f"    {ptype.capitalize()}: {avg_cache:.1f}%")

    # GPU efficiency
    if has_gpu_stats:
        print(f"\nGPU Efficiency Analysis:")
        max_gpu_util = df.loc[df["avg_gpu_util"].idxmax()]
        print(
            f"  Peak GPU utilization: {max_gpu_util['avg_gpu_util']:.1f}% at {max_gpu_util['concurrent_users']} users "
            f"with {max_gpu_util['context_length'] / 1000:.0f}K context"
        )
        print(f"  Peak temperature: {df['max_temperature'].max():.1f}C")
        print(f"  Peak power draw: {df['max_power'].max():.1f}W")
        print(f"  Average power draw: {df['avg_power'].mean():.1f}W")


def sanitize_filename(name: str) -> str:
    """
    Sanitize string for use in filenames.

    Args:
        name: Input string (e.g., model name)

    Returns:
        Filesystem-safe string
    """
    # Replace slashes and special characters
    safe_name = name.replace("/", "_").replace("\\", "_")
    safe_name = re.sub(r"[^\w\-.]", "_", safe_name)
    # Remove consecutive underscores
    safe_name = re.sub(r"_+", "_", safe_name)
    # Trim to reasonable length
    return safe_name[:100]


def ensure_output_directory() -> Path:
    """
    Create output directory if it doesn't exist.

    Returns:
        Path object for output directory
    """
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def warmup_model(model_name: str, output_tokens: int = 100) -> bool:
    """
    Execute warmup inference to initialize GPU kernels and caches.

    Args:
        model_name: Model identifier for API request
        output_tokens: Number of tokens for warmup generation

    Returns:
        True if successful, False otherwise
    """
    console.print("\n[yellow]Warming up model (1K context, single user)...[/yellow]")

    prompt = generate_prompt(1000)
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_tokens,
        "temperature": 0.7,
    }

    try:
        with console.status(
            "[bold yellow]Executing warmup inference...", spinner="dots"
        ):
            start = time.time()
            response = requests.post(API_ENDPOINT, json=data, timeout=REQUEST_TIMEOUT)
            duration = time.time() - start

            if response.status_code == 200:
                result = response.json()
                usage = result.get("usage", {})
                completion_tokens = usage.get("completion_tokens", 0)

                console.print(
                    f"[green]✓[/green] Warmup complete: {duration:.2f}s, "
                    f"{completion_tokens} tokens generated"
                )
                return True
            else:
                console.print(
                    f"[red]✗[/red] Warmup failed: HTTP {response.status_code}"
                )
                return False

    except Exception as e:
        console.print(f"[red]✗[/red] Warmup error: {str(e)}")
        return False


def get_interactive_config() -> Tuple[List[int], List[int], int, List[str]]:
    """
    Interactive CLI for benchmark configuration.

    Returns:
        Tuple of (context_lengths, concurrent_users, output_tokens, prompt_types)
    """
    console.print("\n[bold cyan]••• Benchmark Configuration •••[/bold cyan]\n")

    # Select max context length
    console.print("[bold]Select maximum context length:[/bold]")
    console.print("  [1] 32K")
    console.print("  [2] 64K")
    console.print("  [3] 128K")
    console.print("  [4] 256K")
    console.print("  [5] 512K")
    console.print("  [6] 1024K (1M)\n")
    
    max_choice = IntPrompt.ask("Select max context", default=3, choices=["1", "2", "3", "4", "5", "6"])
    
    max_context_map = {
        1: 32,
        2: 64,
        3: 128,
        4: 256,
        5: 512,
        6: 1024
    }
    max_context = max_context_map[max_choice]
    
    # Generate standard context lengths up to max
    all_standard_contexts = [1, 10, 32, 64, 96, 128, 160, 192, 224, 256, 384, 512, 768, 1024]
    context_lengths = [c * 1000 for c in all_standard_contexts if c <= max_context]
    
    # Select max concurrent users
    console.print("\n[bold]Select maximum concurrent users:[/bold]")
    console.print("  [1] 1 user")
    console.print("  [2] 2 users")
    console.print("  [3] 5 users")
    console.print("  [4] 10 users")
    console.print("  [5] 20 users")
    console.print("  [6] 50 users")
    console.print("  [7] Custom\n")
    
    user_choice = IntPrompt.ask("Select max users", default=4, choices=["1", "2", "3", "4", "5", "6", "7"])
    
    if user_choice == 7:
        # Custom users
        max_users = IntPrompt.ask("Enter max concurrent users", default=10)
    else:
        user_map = {1: 1, 2: 2, 3: 5, 4: 10, 5: 20, 6: 50}
        max_users = user_map[user_choice]
    
    # Generate user counts up to max
    all_user_levels = [1, 2, 5, 10, 20, 50, 100]
    concurrent_users = [u for u in all_user_levels if u <= max_users]
    
    # Output tokens
    console.print("\n[bold]Select output length:[/bold]")
    console.print("  [1] Short summaries (100-200 tokens)")
    console.print("  [2] Standard responses (500 tokens)")
    console.print("  [3] Long reports (1000-2000 tokens)")
    console.print("  [4] Custom\n")
    
    output_choice = IntPrompt.ask("Select output length", default=2, choices=["1", "2", "3", "4"])
    
    if output_choice == 1:
        output_tokens = 150
        output_label = "Short (150 tokens)"
    elif output_choice == 2:
        output_tokens = 500
        output_label = "Standard (500 tokens)"
    elif output_choice == 3:
        output_tokens = 1500
        output_label = "Long (1500 tokens)"
    else:
        output_tokens = IntPrompt.ask("Enter output tokens per request", default=500)
        output_label = f"Custom ({output_tokens} tokens)"
    
    # Prompt type selection
    console.print("\n[bold]Select prompt types to test:[/bold]")
    console.print("  [1] Classic only (default cybersecurity prompt)")
    console.print("  [2] Deterministic only (high cache hit)")
    console.print("  [3] Madlib only (moderate cache hit)")
    console.print("  [4] Random only (low cache hit)")
    console.print("  [5] All three new types (deterministic + madlib + random)")
    console.print("  [6] All four types (classic + deterministic + madlib + random)\n")
    
    prompt_choice = IntPrompt.ask("Select prompt types", default=5, choices=["1", "2", "3", "4", "5", "6"])
    
    if prompt_choice == 1:
        prompt_types = ["classic"]
        prompt_label = "Classic only"
    elif prompt_choice == 2:
        prompt_types = ["deterministic"]
        prompt_label = "Deterministic only"
    elif prompt_choice == 3:
        prompt_types = ["madlib"]
        prompt_label = "Madlib only"
    elif prompt_choice == 4:
        prompt_types = ["random"]
        prompt_label = "Random only"
    elif prompt_choice == 5:
        prompt_types = ["deterministic", "madlib", "random"]
        prompt_label = "All three new types"
    else:  # 6
        prompt_types = ["classic", "deterministic", "madlib", "random"]
        prompt_label = "All four types"
    
    # Show what will be tested
    console.print(f"\n[dim]Context lengths: {', '.join([f'{c//1000}K' for c in context_lengths])}[/dim]")
    console.print(f"[dim]Concurrent users: {', '.join([str(u) for u in concurrent_users])}[/dim]")
    console.print(f"[dim]Prompt types: {', '.join(prompt_types)}[/dim]\n")

    # Display configuration summary
    console.print("[bold green]Configuration Summary:[/bold green]")

    config_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="yellow")

    config_table.add_row("Max Context", f"{max_context}K tokens")
    config_table.add_row(
        "Context Lengths",
        f"{len(context_lengths)} levels: " + ", ".join([f"{c // 1000}K" for c in context_lengths]),
    )
    config_table.add_row("Max Users", str(max_users))
    config_table.add_row(
        "Concurrent Users",
        f"{len(concurrent_users)} levels: " + ", ".join([str(u) for u in concurrent_users]),
    )
    config_table.add_row("Output Length", output_label if 'output_label' in locals() else str(output_tokens))
    config_table.add_row("Prompt Types", prompt_label if 'prompt_label' in locals() else ', '.join(prompt_types))
    config_table.add_row("Total Tests", str(len(context_lengths) * len(concurrent_users) * len(prompt_types)))

    est_time = len(context_lengths) * len(concurrent_users) * len(prompt_types) * 30
    config_table.add_row("Est. Duration", f"{est_time // 60} min")

    console.print(config_table)

    if not Confirm.ask("\nProceed with this configuration?", default=True):
        console.print("[yellow]Benchmark cancelled.[/yellow]")
        sys.exit(0)

    return context_lengths, concurrent_users, output_tokens, prompt_types


def create_live_dashboard(
    test_num: int,
    total_tests: int,
    context_length: int,
    concurrent_users: int,
    elapsed_time: float,
    current_gpu: Optional[Dict] = None,
    all_results: List[Dict] = None,
    remaining_tests: List[Tuple[int, int, str]] = None,
    all_gpu_history: List[Dict] = None,
    total_benchmark_time: float = 0,
) -> Layout:
    """
    Create simple dashboard with progress bars only.

    Args:
        test_num: Current test number
        total_tests: Total number of tests
        context_length: Current context length
        concurrent_users: Current concurrent users
        elapsed_time: Elapsed time for current test
        current_gpu: Real-time GPU statistics
        all_results: All completed results so far
        remaining_tests: List of (context, users) tuples for remaining tests
        all_gpu_history: Historical GPU samples for entire benchmark (persistent)
        total_benchmark_time: Total elapsed time for entire benchmark

    Returns:
        Rich Layout object with dashboard
    """
    # Dynamically adjust remaining tests panel size to show all tests
    remaining_size = min(max(8, len(remaining_tests) + 3 if remaining_tests else 6), 35)
    
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="overall_progress", size=5),
        Layout(name="current_test", size=8),
        Layout(name="remaining", size=remaining_size),
    )

    # Header - Basic info
    header_text = Text()
    header_text.append(f"vLLM Benchmark - ", style="bold cyan")
    header_text.append(f"Test {test_num}/{total_tests}", style="bold yellow")
    
    if total_benchmark_time > 0:
        mins = int(total_benchmark_time // 60)
        secs = int(total_benchmark_time % 60)
        header_text.append(f"  |  Runtime: {mins}m {secs}s", style="dim")

    layout["header"].update(Panel(header_text, style="cyan"))

    # Overall Progress Bar
    progress_pct = (test_num / total_tests) * 100
    bar_width = 60
    filled = int((test_num / total_tests) * bar_width)
    overall_bar = "█" * filled + "░" * (bar_width - filled)
    
    progress_text = Text()
    progress_text.append("OVERALL PROGRESS\n", style="bold green")
    progress_text.append(f"{overall_bar} ", style="green")
    progress_text.append(f"{progress_pct:.1f}%\n", style="bold green")
    progress_text.append(f"Completed: {test_num}  Remaining: {total_tests - test_num}", style="dim")
    
    layout["overall_progress"].update(
        Panel(progress_text, title="Benchmark Progress", border_style="green")
    )

    # Current Test Progress (simulated based on elapsed time)
    test_info = Table(show_header=False, box=box.SIMPLE, border_style="cyan")
    test_info.add_column("", style="cyan", width=15)
    test_info.add_column("", style="yellow")
    
    test_info.add_row("Context", f"{context_length // 1000}K tokens")
    test_info.add_row("Users", str(concurrent_users))
    test_info.add_row("Elapsed", f"{elapsed_time:.1f}s")
    
    if current_gpu:
        util = current_gpu.get("gpu_util", 0)
        util_color = "red" if util > 95 else "yellow" if util > 80 else "green"
        test_info.add_row("GPU", f"[{util_color}]{util:.0f}%[/{util_color}]")
    
    test_info.add_row("Status", "[bold yellow]RUNNING[/bold yellow]")
    
    layout["current_test"].update(
        Panel(
            test_info,
            title=f"Current Test ({test_num}/{total_tests})",
            border_style="yellow"
        )
    )

    # Remaining Tests Queue
    if remaining_tests and len(remaining_tests) > 0:
        queue_text = Text()
        queue_text.append(f"Remaining tests:\n\n", style="bold blue")
        
        for i, (ctx, users, ptype) in enumerate(remaining_tests):
            queue_text.append(f"  {i+1}. {ctx // 1000}K × {users} users × {ptype}\n", style="dim")
        
        layout["remaining"].update(
            Panel(queue_text, title=f"Queue ({len(remaining_tests)} remaining)", border_style="blue")
        )
    else:
        layout["remaining"].update(
            Panel("Final test running", title="Queue", border_style="green")
        )

    return layout


def main():
    """Main benchmark execution routine with interactive CLI and live display."""

    # Header
    console.print(
        Panel.fit(
            "[bold cyan]vLLM Performance Benchmark Suite[/bold cyan]\n"
            "[dim]Enhanced Edition v2.0 - Interactive Mode[/dim]",
            border_style="cyan",
        )
    )

    # Collect system information
    console.print("\n[yellow]Initializing system detection...[/yellow]")
    system_info = SystemInfo.get_system_info()

    # Query vLLM server information
    console.print("[yellow]Querying vLLM server...[/yellow]")
    server_info = VLLMServerInfo.get_server_info()
    model_name = server_info.get("model_name") or DEFAULT_MODEL_NAME

    # Display system information panel
    sys_table = Table(show_header=False, box=box.ROUNDED, border_style="green")
    sys_table.add_column("", style="cyan bold")
    sys_table.add_column("", style="yellow")

    sys_table.add_row("Python", system_info["python_version"])
    sys_table.add_row("Platform", system_info["platform"])
    if system_info["gpu_name"]:
        vram_str = (
            f" ({system_info['total_vram_gb']:.0f}GB)"
            if system_info["total_vram_gb"]
            else ""
        )
        sys_table.add_row("GPU", system_info["gpu_name"] + vram_str)
    if system_info["driver_version"]:
        sys_table.add_row("Driver", system_info["driver_version"])
    if system_info["cuda_version"]:
        sys_table.add_row("CUDA", system_info["cuda_version"])

    console.print(
        Panel(
            sys_table,
            title="[bold green]System Information[/bold green]",
            border_style="green",
        )
    )

    # Display vLLM server information panel
    server_table = Table(show_header=False, box=box.ROUNDED, border_style="blue")
    server_table.add_column("", style="cyan bold")
    server_table.add_column("", style="yellow")

    server_table.add_row("Model", model_name)
    server_table.add_row("Endpoint", API_BASE_URL)
    if server_info.get("version"):
        server_table.add_row("vLLM Version", server_info["version"])
    if server_info.get("backend"):
        server_table.add_row("Attention Backend", server_info["backend"])
    if server_info.get("quantization"):
        server_table.add_row("Quantization", server_info["quantization"])
    if server_info.get("tensor_parallel"):
        server_table.add_row("Tensor Parallel", str(server_info["tensor_parallel"]))
    if server_info.get("pipeline_parallel"):
        server_table.add_row("Pipeline Parallel", str(server_info["pipeline_parallel"]))
    if server_info.get("max_num_seqs"):
        server_table.add_row("Max Batch Size", str(server_info["max_num_seqs"]))
    if server_info.get("gpu_memory_utilization"):
        server_table.add_row("GPU Mem Util", f"{server_info['gpu_memory_utilization']:.1%}")
    if server_info.get("prefix_caching"):
        server_table.add_row("Prefix Caching", "Enabled" if server_info["prefix_caching"] else "Disabled")
    if server_info.get("kv_cache_usage") is not None:
        server_table.add_row("KV Cache Usage", f"{server_info['kv_cache_usage']:.1f}%")
    if server_info.get("max_model_len"):
        server_table.add_row("Max Context", f"{server_info['max_model_len']:,} tokens")

    console.print(
        Panel(
            server_table,
            title="[bold blue]vLLM Configuration[/bold blue]",
            border_style="blue",
        )
    )

    # Interactive configuration
    context_lengths, concurrent_users, output_tokens, prompt_types = get_interactive_config()

    # Warmup phase
    console.print("\n[bold cyan]••• Model Warmup Phase •••[/bold cyan]")
    warmup_success = warmup_model(model_name, output_tokens=100)

    if not warmup_success:
        if not Confirm.ask(
            "[yellow]Warmup failed. Continue with benchmark?[/yellow]", default=False
        ):
            console.print("[red]Benchmark cancelled.[/red]")
            sys.exit(1)

    # Brief pause after warmup
    console.print("[dim]Waiting 3 seconds before benchmark execution...[/dim]")
    time.sleep(3)

    all_results = []
    total_tests = len(context_lengths) * len(concurrent_users) * len(prompt_types)
    current_test = 0
    start_time_all = time.time()
    benchmark_start_time = time.time()  # Track total benchmark time

    console.print("\n[bold green]Starting benchmark execution...[/bold green]\n")

    # Create test queue with prompt types
    test_queue = [(ctx, users, ptype) for ptype in prompt_types for ctx in context_lengths for users in concurrent_users]

    # Initialize shared GPU monitor for all tests
    gpu_monitor = GPUMonitor()
    gpu_monitor.start()
    
    # Persistent GPU history across entire benchmark
    all_gpu_history = []

    # Execute benchmarks with live dashboard
    with Live(console=console, refresh_per_second=DASHBOARD_REFRESH_RATE) as live_display:
        for idx, (context, users, ptype) in enumerate(test_queue):
            current_test = idx + 1
            remaining = test_queue[idx + 1 :]

            test_start_time = time.time()

            # Create test execution thread
            test_result = [None]  # Mutable container for thread result

            def run_test():
                test_result[0] = run_benchmark(
                    context,
                    users,
                    output_tokens=output_tokens,
                    model_name=model_name,
                    live_display=live_display,
                    gpu_monitor=gpu_monitor,
                    prompt_type=ptype,
                )

            import threading as test_threading

            test_thread = test_threading.Thread(target=run_test)
            test_thread.start()

            # Update dashboard during test execution
            while test_thread.is_alive():
                elapsed = time.time() - test_start_time
                total_elapsed = time.time() - benchmark_start_time
                current_gpu = gpu_monitor.get_gpu_stats()

                # Append to persistent history
                if current_gpu:
                    all_gpu_history.append(current_gpu)

                dashboard = create_live_dashboard(
                    test_num=current_test,
                    total_tests=total_tests,
                    context_length=context,
                    concurrent_users=users,
                    elapsed_time=elapsed,
                    current_gpu=current_gpu,
                    all_results=all_results,
                    remaining_tests=remaining,
                    all_gpu_history=all_gpu_history,  # Use persistent history
                    total_benchmark_time=total_elapsed,
                )

                live_display.update(dashboard)
                time.sleep(0.5)

            test_thread.join()
            result = test_result[0]

            if result:
                # Calculate GPU stats from persistent history for this test
                # Find the samples that correspond to this test
                test_gpu_samples = []
                test_end_time = time.time()
                for sample in reversed(all_gpu_history):
                    if sample.get("timestamp", 0) >= test_start_time:
                        test_gpu_samples.insert(0, sample)
                    else:
                        break
                
                if test_gpu_samples:
                    result.update(
                        {
                            "avg_gpu_util": mean(
                                [s.get("gpu_util", 0) for s in test_gpu_samples]
                            ),
                            "max_gpu_util": max(
                                [s.get("gpu_util", 0) for s in test_gpu_samples]
                            ),
                            "avg_mem_used": mean(
                                [s.get("mem_used", 0) for s in test_gpu_samples]
                            ),
                            "max_mem_used": max(
                                [s.get("mem_used", 0) for s in test_gpu_samples]
                            ),
                            "avg_temperature": mean(
                                [s.get("temperature", 0) for s in test_gpu_samples]
                            ),
                            "max_temperature": max(
                                [s.get("temperature", 0) for s in test_gpu_samples]
                            ),
                            "avg_power": mean(
                                [s.get("power_draw", 0) for s in test_gpu_samples]
                            ),
                            "max_power": max(
                                [s.get("power_draw", 0) for s in test_gpu_samples]
                            ),
                            "avg_gpu_clock": mean(
                                [s.get("gpu_clock", 0) for s in test_gpu_samples]
                            ),
                            "max_gpu_clock": max(
                                [s.get("gpu_clock", 0) for s in test_gpu_samples]
                            ),
                            "avg_mem_clock": mean(
                                [s.get("mem_clock", 0) for s in test_gpu_samples]
                            ),
                        }
                    )

                all_results.append(result)

                # Show completion message
                summary = (
                    f"[green]OK[/green] Test {current_test}/{total_tests} - "
                    f"{context // 1000}K ctx x {users} users x {ptype} prompt: "
                    f"[bold yellow]{result['tokens_per_second']:.1f}[/bold yellow] tok/s, "
                    f"[bold cyan]{result['avg_latency']:.2f}s[/bold cyan] latency, "
                    f"[dim]{result['total_time']:.0f}s[/dim]"
                )
                if "avg_gpu_util" in result:
                    summary += f", [bold magenta]{result['avg_gpu_util']:.0f}%[/bold magenta] GPU"

                console.print(summary)

            # Pause between tests
            if current_test < total_tests:
                time.sleep(TEST_PAUSE_DURATION)

    # Stop GPU monitoring
    gpu_monitor.stop()

    total_benchmark_time = time.time() - start_time_all

    # Save results with metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = sanitize_filename(model_name)
    output_path = ensure_output_directory()

    metadata = {
        "timestamp": timestamp,
        "benchmark_duration": total_benchmark_time,
        "system_info": system_info,
        "server_info": server_info,
        "configuration": {
            "context_lengths": context_lengths,
            "concurrent_users": concurrent_users,
            "output_tokens": output_tokens,
            "prompt_types": prompt_types,
            "pause_duration": TEST_PAUSE_DURATION,
        },
    }

    results_package = {"metadata": metadata, "results": all_results}

    json_filename = output_path / f"benchmark_{safe_model_name}_{timestamp}.json"
    with open(json_filename, "w") as f:
        json.dump(results_package, f, indent=2)

    # Display final summary
    console.print(f"\n[bold green]{'=' * 80}[/bold green]")
    console.print(f"[bold cyan]Benchmark Complete[/bold cyan]")
    console.print(f"[bold green]{'=' * 80}[/bold green]\n")

    # Quick summary table
    df = pd.DataFrame(all_results)

    # Save to CSV
    csv_filename = output_path / f"benchmark_{safe_model_name}_{timestamp}.csv"
    df.to_csv(csv_filename)

    summary_table = Table(
        show_header=True,
        header_style="bold magenta",
        box=box.DOUBLE,
        title="Performance Highlights",
    )
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="yellow", justify="right")
    summary_table.add_column("Configuration", style="green")

    max_throughput = df.loc[df["tokens_per_second"].idxmax()]
    summary_table.add_row(
        "Peak Throughput",
        f"{max_throughput['tokens_per_second']:.1f} tok/s",
        f"{max_throughput['concurrent_users']} users @ {max_throughput['context_length'] // 1000}K",
    )

    best_efficiency = df.loc[df["throughput_per_user"].idxmax()]
    summary_table.add_row(
        "Best Efficiency",
        f"{best_efficiency['throughput_per_user']:.1f} tok/s/user",
        f"{best_efficiency['concurrent_users']} users @ {best_efficiency['context_length'] // 1000}K",
    )

    min_latency = df.loc[df["avg_latency"].idxmin()]
    summary_table.add_row(
        "Lowest Latency",
        f"{min_latency['avg_latency']:.2f}s",
        f"{min_latency['concurrent_users']} users @ {min_latency['context_length'] // 1000}K",
    )

    if "avg_gpu_util" in df.columns:
        max_gpu = df.loc[df["avg_gpu_util"].idxmax()]
        summary_table.add_row(
            "Peak GPU Util",
            f"{max_gpu['avg_gpu_util']:.1f}%",
            f"{max_gpu['concurrent_users']} users @ {max_gpu['context_length'] // 1000}K",
        )

    console.print(summary_table)

    # Energy efficiency summary
    if "tokens_per_watt" in df.columns:
        console.print("\n")
        energy_table = Table(
            show_header=True,
            header_style="bold green",
            box=box.DOUBLE,
            title="Energy Efficiency Analysis",
        )
        energy_table.add_column("Metric", style="cyan")
        energy_table.add_column("Value", style="yellow", justify="right")
        energy_table.add_column("Configuration", style="green")

        # Most energy efficient (tokens per watt)
        best_energy = df.loc[df["tokens_per_watt"].idxmax()]
        energy_table.add_row(
            "Best Energy Efficiency",
            f"{best_energy['tokens_per_watt']:.2f} tok/W",
            f"{best_energy['concurrent_users']} users @ {best_energy['context_length'] // 1000}K",
        )

        # Lowest watts per token per user
        if "watts_per_token_per_user" in df.columns:
            lowest_watts = df.loc[df["watts_per_token_per_user"].idxmin()]
            energy_table.add_row(
                "Lowest Power/Tok/User",
                f"{lowest_watts['watts_per_token_per_user']:.4f} W",
                f"{lowest_watts['concurrent_users']} users @ {lowest_watts['context_length'] // 1000}K",
            )

        # Best normalized efficiency (per context)
        if "watts_per_token_per_user_per_1k_context" in df.columns:
            best_normalized = df.loc[
                df["watts_per_token_per_user_per_1k_context"].idxmin()
            ]
            energy_table.add_row(
                "Best Normalized Efficiency",
                f"{best_normalized['watts_per_token_per_user_per_1k_context']:.6f} W/tok/usr/1K",
                f"{best_normalized['concurrent_users']} users @ {best_normalized['context_length'] // 1000}K",
            )

        # Total energy consumed
        if "energy_watt_hours" in df.columns:
            total_wh = df["energy_watt_hours"].sum()
            energy_table.add_row(
                "Total Energy Used",
                f"{total_wh:.4f} Wh ({total_wh * 1000:.2f} mWh)",
                "All tests combined",
            )

        console.print(energy_table)

    # File outputs
    console.print(f"\n[bold cyan]Outputs:[/bold cyan]")
    console.print(
        f"  [green]*[/green] Results: [link=file://{json_filename}]{json_filename}[/link]"
    )

    # Generate visualizations
    console.print(f"\n[yellow]Generating visualizations...[/yellow]")
    viz_filename = visualize_results(all_results, model_name, system_info, server_info, output_tokens)
    console.print(
        f"  [green]*[/green] Charts: [link=file://{viz_filename}]{viz_filename}[/link]"
    )

    # Detailed summary
    if Confirm.ask("\n[cyan]Show detailed summary table?[/cyan]", default=False):
        print_summary_table(all_results)

    # Display total time prominently
    console.print(f"\n[bold green]{'=' * 80}[/bold green]")
    console.print(
        f"[bold cyan]Total Benchmark Time: {total_benchmark_time / 60:.1f} minutes ({total_benchmark_time:.0f} seconds)[/bold cyan]"
    )
    if all_results:
        total_test_time = sum(r["total_time"] for r in all_results)
        overhead = total_benchmark_time - total_test_time
        console.print(
            f"[dim]  Test execution: {total_test_time:.0f}s | Overhead (pauses, etc): {overhead:.0f}s[/dim]"
        )
    console.print(f"[bold green]{'=' * 80}[/bold green]")
    console.print("[bold cyan]Benchmark complete![/bold cyan]\n")


if __name__ == "__main__":
    main()
