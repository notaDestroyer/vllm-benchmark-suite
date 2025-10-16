#!/usr/bin/env python3
"""
vLLM Performance Benchmark Suite

Comprehensive benchmarking tool for evaluating vLLM inference performance across
various context lengths and concurrency levels. Includes real-time GPU monitoring
and detailed performance metrics collection.

Author: System Performance Engineering Team
License: MIT
"""

import requests
import time
import threading
from statistics import mean, stdev
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import subprocess
import re
from typing import Dict, List, Optional, Tuple
import sys

# Configuration Constants
API_BASE_URL = "http://localhost:8000"
API_ENDPOINT = f"{API_BASE_URL}/v1/chat/completions"
API_MODELS_ENDPOINT = f"{API_BASE_URL}/v1/models"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
REQUEST_TIMEOUT = 900  # seconds
GPU_POLL_INTERVAL = 1  # seconds
TEST_PAUSE_DURATION = 5  # seconds between tests


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
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.gr,clocks.mem',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=2)
            
            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                return {
                    'gpu_util': float(values[0]),
                    'mem_used': float(values[1]),
                    'mem_total': float(values[2]),
                    'temperature': float(values[3]),
                    'power_draw': float(values[4]),
                    'gpu_clock': float(values[5]),
                    'mem_clock': float(values[6]),
                    'timestamp': time.time()
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
            'avg_gpu_util': mean([s['gpu_util'] for s in self.stats]),
            'max_gpu_util': max([s['gpu_util'] for s in self.stats]),
            'avg_mem_used': mean([s['mem_used'] for s in self.stats]),
            'max_mem_used': max([s['mem_used'] for s in self.stats]),
            'avg_temperature': mean([s['temperature'] for s in self.stats]),
            'max_temperature': max([s['temperature'] for s in self.stats]),
            'avg_power': mean([s['power_draw'] for s in self.stats]),
            'max_power': max([s['power_draw'] for s in self.stats]),
            'avg_gpu_clock': mean([s['gpu_clock'] for s in self.stats]),
            'max_gpu_clock': max([s['gpu_clock'] for s in self.stats]),
            'avg_mem_clock': mean([s['mem_clock'] for s in self.stats]),
            'samples': len(self.stats)
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
            if 'data' in data and len(data['data']) > 0:
                model_name = data['data'][0].get('id', DEFAULT_MODEL_NAME)
                print(f"[INFO] Detected model: {model_name}")
                return model_name
    except Exception as e:
        print(f"[WARNING] Failed to query model name: {e}", file=sys.stderr)
    
    print(f"[INFO] Using default model name: {DEFAULT_MODEL_NAME}")
    return DEFAULT_MODEL_NAME


def generate_prompt(target_tokens: int) -> str:
    """
    Generate a synthetic prompt of approximately target_tokens length.
    
    Uses cybersecurity threat intelligence text as content to simulate
    realistic workload patterns for CTI use cases.
    
    Args:
        target_tokens: Approximate desired token count
        
    Returns:
        Generated prompt string
    """
    base_text = "Analyze the following cybersecurity threat intelligence data in detail. "
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


def make_request(prompt: str, request_id: int, results: List[Dict], 
                 max_tokens: int = 500, model_name: str = None) -> None:
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
        "temperature": 0.7
    }
    
    try:
        start = time.time()
        response = requests.post(API_ENDPOINT, json=data, timeout=REQUEST_TIMEOUT)
        duration = time.time() - start
        
        if response.status_code == 200:
            result = response.json()
            usage = result.get('usage', {})
            results.append({
                'request_id': request_id,
                'duration': duration,
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0),
                'success': True
            })
        else:
            results.append({
                'request_id': request_id,
                'duration': duration,
                'success': False,
                'error': f"HTTP {response.status_code}"
            })
    except Exception as e:
        results.append({
            'request_id': request_id,
            'success': False,
            'error': str(e)
        })


def run_benchmark(context_length: int, num_concurrent_users: int, 
                  output_tokens: int = 500, model_name: str = None) -> Optional[Dict]:
    """
    Execute benchmark for specific context length and concurrency level.
    
    Args:
        context_length: Input context size in tokens
        num_concurrent_users: Number of concurrent requests
        output_tokens: Target output token count
        model_name: Model identifier
        
    Returns:
        Dictionary containing performance metrics or None on failure
    """
    print(f"\n{'='*100}")
    print(f"Testing: {context_length:,} token context | {num_concurrent_users} concurrent users")
    print(f"{'='*100}")
    
    prompt = generate_prompt(context_length)
    actual_prompt_tokens = len(prompt) // 4  # Rough estimate
    
    results = []
    threads = []
    
    # Initialize GPU monitoring
    gpu_monitor = GPUMonitor()
    gpu_monitor.start()
    
    start_time = time.time()
    
    # Launch concurrent requests
    for i in range(num_concurrent_users):
        t = threading.Thread(
            target=make_request, 
            args=(prompt, i, results, output_tokens, model_name)
        )
        threads.append(t)
        t.start()
    
    # Wait for completion
    for t in threads:
        t.join()
    
    total_time = time.time() - start_time
    
    # Stop GPU monitoring
    gpu_stats = gpu_monitor.stop()
    
    # Calculate statistics
    successful = [r for r in results if r.get('success', False)]
    failed = len(results) - len(successful)
    
    if successful:
        durations = [r['duration'] for r in successful]
        completion_tokens = [r['completion_tokens'] for r in successful]
        prompt_tokens = [r['prompt_tokens'] for r in successful]
        
        avg_duration = mean(durations)
        std_duration = stdev(durations) if len(durations) > 1 else 0
        min_duration = min(durations)
        max_duration = max(durations)
        
        total_completion_tokens = sum(completion_tokens)
        avg_prompt_tokens = mean(prompt_tokens) if prompt_tokens else actual_prompt_tokens
        
        tokens_per_second = total_completion_tokens / total_time
        requests_per_second = len(successful) / total_time
        avg_tokens_per_request = mean(completion_tokens) if completion_tokens else 0
        
        # Estimate Time to First Token (TTFT)
        ttft_estimate = avg_duration * 0.1
        
        # Per-user throughput
        throughput_per_user = tokens_per_second / num_concurrent_users if num_concurrent_users > 0 else 0
        
        # Print results
        print(f"\nResults:")
        print(f"  Total time:              {total_time:.2f}s")
        print(f"  Successful:              {len(successful)}/{num_concurrent_users}")
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
        
        # Print GPU statistics
        if gpu_stats:
            print(f"\nGPU Metrics:")
            print(f"  Avg utilization:         {gpu_stats['avg_gpu_util']:.1f}%")
            print(f"  Max utilization:         {gpu_stats['max_gpu_util']:.1f}%")
            print(f"  Avg memory used:         {gpu_stats['avg_mem_used']:.0f} MB ({gpu_stats['avg_mem_used']/1024:.1f} GB)")
            print(f"  Max memory used:         {gpu_stats['max_mem_used']:.0f} MB ({gpu_stats['max_mem_used']/1024:.1f} GB)")
            print(f"  Avg temperature:         {gpu_stats['avg_temperature']:.1f}C")
            print(f"  Max temperature:         {gpu_stats['max_temperature']:.1f}C")
            print(f"  Avg power draw:          {gpu_stats['avg_power']:.1f} W")
            print(f"  Max power draw:          {gpu_stats['max_power']:.1f} W")
            print(f"  Avg GPU clock:           {gpu_stats['avg_gpu_clock']:.0f} MHz")
            print(f"  Max GPU clock:           {gpu_stats['max_gpu_clock']:.0f} MHz")
            print(f"  Avg memory clock:        {gpu_stats['avg_mem_clock']:.0f} MHz")
        
        result_dict = {
            'context_length': context_length,
            'concurrent_users': num_concurrent_users,
            'total_time': total_time,
            'successful': len(successful),
            'failed': failed,
            'avg_latency': avg_duration,
            'std_latency': std_duration,
            'min_latency': min_duration,
            'max_latency': max_duration,
            'ttft_estimate': ttft_estimate,
            'tokens_per_second': tokens_per_second,
            'requests_per_second': requests_per_second,
            'throughput_per_user': throughput_per_user,
            'avg_prompt_tokens': avg_prompt_tokens,
            'avg_completion_tokens': avg_tokens_per_request
        }
        
        # Merge GPU statistics
        if gpu_stats:
            result_dict.update(gpu_stats)
        
        return result_dict
    else:
        print(f"\n[ERROR] All requests failed!")
        for r in results[:3]:
            if not r.get('success'):
                print(f"  Error: {r.get('error', 'Unknown')}")
        return None


def visualize_results(all_results: List[Dict], model_name: str) -> str:
    """
    Generate comprehensive visualization charts from benchmark results.
    
    Args:
        all_results: List of benchmark result dictionaries
        model_name: Model identifier for chart title
        
    Returns:
        Filename of saved visualization
    """
    df = pd.DataFrame(all_results)
    has_gpu_stats = 'avg_gpu_util' in df.columns
    
    # Configure plot styling
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16
    
    # Create figure with dynamic layout
    if has_gpu_stats:
        fig = plt.figure(figsize=(26, 24))
        gs = fig.add_gridspec(6, 3, hspace=0.4, wspace=0.35, 
                             left=0.06, right=0.98, top=0.95, bottom=0.03)
    else:
        fig = plt.figure(figsize=(24, 18))
        gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.35, 
                             left=0.06, right=0.98, top=0.94, bottom=0.05)
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']
    
    # 1. Primary Metric: Throughput vs Context Length
    ax1 = fig.add_subplot(gs[0, :])
    for idx, users in enumerate(sorted(df['concurrent_users'].unique())):
        data = df[df['concurrent_users'] == users].sort_values('context_length')
        color = colors[idx % len(colors)]
        ax1.plot(data['context_length']/1000, data['tokens_per_second'], 
                marker='o', linewidth=3, markersize=11, label=f'{users} users',
                color=color, markeredgecolor='white', markeredgewidth=1.5)
    ax1.set_xlabel('Context Length (K tokens)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Throughput (tokens/s)', fontsize=14, fontweight='bold')
    ax1.set_title('Throughput vs Context Length by Concurrency Level', 
                 fontsize=16, fontweight='bold', pad=15)
    ax1.legend(title='Concurrent Users', fontsize=11, title_fontsize=12, 
              loc='best', frameon=True, shadow=True, ncol=5)
    ax1.grid(True, alpha=0.25, linestyle='--')
    ax1.set_facecolor('#F8F9FA')
    
    # 2. Latency Distribution
    ax2 = fig.add_subplot(gs[1, 0])
    for idx, users in enumerate(sorted(df['concurrent_users'].unique())):
        data = df[df['concurrent_users'] == users].sort_values('context_length')
        color = colors[idx % len(colors)]
        ax2.plot(data['context_length']/1000, data['avg_latency'], 
                marker='s', linewidth=2.5, markersize=9, label=f'{users} users',
                color=color, markeredgecolor='white', markeredgewidth=1.5)
    ax2.set_xlabel('Context Length (K tokens)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Latency (seconds)', fontsize=13, fontweight='bold')
    ax2.set_title('Average Latency vs Context', fontsize=14, fontweight='bold', pad=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25, linestyle='--')
    ax2.set_facecolor('#F8F9FA')
    
    # 3. Throughput Heatmap
    ax3 = fig.add_subplot(gs[1, 1:])
    pivot_throughput = df.pivot(index='context_length', columns='concurrent_users', values='tokens_per_second')
    sns.heatmap(pivot_throughput, annot=True, fmt='.0f', cmap='RdYlGn', ax=ax3, 
                cbar_kws={'label': 'Tokens/s', 'shrink': 0.8}, linewidths=1.5,
                linecolor='white', annot_kws={'fontsize': 10, 'weight': 'bold'})
    ax3.set_xlabel('Concurrent Users', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Context Length', fontsize=13, fontweight='bold')
    ax3.set_title('Throughput Heatmap (Tokens/Second)', fontsize=14, fontweight='bold', pad=12)
    ax3.set_yticklabels([f'{int(y/1000)}K' for y in pivot_throughput.index], rotation=0)
    
    # 4. Efficiency (Throughput per User)
    ax4 = fig.add_subplot(gs[2, 0])
    for idx, users in enumerate(sorted(df['concurrent_users'].unique())):
        data = df[df['concurrent_users'] == users].sort_values('context_length')
        color = colors[idx % len(colors)]
        ax4.plot(data['context_length']/1000, data['throughput_per_user'], 
                marker='D', linewidth=2.5, markersize=9, label=f'{users} users',
                color=color, markeredgecolor='white', markeredgewidth=1.5)
    ax4.set_xlabel('Context Length (K tokens)', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Tokens/s per User', fontsize=13, fontweight='bold')
    ax4.set_title('Efficiency: Throughput per User', fontsize=14, fontweight='bold', pad=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.25, linestyle='--')
    ax4.set_facecolor('#F8F9FA')
    
    # 5. Requests per Second
    ax5 = fig.add_subplot(gs[2, 1])
    for idx, users in enumerate(sorted(df['concurrent_users'].unique())):
        data = df[df['concurrent_users'] == users].sort_values('context_length')
        color = colors[idx % len(colors)]
        ax5.plot(data['context_length']/1000, data['requests_per_second'], 
                marker='^', linewidth=2.5, markersize=10, label=f'{users} users',
                color=color, markeredgecolor='white', markeredgewidth=1.5)
    ax5.set_xlabel('Context Length (K tokens)', fontsize=13, fontweight='bold')
    ax5.set_ylabel('Requests/Second', fontsize=13, fontweight='bold')
    ax5.set_title('Request Throughput', fontsize=14, fontweight='bold', pad=12)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.25, linestyle='--')
    ax5.set_facecolor('#F8F9FA')
    
    # 6. Time to First Token
    ax6 = fig.add_subplot(gs[2, 2])
    for idx, users in enumerate(sorted(df['concurrent_users'].unique())):
        data = df[df['concurrent_users'] == users].sort_values('context_length')
        color = colors[idx % len(colors)]
        ax6.plot(data['context_length']/1000, data['ttft_estimate']*1000, 
                marker='*', linewidth=2.5, markersize=13, label=f'{users} users',
                color=color, markeredgecolor='white', markeredgewidth=1.5)
    ax6.set_xlabel('Context Length (K tokens)', fontsize=13, fontweight='bold')
    ax6.set_ylabel('TTFT (milliseconds)', fontsize=13, fontweight='bold')
    ax6.set_title('Time to First Token (Estimated)', fontsize=14, fontweight='bold', pad=12)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.25, linestyle='--')
    ax6.set_facecolor('#F8F9FA')
    
    # 7. Context Length Impact (normalized)
    ax7 = fig.add_subplot(gs[3, 0])
    for idx, users in enumerate(sorted(df['concurrent_users'].unique())):
        data = df[df['concurrent_users'] == users].sort_values('context_length')
        if len(data) > 0:
            baseline = data.iloc[0]['tokens_per_second']
            normalized = (data['tokens_per_second'] / baseline) * 100
            color = colors[idx % len(colors)]
            ax7.plot(data['context_length']/1000, normalized, 
                    marker='o', linewidth=2.5, markersize=9, label=f'{users} users',
                    color=color, markeredgecolor='white', markeredgewidth=1.5)
    ax7.axhline(y=100, color='#E63946', linestyle='--', alpha=0.6, linewidth=2, label='Baseline')
    ax7.set_xlabel('Context Length (K tokens)', fontsize=13, fontweight='bold')
    ax7.set_ylabel('Throughput (% of 1K context)', fontsize=13, fontweight='bold')
    ax7.set_title('Context Length Performance Impact', fontsize=14, fontweight='bold', pad=12)
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.25, linestyle='--')
    ax7.set_facecolor('#F8F9FA')
    
    # 8. Success Rate
    ax8 = fig.add_subplot(gs[3, 1])
    df['success_rate'] = (df['successful'] / (df['successful'] + df['failed'])) * 100
    for idx, users in enumerate(sorted(df['concurrent_users'].unique())):
        data = df[df['concurrent_users'] == users].sort_values('context_length')
        color = colors[idx % len(colors)]
        ax8.plot(data['context_length']/1000, data['success_rate'], 
                marker='h', linewidth=2.5, markersize=10, label=f'{users} users',
                color=color, markeredgecolor='white', markeredgewidth=1.5)
    ax8.axhline(y=100, color='#2A9D8F', linestyle='--', alpha=0.5, linewidth=2)
    ax8.set_xlabel('Context Length (K tokens)', fontsize=13, fontweight='bold')
    ax8.set_ylabel('Success Rate (%)', fontsize=13, fontweight='bold')
    ax8.set_title('Request Success Rate', fontsize=14, fontweight='bold', pad=12)
    ax8.set_ylim([95, 105])
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.25, linestyle='--')
    ax8.set_facecolor('#F8F9FA')
    
    # 9. Latency Heatmap
    ax9 = fig.add_subplot(gs[3, 2])
    pivot_latency = df.pivot(index='context_length', columns='concurrent_users', values='avg_latency')
    sns.heatmap(pivot_latency, annot=True, fmt='.1f', cmap='coolwarm', ax=ax9, 
                cbar_kws={'label': 'Seconds', 'shrink': 0.8}, linewidths=1.5,
                linecolor='white', annot_kws={'fontsize': 10, 'weight': 'bold'})
    ax9.set_xlabel('Concurrent Users', fontsize=13, fontweight='bold')
    ax9.set_ylabel('Context Length', fontsize=13, fontweight='bold')
    ax9.set_title('Latency Heatmap (Seconds)', fontsize=14, fontweight='bold', pad=12)
    ax9.set_yticklabels([f'{int(y/1000)}K' for y in pivot_latency.index], rotation=0)
    
    # GPU-specific visualizations
    if has_gpu_stats:
        # 10. GPU Utilization
        ax10 = fig.add_subplot(gs[4, 0])
        for idx, users in enumerate(sorted(df['concurrent_users'].unique())):
            data = df[df['concurrent_users'] == users].sort_values('context_length')
            color = colors[idx % len(colors)]
            ax10.plot(data['context_length']/1000, data['avg_gpu_util'], 
                    marker='o', linewidth=2.5, markersize=9, label=f'{users} users',
                    color=color, markeredgecolor='white', markeredgewidth=1.5)
        ax10.set_xlabel('Context Length (K tokens)', fontsize=13, fontweight='bold')
        ax10.set_ylabel('GPU Utilization (%)', fontsize=13, fontweight='bold')
        ax10.set_title('Average GPU Utilization', fontsize=14, fontweight='bold', pad=12)
        ax10.legend(fontsize=9)
        ax10.grid(True, alpha=0.25, linestyle='--')
        ax10.set_facecolor('#F8F9FA')
        ax10.set_ylim([0, 105])
        
        # 11. GPU Memory Usage
        ax11 = fig.add_subplot(gs[4, 1])
        for idx, users in enumerate(sorted(df['concurrent_users'].unique())):
            data = df[df['concurrent_users'] == users].sort_values('context_length')
            color = colors[idx % len(colors)]
            ax11.plot(data['context_length']/1000, data['avg_mem_used']/1024, 
                    marker='s', linewidth=2.5, markersize=9, label=f'{users} users',
                    color=color, markeredgecolor='white', markeredgewidth=1.5)
        ax11.set_xlabel('Context Length (K tokens)', fontsize=13, fontweight='bold')
        ax11.set_ylabel('VRAM Used (GB)', fontsize=13, fontweight='bold')
        ax11.set_title('GPU Memory Usage', fontsize=14, fontweight='bold', pad=12)
        ax11.legend(fontsize=9)
        ax11.grid(True, alpha=0.25, linestyle='--')
        ax11.set_facecolor('#F8F9FA')
        
        # 12. GPU Power Draw and Temperature
        ax12 = fig.add_subplot(gs[4, 2])
        ax12_temp = ax12.twinx()
        
        max_users = max(df['concurrent_users'].unique())
        data = df[df['concurrent_users'] == max_users].sort_values('context_length')
        
        line1 = ax12.plot(data['context_length']/1000, data['avg_power'], 
                marker='D', linewidth=2.5, markersize=9, label='Power Draw',
                color='#E63946', markeredgecolor='white', markeredgewidth=1.5)
        line2 = ax12_temp.plot(data['context_length']/1000, data['avg_temperature'], 
                marker='^', linewidth=2.5, markersize=9, label='Temperature',
                color='#457B9D', markeredgecolor='white', markeredgewidth=1.5)
        
        ax12.set_xlabel('Context Length (K tokens)', fontsize=13, fontweight='bold')
        ax12.set_ylabel('Power Draw (W)', fontsize=13, fontweight='bold', color='#E63946')
        ax12_temp.set_ylabel('Temperature (C)', fontsize=13, fontweight='bold', color='#457B9D')
        ax12.set_title(f'Power & Temperature ({max_users} users)', fontsize=14, fontweight='bold', pad=12)
        ax12.tick_params(axis='y', labelcolor='#E63946')
        ax12_temp.tick_params(axis='y', labelcolor='#457B9D')
        ax12.grid(True, alpha=0.25, linestyle='--')
        ax12.set_facecolor('#F8F9FA')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax12.legend(lines, labels, fontsize=9, loc='upper left')
        
        # 13. GPU Clock Frequencies
        ax13 = fig.add_subplot(gs[5, 0])
        ax13_mem = ax13.twinx()
        
        for idx, users in enumerate(sorted(df['concurrent_users'].unique())):
            data = df[df['concurrent_users'] == users].sort_values('context_length')
            color = colors[idx % len(colors)]
            ax13.plot(data['context_length']/1000, data['avg_gpu_clock'], 
                    marker='o', linewidth=2.5, markersize=9, label=f'{users} users (GPU)',
                    color=color, markeredgecolor='white', markeredgewidth=1.5, alpha=0.7)
        
        ax13.set_xlabel('Context Length (K tokens)', fontsize=13, fontweight='bold')
        ax13.set_ylabel('GPU Clock (MHz)', fontsize=13, fontweight='bold', color='#2E86AB')
        ax13.set_title('GPU Clock Frequency', fontsize=14, fontweight='bold', pad=12)
        ax13.tick_params(axis='y', labelcolor='#2E86AB')
        ax13.legend(fontsize=8, loc='upper left')
        ax13.grid(True, alpha=0.25, linestyle='--')
        ax13.set_facecolor('#F8F9FA')
        
        # 14. Clock Frequency Heatmap
        ax14 = fig.add_subplot(gs[5, 1:])
        pivot_clock = df.pivot(index='context_length', columns='concurrent_users', values='avg_gpu_clock')
        sns.heatmap(pivot_clock, annot=True, fmt='.0f', cmap='plasma', ax=ax14, 
                    cbar_kws={'label': 'MHz', 'shrink': 0.8}, linewidths=1.5,
                    linecolor='white', annot_kws={'fontsize': 10, 'weight': 'bold'})
        ax14.set_xlabel('Concurrent Users', fontsize=13, fontweight='bold')
        ax14.set_ylabel('Context Length', fontsize=13, fontweight='bold')
        ax14.set_title('GPU Clock Frequency Heatmap (MHz)', fontsize=14, fontweight='bold', pad=12)
        ax14.set_yticklabels([f'{int(y/1000)}K' for y in pivot_clock.index], rotation=0)
    
    # Main title
    gpu_info = " | GPU Monitored (with Frequency)" if has_gpu_stats else ""
    fig.suptitle(
        f'{model_name} Performance Benchmark\n'
        f'RTX Pro 6000 Blackwell (96GB) | FlashInfer | 256K Context{gpu_info}',
        fontsize=18, fontweight='bold', y=0.985 if has_gpu_stats else 0.988,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    )
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'benchmark_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"\n[INFO] Visualization saved: {filename}")
    
    return filename


def print_summary_table(all_results: List[Dict]) -> None:
    """
    Print detailed performance summary tables to console.
    
    Args:
        all_results: List of benchmark result dictionaries
    """
    df = pd.DataFrame(all_results)
    has_gpu_stats = 'avg_gpu_util' in df.columns
    
    print("\n" + "="*160)
    print("DETAILED PERFORMANCE SUMMARY")
    print("="*160)
    
    # Per-context results
    for context in sorted(df['context_length'].unique()):
        context_data = df[df['context_length'] == context].sort_values('concurrent_users')
        print(f"\nContext Length: {context:,} tokens ({context/1000:.0f}K)")
        print("-"*160)
        
        if has_gpu_stats:
            print(f"{'Users':<8} {'Latency(s)':<12} {'Tok/s':<10} {'Req/s':<10} {'TTFT(ms)':<10} "
                  f"{'GPU%':<8} {'VRAM(GB)':<10} {'Temp(C)':<10} {'Power(W)':<10} {'GPU MHz':<10} {'Success%':<10}")
            print("-"*160)
            
            for _, row in context_data.iterrows():
                success_rate = (row['successful'] / (row['successful'] + row['failed'])) * 100
                print(f"{row['concurrent_users']:<8} "
                      f"{row['avg_latency']:<12.2f} "
                      f"{row['tokens_per_second']:<10.1f} "
                      f"{row['requests_per_second']:<10.2f} "
                      f"{row['ttft_estimate']*1000:<10.0f} "
                      f"{row['avg_gpu_util']:<8.1f} "
                      f"{row['avg_mem_used']/1024:<10.1f} "
                      f"{row['avg_temperature']:<10.1f} "
                      f"{row['avg_power']:<10.1f} "
                      f"{row['avg_gpu_clock']:<10.0f} "
                      f"{success_rate:<10.1f}")
        else:
            print(f"{'Users':<8} {'Latency(s)':<12} {'Tokens/s':<12} {'Req/s':<10} "
                  f"{'TTFT(ms)':<12} {'Tok/s/User':<15} {'Success%':<10}")
            print("-"*160)
            
            for _, row in context_data.iterrows():
                success_rate = (row['successful'] / (row['successful'] + row['failed'])) * 100
                print(f"{row['concurrent_users']:<8} "
                      f"{row['avg_latency']:<12.2f} "
                      f"{row['tokens_per_second']:<12.1f} "
                      f"{row['requests_per_second']:<10.2f} "
                      f"{row['ttft_estimate']*1000:<12.0f} "
                      f"{row['throughput_per_user']:<15.1f} "
                      f"{success_rate:<10.1f}")
    
    # Optimal configurations
    print("\n" + "="*160)
    print("OPTIMAL CONFIGURATIONS")
    print("="*160)
    
    max_throughput = df.loc[df['tokens_per_second'].idxmax()]
    print(f"\nMaximum Throughput:")
    print(f"  {max_throughput['tokens_per_second']:.1f} tokens/s at {max_throughput['concurrent_users']} users "
          f"with {max_throughput['context_length']/1000:.0f}K context")
    if has_gpu_stats:
        print(f"  GPU: {max_throughput['avg_gpu_util']:.1f}% util, {max_throughput['avg_mem_used']/1024:.1f}GB VRAM, "
              f"{max_throughput['avg_temperature']:.1f}C, {max_throughput['avg_power']:.1f}W, "
              f"{max_throughput['avg_gpu_clock']:.0f} MHz")
    
    best_efficiency = df.loc[df['throughput_per_user'].idxmax()]
    print(f"\nBest Efficiency (tokens/s per user):")
    print(f"  {best_efficiency['throughput_per_user']:.1f} tokens/s/user at {best_efficiency['concurrent_users']} users "
          f"with {best_efficiency['context_length']/1000:.0f}K context")
    
    min_latency = df.loc[df['avg_latency'].idxmin()]
    print(f"\nLowest Latency:")
    print(f"  {min_latency['avg_latency']:.2f}s at {min_latency['concurrent_users']} users "
          f"with {min_latency['context_length']/1000:.0f}K context")
    
    best_req_throughput = df.loc[df['requests_per_second'].idxmax()]
    print(f"\nHighest Request Throughput:")
    print(f"  {best_req_throughput['requests_per_second']:.2f} req/s at {best_req_throughput['concurrent_users']} users "
          f"with {best_req_throughput['context_length']/1000:.0f}K context")
    
    # Scaling analysis
    print(f"\nContext Scaling Analysis:")
    single_user = df[df['concurrent_users'] == 1].sort_values('context_length')
    if len(single_user) > 1:
        baseline_throughput = single_user.iloc[0]['tokens_per_second']
        max_context_throughput = single_user.iloc[-1]['tokens_per_second']
        degradation = ((baseline_throughput - max_context_throughput) / baseline_throughput) * 100
        print(f"  Throughput degradation from 1K to {single_user.iloc[-1]['context_length']/1000:.0f}K: {degradation:.1f}%")
    
    # GPU efficiency
    if has_gpu_stats:
        print(f"\nGPU Efficiency Analysis:")
        max_gpu_util = df.loc[df['avg_gpu_util'].idxmax()]
        print(f"  Peak GPU utilization: {max_gpu_util['avg_gpu_util']:.1f}% at {max_gpu_util['concurrent_users']} users "
              f"with {max_gpu_util['context_length']/1000:.0f}K context")
        print(f"  Peak VRAM usage: {df['max_mem_used'].max()/1024:.1f}GB")
        print(f"  Peak temperature: {df['max_temperature'].max():.1f}C")
        print(f"  Peak power draw: {df['max_power'].max():.1f}W")
        print(f"  Average power draw: {df['avg_power'].mean():.1f}W")
        print(f"  Average GPU clock: {df['avg_gpu_clock'].mean():.0f} MHz")
        print(f"  Peak GPU clock: {df['max_gpu_clock'].max():.0f} MHz")


def main():
    """Main benchmark execution routine."""
    print("="*130)
    print("vLLM Performance Benchmark Suite")
    print("="*130)
    
    # Query model name from server
    model_name = get_model_name()
    
    print(f"Model: {model_name}")
    print(f"Server: {API_BASE_URL}")
    print(f"GPU: RTX Pro 6000 Blackwell (96GB VRAM)")
    print(f"Backend: FlashInfer")
    print(f"Max Context: 256K tokens")
    print("="*130)
    
    # Benchmark configuration
    context_lengths = [
        1000, 10000, 32000, 64000, 96000,
        128000, 160000, 192000, 224000, 256000
    ]
    
    concurrent_users = [1, 2, 5, 10]
    
    all_results = []
    
    # Execute benchmarks
    total_tests = len(context_lengths) * len(concurrent_users)
    current_test = 0
    
    for context in context_lengths:
        for users in concurrent_users:
            current_test += 1
            print(f"\n[Progress: {current_test}/{total_tests}]")
            
            result = run_benchmark(context, users, output_tokens=500, model_name=model_name)
            if result:
                all_results.append(result)
            
            # Pause between tests
            if current_test < total_tests:
                time.sleep(TEST_PAUSE_DURATION)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_filename = f'benchmark_results_{timestamp}.json'
    
    with open(json_filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[INFO] Raw results saved: {json_filename}")
    
    # Generate reports
    print_summary_table(all_results)
    visualize_results(all_results, model_name)
    
    print("\n" + "="*130)
    print("[INFO] Benchmark Complete")
    print("="*130)


if __name__ == "__main__":
    main()
