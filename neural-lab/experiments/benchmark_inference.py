
import argparse
import gc
import os
import resource
import time
from statistics import mean

import torch

from architecture.config import ModelConfig
from architecture.model import LuciGPT


def get_rss_mb():
    """Retorna a memória residente atual do processo em MB."""
    with open("/proc/self/status", encoding="utf-8") as file:
        for line in file:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0.0


def get_peak_rss_mb():
    """Retorna o pico de RAM do processo em MB."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024


def benchmark_batch(model, config, batch_size, sequence_length, repetitions=5):
    tokens = torch.randint(
        0,
        config.vocab_size,
        (batch_size, sequence_length),
        dtype=torch.long,
    )

    print(f"\n{'=' * 55}")
    print(f"BATCH SIZE: {batch_size}")
    print(f"SEQUENCE LENGTH: {sequence_length}")
    print(f"TOTAL TOKENS: {batch_size * sequence_length:,}")
    print(f"{'=' * 55}")

    # Aquecimento
    with torch.inference_mode():
        warmup_output = model(tokens)

    del warmup_output
    gc.collect()

    execution_times = []

    # Medições
    for iteration in range(repetitions):
        start = time.perf_counter()

        with torch.inference_mode():
            logits = model(tokens)

        elapsed = time.perf_counter() - start
        execution_times.append(elapsed)

        del logits
        gc.collect()

        print(f"Run {iteration + 1}: {elapsed:.3f}s")

    average_time = mean(execution_times)
    total_tokens = batch_size * sequence_length
    tokens_per_second = total_tokens / average_time

    print(f"\nAverage latency: {average_time:.3f}s")
    print(f"Throughput: {tokens_per_second:.2f} tokens/s")
    print(f"Current RSS: {get_rss_mb():.2f} MB")
    print(f"Peak RSS: {get_peak_rss_mb():.2f} MB")

    result = {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "latency": average_time,
        "tokens_per_second": tokens_per_second,
        "peak_rss_mb": get_peak_rss_mb(),
    }

    del tokens
    gc.collect()

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--batches",
        nargs="*",
        type=int,
        default=[1, 2, 4, 8],
    )
        
    parser.add_argument(
        "--threads",
        type=int,
        default=6,
    )

    args = parser.parse_args()
    
    if args.threads <= 0:
        raise ValueError("O número de threads deve ser positivo.")
    
    torch.set_num_threads(args.threads)

    config = ModelConfig()

    print("========== LUCI BATCH BENCHMARK ==========")
    print(f"PyTorch: {torch.__version__}")
    print("Device: CPU")
    print(f"PyTorch threads: {torch.get_num_threads()}")
    print(f"Logical CPUs reported by OS: {os.cpu_count()}")
    print(f"Sequence length: {args.sequence_length}")

    if not 0 < args.sequence_length <= config.max_seq_len:
        raise ValueError(
            f"Sequence length must be between 1 and {config.max_seq_len}."
        )

    gc.collect()

    print(f"\nRSS before model: {get_rss_mb():.2f} MB")

    # Instancia o modelo apenas uma vez
    model = LuciGPT(config)
    model.eval()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Model parameters: {parameter_count:,}")
    print(f"RSS after model: {get_rss_mb():.2f} MB")
    print(f"Peak RSS after model: {get_peak_rss_mb():.2f} MB")

    results = []

    for batch_size in args.batches:
        if batch_size <= 0:
            print(f"Skipping invalid batch size: {batch_size}")
            continue

        try:
            result = benchmark_batch(
                model,
                config,
                batch_size,
                args.sequence_length,
            )
            results.append(result)

        except RuntimeError as error:
            print(f"\nERROR at batch size {batch_size}:")
            print(error)
            break

    print("\n========== FINAL SUMMARY ==========")

    for result in results:
        print(
            f"Batch: {result['batch_size']:2d} | "
            f"Sequence: {result['sequence_length']:4d} | "
            f"Latency: {result['latency']:8.3f}s | "
            f"Throughput: {result['tokens_per_second']:8.2f} tok/s | "
            f"Peak RSS: {result['peak_rss_mb']:8.2f} MB"
        )


if __name__ == "__main__":
    main()
