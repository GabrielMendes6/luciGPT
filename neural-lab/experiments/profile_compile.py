
import torch
from torch.profiler import profile, ProfilerActivity, record_function

from architecture.config import ModelConfig
from architecture.model import LuciGPT


# ============================================================
# CONFIGURATION
# ============================================================

torch.manual_seed(42)
torch.set_num_threads(2)

PROMPT_LENGTH = 128
DECODE_STEPS = 64

config = ModelConfig(
    vocab_size=16384,
    embedding_dim=768,
    max_seq_len=4096,
    num_heads=12,
    num_kv_heads=4,
    num_layers=12,
    ffn_dim=3072,
)

model = LuciGPT(config).eval()

prompt = torch.randint(
    0,
    config.vocab_size,
    (1, PROMPT_LENGTH),
)


# ============================================================
# GENERATION
# ============================================================

@torch.inference_mode()
def run_generation(target_model, optimize_lm_head=True):

    with record_function("LUCI::prefill"):

        if optimize_lm_head:
            logits, cache = target_model(
                prompt,
                use_cache=True,
                logits_to_keep=1,
            )
        else:
            logits, cache = target_model(
                prompt,
                use_cache=True,
            )

        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
        ).unsqueeze(1)

    with record_function("LUCI::decode"):

        for _ in range(DECODE_STEPS):

            logits, cache = target_model(
                next_token,
                kv_cache=cache,
                use_cache=True,
            )

            next_token = torch.argmax(
                logits[:, -1, :],
                dim=-1,
            ).unsqueeze(1)

    return cache.length


# ============================================================
# VALIDATION
# ============================================================

@torch.inference_mode()
def validate_logits_optimization(target_model):

    print("\n" + "=" * 65)
    print("VALIDATING LOGITS OPTIMIZATION")
    print("=" * 65)

    full_logits, _ = target_model(
        prompt,
        use_cache=True,
    )

    optimized_logits, _ = target_model(
        prompt,
        use_cache=True,
        logits_to_keep=1,
    )

    reference = full_logits[:, -1:, :]

    max_diff = (
        reference - optimized_logits
    ).abs().max().item()

    mean_diff = (
        reference - optimized_logits
    ).abs().mean().item()

    equivalent = torch.allclose(
        reference,
        optimized_logits,
        rtol=1e-4,
        atol=1e-5,
    )

    print("Full logits:", tuple(full_logits.shape))
    print("Optimized logits:", tuple(optimized_logits.shape))
    print(f"Max difference: {max_diff:.10e}")
    print(f"Mean difference: {mean_diff:.10e}")
    print("Equivalent:", equivalent)

    if not equivalent:
        raise RuntimeError(
            "Logits optimization failed equivalence validation."
        )


# ============================================================
# MODULE PROFILING HOOKS
# ============================================================

def register_module_ranges(target_model):
    """
    Registra regiões de profiling para módulos selecionados.
    Não altera os cálculos nem os parâmetros do modelo.
    """

    handles = []
    contexts = {}

    def register(module, name):

        contexts[module] = []

        def pre_hook(module, inputs):

            context = record_function(name)
            context.__enter__()

            contexts[module].append(context)

        def post_hook(module, inputs, output):

            context = contexts[module].pop()
            context.__exit__(None, None, None)

        handles.append(
            module.register_forward_pre_hook(pre_hook)
        )

        handles.append(
            module.register_forward_hook(post_hook)
        )

    # Global modules

    register(
        target_model.input_embedding,
        "LUCI::embedding",
    )

    register(
        target_model.final_norm,
        "LUCI::final_norm",
    )

    register(
        target_model.lm_head,
        "LUCI::lm_head",
    )

    # Transformer blocks

    for index, block in enumerate(target_model.transformer_blocks):

        prefix = f"LUCI::layer_{index}"

        register(
            block.attention,
            f"{prefix}::attention",
        )

        register(
            block.attention.qkv_projection,
            f"{prefix}::qkv_projection",
        )

        register(
            block.attention.output_projection,
            f"{prefix}::attention_output",
        )

        register(
            block.feed_forward,
            f"{prefix}::feed_forward",
        )

        register(
            block.feed_forward.network[0],
            f"{prefix}::ffn_linear_1",
        )

        register(
            block.feed_forward.network[2],
            f"{prefix}::ffn_linear_2",
        )

    return handles


# ============================================================
# COMPONENT REPORT
# ============================================================

def print_module_report(prof):

    totals = {}

    categories = (
        "embedding",
        "final_norm",
        "lm_head",
        "attention",
        "qkv_projection",
        "attention_output",
        "feed_forward",
        "ffn_linear_1",
        "ffn_linear_2",
    )

    for event in prof.key_averages():

        if not event.key.startswith("LUCI::"):
            continue

        parts = event.key.split("::")

        if len(parts) < 2:
            continue

        category = parts[-1]

        if category in categories:

            totals[category] = totals.get(
                category,
                0.0,
            ) + event.self_cpu_time_total

    print("\n--- TEMPO PRÓPRIO POR COMPONENTE ---")

    for category, elapsed in sorted(
        totals.items(),
        key=lambda item: item[1],
        reverse=True,
    ):

        print(
            f"{category:20s} "
            f"{elapsed / 1000:10.2f} ms"
        )


# ============================================================
# PROFILER
# ============================================================

def profile_model(name, target_model):

    print("\n" + "=" * 65)
    print(f"PROFILING: {name}")
    print("=" * 65)

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:

        final_cache_length = run_generation(
            target_model,
            optimize_lm_head=True,
        )

    print("\nFinal cache length:", final_cache_length)

    print("\n--- TOP 25 OPERAÇÕES POR TEMPO PRÓPRIO ---")

    print(
        prof.key_averages().table(
            sort_by="self_cpu_time_total",
            row_limit=25,
        )
    )

    print("\n--- TOP 15 OPERAÇÕES POR MEMÓRIA ---")

    print(
        prof.key_averages().table(
            sort_by="self_cpu_memory_usage",
            row_limit=15,
        )
    )

    print("\n--- REGIÕES LUCI ---")

    for event in prof.key_averages():

        if event.key.startswith("LUCI::"):

            print(
                f"{event.key}: "
                f"{event.self_cpu_time_total / 1000:.2f} ms "
                f"(self), "
                f"{event.cpu_time_total / 1000:.2f} ms (total)"
            )

    print("\n--- OPERAÇÕES IMPORTANTES ---")

    interesting_ops = (
        "aten::addmm",
        "aten::mm",
        "aten::gelu",
        "aten::scaled_dot_product_attention",
        "aten::copy_",
        "aten::argmax",
        "Torch-Compiled Region",
    )

    for event in prof.key_averages():

        if any(op in event.key for op in interesting_ops):

            print(
                f"{event.key}: "
                f"{event.count} chamadas | "
                f"{event.self_cpu_time_total / 1000:.2f} ms"
            )

    print_module_report(prof)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("===== LUCI COMPILATION PROFILER =====")
    print("=" * 65)

    print("PyTorch:", torch.__version__)
    print("Threads:", torch.get_num_threads())

    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters()),
    )

    # --------------------------------------------------------
    # Validate LM Head optimization
    # --------------------------------------------------------

    validate_logits_optimization(model)

    # --------------------------------------------------------
    # EAGER
    # --------------------------------------------------------

    print("\nAquecendo Eager...")

    run_generation(model)
    run_generation(model)

    handles = register_module_ranges(model)

    try:

        profile_model(
            "EAGER",
            model,
        )

    finally:

        for handle in handles:
            handle.remove()

    # --------------------------------------------------------
    # INDUCTOR
    # --------------------------------------------------------

    print("\nCompilando modelo com Inductor...")

    compiled_model = torch.compile(
        model,
        backend="inductor",
    )

    # Compilation and warmup outside profiler

    run_generation(compiled_model)
    run_generation(compiled_model)

    profile_model(
        "INDUCTOR",
        compiled_model,
    )

    print("\n" + "=" * 65)
    print("===== PROFILING CONCLUÍDO =====")
    print("=" * 65)


if __name__ == "__main__":
    main()
