
import torch

from architecture.model import LuciGPT


@torch.inference_mode()
def generate(
    model: LuciGPT,
    token_ids: torch.Tensor,
    eos_token_id: int | None = None,
    max_new_tokens: int = 64,
) -> torch.Tensor:

    if token_ids.ndim != 2:
        raise ValueError(
            "A entrada precisa ter formato [batch, sequence]."
        )

    if token_ids.size(0) == 0 or token_ids.size(1) == 0:
        raise ValueError("O batch e o prompt não podem estar vazios.")

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens não pode ser negativo.")

    if token_ids.size(1) + max_new_tokens > model.config.max_seq_len:
        raise ValueError(
            "A geração solicitada excede o contexto máximo do modelo."
        )

    if eos_token_id is not None:
        if not 0 <= eos_token_id < model.config.vocab_size:
            raise ValueError("EOS está fora do vocabulário.")

    generated = token_ids.clone()

    if max_new_tokens == 0:
        return generated

    was_training = model.training
    model.eval()

    try:
        # PREFILL:
        # Processa o prompt inteiro e aloca o KV Cache.
        logits, kv_cache = model(
            generated,
            use_cache=True,
        )

        finished = torch.zeros(
            generated.size(0),
            dtype=torch.bool,
            device=generated.device,
        )

        for step in range(max_new_tokens):

            # Seleciona o próximo token.
            next_token = torch.argmax(
                logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

            # Mantém EOS para sequências já finalizadas.
            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_token, eos_token_id),
                    next_token,
                )

            generated = torch.cat(
                (generated, next_token),
                dim=1,
            )

            if eos_token_id is not None:
                finished |= next_token.squeeze(1).eq(eos_token_id)

                if finished.all():
                    break

            if step + 1 >= max_new_tokens:
                break

            # DECODIFICAÇÃO INCREMENTAL:
            # Atualiza o mesmo KV Cache, sem concatenar K/V.
            logits, kv_cache = model(
                next_token,
                kv_cache=kv_cache,
                use_cache=True,
            )

    finally:
        model.train(was_training)

    return generated
