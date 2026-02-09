import torch

from cocoruta_sayless import (
    print_env_config,
    get_tokenizer_and_model,
    query_model,
    say_less,
)

if __name__ == "__main__":
    print_env_config()

    prompt = "O que é a Amazônia Azul?"  # removed the delimiters ### Pergunta: and ### Resposta:
    model_id = "felipeoes/cocoruta-7b"
    # using the second GPU only
    tokenizer, model = get_tokenizer_and_model(
        model_id=model_id, device_map="cuda:1", torch_dtype=torch.float16
    )
    output = query_model(model, tokenizer, prompt)
    # Copied threshold from /factscore_a=1_alpha=0.15_conf=frequency+gpt.txt.
    # Compute new ones by running factscore.py with desired parameters and setting compute_single_threshold=True
    threshold = 4.8998812119930735
    merged_output, (accepted_subclaims, all_subclaims) = say_less(
        model, tokenizer, prompt, output, threshold
    )
    print("Original output: ")
    print(output)
    print("\n\n\n\n\nModified output: ")
    print(merged_output)
    print("\n\n\n\nAccepted sub-claims: ")
    print(accepted_subclaims)
    print("\n\n\n\nAll sub-claims: ")
    print(all_subclaims)
