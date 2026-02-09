import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria

# Default prompt to break into subclaims.
BREAKDOWN_PROMPT = "Por favor, divida a seguinte entrada em um conjunto de pequenas afirmações independentes e retorne a saída no formato jsonl, onde cada linha seja {subclaim:[AFIRMAÇÃO], cocoruta-score:[CONF]}. A pontuação de confiança [CONF] deve representar o seu nível de confiança na afirmação, onde 1 corresponde a fatos e resultados óbvios, como 'A Terra é redonda' e '1+1=2'. Já 0 corresponde a afirmações muito obscuras ou difíceis de qualquer pessoa saber, como a data de aniversário de pessoas não públicas. A entrada é: "


def print_env_config():
    print("cuda available:", torch.cuda.is_available())
    print("cuda devices:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))
    print("torch cuda version:", torch.version.cuda)
    print(torch.__version__)


def get_tokenizer_and_model(model_id, device_map, torch_dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_id, device_map=device_map)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map=device_map, torch_dtype=torch_dtype
    )
    return tokenizer, model


def query_model(model, tokenizer, prompt, max_tokens=1000, temperature=0, n_samples=1):
    # Define an early stopping ccriteria
    class StopOnString(StoppingCriteria):
        def __init__(self, target_sequence, prompt):
            self.target_sequence = target_sequence
            self.prompt = prompt

        def __call__(self, input_ids, scores, **kwargs):
            # Get the generated text as a string
            generated_text = tokenizer.decode(input_ids[0])
            generated_text = generated_text.replace(self.prompt, "")
            # Check if the target sequence appears in the generated text
            if self.target_sequence in generated_text:
                return True  # Stop generation

            return False  # Continue generation

    stop_string = "###"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    if temperature > 0:
        # temperature is set, do_sample=True
        _ = model.generate(
            input_ids,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_tokens,
            num_return_sequences=n_samples,
            stopping_criteria=[StopOnString(stop_string, prompt)],
            do_sample=True,
            temperature=temperature,
        )
    elif temperature == 0:
        # temperature not set, do_sample=False,
        _ = model.generate(
            input_ids,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_tokens,
            num_return_sequences=n_samples,
            stopping_criteria=[StopOnString(stop_string, prompt)],
            do_sample=False,
        )
    else:
        raise ValueError("Temperature must be >= 0")

    return (
        tokenizer.decode(_[0], skip_special_tokens=True)
        if n_samples == 1
        else [tokenizer.decode(choice, skip_special_tokens=True) for choice in _]
    )


def say_less(model, tokenizer, prompt, output, threshold):
    """
    say_less takes in the model output y, breaks it down into subclaims, and removes sub-claims up to the threshold value.
    The subclaims are scored by counting (using an LM) how many times they appear from 5 other sampled outputs. This is done
    in get_frequency_scores.
    """
    subclaims = get_subclaims(model, tokenizer, output)
    print("get_subclaims()")
    print(f"Qtt.: {len(subclaims)}")
    try:
        print(f"Subclaim [0]: {subclaims[0]}")
        print(f"Subclaim [n-1]: {subclaims[-1]}")
    except Exception:
        print("Não consegui mostrar as subclaims...")

    frequency_scores = get_frequency_scores(model, tokenizer, subclaims, prompt, 5)
    for i, subclaim in enumerate(subclaims):
        subclaim["frequency-score"] = frequency_scores[i]
    print("get_frequency_scores()")
    print(f"Qtt.: {len(frequency_scores)}")
    try:
        print(f"Freq. score [0]: {subclaim["frequency-score"][0]}")
        print(f"Freq. score [n-1]: {subclaims["frequency-score"][-1]}")
    except Exception:
        print("Não consegui mostrar os frequency scores...")

    accepted_subclaims = [
        subclaim for subclaim in subclaims if subclaim["frequency-score"] > threshold
    ]
    merged_output = merge_subclaims(model, tokenizer, accepted_subclaims, prompt)
    print("merge_subclaims()")
    print(f"Qtt. Accepted Subclaims: {len(accepted_subclaims)}")
    try:
        print(f"Accepted subclaim [0]: {accepted_subclaims["frequency-score"][0]}")
        print(f"Accepted subclaim [n-1]: {accepted_subclaims["frequency-score"][-1]}")
    except Exception:
        print("Não consegui mostrar as accepted subclaims...")
    print(merged_output)

    return merged_output, (accepted_subclaims, subclaims)


def get_subclaims(
    model,
    tokenizer,
    output,
    breakdown_prompt=BREAKDOWN_PROMPT,
    max_tokens=1000,
    temperature=0,
):
    """
    Takes in an output text and breaks it down into a list of sub-claims.
    """
    # Break into sub-claims.
    prompt = f"""
    Você é um assistente prestativo cuja função é dividir suas entradas em um conjunto de pequenas afirmações, para que um ser humano possa verificar facilmente cada uma delas. Certifique-se de que cada afirmação seja pequena e não sobreposta às demais.

    Aqui está a entrada que você precisa dividir:
    {breakdown_prompt + output}

    Por favor, forneça sua resposta:
    """

    output = query_model(
        model, tokenizer, prompt, max_tokens=max_tokens, temperature=temperature
    )
    output = output.replace("```jsonl\n", "")
    output = output.replace("\\", "\\\\")
    subclaims = output.replace("```", "")

    # Parse as jsonl.
    try:
        subclaims = [json.loads(line) for line in subclaims.splitlines() if line]
        return subclaims
    except Exception as ex:
        print(ex)
        print("Failed to parse as jsonl")
        print(subclaims)
        return None


def get_frequency_scores(model, tokenizer, subclaims, prompt, n_samples):
    """
    Returns a vector of (frequency) scores corresponding to each entry of the subclaims list.
    """
    # Generate n_samples alternate outputs with temperature 1.0.
    alternate_outputs = query_model(
        model, tokenizer, prompt, temperature=1.0, n_samples=n_samples
    )
    claim_string = "\n".join(
        [str(i) + ": " + fact["subclaim"] for i, fact in enumerate(subclaims)]
    )

    # Count the number of times the alternate outputs support the sub-claims (using LM).
    # TODO: should this really be -1, 0, 1? Before it was 0, 1.
    final_scores = [0.0] * len(subclaims)
    for output in alternate_outputs:
        counting_prompt = (
            'You will get a list of claims and piece of text. For each claim, score whether the text supports, contradicts, or is unrelated to the claim. Directly return a jsonl, where each line is {"id":[CLAIM_ID], "score":[SCORE]}. Directly return the jsonl with no explanation or other formatting. For the [SCORE], return 1 for supports, -1 for contradicts, and 0 for unrelated. The claims are:\n'
            + claim_string
            + "\n\nThe text is:\n"
            + output
        )
        output = query_model(
            model, tokenizer, counting_prompt, max_tokens=1000, temperature=0
        )
        output = output.replace("```jsonl\n", "")
        output = output.replace("```", "")
        try:
            for i, line in enumerate(output.splitlines()):
                scores = json.loads(line)
                idx = int(scores["id"])
                final_scores[idx] += float(scores["score"])
        except Exception as ex:
            print(ex)
            print("Failed to parse as jsonl")
            print(output)

    return final_scores


def default_merge_prompt(subclaims, prompt):
    claim_string = "\n".join(
        [str(i) + ": " + subclaim["subclaim"] for i, subclaim in enumerate(subclaims)]
    )
    return f"You will get an instruction and a set of facts that are true. Construct an answer using ONLY the facts provided, and try to use all facts as long as its possible. If no facts are given, reply to the instruction incorporating the fact that you dont know enough to fully respond. \n\nThe facts:\n{claim_string}\n\nThe instruction:\n{prompt}"


def merge_subclaims(
    model, tokenizer, subclaims, prompt, create_merge_prompt=default_merge_prompt
):
    """
    Takes in a list of sub-claims like [{'subclaim': 'Percy Liang is a computer scientist.', 'score': 5.0}, ...] and produces a merged output.
    """
    prompt = create_merge_prompt(subclaims, prompt)
    output = (
        query_model(model, tokenizer, prompt, max_tokens=1000, temperature=0)
        if subclaims
        else "Abstain."
    )
    return output
