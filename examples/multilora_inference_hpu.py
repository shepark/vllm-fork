"""
This example shows how to use the multi-LoRA functionality
for offline inference.

Requires HuggingFace credentials for access to Llama2.
"""

from typing import List, Optional, Tuple

from huggingface_hub import snapshot_download

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest


def create_test_prompts(
        lora_path: str
) -> List[Tuple[str, SamplingParams, Optional[LoRARequest]]]:
    """Create a list of test prompts with their sampling parameters.

    2 requests for base model, 4 requests for the LoRA. We define 2
    different LoRA adapters (using the same model for demo purposes).
    Since we also set `max_loras=1`, the expectation is that the requests
    with the second LoRA adapter will be ran after all requests with the
    first adapter have finished.
    """
    # TODO Fix issues when enabling paramerters [prompt_logprobs=1, presence_penalty=0.2,
    # (n=3, best_of=3, use_beam_search=True)] in SamplingParams.

    return [
        ("A robot may not injure a human being",
         SamplingParams(temperature=0.0,
                        logprobs=1,
                        prompt_logprobs=1,
                        max_tokens=128), None),
        ("To be or not to be,",
         SamplingParams(temperature=0.8,
                        top_k=5,
                        max_tokens=128), None),
        (
            "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",  # noqa: E501
            SamplingParams(temperature=0.0,
                           logprobs=1,
                           prompt_logprobs=1,
                           max_tokens=128,
                           stop_token_ids=[32003]),
            LoRARequest("sql-lora", 1, lora_path)),
        (
            "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",  # noqa: E501
            SamplingParams(
                           temperature=0,
                           max_tokens=128,
                           stop_token_ids=[32003]),
            LoRARequest("sql-lora", 1, lora_path)),
        (
            "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",  # noqa: E501
            SamplingParams(temperature=0.0,
                           logprobs=1,
                           prompt_logprobs=1,
                           max_tokens=128,
                           stop_token_ids=[32003]),
            LoRARequest("sql-lora2", 2, lora_path)),
        (
            "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",  # noqa: E501
            SamplingParams(temperature=0,
                           max_tokens=128,
                           stop_token_ids=[32003]),
            LoRARequest("sql-lora", 1, lora_path)),
    ]


def process_requests(engine: LLMEngine,
                     test_prompts: List[Tuple[str, SamplingParams,
                                              Optional[LoRARequest]]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0
    result = {}

    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params, lora_request = test_prompts.pop(0)
            engine.add_request(str(request_id),
                               prompt,
                               sampling_params,
                               lora_request=lora_request)
            request_id += 1

        request_outputs: List[RequestOutput] = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                result[request_output.request_id] = request_output.outputs[0].text
    return result


def initialize_engine() -> LLMEngine:
    """Initialize the LLMEngine."""
    # max_loras: controls the number of LoRAs that can be used in the same
    #   batch. Larger numbers will cause higher memory usage, as each LoRA
    #   slot requires its own preallocated tensor.
    # max_lora_rank: controls the maximum supported rank of all LoRAs. Larger
    #   numbers will cause higher memory usage. If you know that all LoRAs will
    #   use the same rank, it is recommended to set this as low as possible.
    # max_cpu_loras: controls the size of the CPU LoRA cache.
    engine_args = EngineArgs(model="meta-llama/Llama-2-7b-hf",
                             enable_lora=True,
                             max_loras=6,
                             max_lora_rank=8,
                             max_num_seqs=16,
                             dtype='bfloat16')
    return LLMEngine.from_engine_args(engine_args)

# References from GPU with dtype=bfloat16
expected_output = [
" or, through inaction, allow a human being to come to harm.\nA robot must obey the orders given it by human beings except where such orders would conflict with the First Law.\nA robot must protect its own existence as long as such protection does not conflict with the First or Second Law.\nThe Three Laws of Robotics were created by Isaac Asimov in 1942. They are the foundation of robotics and artificial intelligence.\nThe Three Laws of Robotics are the foundation of robotics and artificial intelligence. They were created by Isaac Asimov in 194", 
" that is the question.\nI am not sure what I would do if I had to make a decision to live or die.\nI would probably die, just to be sure.\nI think I would die, because if I were to live, I would not be able to live.\nIf I were dead, I could live, but I would not be happy.\nIf I had to choose between living and dying, I would die.\nIf I chose to live, I would die, but I would be happy, because I would be dead.\nSo if I were alive, I would die and be happy.",
"  SELECT icao FROM table_name_74 WHERE airport = 'lilongwe international airport' ",
"  SELECT nationality FROM table_name_11 WHERE elector = 'Anchero Pantaleone' ",
"  SELECT icao FROM table_name_74 WHERE airport = 'lilongwe international airport' ",
"  SELECT nationality FROM table_name_11 WHERE elector = 'Anchero Pantaleone' "
]

def main():
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine()
    lora_path = snapshot_download(repo_id="yard1/llama-2-7b-sql-lora-test")
    test_prompts = create_test_prompts(lora_path)
    result = process_requests(engine, test_prompts)
    for idx in result:
        generated_text = result[idx]
        matching = expected_output[int(idx)] == generated_text
        if not matching:
            print(f"{idx} matching::{matching} Generated text: {generated_text!r} expected_output: {expected_output[int(idx)]!r}")
        else:
            print(f"{idx} matching::{matching}")


if __name__ == '__main__':
    main()
