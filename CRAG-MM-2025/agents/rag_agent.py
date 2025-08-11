from typing import Dict, List, Any, Tuple
import os
import torch
from PIL import Image
from agents.base_agent import BaseAgent
from cragmm_search.search import UnifiedSearchPipeline
from crag_web_result_fetcher import WebSearchResult
import vllm
from sentence_transformers import SentenceTransformer, util

# Constants
AICROWD_SUBMISSION_BATCH_SIZE = 8
VLLM_TENSOR_PARALLEL_SIZE = 1
VLLM_GPU_MEMORY_UTILIZATION = 0.75
MAX_MODEL_LEN = 32768
MAX_NUM_SEQS = 2
MAX_GENERATION_TOKENS = 60
NUM_SEARCH_RESULTS = 10

# hallucination control thresholds
LOGPROB_THRESHOLD = -1.0
SIMILARITY_THRESHOLD = 0.85
STABILITY_SIM_THRESHOLD = 0.88

# Max query len for summarize
MAX_QUERY_LEN = 512
MAX_SUMMARIZE_PROMPT_TOKENS = 4096

class SimpleRAGAgent(BaseAgent):
    def __init__(
        self,
        search_pipeline: UnifiedSearchPipeline,
        model_name: str = "/root/autodl-tmp/meta-comprehensive-rag-benchmark-starter-kit-main321/model/Qwen",
        max_gen_len: int = 64
    ):
        super().__init__(search_pipeline)
        if search_pipeline is None:
            raise ValueError("Search pipeline is required for RAG agent")
        self.model_name = model_name
        self.max_gen_len = max_gen_len
        self.initialize_models()

    def initialize_models(self):
        print(f"Initializing {self.model_name} with vLLM...")
        self.llm = vllm.LLM(
            self.model_name,
            tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=MAX_NUM_SEQS,
            trust_remote_code=True,
            dtype="bfloat16",
            enforce_eager=True,
            limit_mm_per_prompt={"image": 1}
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("Models loaded successfully")

    def get_batch_size(self) -> int:
        return AICROWD_SUBMISSION_BATCH_SIZE

    def batch_summarize_images(self, images: List[Image.Image], queries: List[str]) -> List[str]:
        summaries = []
        for image, query in zip(images, queries):
            short_query = query[:MAX_QUERY_LEN]
            summarize_prompt = f"Summarize the parts of the image relevant to the question: '{short_query}' in one concise sentence."
            messages = [
                {"role": "system", "content": "You are an expert visual analyst. Provide image summaries relevant to the user's question."},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": summarize_prompt}]},
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            encoded_prompt = self.tokenizer.encode(formatted_prompt)
            if len(encoded_prompt) > MAX_SUMMARIZE_PROMPT_TOKENS:
                print(f"[WARN] summarize prompt too long ({len(encoded_prompt)}), truncating...")
                encoded_prompt = encoded_prompt[:MAX_SUMMARIZE_PROMPT_TOKENS]
                formatted_prompt = self.tokenizer.decode(encoded_prompt)

            inputs = [{
                "prompt": formatted_prompt,
                "multi_modal_data": {"image": image}
            }]
            output = self.llm.generate(
                inputs,
                sampling_params=vllm.SamplingParams(
                    temperature=0.1,
                    top_p=0.9,
                    max_tokens=40,
                    skip_special_tokens=True
                )
            )
            summary = output[0].outputs[0].text.strip()
            summaries.append(summary)
        print(f"Generated {len(summaries)} question-aware image summaries")
        return summaries

    def prepare_rag_enhanced_inputs(
        self,
        queries: List[str],
        images: List[Image.Image],
        image_summaries: List[str],
        message_histories: List[List[Dict[str, Any]]]
    ) -> List[dict]:
        search_results_batch = []
        search_queries = [f"{query} {summary}" for query, summary in zip(queries, image_summaries)]
        for search_query in search_queries:
            results = self.search_pipeline(search_query, k=NUM_SEARCH_RESULTS)
            search_results_batch.append(results)

        inputs = []
        for idx, (query, image, history, search_results) in enumerate(
            zip(queries, images, message_histories, search_results_batch)
        ):
            SYSTEM_PROMPT = (
                "You are a helpful and context-aware vision-language assistant.\n"
                "You are in a multi-turn conversation. Answer naturally and consistently with the prior dialogue.\n"
                "Use the image and context to answer the user's query truthfully.\n"
                "Avoid hallucinations. If unsure, say 'I don't know'."
            )

            rag_context = ""
            if search_results:
                query_image_text = f"{query} {image_summaries[idx]}"
                query_image_embed = self.embedder.encode(query_image_text, convert_to_tensor=True)

                scored_results = []
                for result in search_results:
                    result_obj = WebSearchResult(result)
                    snippet = result_obj.get("page_snippet", "")
                    if snippet.strip():
                        snippet_embed = self.embedder.encode(snippet, convert_to_tensor=True)
                        sim = util.cos_sim(query_image_embed, snippet_embed).item()
                        score = result.get("score", 0.0)
                        combined_score = 0.4 * score + 0.6 * sim
                        scored_results.append((combined_score, result_obj))

                sorted_results = sorted(scored_results, key=lambda x: x[0], reverse=True)[:4]
                rag_context = "Here is some additional information that may help:\n\n"
                for i, (combined_score, result_obj) in enumerate(sorted_results):
                    snippet = result_obj.get("page_snippet", "")
                    if snippet:
                        rag_context += f"[Info {i+1}] {snippet}\n\n"

            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            if history:
                messages += history
            if image:
                messages.append({"role": "user", "content": [{"type": "image"}]})
            if rag_context:
                messages.append({"role": "user", "content": rag_context})
            messages.append({"role": "user", "content": query})

            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            inputs.append({
                "prompt": formatted_prompt,
                "multi_modal_data": {"image": image} if image else {}
            })
        return inputs

    def is_low_confidence(self, text: str, logprobs: List[Dict[str, float]]) -> bool:
        if not text.strip():
            return True
        avg_logprob = (
            sum(p["logprob"] for p in logprobs if "logprob" in p) / len(logprobs)
            if logprobs else -10.0
        )
        if avg_logprob < LOGPROB_THRESHOLD:
            return True
        idk_embed = self.embedder.encode("I don't know", convert_to_tensor=True)
        resp_embed = self.embedder.encode(text.strip(), convert_to_tensor=True)
        similarity = util.cos_sim(resp_embed, idk_embed).item()
        return similarity > SIMILARITY_THRESHOLD

    def generate_single_response(self, input_data: dict, strict: bool = False) -> Tuple[str, List[Dict[str, float]]]:
        temperature = 0.07 if strict else 0.10
        if not strict:
            prompt_data = input_data
        else:
            messages = self.tokenizer.parse_chat_template(input_data["prompt"])
            system_msg = {"role": "system", "content": (
                "You are a highly cautious and context-aware vision-language assistant.\n"
                "Avoid guessing. Use image and dialogue history only. If unsure, say 'I don't know'."
            )}
            if messages and messages[0]["role"] == "system":
                messages[0] = system_msg
            else:
                messages.insert(0, system_msg)
            new_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            prompt_data = {
                "prompt": new_prompt,
                "multi_modal_data": input_data["multi_modal_data"]
            }
        output = self.llm.generate(
            [prompt_data],
            sampling_params=vllm.SamplingParams(
                temperature=temperature,
                top_p=0.85,
                max_tokens=MAX_GENERATION_TOKENS,
                logprobs=1,
                skip_special_tokens=True
            )
        )
        result = output[0].outputs[0]
        return result.text.strip(), result.logprobs

    def batch_generate_response(
        self,
        queries: List[str],
        images: List[Image.Image],
        message_histories: List[List[Dict[str, Any]]],
    ) -> List[str]:
        print(f"Processing batch of {len(queries)} queries with multi-pass generation and stability check...")
        image_summaries = self.batch_summarize_images(images, queries)
        rag_inputs = self.prepare_rag_enhanced_inputs(queries, images, image_summaries, message_histories)

        final_responses = []
        for input_data in rag_inputs:
            candidates = []
            text, logprobs = self.generate_single_response(input_data, strict=False)
            candidates.append(text)
            low_conf = self.is_low_confidence(text, logprobs)

            if low_conf:
                print("Low confidence detected in first generation, retrying with strict prompt...")
                text_strict, logprobs_strict = self.generate_single_response(input_data, strict=True)
                candidates.append(text_strict)
                low_conf = self.is_low_confidence(text_strict, logprobs_strict)

                if low_conf:
                    print("Still low confidence after second generation, retrying once more with strict prompt...")
                    text_strict2, logprobs_strict2 = self.generate_single_response(input_data, strict=True)
                    candidates.append(text_strict2)

                    embeds = self.embedder.encode(candidates, convert_to_tensor=True)
                    sim12 = util.cos_sim(embeds[0], embeds[1]).item()
                    sim13 = util.cos_sim(embeds[0], embeds[2]).item()
                    sim23 = util.cos_sim(embeds[1], embeds[2]).item()
                    avg_sim = (sim12 + sim13 + sim23) / 3

                    if avg_sim < STABILITY_SIM_THRESHOLD:
                        print(f"Generation unstable (avg_sim={avg_sim:.3f}), outputting 'I don't know'")
                        final_responses.append("I don't know")
                        continue
                    else:
                        final_responses.append(text_strict2)
                        continue
                else:
                    final_responses.append(text_strict)
                    continue
            else:
                final_responses.append(text)

        print(f"Generated {len(final_responses)} final responses after multi-pass confidence check")
        return final_responses
