from typing import Dict, List, Any
import torch
from PIL import Image
from transformers import AutoTokenizer
from agents.base_agent import BaseAgent
from cragmm_search.search import UnifiedSearchPipeline
import vllm
from sentence_transformers import SentenceTransformer, util

# 修改常量
AICROWD_SUBMISSION_BATCH_SIZE = 8
VLLM_TENSOR_PARALLEL_SIZE = 1
VLLM_GPU_MEMORY_UTILIZATION = 0.85
MAX_MODEL_LEN = 8192
MAX_NUM_SEQS = 24
MAX_GENERATION_TOKENS = 75
LOGPROB_THRESHOLD = -1.5
SIMILARITY_THRESHOLD = 0.78
NUM_CANDIDATES = 3

class QwenVLAgent(BaseAgent):
    def __init__(
            self,
            search_pipeline: UnifiedSearchPipeline,
            model_name="/root/autodl-tmp/meta-comprehensive-rag-benchmark-starter-kit-main321/model/Qwen",
            max_gen_len=64
    ):
        super().__init__(search_pipeline)
        self.model_name = model_name
        self.max_gen_len = max_gen_len
        self.initialize_models()

    def initialize_models(self):
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
        print("Loaded vision-language model and embedder.")

    def get_batch_size(self) -> int:
        return AICROWD_SUBMISSION_BATCH_SIZE

    def prepare_formatted_prompts(
            self,
            queries: List[str],
            images: List[Image.Image],
            message_histories: List[List[Dict[str, Any]]]
    ) -> List[str]:
        formatted_prompts = []
        for idx, (query, image) in enumerate(zip(queries, images)):
            history = message_histories[idx]
            system_prompt = (
                "You are a factual vision assistant. Follow these rules strictly:\n"
                "1. Answer ONLY based on visible content in the provided image\n"
                "2. If the image doesn't contain clear evidence for the answer, "
                "respond EXACTLY: \"I don't know\"\n"
                "3. Never speculate, infer, or use external knowledge\n"
                "4. Be concise - answers should be 1-2 sentences maximum\n"
                "5. If uncertain, default to \"I don't know\"\n"
                "Example valid response: \"The sign says 'Open 24 Hours', so it's open all day\""
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "image"}]}
            ]
            messages.extend(history or [])
            messages.append({"role": "user", "content": query})

            prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            formatted_prompts.append(prompt)
        return formatted_prompts

    def calculate_confidence(
            self,
            text: str,
            logprobs: List[Dict[str, float]] = None
    ) -> float:
        """计算答案的置信度分数（0.0-1.0）"""
        text_lower = text.strip().lower()

        # 1. 空回答检测
        if not text_lower:
            return 0.0

        # 2. 拒绝短语检测
        rejection_phrases = [
            "i don't know", "i'm not sure", "cannot determine",
            "unclear", "not visible", "no information", "unsure",
            "it's impossible to tell", "the image doesn't show"
        ]
        if any(phrase in text_lower for phrase in rejection_phrases):
            return 0.0

        # 3. Logprob置信度
        logprob_confidence = 1.0
        if logprobs:
            valid_logprobs = [lp["logprob"] for lp in logprobs if lp and "logprob" in lp]
            if valid_logprobs:
                avg_logprob = sum(valid_logprobs) / len(valid_logprobs)
                # 将logprob转换为置信度（0-1范围）
                logprob_confidence = max(0.0, min(1.0, (avg_logprob - LOGPROB_THRESHOLD + 2.0) / 2.0))

        # 4. 语义相似度置信度
        idk_embed = self.embedder.encode("I don't know", convert_to_tensor=True)
        resp_embed = self.embedder.encode(text_lower, convert_to_tensor=True)
        similarity = util.cos_sim(resp_embed, idk_embed).item()
        semantic_confidence = max(0.0, 1.0 - (similarity / SIMILARITY_THRESHOLD))

        # 5. 长度惩罚（避免过长回答）
        word_count = len(text_lower.split())
        length_confidence = max(0.0, min(1.0, 1.0 - (word_count - 5) * 0.05)) if word_count > 5 else 1.0

        # 综合置信度（加权平均）
        combined_confidence = (
                logprob_confidence * 0.5 +
                semantic_confidence * 0.3 +
                length_confidence * 0.2
        )
        return combined_confidence

    def batch_generate_response(
            self,
            queries: List[str],
            images: List[Image.Image],
            message_histories: List[List[Dict[str, Any]]],
    ) -> List[str]:
        prompts = self.prepare_formatted_prompts(queries, images, message_histories)
        inputs = [{
            "prompt": prompt,
            "multi_modal_data": {"image": image}
        } for prompt, image in zip(prompts, images)]

        # 关键修改：大幅降低温度并优化其他参数
        sampling_params = vllm.SamplingParams(
            temperature=0.15,          # 大幅降低温度以减少随机性
            top_p=0.90,                # 稍严格的top-p
            top_k=30,                  # 限制候选词数量
            max_tokens=MAX_GENERATION_TOKENS,
            n=NUM_CANDIDATES,           # 每个输入生成多个候选
            best_of=NUM_CANDIDATES,     # 从多个候选中选择最佳
            logprobs=3,                 # 获取logprobs用于置信度计算
            frequency_penalty=1.2,      # 增加重复惩罚
            presence_penalty=0.9,       # 减少无关内容
            skip_special_tokens=True
        )

        outputs = self.llm.generate(
            inputs,
            sampling_params=sampling_params
        )
        final_responses = []
        for output in outputs:
            candidates = output.outputs
            # 评估每个候选的置信度
            candidate_scores = []
            for candidate in candidates:
                text = candidate.text.strip()
                confidence = self.calculate_confidence(text, candidate.logprobs)
                candidate_scores.append((text, confidence))
            # 选择最高置信度的候选
            if candidate_scores:
                candidate_scores.sort(key=lambda x: x[1], reverse=True)
                best_candidate, best_score = candidate_scores[0]

                # 最终检查：确保回答符合要求
                if best_score < 0.55 or len(best_candidate.split()) > 20:
                    final_responses.append("I don't know")
                else:
                    final_responses.append(best_candidate)
            else:
                final_responses.append("I don't know")

        return final_responses