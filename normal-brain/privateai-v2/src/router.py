"""
AI Router for PrivateAI
Routes queries to appropriate brain: Normal, Complex, or Coder
"""

import re
from typing import Literal, Dict, Any
from enum import Enum


class BrainType(Enum):
    NORMAL = "normal"
    COMPLEX = "complex"
    CODER = "coder"


class AIRouter:
    """Routes queries to appropriate brain based on keywords and patterns"""
    
    def __init__(self):
        # Coder patterns - programming related
        self.coder_patterns = [
            r"\b(code|program|function|class|debug|error|bug|script|api|library|framework)\b",
            r"\b(python|javascript|java|c\+\+|rust|go|sql|html|css|react|vue|django|flask)\b",
            r"\b(variable|loop|array|list|dict|object|method|parameter|return|import)\b",
            r"\b(github|git|commit|push|pull|merge|branch|repository)\b",
            r"\b(algorithm|data structure|complexity|big o|leetcode|hackerrank)\b",
            r"\b(docker|kubernetes|aws|azure|gcp|ci/cd|pipeline|deploy)\b",
            r"\b(component|hook|props|state|jsx|typescript|npm|pip|package)\b",
            r"\b(segfault|segmentation fault|memory leak|stack trace|exception)\b",
            r"\b(refactor|optimize|benchmark|profile|unit test|integration test)\b",
        ]
        
        # Complex patterns - math, science, reasoning
        self.complex_patterns = [
            r"\b(calculate|compute|solve|equation|formula|proof|theorem|derive)\b",
            r"\b(integral|derivative|limit|matrix|vector|probability|statistics)\b",
            r"\b(physics|chemistry|biology|astronomy|quantum|relativity)\b",
            r"\b(logic|reasoning|deduction|inference|premise|conclusion)\b",
            r"\b(optimize|minimize|maximize|constraint|linear programming)\b",
            r"\b(explain (why|how)|analyze|compare|contrast|evaluate)\b",
            r"\b(what is the (derivative|integral|probability|area|volume))\b",
            r"\b(entropy|thermodynamics|electromagnetism|mechanics|optics)\b",
            r"\b(hypothesis|experiment|observation|data analysis|regression)\b",
        ]
        
        # Compile patterns for efficiency
        self.coder_regex = [re.compile(p, re.IGNORECASE) for p in self.coder_patterns]
        self.complex_regex = [re.compile(p, re.IGNORECASE) for p in self.complex_patterns]
        
        # Confidence thresholds
        self.coder_threshold = 2
        self.complex_threshold = 2
        
    def route(self, query: str) -> Dict[str, Any]:
        """
        Route query to appropriate brain
        Returns: {"brain": BrainType, "confidence": float, "reason": str}
        """
        query_lower = query.lower()
        
        # Count matches
        coder_matches = sum(1 for r in self.coder_regex if r.search(query))
        complex_matches = sum(1 for r in self.complex_regex if r.search(query))
        
        # Determine route
        if coder_matches >= self.coder_threshold:
            return {
                "brain": BrainType.CODER,
                "confidence": min(coder_matches / self.coder_threshold, 1.0),
                "reason": f"Coder keywords matched: {coder_matches}"
            }
        elif complex_matches >= self.complex_threshold:
            return {
                "brain": BrainType.COMPLEX,
                "confidence": min(complex_matches / self.complex_threshold, 1.0),
                "reason": f"Complex keywords matched: {complex_matches}"
            }
        else:
            return {
                "brain": BrainType.NORMAL,
                "confidence": 0.7,
                "reason": "Default to normal brain"
            }
    
    def route_with_context(self, query: str, context: str = "") -> Dict[str, Any]:
        """Route with additional context"""
        full_text = query + " " + context
        return self.route(full_text)


class ModelManager:
    """Manages loading and inference for all three brains"""
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = models_dir
        self.models = {}
        self.router = AIRouter()
        
    def load_model(self, brain_type: BrainType):
        """Load a specific brain model (lazy loading)"""
        if brain_type in self.models:
            return self.models[brain_type]
        
        model_paths = {
            BrainType.NORMAL: "privateai-normal-early-q8.gguf",
            BrainType.COMPLEX: "complex_brain_q4.gguf",  # placeholder
            BrainType.CODER: "coder_brain_q4.gguf",       # placeholder
        }
        
        try:
            from llama_cpp import Llama
            model_path = f"{self.models_dir}/{model_paths[brain_type]}"
            
            # Context sizes per brain
            ctx_sizes = {
                BrainType.NORMAL: 2048,
                BrainType.COMPLEX: 4096,  # More context for reasoning
                BrainType.CODER: 4096,    # More context for code
            }
            
            model = Llama(
                model_path=model_path,
                n_ctx=ctx_sizes[brain_type],
                n_threads=4,
                n_gpu_layers=0,
                verbose=False,
            )
            self.models[brain_type] = model
            return model
            
        except Exception as e:
            print(f"Failed to load {brain_type.value} brain: {e}")
            return None
    
    def chat(self, query: str, context: str = "") -> str:
        """Route query and generate response"""
        route_info = self.router.route_with_context(query, context)
        brain_type = route_info["brain"]
        
        model = self.load_model(brain_type)
        if not model:
            return f"Error: {brain_type.value} brain not available"
        
        # Format prompt based on brain
        prompts = {
            BrainType.NORMAL: f"<|system|>\nYou are PrivateAI, a helpful local AI assistant.\n<|user|>\n{query}\n<|assistant|>\n",
            BrainType.COMPLEX: f"<|system|>\nYou are PrivateAI Complex Brain, specialized in mathematics, science, and reasoning. Show your work step by step.\n<|user|>\n{query}\n<|assistant|>\n",
            BrainType.CODER: f"<|system|>\nYou are PrivateAI Coder Brain, specialized in programming. Write clean, efficient code with explanations.\n<|user|>\n{query}\n<|assistant|>\n",
        }
        
        prompt = prompts[brain_type]
        
        try:
            output = model(
                prompt,
                max_tokens=512,
                temperature=0.7,
                top_p=0.9,
                stop=["<|user|>", "<|system|>", "<|endoftext|>"],
                echo=False,
            )
            return output['choices'][0]['text'].strip()
        except Exception as e:
            return f"Generation error: {e}"
    
    def get_route_info(self, query: str) -> Dict[str, Any]:
        """Get routing info without generating"""
        return self.router.route(query)


def demo():
    """Test router without models"""
    router = AIRouter()
    
    test_queries = [
        "What is the capital of France?",
        "Write a Python function to reverse a string",
        "Calculate the derivative of x^2 + 3x + 2",
        "How do I debug a segmentation fault in C++?",
        "Explain quantum entanglement",
        "Write a React component for a login form",
        "What's the weather like today?",
        "Solve this integral: ∫(x^2)dx from 0 to 1",
    ]
    
    print("Router Demo:")
    print("=" * 60)
    for q in test_queries:
        result = router.route(q)
        print(f"Q: {q}")
        print(f"  → {result['brain'].value.upper()} (confidence: {result['confidence']:.1f})")
        print(f"  Reason: {result['reason']}")
        print()


if __name__ == "__main__":
    demo()