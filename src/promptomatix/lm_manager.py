from typing import Optional
import dspy
from dspy.backends import OpenAI, Anthropic, Cohere

class LMManager:
    """Manages Language Model initialization and configuration for different providers"""
    
    SUPPORTED_PROVIDERS = {
        'openai': OpenAI,
        'anthropic': Anthropic,
        'cohere': Cohere,
        # Add more providers as needed
        'bedrock': dspy.LM
    }

    @classmethod
    def get_lm(cls, 
               provider: str, 
               model_name: str, 
               api_key: str, 
               api_base: Optional[str] = None,
               temperature: float = 0.0,
               max_tokens: int = 4000,
               **kwargs):
        """
        Initialize and return appropriate language model based on provider.
        
        Args:
            provider: The model provider (e.g., 'openai', 'anthropic')
            model_name: Name of the model to use
            api_key: API key for the provider
            api_base: Optional API base URL
            temperature: Sampling temperature
            max_tokens: Maximum tokens for generation
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Initialized language model instance
            
        Raises:
            ValueError: If provider is not supported
        """
        provider = provider.lower()
        if provider not in cls.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported providers are: {list(cls.SUPPORTED_PROVIDERS.keys())}"
            )

        # Get the appropriate LM class
        lm_class = cls.SUPPORTED_PROVIDERS[provider]
        bedrock_model_names = [
            "us.anthropic.claude-3-haiku-20240307-v1:0",
            "us.anthropic.claude-3-opus-20240229-v1:0",
            "us.anthropic.claude-3-sonnet-20240229-v1:0",
            "us.anthropic.claude-3-5-haiku-20241022-v1:0",
            "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "us.anthropic.claude-opus-4-20250514-v1:0",
            "us.anthropic.claude-sonnet-4-20250514-v1:0",
            "us.deepseek.r1-v1:0",
            "us.meta.llama4-maverick-17b-instruct-v1:0",
            "us.meta.llama4-scout-17b-instruct-v1:0",
            "us.meta.llama3-1-70b-instruct-v1:0",
            "us.meta.llama3-1-8b-instruct-v1:0",
            "us.meta.llama3-2-11b-instruct-v1:0",
            "us.meta.llama3-2-1b-instruct-v1:0",
            "us.meta.llama3-2-3b-instruct-v1:0",
            "us.meta.llama3-2-90b-instruct-v1:0",
            "us.meta.llama3-3-70b-instruct-v1:0",
            "us.mistral.pixtral-large-2502-v1:0",
            "us.amazon.nova-lite-v1:0",
            "us.amazon.nova-micro-v1:0",
            "us.amazon.nova-premier-v1:0",
            "us.amazon.nova-pro-v1:0",
        ]
        if provider == 'bedrock':
            try:
                import boto3  # noqa: F401
            except ImportError:
                raise ImportError(
                    "boto3 is required for Bedrock provider but is not installed. "
                    "Please install it with 'pip install boto3[crt]'."
                )
            if 'bedrock/' in model_name:
                raise ValueError(f"Model name {model_name} is not a valid Bedrock model name. If model_name begins with 'bedrock/', please remove the 'bedrock/' prefix.")
            if model_name not in bedrock_model_names:
                raise Warning(f"Model name {model_name} may not be a valid Bedrock model name. Please use cross inference model ID to avoid bedrock errors.")
            return dspy.LM(
                model=f"bedrock/{model_name}",
                max_tokens=max_tokens,
                temperature=temperature,
            )
        # Prepare base arguments
        lm_args = {
            "model": model_name,
            "api_key": api_key,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        # Add provider-specific configurations
        if provider == 'openai' and api_base:
            lm_args["api_base"] = api_base
            
        # Initialize the LM
        try:
            return lm_class(**lm_args)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize {provider} LM: {str(e)}")

    @staticmethod
    def configure_dspy(lm) -> None:
        """Configure DSPy with the given language model"""
        dspy.configure(lm=lm) 