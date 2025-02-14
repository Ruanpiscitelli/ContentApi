"""
Wrapper para integração com LangChain.
"""
import logging
from typing import Any, Dict, List, Optional, Union
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from vllm import AsyncLLMEngine, SamplingParams

from config import VLLM_CONFIG, MODEL_CONFIG
from metrics import metrics

logger = logging.getLogger(__name__)

class VLLMWrapper(LLM):
    """Wrapper do vLLM para uso com LangChain."""
    
    def __init__(
        self,
        model_name: str,
        engine: AsyncLLMEngine,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        **kwargs
    ):
        """
        Inicializa o wrapper.
        
        Args:
            model_name: Nome do modelo
            engine: Engine vLLM
            max_tokens: Número máximo de tokens
            temperature: Temperatura para sampling
            top_p: Valor de top-p para sampling
            presence_penalty: Penalidade de presença
            frequency_penalty: Penalidade de frequência
            **kwargs: Argumentos adicionais
        """
        super().__init__(**kwargs)
        
        self.model_name = model_name
        self.engine = engine
        
        # Usa configurações do modelo ou valores padrão
        model_config = MODEL_CONFIG["models"].get(model_name, {})
        self.max_tokens = max_tokens or model_config.get("max_tokens", 100)
        self.temperature = temperature or model_config.get("temperature", 0.7)
        self.top_p = top_p or model_config.get("top_p", 0.95)
        self.presence_penalty = presence_penalty or model_config.get("presence_penalty", 0.0)
        self.frequency_penalty = frequency_penalty or model_config.get("frequency_penalty", 0.0)
    
    @property
    def _llm_type(self) -> str:
        """Retorna o tipo do LLM."""
        return "vllm"
    
    def _get_sampling_params(self, **kwargs) -> SamplingParams:
        """
        Cria parâmetros de sampling.
        
        Args:
            **kwargs: Parâmetros adicionais
            
        Returns:
            Parâmetros de sampling
        """
        return SamplingParams(
            n=1,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            top_p=kwargs.get("top_p", self.top_p),
            presence_penalty=kwargs.get("presence_penalty", self.presence_penalty),
            frequency_penalty=kwargs.get("frequency_penalty", self.frequency_penalty)
        )
    
    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> List[str]:
        """
        Gera texto de forma assíncrona.
        
        Args:
            prompts: Lista de prompts
            stop: Lista de strings de parada
            run_manager: Gerenciador de callbacks
            **kwargs: Parâmetros adicionais
            
        Returns:
            Lista de textos gerados
        """
        sampling_params = self._get_sampling_params(**kwargs)
        if stop:
            sampling_params.stop = stop
            
        try:
            outputs = []
            for prompt in prompts:
                # Gera texto
                result = await self.engine.generate(prompt, sampling_params)
                
                # Extrai texto gerado
                generated_text = result[0].outputs[0].text
                outputs.append(generated_text)
                
                # Registra métricas
                metrics.track_tokens(
                    self.model_name,
                    len(result[0].prompt_token_ids),
                    len(result[0].outputs[0].token_ids)
                )
                
                # Callbacks
                if run_manager:
                    await run_manager.on_llm_new_token(generated_text)
            
            return outputs
            
        except Exception as e:
            logger.error(f"Erro na geração com vLLM: {e}")
            metrics.track_error(self.model_name, type(e).__name__)
            raise
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        """
        Gera texto de forma síncrona.
        
        Args:
            prompt: Prompt de entrada
            stop: Lista de strings de parada
            run_manager: Gerenciador de callbacks
            **kwargs: Parâmetros adicionais
            
        Returns:
            Texto gerado
        """
        # Usa versão assíncrona
        import asyncio
        return asyncio.run(
            self._agenerate([prompt], stop, run_manager, **kwargs)
        )[0] 