"""
Pipeline customizado do FastHunyuan para geração de vídeos.
"""
import torch
import torch.nn.functional as F
from typing import List, Optional, Union, Tuple
from dataclasses import dataclass

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput
from transformers import CLIPTextModel, CLIPTokenizer

@dataclass
class FastHunyuanPipelineOutput(BaseOutput):
    """
    Saída do pipeline FastHunyuan.
    
    Args:
        videos: Lista de tensores representando os vídeos gerados
    """
    videos: List[torch.FloatTensor]

class FastHunyuanPipeline(DiffusionPipeline):
    """
    Pipeline para geração de vídeos usando FastHunyuan.
    """
    def __init__(
        self,
        vae,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet,
        scheduler,
    ):
        super().__init__()
        
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )
        
        # Configurações padrão
        self.vae_scale_factor = 8
        
    def enable_vae_slicing(self):
        """Habilita VAE slicing para economia de memória."""
        self.vae.enable_slicing()
        
    def enable_sequential_cpu_offload(self):
        """Habilita offload sequencial para CPU."""
        self.device = torch.device("cuda")
        for name, module in self.named_modules():
            if "text_" in name:
                device_map = {"": torch.device("cpu")}
            else:
                device_map = {"": self.device}
            module.to(device_map[""])
            
    def prepare_latents(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: Union[str, torch.device],
        generator: Optional[torch.Generator] = None,
    ) -> torch.FloatTensor:
        """
        Prepara os latents iniciais para a geração.
        """
        shape = (
            batch_size,
            self.unet.config.in_channels,
            num_frames,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        
        if isinstance(generator, list):
            latents = [
                torch.randn(shape, generator=generator[i], device=device, dtype=dtype)
                for i in range(len(generator))
            ]
            latents = torch.cat(latents, dim=0)
        else:
            latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
            
        return latents
    
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Union[str, torch.device],
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
        negative_prompt: Optional[Union[str, List[str]]] = None,
    ) -> torch.FloatTensor:
        """
        Codifica o prompt para embeddings de texto.
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        text_embeddings = self.text_encoder(text_input_ids.to(device))[0]
        
        # Processa prompt negativo se necessário
        if do_classifier_free_guidance:
            uncond_tokens = negative_prompt or ""
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            
        text_embeddings = text_embeddings.repeat_interleave(num_videos_per_prompt, dim=0)
        return text_embeddings
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = 576,
        width: Optional[int] = 1024,
        num_frames: Optional[int] = 16,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 9.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        use_fp8: bool = True,
    ) -> FastHunyuanPipelineOutput:
        """
        Função principal de geração de vídeos.
        """
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # Codifica prompt
        text_embeddings = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )
        
        # Prepara latents
        latents = self.prepare_latents(
            batch_size=1,
            num_frames=num_frames,
            height=height,
            width=width,
            dtype=text_embeddings.dtype,
            device=device,
            generator=generator,
        )
        
        # Define timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # Loop de geração
        for i, t in enumerate(timesteps):
            # Expande latents para guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            
            # Predição do ruído residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
            ).sample
            
            # Guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Passo de denoising
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decodifica latents
        video = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        
        # Normaliza vídeo
        video = (video / 2 + 0.5).clamp(0, 1)
        
        return FastHunyuanPipelineOutput(videos=[video]) 