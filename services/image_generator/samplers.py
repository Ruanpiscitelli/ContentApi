import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Callable, Dict, Any
from tqdm import tqdm

class BaseSampler(ABC):
    """Classe base para todos os samplers"""
    
    @abstractmethod
    def sample(self, model, x, sigmas, extra_args=None, callback=None, disable=False):
        """
        Método abstrato para sampling
        Args:
            model: O modelo de difusão
            x: Tensor inicial (ruído ou latente)
            sigmas: Schedule de ruído
            extra_args: Argumentos adicionais para o modelo
            callback: Função de callback para progresso
            disable: Desabilitar barra de progresso
        """
        pass

class DDIMSampler(BaseSampler):
    def sample(self, model, x, sigmas, extra_args=None, callback=None, disable=False):
        """
        DDIM Sampling
        Referência: https://arxiv.org/abs/2010.02502
        """
        extra_args = extra_args or {}
        s_in = x.new_ones([x.shape[0]])
        
        for i in tqdm(range(len(sigmas) - 1), disable=disable):
            denoised = model(x, sigmas[i] * s_in, **extra_args)
            sigma_up, sigma_down = sigmas[i], sigmas[i + 1]
            t_steps = (sigma_down - sigma_up) / (sigma_down + 1e-8)
            
            x = denoised + t_steps[:, None, None, None] * (x - denoised)
            
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_down, 
                         'denoised': denoised})
        return x

class DPMSolverMultistepSampler(BaseSampler):
    def sample(self, model, x, sigmas, extra_args=None, callback=None, disable=False):
        """
        DPM-Solver++ (2M) Sampling
        Referência: https://arxiv.org/abs/2211.01095
        """
        extra_args = extra_args or {}
        s_in = x.new_ones([x.shape[0]])
        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()
        
        old_denoised = None
        h_last = None
        
        for i in tqdm(range(len(sigmas) - 1), disable=disable):
            denoised = model(x, sigmas[i] * s_in, **extra_args)
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            
            if old_denoised is None or h_last is None:
                x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
            else:
                h_frac = h / h_last
                x = (sigma_fn(t_next) / sigma_fn(t)) * x - (h * ((1 + 1 / (2 * h_frac)) * denoised - 
                     1 / (2 * h_frac) * old_denoised))
            
            old_denoised = denoised
            h_last = h
            
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i + 1],
                         'denoised': denoised})
        return x

class EulerSampler(BaseSampler):
    def sample(self, model, x, sigmas, extra_args=None, callback=None, disable=False):
        """
        Euler sampling method
        """
        extra_args = extra_args or {}
        s_in = x.new_ones([x.shape[0]])
        
        for i in tqdm(range(len(sigmas) - 1), disable=disable):
            denoised = model(x, sigmas[i] * s_in, **extra_args)
            d = (x - denoised) / sigmas[i]
            dt = sigmas[i + 1] - sigmas[i]
            x = x + d * dt
            
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i + 1],
                         'denoised': denoised})
        return x

class EulerAncestralSampler(BaseSampler):
    def sample(self, model, x, sigmas, extra_args=None, callback=None, disable=False):
        """
        Euler-Ancestral sampling method
        """
        extra_args = extra_args or {}
        s_in = x.new_ones([x.shape[0]])
        noise = torch.randn_like(x)
        
        for i in tqdm(range(len(sigmas) - 1), disable=disable):
            denoised = model(x, sigmas[i] * s_in, **extra_args)
            sigma_down, sigma_up = sigmas[i + 1], sigmas[i]
            
            # Euler step
            d = (x - denoised) / sigma_up
            dt = sigma_down - sigma_up
            x = x + d * dt
            
            # Adiciona ruído
            if sigma_down > 0:
                noise = torch.randn_like(x)
                x = x + noise * torch.sqrt(sigma_down ** 2 - sigma_up ** 2)
            
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_down,
                         'denoised': denoised})
        return x

class UniPCSampler(BaseSampler):
    def sample(self, model, x, sigmas, extra_args=None, callback=None, disable=False):
        """
        UniPC (Unified Predictor-Corrector) Sampling
        Referência: https://arxiv.org/abs/2302.04867
        """
        extra_args = extra_args or {}
        s_in = x.new_ones([x.shape[0]])
        
        # Parâmetros do UniPC
        order = 3
        t = sigmas.log().neg()
        h = t[1:] - t[:-1]
        r1 = h[1:] / h[:-1]
        
        denoised_list = []
        for i in tqdm(range(len(sigmas) - 1), disable=disable):
            denoised = model(x, sigmas[i] * s_in, **extra_args)
            denoised_list.append(denoised)
            
            if len(denoised_list) >= order:
                # Cálculo dos coeficientes UniPC
                if len(denoised_list) == order:
                    phi_1 = torch.ones_like(r1[i-1])
                    phi_2 = (1 + r1[i-1]) * phi_1
                    phi_3 = (1 + r1[i-1] + r1[i-1]**2) * phi_1
                    
                    # Predição UniPC
                    x = (1 + phi_1 + phi_2 + phi_3) * denoised_list[-1] - \
                        (phi_1 + phi_2) * denoised_list[-2] + \
                        phi_1 * denoised_list[-3]
                else:
                    # Euler step para os primeiros passos
                    d = (x - denoised) / sigmas[i]
                    dt = sigmas[i + 1] - sigmas[i]
                    x = x + d * dt
            else:
                # Euler step para os primeiros passos
                d = (x - denoised) / sigmas[i]
                dt = sigmas[i + 1] - sigmas[i]
                x = x + d * dt
            
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i + 1],
                         'denoised': denoised})
        return x

SAMPLER_REGISTRY = {
    'ddim': DDIMSampler,
    'dpm_solver_multistep': DPMSolverMultistepSampler,
    'euler': EulerSampler,
    'euler_ancestral': EulerAncestralSampler,
    'unipc': UniPCSampler
}

def get_sampler(name: str) -> BaseSampler:
    """
    Retorna uma instância do sampler pelo nome
    Args:
        name: Nome do sampler
    Returns:
        Instância do sampler
    Raises:
        ValueError: Se o sampler não existir
    """
    if name not in SAMPLER_REGISTRY:
        raise ValueError(f"Sampler {name} não encontrado. Opções disponíveis: {list(SAMPLER_REGISTRY.keys())}")
    return SAMPLER_REGISTRY[name]() 