"""
Procesadores de inputs para modelos de deep learning.

Este módulo define interfaces y implementaciones para procesar diferentes tipos
de datos de entrada (frames, comportamiento, posición de pupila) y convertirlos
en tensores de PyTorch listos para entrenar modelos.
"""

import abc
from typing import Type, Optional, Tuple
import numpy as np
import torch


class InputsProcessor(metaclass=abc.ABCMeta):
    """
    Interfaz abstracta para procesadores de inputs.
    
    Define el contrato que deben cumplir todos los procesadores de inputs
    para convertir datos brutos en tensores de PyTorch.
    """
    
    @abc.abstractmethod
    def __call__(
        self, 
        frames: np.ndarray, 
        behavior: np.ndarray, 
        pupil_center: np.ndarray
    ) -> torch.Tensor:
        """
        Procesa los inputs brutos y los convierte en un tensor de PyTorch.
        
        Args:
            frames: Array de frames de video con shape (height, width, time)
            behavior: Array de datos de comportamiento con shape (2, time)  
            pupil_center: Array de posición de pupila con shape (2, time)
            
        Returns:
            torch.Tensor: Tensor procesado listo para el modelo
            
        Raises:
            ValueError: Si los inputs tienen dimensiones incompatibles
        """
        pass


class StackInputsProcessor(InputsProcessor):
    """
    Procesador que combina frames, comportamiento y pupila en un tensor apilado.
    
    Crea un tensor de 5 canales:
    - Canal 0: Frames de video (centrados y con padding)
    - Canales 1-2: Datos de comportamiento (broadcast a todas las posiciones)
    - Canales 3-4: Posición de pupila (broadcast a todas las posiciones)
    """
    
    def __init__(
        self,
        size: Tuple[int, int],
        pad_fill_value: float = 0.0
    ) -> None:
        """
        Inicializa el procesador con parámetros de configuración.
        
        Args:
            size: Tupla (width, height) del tamaño objetivo de salida
            pad_fill_value: Valor para rellenar el padding (default: 0.0)
            
        Raises:
            ValueError: Si size contiene valores no positivos
        """
        if not all(dim > 0 for dim in size):
            raise ValueError("Todas las dimensiones de size deben ser positivas")
        
        self.size = size
        self.pad_fill_value = float(pad_fill_value)
    
    def __call__(
        self, 
        frames: np.ndarray, 
        behavior: np.ndarray, 
        pupil_center: np.ndarray
    ) -> torch.Tensor:
        """
        Procesa y combina todos los inputs en un tensor unificado.
        
        Args:
            frames: Frames de video con shape (height, width, time)
            behavior: Datos de comportamiento con shape (2, time)
            pupil_center: Posición de pupila con shape (2, time)
            
        Returns:
            torch.Tensor: Tensor con shape (5, time, height, width)
            
        Raises:
            ValueError: Si las dimensiones de los inputs son incompatibles
        """
        # Validar inputs
        self._validate_inputs(frames, behavior, pupil_center)
        
        length = frames.shape[-1]
        
        # Crear array de salida con padding
        input_array = np.full(
            (5, length, self.size[1], self.size[0]), 
            self.pad_fill_value, 
            dtype=np.float32
        )
        
        # Procesar frames con padding centrado
        frames_processed = self._process_frames(frames)
        input_array[0] = frames_processed
        
        # Añadir datos de comportamiento y pupila
        input_array[1:3] = behavior[:, :, None, None]
        input_array[3:5] = pupil_center[:, :, None, None]
        
        return torch.from_numpy(input_array)
    
    def _validate_inputs(
        self, 
        frames: np.ndarray, 
        behavior: np.ndarray, 
        pupil_center: np.ndarray
    ) -> None:
        """
        Valida que los inputs tengan las dimensiones correctas.
        
        Args:
            frames: Array de frames
            behavior: Array de comportamiento  
            pupil_center: Array de posición de pupila
            
        Raises:
            ValueError: Si algún input tiene dimensiones incorrectas
        """
        if frames.ndim != 3:
            raise ValueError(f"frames debe tener 3 dimensiones, recibido: {frames.ndim}")
        
        if behavior.shape[0] != 2:
            raise ValueError(f"behavior debe tener shape (2, time), recibido: {behavior.shape}")
            
        if pupil_center.shape[0] != 2:
            raise ValueError(f"pupil_center debe tener shape (2, time), recibido: {pupil_center.shape}")
        
        # Verificar que todas las secuencias temporales tengan la misma longitud
        time_length = frames.shape[-1]
        if behavior.shape[1] != time_length:
            raise ValueError(f"Longitud temporal inconsistente: frames={time_length}, behavior={behavior.shape[1]}")
        
        if pupil_center.shape[1] != time_length:
            raise ValueError(f"Longitud temporal inconsistente: frames={time_length}, pupil_center={pupil_center.shape[1]}")
    
    def _process_frames(self, frames: np.ndarray) -> np.ndarray:
        """
        Procesa los frames aplicando transposición y padding centrado.
        
        Args:
            frames: Array de frames con shape (height, width, time)
            
        Returns:
            np.ndarray: Frames procesados con padding centrado
        """
        # Transponer para formato (time, height, width)
        frames_transposed = np.transpose(frames.astype(np.float32), (2, 0, 1))
        height, width = frames_transposed.shape[-2:]
        
        # Calcular posiciones para centrar
        height_start = (self.size[1] - height) // 2
        width_start = (self.size[0] - width) // 2
        
        # Crear array con padding
        padded_frames = np.full(
            (frames_transposed.shape[0], self.size[1], self.size[0]),
            self.pad_fill_value,
            dtype=np.float32
        )
        
        # Insertar frames centrados
        padded_frames[
            :, 
            height_start:height_start + height, 
            width_start:width_start + width
        ] = frames_transposed
        
        return padded_frames


# Registry de procesadores disponibles
INPUTS_PROCESSOR_REGISTRY: dict[str, Type[InputsProcessor]] = {
    "stack_inputs": StackInputsProcessor,
}


def get_inputs_processor(name: str, processor_params: dict) -> InputsProcessor:
    """
    Factory function para crear procesadores de inputs.
    
    Args:
        name: Nombre del procesador a crear
        processor_params: Parámetros de configuración del procesador
        
    Returns:
        InputsProcessor: Instancia del procesador solicitado
        
    Raises:
        KeyError: Si el nombre del procesador no está registrado
        TypeError: Si los parámetros son incorrectos
    """
    if name not in INPUTS_PROCESSOR_REGISTRY:
        available = list(INPUTS_PROCESSOR_REGISTRY.keys())
        raise KeyError(f"Procesador '{name}' no encontrado. Disponibles: {available}")
    
    try:
        return INPUTS_PROCESSOR_REGISTRY[name](**processor_params)
    except TypeError as e:
        raise TypeError(f"Error al crear procesador '{name}': {e}") from e
