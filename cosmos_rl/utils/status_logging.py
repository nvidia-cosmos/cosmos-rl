# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Status logging system for Cosmos-RL based on TAO Deploy implementation."""

from abc import abstractmethod
import atexit
from datetime import datetime
import json
import logging
import os
from typing import Dict, Any, Optional

# Import cosmos-rl logger
from cosmos_rl.utils.logging import logger
from nvidia_tao_core.cloud_handlers.utils import status_callback


class Verbosity:
    """Verbosity levels for status logging."""

    DISABLE = 0
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


# Defining a log level to name dictionary.
log_level_to_name = {
    Verbosity.DISABLE: "DISABLE",
    Verbosity.DEBUG: 'DEBUG',
    Verbosity.INFO: 'INFO',
    Verbosity.WARNING: 'WARNING',
    Verbosity.ERROR: 'ERROR',
    Verbosity.CRITICAL: 'CRITICAL'
}


class Status:
    """Status levels for tracking execution states."""

    SUCCESS = 0
    FAILURE = 1
    STARTED = 2
    RUNNING = 3
    SKIPPED = 4


status_level_to_name = {
    Status.SUCCESS: 'SUCCESS',
    Status.FAILURE: 'FAILURE',
    Status.STARTED: 'STARTED',
    Status.RUNNING: 'RUNNING',
    Status.SKIPPED: 'SKIPPED'
}


class BaseStatusLogger:
    """Base status logger class for Cosmos-RL."""

    def __init__(self, is_master: bool = False, verbosity: int = Verbosity.INFO):
        """Initialize base logger class.
        
        Args:
            is_master: Whether this is the master process
            verbosity: Logging verbosity level
        """
        self.is_master = is_master
        self.verbosity = verbosity
        self.categorical = {}
        self.graphical = {}
        self.kpi = {}

    @property
    def date(self) -> str:
        """Get current date string."""
        date_time = datetime.now()
        date_object = date_time.date()
        return f"{date_object.month}/{date_object.day}/{date_object.year}"

    @property
    def time(self) -> str:
        """Get current time string."""
        date_time = datetime.now()
        time_object = date_time.time()
        return f"{time_object.hour}:{time_object.minute}:{time_object.second}"

    @property
    def categorical(self) -> Dict[str, Any]:
        """Categorical data to be logged."""
        return self._categorical

    @categorical.setter
    def categorical(self, value: Dict[str, Any]):
        """Set categorical data to be logged."""
        self._categorical = value

    @property
    def graphical(self) -> Dict[str, Any]:
        """Graphical data to be logged."""
        return self._graphical

    @graphical.setter
    def graphical(self, value: Dict[str, Any]):
        """Set graphical data to be logged."""
        self._graphical = value

    @property
    def kpi(self) -> Dict[str, Any]:
        """KPI data to be logged."""
        return self._kpi

    @kpi.setter
    def kpi(self, value: Dict[str, Any]):
        """Set KPI data to be logged."""
        self._kpi = value


    def flush(self):
        """Flush the logger."""
        pass

    def format_data(self, data: Dict[str, Any]) -> str:
        """Format the data as JSON string.
        
        Args:
            data: Dictionary data to format
            
        Returns:
            JSON formatted string
        """
        if not isinstance(data, dict):
            raise TypeError(f"Data must be a dictionary and not type {type(data)}.")
        return json.dumps(data, indent=2, default=str)

    def log(self, level: int, string: str):
        """Log the data string.
        
        Args:
            level: Log level
            string: String to log
        """
        if level >= self.verbosity:
            logger.log(level, string)

    @abstractmethod
    def write(self, 
              data: Optional[Dict[str, Any]] = None,
              status_level: int = Status.RUNNING,
              verbosity_level: int = Verbosity.INFO,
              message: Optional[str] = None,
              step: Optional[int] = None,
              epoch: Optional[int] = None,
              replica_name: Optional[str] = None):
        """Write data out to the log file.
        
        Args:
            data: Additional data to log
            status_level: Status level
            verbosity_level: Verbosity level
            message: Log message
            step: Training step
            epoch: Training epoch
            replica_name: Name of replica if applicable
        """
        if self.verbosity > Verbosity.DISABLE:
            if not data:
                data = {}
            
            # Define generic data
            data["date"] = self.date
            data["time"] = self.time
            data["status"] = status_level_to_name.get(status_level, "RUNNING")
            data["verbosity"] = log_level_to_name.get(verbosity_level, "INFO")

            if message:
                data["message"] = message
            
            if step is not None:
                data["step"] = step
            
            if epoch is not None:
                data["epoch"] = epoch
                
            if replica_name:
                data["replica_name"] = replica_name

            if self.categorical:
                data["categorical"] = self.categorical

            if self.graphical:
                data["graphical"] = self.graphical

            if self.kpi:
                data["kpi"] = self.kpi

            data_string = self.format_data(data)
            if self.is_master:
                self.log(verbosity_level, data_string)
            self.flush()
            
            # Call any callbacks if needed
            self._call_status_callback(data_string)

    def _call_status_callback(self, data_string: str):
        """Call status callback if available.
        
        Args:
            data_string: Formatted data string
        """
        status_callback(data_string)


class StatusLogger(BaseStatusLogger):
    """File-based status logger for Cosmos-RL."""

    def __init__(self, 
                 filename: Optional[str] = None,
                 is_master: bool = False,
                 verbosity: int = Verbosity.INFO,
                 append: bool = True):
        """Initialize status logger.
        
        Args:
            filename: Path to log file
            is_master: Whether this is the master process
            verbosity: Logging verbosity level
            append: Whether to append to existing file
        """
        super().__init__(is_master=is_master, verbosity=verbosity)
        
        if filename:
            self.log_path = os.path.realpath(filename)
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            
            self.append = append
            if os.path.exists(self.log_path) and append:
                logger.info(f"Status log file already exists at {self.log_path}, appending")
            
            if is_master:
                # Test write access
                with open(self.log_path, "a" if append else "w", encoding="utf-8") as _:
                    pass
        else:
            self.log_path = None

    def log(self, level: int, string: str):
        """Log the data string to file.
        
        Args:
            level: Log level
            string: String to log
        """
        if level >= self.verbosity and self.log_path and self.is_master:
            with open(self.log_path, "a", encoding="utf-8") as file:
                file.write(string + "\n")

    def write(self,
              data: Optional[Dict[str, Any]] = None,
              status_level: int = Status.RUNNING,
              verbosity_level: int = Verbosity.INFO,
              message: Optional[str] = None,
              step: Optional[int] = None,
              epoch: Optional[int] = None,
              replica_name: Optional[str] = None):
        """Write status data to log file.
        
        Compatible with TAO Deploy/PyTorch - adds Cosmos-RL fields to data dict.
        """
        # Enhance data with Cosmos-RL specific fields
        if not data:
            data = {}
            
        # Add Cosmos-RL specific fields to data dict (compatible approach)
        if step is not None:
            data['step'] = step
        if epoch is not None:
            data['epoch'] = epoch
        if replica_name is not None:
            data['replica_name'] = replica_name
            
        # Call parent write method with standard TAO Deploy/PyTorch signature
        super().write(data, status_level, verbosity_level, message)


class CosmosStatusLogger(BaseStatusLogger):
    """Extended status logger with Cosmos-RL specific features."""

    def __init__(self, 
                 filename: Optional[str] = None,
                 is_master: bool = False,
                 verbosity: int = Verbosity.INFO,
                 append: bool = True,
                 enable_wandb: bool = False):
        """Initialize Cosmos status logger.
        
        Args:
            filename: Path to log file
            is_master: Whether this is the master process
            verbosity: Logging verbosity level
            append: Whether to append to existing file
            enable_wandb: Whether to enable wandb logging
        """
        super().__init__(is_master=is_master, verbosity=verbosity)
        
        self.enable_wandb = enable_wandb
        if filename:
            self.file_logger = StatusLogger(filename, is_master, verbosity, append)
        else:
            self.file_logger = None

    def write(self,
              data: Optional[Dict[str, Any]] = None,
              status_level: int = Status.RUNNING,
              verbosity_level: int = Verbosity.INFO,
              message: Optional[str] = None,
              step: Optional[int] = None,
              epoch: Optional[int] = None,
              replica_name: Optional[str] = None):
        """Write status data with Cosmos-RL extensions.
        
        Compatible with TAO Deploy/PyTorch interface.
        """
        # Log to file if available
        if self.file_logger:
            # Sync KPI and other fields to the file logger
            self.file_logger.kpi = self.kpi
            self.file_logger.categorical = self.categorical
            self.file_logger.graphical = self.graphical
            
            # Use compatible interface for file logger
            self.file_logger.write(data, status_level, verbosity_level, message, step, epoch, replica_name)
        
        # Log to console using compatible interface
        super().write(data, status_level, verbosity_level, message, step, epoch, replica_name)
        
        # Log to wandb if enabled and step is provided
        if self.enable_wandb and step is not None:
            self._log_to_wandb(data, step, epoch, replica_name)

    def _log_to_wandb(self, data: Dict[str, Any], step: int, epoch: Optional[int], replica_name: Optional[str]):
        """Log to wandb if available.
        
        Args:
            data: Data to log
            step: Training step
            epoch: Training epoch
            replica_name: Replica name
        """
        try:
            from cosmos_rl.utils.wandb_logger import is_wandb_available, log_wandb
            
            if is_wandb_available():
                wandb_data = {}
                
                # Add KPIs to wandb
                if self.kpi:
                    for key, value in self.kpi.items():
                        wandb_data[f"status_{key}"] = value
                
                # Metadata-like info can be added to kpi instead
                
                if wandb_data:
                    log_wandb(wandb_data, step=step)
                    
        except ImportError:
            pass  # wandb not available


# Define the global logger
_STATUS_LOGGER = BaseStatusLogger()


def set_status_logger(status_logger: BaseStatusLogger):
    """Set the global status logger.

    Args:
        status_logger: An instance of the status logger class.
    """
    global _STATUS_LOGGER
    _STATUS_LOGGER = status_logger


def get_status_logger() -> BaseStatusLogger:
    """Get the global status logger.
    
    Returns:
        Current status logger instance
    """
    global _STATUS_LOGGER
    return _STATUS_LOGGER
