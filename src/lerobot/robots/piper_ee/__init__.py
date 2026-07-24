# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AgileX Piper 6-DOF with end-effector control (xyz+rpy → IK → joints).

Reuses :class:`PiperFullSDK` from ``piper_full`` for hardware I/O, but exposes
a Cartesian action space and runs a Damped-Least-Squares inverse-kinematics
solver against a Pinocchio-built reduced model of the Piper URDF.

Importing this module does **not** import Pinocchio — the kinematics module
is loaded lazily inside :meth:`PiperEE.__init__`.
"""

from .config_piper_ee import PiperEEConfig
from .piper_ee import PiperEE

__all__ = ["PiperEE", "PiperEEConfig"]
