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

"""QP-based chunk smoother for action chunks predicted by a policy.

Public surface:

- ``QPSmoother``: stateful, OSQP-backed solver that enforces velocity caps and
  penalises acceleration/jerk on each joint independently.
"""

from .qp_smoother import QPSmoother

__all__ = ["QPSmoother"]
