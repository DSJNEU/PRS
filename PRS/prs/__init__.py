from __future__ import annotations

__version__ = "1.0.0"
__author__ = "PRS"
__license__ = "MIT"

from prs.checkpoint import CheckpointManager, CheckpointMetadata, HierarchicalCheckpointManager, IncrementalCheckpoint
from prs.distributed import (
    DistributedCoordinator,
    Message,
    MessageType,
    NodeConfig,
    TCPConnection,
    find_free_port,
    get_local_ip,
    parse_node_config_from_env,
)
from prs.elastic import (
    DistributedEnvironmentManager,
    ElasticController,
    FailureDetector,
    HeartbeatMonitor,
    NodeInfo,
    NodeStatus,
    ReconfigurationEvent,
)
from prs.engine import PRSEngine, TrainingState, create_demo_engine, create_demo_profiles
from prs.experiments import (
    FaultInjectionConfig,
    MetricLogger,
    TorchExperimentRunner,
    TorchTrainingConfig,
    build_experiment_report,
    run_with_restarts,
)
from prs.launcher import ClusterConfig, ClusterLauncher, NodeLauncher, NodeSpec
from prs.moe import (
    ExpertGroup,
    ExpertLoad,
    ExpertReplica,
    FaultAwareReplicaAllocator,
    LoadHeatmapMonitor,
    MROReplicaPlacer,
    MoEExpertParallelManager,
    TokenBatch,
    ZeroPaddingCommunicator,
)
from prs.pipeline import Microbatch, MultiNodePipelineTrainer, PipelineExecutor, PipelineStage, StageConfig
from prs.planning import LayerProfile, PipelinePlanner, PipelineTemplate, create_layer_profiles_from_model
from prs.plugin import (
    PRSPlugin,
    PipelineConfig,
    Precision,
    StageManager,
    get_debug_config,
    get_default_config,
    get_large_model_config,
)
from prs.scheduler import (
    ARIMANodePredictor,
    DPConfigSearcher,
    LiveputCalculator,
    NodeHealth,
    ParallelConfig,
    PredictionResult,
    PredictiveResourceScheduler,
    ProactiveResourceAdjuster,
    SchedulingDecision,
)

__all__ = [
    "__version__",
    "PRSPlugin",
    "PRSEngine",
    "TrainingState",
    "create_demo_engine",
    "create_demo_profiles",
    "MetricLogger",
    "FaultInjectionConfig",
    "TorchTrainingConfig",
    "TorchExperimentRunner",
    "build_experiment_report",
    "run_with_restarts",
    "LayerProfile",
    "PipelineTemplate",
    "PipelinePlanner",
    "create_layer_profiles_from_model",
    "CheckpointManager",
    "CheckpointMetadata",
    "IncrementalCheckpoint",
    "HierarchicalCheckpointManager",
    "ElasticController",
    "NodeInfo",
    "NodeStatus",
    "HeartbeatMonitor",
    "FailureDetector",
    "ReconfigurationEvent",
    "DistributedEnvironmentManager",
    "ExpertLoad",
    "ExpertReplica",
    "ExpertGroup",
    "TokenBatch",
    "LoadHeatmapMonitor",
    "FaultAwareReplicaAllocator",
    "MROReplicaPlacer",
    "ZeroPaddingCommunicator",
    "MoEExpertParallelManager",
    "NodeHealth",
    "ParallelConfig",
    "PredictionResult",
    "SchedulingDecision",
    "ARIMANodePredictor",
    "LiveputCalculator",
    "DPConfigSearcher",
    "ProactiveResourceAdjuster",
    "PredictiveResourceScheduler",
    "PipelineConfig",
    "Precision",
    "StageManager",
    "get_default_config",
    "get_large_model_config",
    "get_debug_config",
    "MessageType",
    "Message",
    "TCPConnection",
    "DistributedCoordinator",
    "NodeConfig",
    "parse_node_config_from_env",
    "get_local_ip",
    "find_free_port",
    "StageConfig",
    "Microbatch",
    "PipelineStage",
    "PipelineExecutor",
    "MultiNodePipelineTrainer",
    "NodeSpec",
    "ClusterConfig",
    "NodeLauncher",
    "ClusterLauncher",
]


def get_version() -> str:
    return __version__


def info() -> None:
    print(f"PRS {__version__}: 面向分布式训练的弹性容错与资源调度系统")
