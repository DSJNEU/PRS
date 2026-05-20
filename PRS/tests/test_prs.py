from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from prs import (
    CheckpointManager,
    ClusterConfig,
    ElasticController,
    FaultInjectionConfig,
    LayerProfile,
    MetricLogger,
    NodeInfo,
    NodeSpec,
    PipelinePlanner,
    PRSPlugin,
    ParallelConfig,
    PredictiveResourceScheduler,
    TorchTrainingConfig,
    build_experiment_report,
    create_demo_engine,
    create_demo_profiles,
)


class PlanningTests(unittest.TestCase):
    def test_dynamic_programming_template(self) -> None:
        profiles = create_demo_profiles(8)
        planner = PipelinePlanner(profiles, gpu_memory=8 * 1024 * 1024 * 1024)
        templates = planner.create_templates(1, 4)

        self.assertIn(4, templates)
        self.assertTrue(planner.validate_template(templates[4]))
        self.assertEqual(
            [layer for stage in templates[4].stage_layers for layer in stage],
            list(range(8)),
        )

    def test_layer_profile_intensity(self) -> None:
        profile = LayerProfile(0, "layer", 1.0, 2.0, 3 * 1024 * 1024)
        self.assertAlmostEqual(profile.compute_intensity, 1.0)


class CheckpointTests(unittest.TestCase):
    def test_incremental_restore(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = CheckpointManager(Path(tmp), interval=1, max_to_keep=10, full_interval=3)
            manager.save_state(1, {"step": 1, "loss": 3.0, "model": {"w": 1}})
            manager.save_state(2, {"step": 2, "loss": 2.5, "model": {"w": 1}})
            manager.save_state(3, {"step": 3, "loss": 2.0, "model": {"w": 2}})

            restored = manager.load(3)
            self.assertEqual(restored["step"], 3)
            self.assertEqual(restored["model"]["w"], 2)
            manager.close()


class ElasticTests(unittest.TestCase):
    def test_failure_event(self) -> None:
        controller = ElasticController(min_nodes=1)
        controller.initialize([NodeInfo(0, "127.0.0.1"), NodeInfo(1, "127.0.0.2")])
        event = controller.mark_failed(1)

        self.assertEqual(event.failed_nodes, [1])
        self.assertEqual(controller.get_num_nodes(), 1)
        self.assertTrue(controller.can_continue_training())


class SchedulerTests(unittest.TestCase):
    def test_scheduler_returns_feasible_config(self) -> None:
        scheduler = PredictiveResourceScheduler(initial_nodes=4, initial_config=ParallelConfig(1, 2))
        for nodes in [4, 4, 3, 3]:
            scheduler.update_state(node_count=nodes, throughput=4.0)
        decision = scheduler.step()

        self.assertLessEqual(decision.target_config.required_nodes, scheduler.current_nodes)
        self.assertGreater(decision.expected_liveput, 0)


class EngineTests(unittest.TestCase):
    def test_demo_engine_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = create_demo_engine(Path(tmp), nodes=4)
            outputs = engine.run_steps(5)
            event = engine.simulate_failure()
            decision = engine.scheduler_step()
            snapshot = engine.snapshot()

            self.assertEqual(len(outputs), 5)
            self.assertTrue(event.selected_template_id)
            self.assertIsNotNone(decision)
            self.assertGreaterEqual(len(snapshot["checkpoints"]), 1)
            engine.shutdown()

    def test_plugin_validation(self) -> None:
        with self.assertRaises(ValueError):
            PRSPlugin(global_batch_size=7, microbatch_size=4)


class ExperimentTests(unittest.TestCase):
    def test_metric_logger_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = MetricLogger(Path(tmp), rank=0)
            logger.log(step=1, loss=2.0, throughput=4.0, step_time=0.5)
            logger.log(step=2, loss=1.5, throughput=6.0, step_time=0.4)
            summary = logger.summarize()
            logger.close()

            self.assertEqual(summary["steps"], 2)
            self.assertEqual(summary["final_loss"], 1.5)
            self.assertTrue((Path(tmp) / "summary.json").exists())
            report = build_experiment_report(Path(tmp))
            self.assertEqual(report["record_count"], 2)

    def test_torch_training_config(self) -> None:
        config = TorchTrainingConfig(
            total_steps=4,
            fault=FaultInjectionConfig(enabled=True, step=2, rank=0),
        )

        self.assertTrue(config.fault.should_fire(2, 0))
        self.assertFalse(config.fault.should_fire(3, 0))

    def test_torchrun_command_builder(self) -> None:
        config = ClusterConfig(
            nodes=[NodeSpec(0, "127.0.0.1", num_gpus=2)],
            master_port=29500,
            script_args=["--steps", "2"],
        )
        command = config.torchrun_command(0)

        self.assertIn("torch.distributed.run", command)
        self.assertIn("--nproc_per_node=2", command)
        self.assertIn("--steps", command)


if __name__ == "__main__":
    unittest.main()
