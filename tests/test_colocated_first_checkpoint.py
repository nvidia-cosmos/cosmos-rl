"""The initial colocated command must honor checkpoint boundaries on resume."""

from types import SimpleNamespace
from unittest.mock import Mock, patch


def test_initial_colocated_command_honors_checkpoint_schedule():
    from cosmos_rl.dispatcher.status import PolicyStatusManager

    for step, horizon, freq, enabled, expected in [
        (5, 6, 6, True, True),
        (5, 20, 6, True, True),
        (0, 1, 100, True, True),
        (0, 5, 5, True, False),
        (5, 6, 6, False, False),
    ]:
        config = SimpleNamespace(
            mode="colocated",
            validation=SimpleNamespace(enable=False),
            policy=SimpleNamespace(parallelism=SimpleNamespace(n_init_replicas=1)),
            train=SimpleNamespace(
                train_batch_per_replica=16,
                ckpt=SimpleNamespace(
                    save_freq=freq, save_freq_in_epoch=0, enable_checkpoint=enabled
                ),
            ),
        )
        replica = SimpleNamespace(name="policy", start_time=0)
        manager = SimpleNamespace(
            config=config,
            policy_init_done=False,
            trigger_rebuild_mesh=Mock(),
            set_status=Mock(),
            current_step=step,
            total_steps=horizon,
            remain_samples_num=100,
            redis_handler=Mock(),
            training_horizon=lambda: horizon,
        )
        manager.check_checkpoint_saving = lambda count: (
            PolicyStatusManager.check_checkpoint_saving(manager, count)
        )
        with patch(
            "cosmos_rl.dispatcher.status.command.DataFetchCommand.trigger"
        ) as emit:
            PolicyStatusManager.post_register_hook(
                manager, [replica], replica, config, Mock()
            )
        assert emit.call_args.kwargs["do_save"] is expected
        assert emit.call_args.kwargs["global_step"] == step + 1


def test_initial_colocated_replicas_share_epoch_save_decision_and_metadata():
    from cosmos_rl.dispatcher.status import PolicyStatusManager

    config = SimpleNamespace(
        mode="colocated",
        validation=SimpleNamespace(enable=False),
        policy=SimpleNamespace(parallelism=SimpleNamespace(n_init_replicas=2)),
        train=SimpleNamespace(
            train_batch_per_replica=16,
            epoch=2,
            ckpt=SimpleNamespace(
                save_freq=100, save_freq_in_epoch=1, enable_checkpoint=True
            ),
        ),
    )
    replicas = [
        SimpleNamespace(
            name=f"policy-{i}",
            start_time=i,
            weights_loaded_in_view_of_command=i == 0,
        )
        for i in range(2)
    ]
    manager = SimpleNamespace(
        config=config,
        policy_init_done=False,
        trigger_rebuild_mesh=Mock(),
        set_status=Mock(),
        current_step=0,
        total_steps=20,
        remain_samples_num=60,
        samples_per_epoch=32,
        redis_handler=Mock(),
        training_horizon=lambda: 20,
    )
    manager.check_checkpoint_saving = Mock(
        side_effect=lambda count: PolicyStatusManager.check_checkpoint_saving(
            manager, count
        )
    )
    with (
        patch("cosmos_rl.dispatcher.status.command.DataFetchCommand.trigger") as emit,
        patch(
            "cosmos_rl.dispatcher.status.command.PolicyToPolicyBroadcastCommand.trigger"
        ),
    ):
        PolicyStatusManager.post_register_hook(
            manager, replicas, replicas[-1], config, Mock()
        )
    assert emit.call_count == 2
    assert [call.kwargs["do_save"] for call in emit.call_args_list] == [True, True]
    assert [call.kwargs["remain_samples_num"] for call in emit.call_args_list] == [
        28,
        28,
    ]
    manager.check_checkpoint_saving.assert_called_once_with(32)
