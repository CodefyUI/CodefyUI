# Training a vision-language-action policy on PushWorld

The shipped example is `examples/VLA/TrainVLA-PushWorld`. Its wiring:

```
Start --trigger--> PushWorldEnv, VLAModel

PushWorldEnv.env --> PushWorldDemos.env, VLARollout.env
PushWorldDemos.dataset --> DataLoader(train) --> TrainingLoop.dataloader
PushWorldDemos.holdout --> DataLoader(val)   --> TrainingLoop.val_dataloader
PushWorldDemos.holdout --> VLAActionEval.dataset
VLAModel.model   --> Optimizer.model, TrainingLoop.model
VLAModel.loss_fn --> TrainingLoop.loss_fn
Optimizer.optimizer --> LRScheduler.optimizer --> TrainingLoop.lr_scheduler
TrainingLoop.model --> VLARollout.model, VLAActionEval.model
VLARollout.frames, PushWorldDemos.demo_video --> VideoWrite.frames
```

Rules:

- `VLAModel` supplies BOTH the model and its `loss_fn`. Do not use a generic `Loss` node:
  the loss depends on `head_type` (`flow_matching` or `regression`).
- Keep `chunk` equal on `PushWorldDemos` and `VLAModel`, and `image_size` equal on
  `PushWorldEnv` and `VLAModel`.
- Keep `PushWorldDemos.demo_noise` above 0: noisy demonstrations teach recovery.
- `VLARollout.success_rate` (closed loop, out of `episodes`) is the headline result.
  `VLAActionEval.action_mse` on the holdout split is an open-loop diagnostic, not a
  substitute.
- `VLARollout.instruction_mode = swapped` gives each episode the wrong instruction; the
  gap between `normal` and `swapped` success rates tests whether the policy actually
  uses language.
- Other knobs worth an ablation: `head_type`, `VLARollout.execute_k` (how many actions
  of each predicted chunk are executed), `vision_stem` (`conv` or `patchify`),
  `flow_steps`, model size (`d_model`, `n_layers`).
- `VideoWrite` turns `frames` or `demo_video` into a playable clip (`mp4` or `gif`) that
  the Execution Log shows.
