# Reading a loss curve

| What the curves show | Likely cause | What to try |
| --- | --- | --- |
| Training loss flat from the first epoch | Learning rate far too low; the optimizer is built on a different model output than the loop trains; a frozen or disconnected model | Check `Optimizer.model` and `TrainingLoop.model` share one source; raise `lr` by 10x |
| Training loss jumps up or becomes `nan` | Learning rate too high; exploding gradients | Lower `lr`; set `grad_clip_norm`; turn on `log_grad_norm` to see it |
| Training loss falls, validation loss rises | Overfitting | More data or augmentation, `Dropout`, `weight_decay`, early stopping on `val_loss` |
| Both losses fall slowly and are still falling | Undertrained | More epochs, or a higher rate with a schedule |
| Validation loss noisy epoch to epoch | Small validation set | A larger fixed split; compare over several seeds |
| Loss rises again late in a cosine run | `T_max` smaller than `epochs` | Set `LRScheduler.T_max` to `epochs` |
| Accuracy near chance on a classification task | Labels and outputs misaligned, wrong final `Linear.out_features`, or the wrong loss | Check the class count and use `CrossEntropyLoss` on logits |

Two cautions:

- A single run is one sample. Before saying one setting beats another, compare against
  the spread across seeds (see the `run-experiments` skill).
- Report the series you read by name (`val_loss`, `eval_accuracy`) and the epoch, and
  quote the number the run recorded, not a value read off a chart.
