import sys
from pathlib import Path
import yaml
from llamafactory.hparams import get_train_args

path = sys.argv[1]
config = yaml.safe_load(Path(path).read_text())
model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(config)
print("CONFIG OK")
print("layout:", data_args.multitask_task_layout)
print("allocation:", data_args.multitask_microbatch_allocation)
print("gradnorm_tasks:", data_args.multitask_gradnorm_tasks)
print("dataset:", data_args.dataset)
print("output_dir:", training_args.output_dir)
print("lr:", training_args.learning_rate, "max_steps:", training_args.max_steps)
print("gradient_checkpointing:", getattr(model_args, "gradient_checkpointing_layer_ratio", None))
