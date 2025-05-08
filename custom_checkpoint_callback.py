from stable_baselines3.common.callbacks import BaseCallback
import os

class CustomCheckpointCallback(BaseCallback):
    """
    Save a model every `save_freq` steps, starting from `save_start` steps.
    """
    def __init__(self, save_freq, save_path, name_prefix="rl_model", save_start=0, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_start = save_start
        self.last_save_step = ((save_start // save_freq) - 1) * save_freq  # e.g. 50000 for 60000 start

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        current_step = self.num_timesteps + self.save_start
        next_save = self.last_save_step + self.save_freq
        if current_step >= next_save:
            self.last_save_step = current_step
            path = os.path.join(self.save_path, f"{self.name_prefix}_{current_step}.zip")
            self.model.save(path)
            if self.verbose > 0:
                print(f"Saving model checkpoint to {path}")
        return True
