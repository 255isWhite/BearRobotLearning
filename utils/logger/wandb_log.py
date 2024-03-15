import wandb

class BaseLogger():
       def __init__(self, config):
            self.config = config
       
       def log_metrics(self, metrics, step=None):
              raise NotImplementedError

       def finish(self):
              raise NotImplementedError
       

class WandbLogger(BaseLogger):
    def __init__(self, project_name, config):
              super().__init__(config)
              """
              project_name (str): wandb project name
              config (dict): dict
              """
              wandb.init(project=project_name, config=config)
              self.config = wandb.config

    def log_metrics(self, metrics, step=None):
        """
            metrics (dict):
            step (int, optional): epoch or step
        """
        wandb.log(metrics, step=step)

    def finish(self):
        wandb.finish()
