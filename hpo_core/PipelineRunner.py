from hpo_core.HpoArgs import HpoArgs
from hpo_core.DataManager import DataManager
from hpo_core.WorkDir import WorkDir
from hpo_core.OptunaParams import OptunaParams

class PipelineRunner:
    """
    This class is responsible for running the pipeline.
    """

    def __init__(self, args: HpoArgs, data_manager: DataManager, work_dir: WorkDir):
        self.args = args
        self.data_manager = data_manager
        self.work_dir = work_dir

    def run(self):
        pass