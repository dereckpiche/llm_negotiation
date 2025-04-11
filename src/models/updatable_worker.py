class UpdatableWorkerExtension:
    """
    based on https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/rlhf_utils.py, and modified for single GPU
    """

    def update_weight(self, name, weight):
        self.model_runner.model.load_weights(weights=[(name, weight)])