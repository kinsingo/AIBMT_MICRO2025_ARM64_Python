import os
import numpy as np
import cv2
from GUI_Mananger import ExecuteGUI, bmt, current_dir
from dx_engine import InferenceEngine

# Define the interface class for Classification using ONNX
class SubmitterImplementation(bmt.AI_BMT_Interface):
    def __init__(self):
        super().__init__()
        self.ie = None
        
    def getOptionalData(self):
        optional = bmt.Optional_Data()
        optional.cpu_type = ""
        optional.accelerator_type = ""  # e.g., "DeepX M1(NPU)"
        optional.submitter = ""         # e.g., "DeepX"
        optional.cpu_core_count = ""
        optional.cpu_ram_capacity = ""  # e.g., "32GB"
        optional.cooling = ""           # e.g., "Air"
        optional.cooling_option = ""    # e.g., "Active"
        optional.cpu_accelerator_interconnect_interface = ""  # e.g., "PCIe Gen5 x16"
        optional.benchmark_model = ""
        optional.operating_system = ""
        return optional

    def Initialize(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.ie = InferenceEngine(model_path)
        return True

    def convertToPreprocessedDataForInference(self, image_path: str):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return [np.frombuffer(img, dtype=np.uint8)]

    def runInference(self, preprocessed_data_list):
        results = []
        for _, preprocessed_data in enumerate(preprocessed_data_list):
            outputs = self.ie.run(preprocessed_data)
            result = bmt.BMTResult()
            result.classProbabilities = outputs[0].flatten()[:1000]
            results.append(result)
        return results

if __name__ == "__main__":
    interface = SubmitterImplementation()
    ExecuteGUI(interface)