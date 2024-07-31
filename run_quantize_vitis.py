import os
from pathlib import Path
import vai_q_onnx

current_dir = Path(__file__).parent
print(current_dir)

input_model_path = Path(current_dir / "models/AnimeGANv3_Hayao_36.onnx")
print(input_model_path)

output_model_path = Path(current_dir / "models/quantized-models/AnimeGANv3_Hayao_36_fp8.onnx")
print(output_model_path)

vai_q_onnx.quantize_static(
    input_model_path,
    output_model_path,
    calibration_data_reader=None,
    quant_format=vai_q_onnx.QuantFormat.QDQ,
    calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
    activation_type=vai_q_onnx.QuantType.QUInt8,
    weight_type=vai_q_onnx.QuantType.QInt8,
    enable_ipu_cnn=True,
    extra_options={'ActivationSymmetric':True}
)

print("Quantized and Callibrated model saved at:", output_model_path)