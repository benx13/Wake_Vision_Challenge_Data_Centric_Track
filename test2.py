from mcunet.model_zoo import net_id_list, build_model
print(net_id_list)  # the list of models in the model zoo

# pytorch fp32 model
model, image_size, description = build_model(net_id="mcunet-tiny2", pretrained=False)  # you can replace net_id with any other option from net_id_list
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

# download tflite file to tflite_path