from mcunet.model_zoo import net_id_list, build_model, download_tflite
print(net_id_list)  # the list of models in the model zoo

# pytorch fp32 model
model, image_size, description = build_model(net_id="mcunet-vww2", pretrained=True)  # you can replace net_id with any other option from net_id_list

print(image_size, description)

print(model)