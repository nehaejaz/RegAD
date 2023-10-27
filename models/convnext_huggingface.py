from transformers import AutoImageProcessor, ConvNextModel, ConvNextConfig

def HF_Convnext(pretrained=True,in_22k=False,**kwargs):
    # Initializing a ConvNext convnext-tiny-224 style configuration
    configuration = ConvNextConfig(
        out_features=["stage1", "stage2", "stage3", "stage4"],
        output_hidden_states=True
    )

    model = ConvNextModel.from_pretrained("facebook/convnext-tiny-224", config=configuration)

    # Accessing the model configuration
    configuration = model.config
    # print(model)
    return model



