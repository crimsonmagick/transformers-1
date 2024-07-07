from .model_loader import ModelLoader, NativeModelLoader, QuantizedModelLoader


def get_model_loader(model_id, *, model_directory='.models', quantize: bool = False, ) -> ModelLoader:
    if quantize:
        loader = QuantizedModelLoader(model_id, model_directory)
    else:
        loader = NativeModelLoader(model_id)
    return loader
