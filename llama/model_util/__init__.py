from .model_loader import ModelLoader, AwqModelLoader, NativeModelLoader, NormalizedFloatModelLoader


def get_model_loader(model_id, *, model_directory='.models', quantize: bool = False, ) -> ModelLoader:
    if quantize:
        loader = AwqModelLoader(model_id, model_directory)
    else:
        loader = NativeModelLoader(model_id)
    return loader
