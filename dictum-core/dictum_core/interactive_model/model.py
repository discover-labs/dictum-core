from dictum_core.model import model


class InteractiveModelMeta(type):
    def __new__(cls, name, bases, attrs):
        attrs["_model"] = model.Model(name=name)
        return super().__new__(cls, name, bases, attrs)


class InteractiveModel(metaclass=InteractiveModelMeta):
    _model: model.Model

    def __init_subclass__(cls):
        # runs after __set_name__ is called for all child descriptors
        pass
