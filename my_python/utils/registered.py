class RegisteredFunctor:
    @classmethod
    def __init_subclass__(cls, name=None, **kwargs):
        super().__init_subclass__()
        
        if name is None: # superclass setting registry object
            cls.registry = dict()
        else: # subclass registering itself
            setattr(cls, 'name', name)
            cls.registry[name] = cls.__call__
