import functools as _ftl

class memoised:
    def __init__(self, func):
        self.func = func
        self.name = '__cache_' + func.__name__
        
        self.__doc__ = func.__doc__
        self.__name__ = func.__name__
        self.__module__ = func.__module__
    
    def __call__(self, *args):
        if not hasattr(self, 'cache'):
            # Create cache for free function   
            self.cache = {}
        if not self.name in self.cache:
            # This function has not been cached before
            self.cache[self.name] = {}
        try:
            # Cache hit if function has been called with these args before
            return self.cache[self.name][args]
        except KeyError:
            # Cache miss
            value = self.func(*args)
            self.cache[self.name][args] = value
            return value
        except TypeError:
            # uncachable -- for instance, passing a list as an argument.
            # Better to not cache than to blow up entirely.
            return self.func(*args)

    def __get__(self, obj, objtype):
        """Support instance methods."""
        if obj is None:
            # Class method
            return self
                
        fn = _ftl.partial(self.__call__, obj)
        fn.__doc__ = self.__doc__
        fn.__name__ = self.__name__
        fn.__module__ = self.__module__

        # Use the object's __dict__ as the cache 
        if not hasattr(obj, '__dict__'):
            obj.__dict__ = {}
        self.cache = obj.__dict__
        return fn

class memoised_property:
    def __init__(self, func):
        self.__doc__ = func.__doc__
        self.func = func

    def __get__(self, obj, objtype):
        if obj is None:
            # Class method
            return self
        
        attr_name = '__cache_' + self.func.__name__
        
        if attr_name not in obj.__dict__:
            obj.__dict__[attr_name] = self.func(obj)
        
        return obj.__dict__[attr_name]
