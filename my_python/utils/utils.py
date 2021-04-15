import functools as ftl
import itertools as it

__all__ = ['memoised', 'memoised_property', 'pairwise', 'RegisteredFunctor', 'require_attrs']

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
                
        fn = ftl.partial(self.__call__, obj)
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


def pairwise(iterable):
    """
    s -> (s0,s1), (s1,s2), (s2, s3), ...
    
    from https://docs.python.org/3/library/itertools.html
    """
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)


class RegisteredFunctor:
    @classmethod
    def __init_subclass__(cls, name=None, **kwargs):
        super().__init_subclass__()
        
        if name is None: # superclass setting registry object
            cls.registry = dict()
        else: # subclass registering itself
            setattr(cls, 'name', name)
            cls.registry[name] = cls.__call__


def require_attrs(required_attrs):
    """
    Decorator to require that an object has one or more attributes in order to call this method.

    method:
        The method to decorate.
    required_attrs:
        str or sequence of str, the attribute(s) that must exist.
        A string containg multiple attribute names separated by '|' characters is treated as an alternation
        i.e. one or more of the attributes must exist.
    """
    if isinstance(required_attrs, str):
        required_attrs = [required_attrs]

    def decorator_reqattrs(func):
        def _checkattr(attr):
            if '|' in attr: # alternation
                return any(hasattr(self, a) for a in attr.split('|'))
            else:
                return hasattr(self, attr)

        @ftl.wraps(func)
        def _wrapper(self, *args, **kwargs):
            if all(_checkattr(self, attr) for attr in required_attrs):
                return func(self, *args, **kwargs)
            else:
                raise NotImplementedError(
                    f"method {func.__name__} is not implemented for object {repr(self)} of type {type(self).__name__}"
                )
        return _wrapper
    return decorator_reqattrs
