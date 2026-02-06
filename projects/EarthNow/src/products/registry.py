PRODUCTS = {}

def register(name):
    def decorator(func):
        PRODUCTS[name] = func
        return func
    return decorator

