class Conditional(object):

    def __init__(self, dependency):

        self.dependency = dependency
        self.conditions = {}

    def enter(self, condition, result):

        self.conditions[condition] = result

    def get(self, request):

        default = self.conditions.get('default', None)

        condition = request.get(self.dependency, None)
        if condition is None:
            return None

        return self.conditions.get(condition, default)

    __call__ = enter
