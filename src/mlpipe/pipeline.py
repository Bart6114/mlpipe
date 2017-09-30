from functools import partial, reduce
import dill
import warnings
import os

SKLEARN_VERBS = (
    'fit',
    'fit_transform',
    'predict',
    'predict_log_proba',
    'predict_proba',
    'score',
)


try:
    if os.environ.get('MLPIPE_WITHOUT_SKLEARN') is not None:
        raise ImportError
    else:
        import sklearn.base
        SKLEARN_AVAILABLE = True

except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("optional 'sklearn' dependency not available", ImportWarning)


class Pipe(object):
    """Pipe class.

    An instance of this class can hold a number of Segments and in whole forms the pipeline.
    """

    def __init__(self):
        self.segments = []
    def __add__(self, segment):
        """Internal function, will be called when adding a Segment to the Pipe by using the '+' operator.

        Args:
            segment: An instance of Segment, this can contain any type of callable. A callable non-Segment object
                will be automatically encapsulated by a Segment.

        Returns: the Pipe instance

        """

        if not isinstance(segment, Segment):
            segment = Segment(segment)

        self.segments.append(segment)
        segment._set_pipe(self)
        return self

    def __repr__(self):
        repr_str = "a pipe, consisting of:\n"

        for i, segment in enumerate(self.segments):
            repr_str += "segment {}: {}\n".format(i, segment)

        return repr_str

    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            return super(Pipe, self).__getattr__(name)

        else:
            return partial(self.__eval, name)

    def __call__(self, *args, **kwargs):
        attr = kwargs.get('attr', '__call__')
        return self.__eval(attr, *args, **kwargs)

    def __assert_list(self, args):
        if type(args) not in (list, tuple,):
            # force iterability of args until last evaluation
            args = (args,)

        return args

    def __eval(self, attr, *args, **kwargs):

        if attr in SKLEARN_VERBS and SKLEARN_AVAILABLE:
            # dispatch to sklearn compatible pipe evaluator
            return self.__eval_sk_style(attr, *args)

        elif attr in SKLEARN_VERBS and not SKLEARN_AVAILABLE:
            warnings.warn('sklearn warning: \'{}\' seems to be an sklearn verb, but sklearn is not available,'
                          'continuing processing without sklearn logic'.format(attr), UserWarning)


        for i, segment in enumerate(self.segments):
            args = getattr(segment, attr)(*args)

            if i < len(self.segments) - 1:
                args = self.__assert_list(args)


        return args

    def __eval_sk_style(self, attr, *args):
        # todo
        # check if obj is one inheriting an sklearn class
        # if not process as regular func

        if attr == 'fit':

            X, y, var_args = args[0], args[1], args[2:]

            for i, segment in enumerate(self.segments):

                if self.__is_sklearn_obj(segment.obj):

                    getattr(segment, 'fit')(X, y, *var_args)

                    if i < len(self.segments) - 1:
                        X = getattr(segment, 'transform')(X, *var_args)

                else:
                    eval_val = getattr(segment, attr)(X, y, *var_args)
                    X, y, var_args = eval_val[0], eval_val[1], eval_val[2:]

            return self


        elif attr in ('predict', 'predict_log_proba', 'score', 'predict_proba', ):

            X, var_args = args[0], args[1:]

            for i, segment in enumerate(self.segments):

                if self.__is_sklearn_obj(segment.obj):

                    if i < len(self.segments) - 1:
                        X = getattr(segment, 'transform')(X, *var_args)

                    else:
                        output = getattr(segment, attr)(X, *var_args)

                else:
                    eval_val = getattr(segment, attr)(X, *var_args)
                    X, var_args = eval_val[0], eval_val[1:]


            return output

    def __is_sklearn_obj(self, obj):
        return issubclass(obj.__class__, (sklearn.base.BaseEstimator,
                                          sklearn.base.TransformerMixin,
                                          sklearn.base.ClassifierMixin,))

    @classmethod
    def _load(cls, filename):
        with open(filename, 'rb') as f:
            return dill.load(f)

    def _save(self, filename):
        with open(filename, 'wb') as f:
            dill.dump(self, f)

    def _dump(self, *args, **kwargs):
        self._save(*args, **kwargs)

    def _dumps(self):
        return dill.dumps(self)




class Segment(object):
    """The Segment class can hold any callable object.

            Args:
                obj: a callable object
                description: an optional description of the segment/object
                args: additional \*args which should always be passed to the object when called
                kwargs: additional \**kwargs which should always be passed to the object when called

    """
    def __init__(self, obj, description='anonymous segment', **kwargs):
        self.obj = obj
        self.description = description
        self.pipe = None

        self.kwargs = kwargs

    def _set_pipe(self, pipe):
        self.pipe = pipe

    def __repr__(self):
        return self.description

    def __call__(self, *args, **kwargs):
        try:
            return partial(self.obj, **self.kwargs)(*args, **kwargs)
        except TypeError:
            raise TypeError("Segment '{}' object '{}' not callable".format(self.description, self.obj))


    def __getattr__(self, name):

        if name.startswith('__') and name.endswith('__'):
            return super(Segment, self).__getattr__(name)

        if hasattr(self.obj, name):
            return partial(getattr(self.obj, name),**self.kwargs)
        else:
            # return main object if attrs isn't found (and assume it is callable)
            return partial(getattr(self.obj, '__call__'),  **self.kwargs)

if __name__ == '__main__':
    pass
