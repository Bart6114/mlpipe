from functools import partial, reduce
import dill

SKLEARN_VERBS = (
    'fit',
    'fit_transform',
    'predict',
    'predict_log_proba',
    'predict_proba',
    'score',
)

import sklearn.base as sb

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

            if hasattr(segment, '__call__'):
                segment = Segment(segment)

            else:
                raise "Cannot add non-callable objects"


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

    def __call__(self, *args,  attr='__call__', **kwargs):
        return self.__eval(attr, *args, **kwargs)

    def __assert_list(self, args):
        if type(args) not in (list, tuple,):
            # force iterability of args until last evaluation
            args = (args,)

        return args

    def __eval(self, attr, *args, **kwargs):

        if attr in SKLEARN_VERBS:
            # dispatch to sklearn compatible pipe evaluator
            return self.__eval_sk_style(attr, *args)

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

            X, y, *_ = args

            for i, segment in enumerate(self.segments):

                if self.__is_sklearn_obj(segment.obj):

                    getattr(segment, 'fit')(X, y, *_)

                    if i < len(self.segments) - 1:
                        X = getattr(segment, 'transform')(X, *_)

                else:
                    X, y, *_ = getattr(segment, attr)(X, y, *_)

            return self


        elif attr in ('predict', 'predict_log_proba', 'score', 'predict_proba', ):

            X, *_ = args

            for i, segment in enumerate(self.segments):

                if self.__is_sklearn_obj(segment.obj):

                    if i < len(self.segments) - 1:
                        X = getattr(segment, 'transform')(X, *_)

                    else:
                        output = getattr(segment, attr)(X, *_)

                else:
                    X, *_ = getattr(segment, attr)(X, *_)


            return output

    def __is_sklearn_obj(self, obj):
        return issubclass(obj.__class__, (sb.BaseEstimator, sb.TransformerMixin, ))

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
    def __init__(self, obj, description='anonymous segment', *args, **kwargs):
        self.obj = obj
        self.description = description
        self.pipe = None

        self.args = args
        self.kwargs = kwargs

    def _set_pipe(self, pipe):
        self.pipe = pipe

    def __repr__(self):
        return self.description

    def __call__(self, *args, **kwargs):
        return self.obj(*args, **kwargs)

    def __getattr__(self, name):

        if name.startswith('__') and name.endswith('__'):
            return super(Segment, self).__getattr__(name)

        if hasattr(self.obj, name):
            return getattr(self.obj, name)
        else:
            # return main object if attrs isn't found (and assume it is callable)
            return getattr(self.obj, '__call__')

if __name__ == '__main__':
    pass