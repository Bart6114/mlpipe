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
    def __init__(self):
        self.segments = []

    def __add__(self, segment):

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

def dummy(*args):
    return args

if __name__ == '__main__':
    from src.mlpipe import Pipe, Segment



    p = Pipe() + \
        Segment(lambda x: x + 1, "step1") + \
        Segment(lambda x: x + 1, "step2")

    print(p)

    # print(888, p._dumps())

    # p = Pipe() +\
    #     Segment(lambda x: x + 2, "step1") + \
    #     Segment(lambda x: x + 3) + \
    #     Segment(lambda x: x + 1, "another step")
    #
    # print(p)
    # print(p.bla(4))
    # print(p(4))
    #
    #
    #
    # from sklearn import svm
    # from sklearn.datasets import samples_generator
    # from sklearn.feature_selection import SelectKBest
    # from sklearn.feature_selection import f_regression
    #
    # # generate some data to play with
    # X, y = samples_generator.make_classification(
    #     n_informative=5, n_redundant=0, random_state=42)
    # # ANOVA SVM-C
    # anova_filter = SelectKBest(f_regression, k=5)
    # # print(dir(anova_filter))
    # clf = svm.SVC(kernel='linear')
    # # anova_svm = Pipeline([('anova', anova_filter), ('svc', clf)])
    # # You can set the parameters using the names issued
    # # For instance, fit using a k of 10 in the SelectKBest
    # # and a parameter 'C' of the svm
    # # anova_svm.set_params(anova__k=10, svc__C=.1).fit(X, y)
    #
    # # prediction = anova_svm.predict(X)
    # # anova_svm.score(X, y)
    # #
    # # # getting the selected features chosen by anova_filter
    # # anova_svm.named_steps['anova'].get_support()
    # #
    # # # Another way to get selected features chosen by anova_filter
    # # anova_svm.named_steps.anova.get_support()
    #
    # p2 = Pipe() +\
    #     Segment(anova_filter, 'anova') +\
    #     Segment(dummy) +\
    #     Segment(clf, 'svc')
    #
    # print(p2)
    # print(p2.segments[0].obj.__class__)
    # print(issubclass(p2.segments[0].obj.__class__, sb.BaseEstimator), 999)
    # print(1111, p._is_sklearn_obj(anova_filter), p._is_sklearn_obj(lambda x: 3))
    # print(p2.fit(X, y))
    # print("fit finished")
    #
    # print(p2.predict(X))
    #

