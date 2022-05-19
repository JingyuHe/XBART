# This file was automatically generated by SWIG (http://www.swig.org).
# Version 4.0.2
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

import _xbart_cpp_

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "thisown":
            self.this.own(value)
        elif name == "this":
            set(self, name, value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


class XBARTcppParams(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    num_trees = property(_xbart_cpp_.XBARTcppParams_num_trees_get, _xbart_cpp_.XBARTcppParams_num_trees_set)
    num_sweeps = property(_xbart_cpp_.XBARTcppParams_num_sweeps_get, _xbart_cpp_.XBARTcppParams_num_sweeps_set)
    max_depth = property(_xbart_cpp_.XBARTcppParams_max_depth_get, _xbart_cpp_.XBARTcppParams_max_depth_set)
    Nmin = property(_xbart_cpp_.XBARTcppParams_Nmin_get, _xbart_cpp_.XBARTcppParams_Nmin_set)
    Ncutpoints = property(_xbart_cpp_.XBARTcppParams_Ncutpoints_get, _xbart_cpp_.XBARTcppParams_Ncutpoints_set)
    burnin = property(_xbart_cpp_.XBARTcppParams_burnin_get, _xbart_cpp_.XBARTcppParams_burnin_set)
    mtry = property(_xbart_cpp_.XBARTcppParams_mtry_get, _xbart_cpp_.XBARTcppParams_mtry_set)
    alpha = property(_xbart_cpp_.XBARTcppParams_alpha_get, _xbart_cpp_.XBARTcppParams_alpha_set)
    beta = property(_xbart_cpp_.XBARTcppParams_beta_get, _xbart_cpp_.XBARTcppParams_beta_set)
    tau = property(_xbart_cpp_.XBARTcppParams_tau_get, _xbart_cpp_.XBARTcppParams_tau_set)
    kap = property(_xbart_cpp_.XBARTcppParams_kap_get, _xbart_cpp_.XBARTcppParams_kap_set)
    s = property(_xbart_cpp_.XBARTcppParams_s_get, _xbart_cpp_.XBARTcppParams_s_set)
    tau_kap = property(_xbart_cpp_.XBARTcppParams_tau_kap_get, _xbart_cpp_.XBARTcppParams_tau_kap_set)
    tau_s = property(_xbart_cpp_.XBARTcppParams_tau_s_get, _xbart_cpp_.XBARTcppParams_tau_s_set)
    verbose = property(_xbart_cpp_.XBARTcppParams_verbose_get, _xbart_cpp_.XBARTcppParams_verbose_set)
    sampling_tau = property(_xbart_cpp_.XBARTcppParams_sampling_tau_get, _xbart_cpp_.XBARTcppParams_sampling_tau_set)
    parallel = property(_xbart_cpp_.XBARTcppParams_parallel_get, _xbart_cpp_.XBARTcppParams_parallel_set)
    nthread = property(_xbart_cpp_.XBARTcppParams_nthread_get, _xbart_cpp_.XBARTcppParams_nthread_set)
    seed = property(_xbart_cpp_.XBARTcppParams_seed_get, _xbart_cpp_.XBARTcppParams_seed_set)
    sample_weights = property(_xbart_cpp_.XBARTcppParams_sample_weights_get, _xbart_cpp_.XBARTcppParams_sample_weights_set)

    def __init__(self):
        _xbart_cpp_.XBARTcppParams_swiginit(self, _xbart_cpp_.new_XBARTcppParams())
    __swig_destroy__ = _xbart_cpp_.delete_XBARTcppParams

# Register XBARTcppParams in _xbart_cpp_:
_xbart_cpp_.XBARTcppParams_swigregister(XBARTcppParams)

class XBARTcpp(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    params = property(_xbart_cpp_.XBARTcpp_params_get, _xbart_cpp_.XBARTcpp_params_set)
    trees = property(_xbart_cpp_.XBARTcpp_trees_get, _xbart_cpp_.XBARTcpp_trees_set)
    y_mean = property(_xbart_cpp_.XBARTcpp_y_mean_get, _xbart_cpp_.XBARTcpp_y_mean_set)
    n_train = property(_xbart_cpp_.XBARTcpp_n_train_get, _xbart_cpp_.XBARTcpp_n_train_set)
    n_test = property(_xbart_cpp_.XBARTcpp_n_test_get, _xbart_cpp_.XBARTcpp_n_test_set)
    p = property(_xbart_cpp_.XBARTcpp_p_get, _xbart_cpp_.XBARTcpp_p_set)
    yhats_xinfo = property(_xbart_cpp_.XBARTcpp_yhats_xinfo_get, _xbart_cpp_.XBARTcpp_yhats_xinfo_set)
    yhats_test_xinfo = property(_xbart_cpp_.XBARTcpp_yhats_test_xinfo_get, _xbart_cpp_.XBARTcpp_yhats_test_xinfo_set)
    sigma_draw_xinfo = property(_xbart_cpp_.XBARTcpp_sigma_draw_xinfo_get, _xbart_cpp_.XBARTcpp_sigma_draw_xinfo_set)
    mtry_weight_current_tree = property(_xbart_cpp_.XBARTcpp_mtry_weight_current_tree_get, _xbart_cpp_.XBARTcpp_mtry_weight_current_tree_set)
    model = property(_xbart_cpp_.XBARTcpp_model_get, _xbart_cpp_.XBARTcpp_model_set)
    yhats_test_multinomial = property(_xbart_cpp_.XBARTcpp_yhats_test_multinomial_get, _xbart_cpp_.XBARTcpp_yhats_test_multinomial_set)
    num_classes = property(_xbart_cpp_.XBARTcpp_num_classes_get, _xbart_cpp_.XBARTcpp_num_classes_set)
    sigma_draws = property(_xbart_cpp_.XBARTcpp_sigma_draws_get, _xbart_cpp_.XBARTcpp_sigma_draws_set)
    resid = property(_xbart_cpp_.XBARTcpp_resid_get, _xbart_cpp_.XBARTcpp_resid_set)

    def __init__(self, *args):
        _xbart_cpp_.XBARTcpp_swiginit(self, _xbart_cpp_.new_XBARTcpp(*args))

    def _fit(self, n: "int", n_y: "int", p_cat: "size_t") -> "void":
        return _xbart_cpp_.XBARTcpp__fit(self, n, n_y, p_cat)

    def _predict(self, n: "int") -> "void":
        return _xbart_cpp_.XBARTcpp__predict(self, n)

    def _predict_gp(self, n: "int", n_y: "int", n_t: "int", p_cat: "size_t", theta: "double", tau: "double") -> "void":
        return _xbart_cpp_.XBARTcpp__predict_gp(self, n, n_y, n_t, p_cat, theta, tau)

    def np_to_vec_d(self, n: "int", y_std: "vec_d &") -> "void":
        return _xbart_cpp_.XBARTcpp_np_to_vec_d(self, n, y_std)

    def np_to_col_major_vec(self, n: "int", x_std: "vec_d &") -> "void":
        return _xbart_cpp_.XBARTcpp_np_to_col_major_vec(self, n, x_std)

    def xinfo_to_np(self, x_std: "matrix< double >", arr: "double *") -> "void":
        return _xbart_cpp_.XBARTcpp_xinfo_to_np(self, x_std, arr)

    def vec_d_to_np(self, y_std: "vec_d &", arr: "double *") -> "void":
        return _xbart_cpp_.XBARTcpp_vec_d_to_np(self, y_std, arr)

    def compute_Xorder(self, n: "size_t", d: "size_t", x_std_flat: "vec_d const &", Xorder_std: "matrix< size_t > &") -> "void":
        return _xbart_cpp_.XBARTcpp_compute_Xorder(self, n, d, x_std_flat, Xorder_std)
    seed = property(_xbart_cpp_.XBARTcpp_seed_get, _xbart_cpp_.XBARTcpp_seed_set)
    seed_flag = property(_xbart_cpp_.XBARTcpp_seed_flag_get, _xbart_cpp_.XBARTcpp_seed_flag_set)
    no_split_penality = property(_xbart_cpp_.XBARTcpp_no_split_penality_get, _xbart_cpp_.XBARTcpp_no_split_penality_set)

    def _to_json(self) -> "std::string":
        return _xbart_cpp_.XBARTcpp__to_json(self)

    def get_num_trees(self) -> "int":
        return _xbart_cpp_.XBARTcpp_get_num_trees(self)

    def get_num_sweeps(self) -> "int":
        return _xbart_cpp_.XBARTcpp_get_num_sweeps(self)

    def get_burnin(self) -> "int":
        return _xbart_cpp_.XBARTcpp_get_burnin(self)

    def get_yhats(self, size: "int") -> "void":
        return _xbart_cpp_.XBARTcpp_get_yhats(self, size)

    def get_yhats_test(self, size: "int") -> "void":
        return _xbart_cpp_.XBARTcpp_get_yhats_test(self, size)

    def get_yhats_test_multinomial(self, size: "int") -> "void":
        return _xbart_cpp_.XBARTcpp_get_yhats_test_multinomial(self, size)

    def get_sigma_draw(self, size: "int") -> "void":
        return _xbart_cpp_.XBARTcpp_get_sigma_draw(self, size)

    def get_residuals(self, size: "int") -> "void":
        return _xbart_cpp_.XBARTcpp_get_residuals(self, size)

    def _get_importance(self, size: "int") -> "void":
        return _xbart_cpp_.XBARTcpp__get_importance(self, size)
    __swig_destroy__ = _xbart_cpp_.delete_XBARTcpp

# Register XBARTcpp in _xbart_cpp_:
_xbart_cpp_.XBARTcpp_swigregister(XBARTcpp)



