#%%
import copy as cp

class Expr():
    def __init__(self, w_nom = [], w_denom = [], cause_var = {}, kernel_flag = False):
        self.w_nom = cp.deepcopy(w_nom)
        self.w_denom = cp.deepcopy(w_denom)
        self.cause_var = cp.deepcopy(cause_var)
        self.kernel_flag = cp.deepcopy(kernel_flag)
        self.simplify()

    def _repr_latex_(self):
        w_nom = self.w_nom.copy()
        w_denom = self.w_denom.copy()

        nom_terms = self.build_p_term(w_nom)
        denom_terms = self.build_p_term(w_denom)
        kernel_term = self.build_kernel_term()
        if self.kernel_flag:
            kernel_term = [f"\\mathcal{{{kernel_term_i[0]}}}{kernel_term_i[1:]}" for kernel_term_i in kernel_term]
            kernel_latex = "".join(kernel_term)
        else:
            kernel_latex = ""
        nom_latex = "".join(nom_terms)   if nom_terms   else "1"
        denom_latex = "".join(denom_terms) if denom_terms else "1"
        lhs_latex = f"w_{{n}}"
        rhs_latex = f"\\frac{{{nom_latex}{kernel_latex}}}{{{denom_latex}}}"
        return f"${{{lhs_latex}}}={{{rhs_latex}}}$"

    def rv2eval(self, var: str) -> str:
        return (var.lower() + "_{i}") if var in self.cause_var else (var.replace("'", "").lower() + "_{n}")

    def build_p_term(self, terms: list) -> list:
        p_terms = []
        for term_i in terms:
            var_eval = [self.rv2eval(var) for var in term_i]
            joint_var_str = ",".join(sorted(var_eval))
            p_terms.append(f"P({joint_var_str})")
        return p_terms

    def build_kernel_term(self) -> list:
        kernel_term = []
        if self.kernel_flag:
            var_eval = [self.rv2eval(var) for var in self.cause_var]
            intv_eval = [f"{var.lower()}_{{n}}" for var in self.cause_var]
            for v_e, v_i in zip(var_eval, intv_eval):
                kernel_term.append(f"K[{v_e},{v_i}]")
        return kernel_term

    def simplify(self) -> bool:
        w_nom = self.w_nom.copy()
        w_denom = self.w_denom.copy()
        simplified = False
        cancel = True
        while cancel:
            cancel = False
            for term in self.w_nom:
                if term in w_denom:
                    w_denom.remove(term)
                    w_nom.remove(term)
                    cancel = True
                    simplified = True
        if simplified:
            self.w_nom = w_nom.copy()
            self.w_denom = w_denom.copy()
        
        return simplified

    def tostr(self, sep: str = "") -> str:
        w_nom = self.w_nom.copy()
        w_denom = self.w_denom.copy()

        nom_terms = self.build_p_term(w_nom)
        denom_terms = self.build_p_term(w_denom)
        kernel_term = self.build_kernel_term()
        nom_str   = sep.join(nom_terms)   if nom_terms   else "1"
        denom_str = sep.join(denom_terms) if denom_terms else "1"
        kernel_term = sep.join(kernel_term) if self.kernel_flag else ""

        return f"w_{{n}}=[{nom_str}{sep}{kernel_term}]/[N{denom_str}]"

# %%
