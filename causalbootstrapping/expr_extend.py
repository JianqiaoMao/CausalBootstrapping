#%%
import copy as cp
import grapl.expr as expr  # assuming Expr is in original.py or install path
import grapl.util as util

class weightExpr():
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
    
class DOExpr(expr.Expr):
    """Expr subclass tracking do-variables per term."""
    def __init__(self, num=None, den=None, mrg=None, dov=None):
        super().__init__(num or [], den or [], mrg or set(), set())
        # term-level do sets: list aligned with self.num and self.den
        self._num_dovs = [set() for _ in self.num]
        self._den_dovs = [set() for _ in self.den]
        # if initial dov passed, assign to all existing terms
        if dov:
            for dset in (self._num_dovs + self._den_dovs):
                dset.update(dov)

    def addvars(self, num=None, den=None, mrg=None, dov=None):
        # 标准默认值处理（避免可变默认参数）
        if num is None: num = []
        if den is None: den = []
        if mrg is None: mrg = set()
        if dov is None: dov = set()

        # 记录新增项的数量
        n_num_before = len(self.num)
        n_den_before = len(self.den)

        # 调父类，维护 self.num / self.den / self.mrg
        super().addvars(num=num, den=den, mrg=mrg)

        # 计算新增项数
        n_num_added = len(self.num) - n_num_before
        n_den_added = len(self.den) - n_den_before

        # 给“新增的项”附上本次 do-set（逐项一份拷贝）
        self._num_dovs.extend([set(dov) for _ in range(n_num_added)])
        self._den_dovs.extend([set(dov) for _ in range(n_den_added)])

        # 保障长度一致（调试时很有用）
        assert len(self.num) == len(self._num_dovs)
        assert len(self.den) == len(self._den_dovs)

    def combine(self, exprs):
        # combine multiple DOExpr instances
        for ex in exprs:
            if not isinstance(ex, DOExpr):
                raise ValueError("Can only combine DOExpr instances")
            # avoid mutation of ex
            ex_copy = cp.deepcopy(ex)
            # merge numerator terms and their do-sets
            self.num.extend(ex_copy.num)
            self._num_dovs.extend(ex_copy._num_dovs)
            # merge denominator terms and their do-sets
            self.den.extend(ex_copy.den)
            self._den_dovs.extend(ex_copy._den_dovs)
            # merge marginals
            self.mrg |= ex_copy.mrg
        return self

    def cancel(self):
        # 只删匹配且 do‐sets 相同的 term
        new_num, new_num_dov = [], []
        new_den, new_den_dov = [], []
        changed = False

        # 先把所有分母项拷贝到 new_den/new_den_dov
        for term, dov in zip(self.den, self._den_dovs):
            new_den.append(term)
            new_den_dov.append(dov)

        # 遍历分子项，尝试在 new_den/new_den_dov 中匹配并删除
        for term, dov in zip(self.num, self._num_dovs):
            matched = False
            for i, (dterm, ddov) in enumerate(zip(new_den, new_den_dov)):
                if term == dterm and dov == ddov:
                    # 找到一模一样的分母项，就删掉
                    new_den.pop(i)
                    new_den_dov.pop(i)
                    matched = True
                    changed = True
                    break
            if not matched:
                # 没匹配到，则保留这个分子项
                new_num.append(term)
                new_num_dov.append(dov)

        if changed:
            # 把删掉匹配项后的结果写回 self
            self.num = new_num
            self._num_dovs = new_num_dov
            self.den = new_den
            self._den_dovs = new_den_dov

        return changed

    def marginal(self):
        can_marg = set()
        for node in list(self.mrg):
            # 分母出现则不能边缘化
            if any(node in term for term in self.den):
                continue

            # 在分子中出现的位置，且“该位置的 do-set 不含此 node”
            occ_idx = [i for i, term in enumerate(self.num)
                    if (node in term) and (node not in self._num_dovs[i])]

            # 仅当恰好出现一次时安全边缘化
            if len(occ_idx) == 1:
                can_marg.add(node)

        if not can_marg:
            return False

        # 从分子项中移除这些可边缘化变量
        new_num, new_num_dov = [], []
        for term, dov in zip(self.num, self._num_dovs):
            stripped = set(term) - can_marg
            new_num.append(stripped)
            new_num_dov.append(dov)

        self.num = new_num
        self._num_dovs = new_num_dov
        self.mrg -= can_marg
        return True

    def simplify(self):
        changed = True
        any_change = False
        while changed:
            c1 = self.cancel()
            c2 = self.marginal()
            changed = c1 or c2
            any_change |= changed
        return any_change

    def tostr(self):
        expr_str = ''
        if self.mrg:
            expr_str += util.mrgstr(self.mrg)
        if len(self.num) > 1 and self.mrg:
            expr_str += '['
        # numerator
        for term, dov in zip(self.num, self._num_dovs):
            expr_str += util.probstr(term, do_nodes=dov)
        # denominator
        if self.den:
            expr_str += '/'
            if len(self.den) > 1:
                expr_str += '{'
            for term, dov in zip(self.den, self._den_dovs):
                expr_str += util.probstr(term, do_nodes=dov)
            if len(self.den) > 1:
                expr_str += '}'
        if len(self.num) > 1 and self.mrg:
            expr_str += ']'
        return expr_str

    def tocondstr(self):
        expr_str = ''
        if self.mrg:
            expr_str += util.mrgstr(self.mrg)
        if len(self.num) > 1 and self.mrg:
            expr_str += '['
        # sort terms for consistent output
        num_ord = list(zip(self.num, self._num_dovs))
        num_ord.sort(key=lambda x: len(x[0]), reverse=True)
        den_ord = list(zip(self.den, self._den_dovs))
        den_ord.sort(key=lambda x: len(x[0]), reverse=True)
        # numerator conditional conversion
        while num_ord:
            term, dov = num_ord.pop(0)
            assigned = False
            for j, (dterm, ddov) in enumerate(den_ord):
                if dterm.issubset(term) and dov == ddov:
                    den_ord.pop(j)
                    expr_str += util.probcndstr(term - dterm, cnd_nodes=dterm, do_nodes=dov)
                    assigned = True
                    break
            if not assigned:
                expr_str += util.probstr(term, do_nodes=dov)
        # leftover denominator
        if den_ord:
            expr_str += '/'
            for term, dov in den_ord:
                expr_str += util.probstr(term, do_nodes=dov)
        if len(self.num) > 1 and self.mrg:
            expr_str += ']'
        return expr_str

# %%
