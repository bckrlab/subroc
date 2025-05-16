import pysubgroup as ps


def convert_EqualitySelector(selector: ps.EqualitySelector) -> str:
    attribute_value_tex_string = str(selector.attribute_value)
    if isinstance(selector.attribute_value, str):
        attribute_value_tex_string = fr"\text{{{str(selector.attribute_value)}}}"

    return fr"\text{{{selector.attribute_name}}}={attribute_value_tex_string}"


def convert_NegatedSelector(selector: ps.NegatedSelector) -> str:
    return fr"\neg{convert_SelectorBase(selector._selector)}"


def convert_IntervalSelector(selector: ps.IntervalSelector) -> str:
    left_bracket = "[" if selector.lower_bound != float("-inf") else "("
    lower_bound_str = str(selector.lower_bound) if selector.lower_bound != float("-inf") else r"\infty"
    upper_bound_str = str(selector.upper_bound) if selector.upper_bound != float("inf") else r"\infty"
    return fr"\text{{{selector.attribute_name}}}\in{left_bracket}{lower_bound_str},{upper_bound_str})"


def convert_SelectorBase(selector: ps.SelectorBase) -> str:
    if isinstance(selector, ps.EqualitySelector):
        return convert_EqualitySelector(selector)
    elif isinstance(selector, ps.NegatedSelector):
        return convert_NegatedSelector(selector)
    elif isinstance(selector, ps.IntervalSelector):
        return convert_IntervalSelector(selector)

    return "Unknown Selector Type"


def convert_Conjunction(conjunction: ps.Conjunction) -> str:
    pattern_latex_string = ""

    if len(conjunction.selectors) == 0:
        return r"\emptyset"

    for sel_i, selector in enumerate(conjunction.selectors):
        if sel_i != 0:
            pattern_latex_string += r"\:\wedge$\\$"

        if isinstance(selector, ps.SelectorBase):
            pattern_latex_string += convert_SelectorBase(selector)

    return pattern_latex_string

