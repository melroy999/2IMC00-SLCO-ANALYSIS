import pandas


def float_notation(float_number):
    """Pretty print a float number using latex scientific notation."""
    e_float = f"{float_number:.2e}"
    if "e" not in e_float:
        return f"${e_float}$"
    base, exponent = e_float.split("e")
    if int(exponent) == 0:
        return f"${base}$"
    else:
        return f"${base} \\cdot 10^{{{int(exponent)}}}$"


def render_tabular(data: pandas.DataFrame) -> str:
    """Render the data in the given table as a latex object."""
    with pandas.option_context("max_colwidth", 1000):
        latex_code = data.to_latex(
            float_format=float_notation, escape=False, multirow=True
        )
        return latex_code.replace("_", "\\_")
