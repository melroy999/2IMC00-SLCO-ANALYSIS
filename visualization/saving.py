import os
from pathlib import Path

from matplotlib import pyplot as plt

# Keep a constant that denotes whether figures should be shown after being saved.
show_plots = False


def save_plot_as_pgf(target_model: str, file_name: str) -> str:
    """Save the current plot at the given location as a pgf file."""
    target_path = os.path.join("figures", target_model)
    target_location = os.path.join("figures", target_model, f"{file_name}_plot.pgf")
    with plt.rc_context({
        'text.usetex': True,
        'pgf.rcfonts': False,
    }):
        Path(target_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(target_location)
    return target_location


def save_plot_as_png(target_model: str, file_name: str) -> str:
    """Save the current plot at the given location as a png file."""
    target_path = os.path.join("figures", target_model)
    target_location = os.path.join("figures", target_model, f"{file_name}_plot.png")
    Path(target_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(target_location)
    if show_plots:
        plt.show()
    return target_location


def save_table_as_tex(
        target_model: str,
        table_contents: str,
        caption: str,
        label: str,
        file_name: str,
        float_modifier: str = "htbp",
        include_resize_box: bool = True
):
    """Save the given latex table at the given location as a tex file."""
    if include_resize_box:
        lines = [
            f"\\begin{{table}}[{float_modifier}]",
            "\\centering",
            """\\resizebox{%
\\ifdim\\width>\\columnwidth
    \\columnwidth
\\else
    \\width
\\fi
}{!}{ %""",
            table_contents.strip(),
            "}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{table}"
        ]
    else:
        lines = [
            f"\\begin{{table}}[{float_modifier}]",
            "\\centering",
            table_contents.strip(),
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{table}"
        ]

    file_contents = "\n".join(lines)

    target_path = os.path.join("tables", target_model)
    target_location = os.path.join("tables", target_model, f"{file_name}_table.tex")
    Path(target_path).mkdir(parents=True, exist_ok=True)

    with open(target_location, "w") as file:
        file.write(file_contents.strip())


def save_pgf_figure_as_tex(
        target_model: str, target_figure: str, caption: str, label: str, file_name: str, float_modifier: str = "htbp"
):
    """Include the code required to show the given pgf figure as a tex file."""
    target_figure = target_figure.replace("\\", "/")

    file_contents = f"""
\\begin{{figure}}[{float_modifier}]
\\centering
\\begin{{minipage}}{{1\\textwidth}}
  \\centering
  \\makebox[\\textwidth][c]{{ %
        \\resizebox{{1.19\\textwidth}}{{!}}{{ %
            \\input{{{target_figure}}}
        }}%
    }}%
\\end{{minipage}}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{figure}}
"""

    target_path = os.path.join("figures", target_model)
    target_location = os.path.join("figures", target_model, f"{file_name}_figure.tex")
    Path(target_path).mkdir(parents=True, exist_ok=True)

    with open(target_location, "w") as file:
        file.write(file_contents.strip())


def save_png_figure_as_tex(
        target_model: str, target_figure: str, caption: str, label: str, file_name: str, float_modifier: str = "htbp"
):
    """Include the code required to show the given png figure as a tex file."""
    target_figure = target_figure.replace("\\", "/")

    file_contents = f"""
\\begin{{figure}}[{float_modifier}]
\\centering
\\begin{{minipage}}{{1\\textwidth}}
  \\centering
  \\makebox[\\textwidth][c]{{ %
        \\resizebox{{1.2\\textwidth}}{{!}}{{ %
            \\includegraphics{{{target_figure}}}
        }}%
    }}%
\\end{{minipage}}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{figure}}
"""

    target_path = os.path.join("figures", target_model)
    target_location = os.path.join("figures", target_model, f"{file_name}_figure.tex")
    Path(target_path).mkdir(parents=True, exist_ok=True)

    with open(target_location, "w") as file:
        file.write(file_contents.strip())
