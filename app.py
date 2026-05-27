import numpy as np
import matplotlib.pyplot as plt

from shiny import App, ui, render, reactive


# ============================================================
# Objective Functions
# ============================================================

def objective(x, y, function_name):
    if function_name == "Quadratic Bowl":
        return x**2 + 3*y**2

    elif function_name == "Rosenbrock":
        return (1 - x)**2 + 100*(y - x**2)**2

    elif function_name == "Saddle Point":
        return x**2 - y**2

    else:
        raise ValueError("Unknown function selected.")


def gradient(x, y, function_name):
    if function_name == "Quadratic Bowl":
        grad_x = 2*x
        grad_y = 6*y

    elif function_name == "Rosenbrock":
        grad_x = -2*(1 - x) - 400*x*(y - x**2)
        grad_y = 200*(y - x**2)

    elif function_name == "Saddle Point":
        grad_x = 2*x
        grad_y = -2*y

    else:
        raise ValueError("Unknown function selected.")

    return grad_x, grad_y



def function_info(function_name):
    """
    Returns display text and learning-rate slider settings for each objective.
    The ranges are chosen to show useful convergence/divergence behavior.
    """
    if function_name == "Quadratic Bowl":
        return {
            "formula": "f(x,y) = x² + 3y²",
            "description": "Convex bowl with a single global minimum at (0, 0).",
            "lr_min": 0.001,
            "lr_max": 0.5,
            "lr_value": 0.05,
            "lr_step": 0.001,
        }

    elif function_name == "Rosenbrock":
        return {
            "formula": "f(x,y) = (1 - x)² + 100(y - x²)²",
            "description": "Narrow curved valley with global minimum at (1, 1).",
            "lr_min": 0.00001,
            "lr_max": 0.005,
            "lr_value": 0.001,
            "lr_step": 0.00001,
        }

    elif function_name == "Saddle Point":
        return {
            "formula": "f(x,y) = x² - y²",
            "description": "Saddle-shaped surface with no true minimum.",
            "lr_min": 0.001,
            "lr_max": 0.25,
            "lr_value": 0.05,
            "lr_step": 0.001,
        }

    else:
        raise ValueError("Unknown function selected.")


# ============================================================
# Gradient Descent
# ============================================================

def gradient_descent(function_name, x0, y0, learning_rate, max_iter):

    x = x0
    y = y0

    xs = []
    ys = []
    losses = []
    grad_norms = []

    status = "Converged"

    for _ in range(max_iter):

        try:

            loss = objective(x,y,function_name)

            if not np.isfinite(loss):
                status = "Diverged"
                break

            grad_x, grad_y = gradient(
                x,
                y,
                function_name
            )

            grad_norm = np.sqrt(
                grad_x**2 + grad_y**2
            )

            xs.append(x)
            ys.append(y)
            losses.append(loss)
            grad_norms.append(grad_norm)

            x = x - learning_rate*grad_x
            y = y - learning_rate*grad_y

            if abs(x) > 1e6 or abs(y) > 1e6:
                status = "Diverged"
                break

        except OverflowError:
            status = "Overflow"
            break

    return (
        np.array(xs),
        np.array(ys),
        np.array(losses),
        np.array(grad_norms),
        status
    )


# ============================================================
# Shiny UI
# ============================================================

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h3("Controls"),

        ui.input_select(
            "function_name",
            "Objective Function",
            choices=["Quadratic Bowl", "Rosenbrock", "Saddle Point"],
            selected="Quadratic Bowl"
        ),

        ui.output_ui("function_display"),

        ui.input_slider(
            "learning_rate",
            "Learning Rate",
            min=0.001,
            max=0.5,
            value=0.05,
            step=0.001
        ),

        ui.input_slider(
            "max_iter",
            "Iterations",
            min=5,
            max=300,
            value=50,
            step=5
        ),

        ui.input_numeric(
            "x0",
            "Starting x",
            value=2.0,
            step=0.1
        ),

        ui.input_numeric(
            "y0",
            "Starting y",
            value=2.0,
            step=0.1
        ),

        ui.hr(),

        ui.HTML("""
<h4>Method</h4>

<p>
Gradient descent moves in the direction of <b>steepest decrease</b>.
</p>

<p><b>Update rule:</b></p>

<div style="
    background-color:#f5f5f5;
    padding:10px;
    border-radius:6px;
    font-family:monospace;
    font-size:15px;
">
x<sub>k+1</sub> = x<sub>k</sub> − α ∇f(x<sub>k</sub>)
</div>

<p><b>where:</b></p>

<ul>
    <li><b>α</b> = learning rate</li>
    <li><b>∇f(x<sub>k</sub>)</b> = gradient</li>
    <li>smaller objective values correspond to better solutions</li>
</ul>

<p><b>Interpretation:</b></p>

<p>
At each iteration, the algorithm computes the local slope and takes a step downhill toward a minimum.
</p>
""")
    ),


    ui.h1("Optimization Playground"),
    ui.h4("An Interactive App for Gradient Descent"),

    ui.markdown(
        """
        This app visualizes **gradient descent**, a standard numerical optimization method.
        The contour plot shows the objective function landscape, while the red path shows how
        the algorithm moves from the starting point toward a minimum.
        """
    ),

    ui.layout_columns(
        ui.card(
            ui.card_header("Contour Plot with Optimization Path"),
            ui.output_plot("contour_plot")
        ),
        ui.card(
            ui.card_header("Loss vs Iteration"),
            ui.output_plot("loss_plot")
        ),
        col_widths=[7, 5]
    ),

    ui.layout_columns(
        ui.card(
            ui.card_header("Gradient Norm vs Iteration"),
            ui.output_plot("gradient_norm_plot")
        ),
        ui.card(
            ui.card_header("Final Results"),
            ui.output_text_verbatim("summary")
        ),
        col_widths=[7, 5]
    ),

    ui.card(
        ui.card_header("Interpretation"),
        ui.output_ui("interpretation")
    )
)


# ============================================================
# Shiny Server
# ============================================================

def server(input, output, session):

    @reactive.effect
    def update_learning_rate_slider():
        """
        Automatically changes the learning-rate slider range
        when the user selects a different objective function.
        """
        info = function_info(input.function_name())

        ui.update_slider(
            "learning_rate",
            min=info["lr_min"],
            max=info["lr_max"],
            value=info["lr_value"],
            step=info["lr_step"]
        )

    @output
    @render.ui
    def function_display():
        info = function_info(input.function_name())

        return ui.HTML(f"""
        <div style="
            background-color:#f5f5f5;
            padding:10px;
            border-radius:6px;
            margin-top:8px;
            margin-bottom:12px;
            border-left:4px solid #999;
        ">
            <div style="font-family:monospace; font-size:14px; margin-bottom:6px;">
                {info["formula"]}
            </div>
            <div style="font-size:13px;">
                {info["description"]}
            </div>
        </div>
        """)

    @reactive.calc
    def gd_results():
        return gradient_descent(
            function_name=input.function_name(),
            x0=input.x0(),
            y0=input.y0(),
            learning_rate=input.learning_rate(),
            max_iter=input.max_iter()
        )

    @output
    @render.plot
    def contour_plot():
        function_name = input.function_name()
        xs, ys, losses, grad_norms, status = gd_results()

        if function_name == "Rosenbrock":
            xgrid = np.linspace(-2, 2, 300)
            ygrid = np.linspace(-1, 3, 300)
        else:
            xgrid = np.linspace(-4, 4, 300)
            ygrid = np.linspace(-4, 4, 300)

        X, Y = np.meshgrid(xgrid, ygrid)
        Z = objective(X, Y, function_name)

        fig, ax = plt.subplots(figsize=(7, 6))

        if function_name == "Rosenbrock":
            levels = np.logspace(-1, 3, 25)
            ax.contour(X, Y, Z, levels=levels)
        else:
            ax.contour(X, Y, Z, levels=25)

        ax.plot(xs, ys, marker="o", linewidth=2, markersize=4)
        ax.scatter(xs[0], ys[0], s=80, label="Start")
        ax.scatter(xs[-1], ys[-1], s=80, label="End")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Gradient Descent on {function_name}")
        ax.legend()

        return fig

    @output
    @render.plot
    def loss_plot():
        xs, ys, losses, grad_norms, status = gd_results()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(losses, marker="o", markersize=3)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective Value")
        ax.set_title("Loss vs Iteration")

        return fig

    @output
    @render.plot
    def gradient_norm_plot():
        xs, ys, losses, grad_norms, status = gd_results()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(grad_norms, marker="o", markersize=3)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Gradient Norm")
        ax.set_title("Gradient Norm vs Iteration")

        return fig

    @output
    @render.text
    def summary():
        xs, ys, losses, grad_norms, status = gd_results()

        return f"""
Optimization Status: {status}
Function: {input.function_name()}

Learning rate: {input.learning_rate()}
Iterations completed: {len(losses)}

Starting point: ({xs[0]:.4f}, {ys[0]:.4f})
Final point: ({xs[-1]:.4f}, {ys[-1]:.4f})

Initial loss: {losses[0]:.6f}
Final loss: {losses[-1]:.6f}

Final gradient norm: {grad_norms[-1]:.6f}
"""

    @output
    @render.ui
    def interpretation():

        xs, ys, losses, grad_norms, status = gd_results()

        if status == "Overflow":

            message = """
    ### Numerical Error

    The objective became too large for numerical computation.

    This commonly occurs for the **Rosenbrock function** when the learning rate is too large.

    Try:

    - decreasing the learning rate
    - moving the starting point closer to the minimum
    - reducing the number of iterations
    """

        elif status == "Diverged":

            message = """
    ### Divergence Detected

    Gradient descent became unstable and moved far away from the minimum.

    Possible causes:

    - learning rate too large
    - steep objective geometry
    - difficult starting point

    Try lowering the learning rate.
    """

        else:

            message = """
    ### Successful Optimization

    The objective decreased successfully.

    Experiment with:

    - different learning rates
    - different starting points
    - different objective functions

    to observe how convergence behavior changes.
    """

        return ui.markdown(message)


app = App(app_ui, server)