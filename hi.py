import streamlit as st
import numpy as np
from sympy import symbols, diff, lambdify
import plotly.graph_objects as go

def newton_forward_interpolation(x, y, x_value):
    n = len(x)
    forward_diff = np.zeros((n, n))
    forward_diff[:, 0] = y

    for i in range(1, n):
        for j in range(n - i):
            forward_diff[j][i] = (forward_diff[j + 1][i - 1] - forward_diff[j][i - 1]) / (x[j + i] - x[j])

    result = forward_diff[0][0]
    x_product = 1

    for i in range(1, n):
        x_product *= (x_value - x[i - 1])
        result += x_product * forward_diff[0][i]

    formula = "f(x) = " + str(forward_diff[0][0])
    for i in range(1, n):
        formula += f" + ({forward_diff[0][i]:.6f}) * (x - {x[i - 1]:.6f})"
    
    return result, formula

def newton_backward_interpolation(x, y, x_value):
    n = len(x)
    backward_diff = np.zeros((n, n))
    backward_diff[:, 0] = y

    for i in range(1, n):
        for j in range(n - i):
            backward_diff[j][i] = (backward_diff[j][i - 1] - backward_diff[j - 1][i - 1]) / (x[j] - x[j - i])

    result = backward_diff[0][0]
    x_product = 1

    for i in range(1, n):
        x_product *= (x_value - x[i])
        result += x_product * backward_diff[0][i]

    formula = "f(x) = " + str(backward_diff[0][0])
    for i in range(1, n):
        formula += f" + ({backward_diff[0][i]:.6f}) * (x - {x[n-1-i]:.6f})"
    
    return result, formula

def newton_raphson(equation, x0, tolerance=1e-6, max_iterations=100):
    x = symbols('x')
    f = lambdify(x, equation)
    f_prime = lambdify(x, diff(equation, x))

    for _ in range(max_iterations):
        x1 = x0 - f(x0) / f_prime(x0)
        if abs(x1 - x0) < tolerance:
            return x1
        x0 = x1

    raise ValueError("Newton-Raphson method did not converge.")

# Streamlit web app starts here
def main():
    st.title("Numerical Methods with Streamlit")
    st.markdown(
        """<style>
        .reportview-container {
            background: linear-gradient(45deg, #e6e6e6, #f9f9f9);
        }
        .sidebar .sidebar-content {
            background: linear-gradient(45deg, #c5c5c5, #f9f9f9);
        }
        </style>""",
        unsafe_allow_html=True
    )

    st.sidebar.header("Select Method")
    method = st.sidebar.selectbox("", ["Newton Forward", "Newton Backward", "Newton-Raphson"])

    formula = ""

    if method == "Newton Forward":
        st.header("Newton Forward Interpolation")
        st.sidebar.title("Newton Forward Interpolation")
        
        n_points = st.sidebar.number_input("Enter the number of data points:", value=3, min_value=3, max_value=10, step=1)

        x_values = []
        y_values = []

        st.sidebar.write("Enter data points:")
        for i in range(n_points):
            x_values.append(st.sidebar.number_input(f"x{i+1}", value=i, key=f"x{i+1}"))
            y_values.append(st.sidebar.number_input(f"y{i+1}", value=i, key=f"y{i+1}"))

        x_value_to_find = st.sidebar.number_input("Enter the x value to find the corresponding y value:", value=0.5)

        if st.sidebar.button("Interpolate"):
            result, formula = newton_forward_interpolation(x_values, y_values, x_value_to_find)
            st.success(f"The interpolated value at x={x_value_to_find} is y={result:.6f}")
            st.markdown("### Newton Forward Interpolation Formula")
            st.write(formula)

        st.sidebar.markdown("### Newton Forward Interpolation Formula")
        st.sidebar.write(formula)

        # Plotting
        x_interp = np.linspace(min(x_values), max(x_values), 100)
        y_interp = [newton_forward_interpolation(x_values, y_values, x)[0] for x in x_interp]
       
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='markers', name='Data Points'))
        fig.add_trace(go.Scatter(x=x_interp, y=y_interp, mode='lines', name='Interpolation'))
        fig.update_layout(title='Newton Forward Interpolation', xaxis_title='x', yaxis_title='y')
        st.plotly_chart(fig)

        
    elif method == "Newton Backward":
        st.header("Newton Backward Interpolation")
        st.sidebar.title("Newton Backward Interpolation")

        n_points = st.sidebar.number_input("Enter the number of data points:", value=3, min_value=3, max_value=10, step=1)

        x_values = []
        y_values = []

        st.sidebar.write("Enter data points:")
        for i in range(n_points):
            x_values.append(st.sidebar.number_input(f"x{i+1}", value=i, key=f"x{i+1}"))
            y_values.append(st.sidebar.number_input(f"y{i+1}", value=i, key=f"y{i+1}"))

        x_value_to_find = st.sidebar.number_input("Enter the x value to find the corresponding y value:", value=0.5)

        if st.sidebar.button("Interpolate"):
            result, formula = newton_backward_interpolation(x_values, y_values, x_value_to_find)
            st.success(f"The interpolated value at x={x_value_to_find} is y={result:.6f}")
            st.sidebar.markdown("### Newton Backward Interpolation Formula")
            st.sidebar.write(formula)
            

        st.sidebar.markdown("### Newton Backward Interpolation Formula")
        st.sidebar.write(formula)
        
        st.markdown("### Newton Backward Interpolation Formula")
        st.write(formula)


        # Plotting
        x_interp = np.linspace(min(x_values), max(x_values), 100)
        y_interp = [newton_backward_interpolation(x_values, y_values, x)[0] for x in x_interp]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='markers', name='Data Points'))
        fig.add_trace(go.Scatter(x=x_interp, y=y_interp, mode='lines', name='Interpolation'))
        fig.update_layout(title='Newton Backward Interpolation', xaxis_title='x', yaxis_title='y')
        st.plotly_chart(fig)

        
    elif method == "Newton-Raphson":
        st.header("Newton-Raphson Method")
        st.sidebar.title("Newton-Raphson Method")

        equation_input = st.sidebar.text_input("Enter the equation in terms of 'x' (e.g., x**2 - 4*x + 4):")

        x0 = st.sidebar.number_input("Enter the initial guess (x0):", value=1.0)

        if st.sidebar.button("Find Root"):
            try:
                root = newton_raphson(equation_input, x0)
                st.success(f"The root of the equation is x={root:.6f}")

                
                #3##3
                st.markdown("### Newton-Raphson Method Formula")
                st.write("Newton-Raphson method finds the root of the equation f(x) = 0 using the iterative formula:")
                st.write("x1 = x0 - f(x0) / f'(x0)")
                # Plotting
                x_interp = np.linspace(x0 - 5, x0 + 5, 100)
                y_interp = [eval(equation_input) for x in x_interp]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_interp, y=y_interp, mode='lines', name='f(x)'))
                fig.add_shape(
                    go.layout.Shape(
                        type='line', x0=root, x1=root, y0=min(y_interp), y1=max(y_interp),
                        line=dict(color='green', width=2, dash='dash'),
                    )
                )
                fig.update_layout(title='Newton-Raphson Method', xaxis_title='x', yaxis_title='y')
                st.plotly_chart(fig)

              

            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()

