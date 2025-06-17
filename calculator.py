import streamlit as st
from functools import partial

def render_calculator():
    """
    Renders a fully functional calculator UI and handles its logic.
    """
    st.markdown("<h2 style='text-align: center; color: #4A90E2;'>🧮 Calculator</h2>", unsafe_allow_html=True)

    # --- State Initialization ---
    # We use session state to keep track of the calculator's current state.
    if 'calc_display' not in st.session_state:
        st.session_state.calc_display = '0'
        st.session_state.first_operand = None
        st.session_state.operator = None
        st.session_state.waiting_for_second_operand = False

    # --- Callback Functions ---
    # These functions modify the state in response to button clicks.

    def handle_digit(digit):
        """Appends a digit to the display."""
        if st.session_state.waiting_for_second_operand:
            st.session_state.calc_display = digit
            st.session_state.waiting_for_second_operand = False
        else:
            st.session_state.calc_display = st.session_state.calc_display + digit if st.session_state.calc_display != '0' else digit

    def handle_decimal():
        """Adds a decimal point if one doesn't already exist."""
        if '.' not in st.session_state.calc_display:
            st.session_state.calc_display += '.'

    def handle_operator(op):
        """Handles an operator click (+, -, *, /)."""
        current_value = float(st.session_state.calc_display)
        
        # This block handles chained operations like 5 * 2 + (result is 10, then we wait for next number)
        if st.session_state.first_operand is not None and st.session_state.operator is not None and not st.session_state.waiting_for_second_operand:
            handle_equals()
            st.session_state.first_operand = float(st.session_state.calc_display)
        else:
            st.session_state.first_operand = current_value

        st.session_state.operator = op
        st.session_state.waiting_for_second_operand = True
        
    def handle_equals():
        """Performs the calculation."""
        if st.session_state.operator is None or st.session_state.first_operand is None:
            return

        second_operand = float(st.session_state.calc_display)
        first_operand = st.session_state.first_operand
        operator = st.session_state.operator
        
        if operator == '+':
            result = first_operand + second_operand
        elif operator == '-':
            result = first_operand - second_operand
        elif operator == '*':
            result = first_operand * second_operand
        elif operator == '/':
            if second_operand == 0:
                result = 'Error'
            else:
                result = first_operand / second_operand

        st.session_state.calc_display = str(result)
        st.session_state.first_operand = result # So you can chain operations with the result
        st.session_state.operator = None
        st.session_state.waiting_for_second_operand = True


    def handle_clear():
        """Resets the calculator to its initial state."""
        st.session_state.calc_display = '0'
        st.session_state.first_operand = None
        st.session_state.operator = None
        st.session_state.waiting_for_second_operand = False

    # --- UI Layout ---
    # Display screen
    st.text_input("Result", st.session_state.calc_display, key="display", disabled=True)

    # Calculator buttons layout
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.button('7', on_click=partial(handle_digit, '7'), use_container_width=True)
        st.button('4', on_click=partial(handle_digit, '4'), use_container_width=True)
        st.button('1', on_click=partial(handle_digit, '1'), use_container_width=True)
        st.button('0', on_click=partial(handle_digit, '0'), use_container_width=True)

    with col2:
        st.button('8', on_click=partial(handle_digit, '8'), use_container_width=True)
        st.button('5', on_click=partial(handle_digit, '5'), use_container_width=True)
        st.button('2', on_click=partial(handle_digit, '2'), use_container_width=True)
        st.button('.', on_click=handle_decimal, use_container_width=True)

    with col3:
        st.button('9', on_click=partial(handle_digit, '9'), use_container_width=True)
        st.button('6', on_click=partial(handle_digit, '6'), use_container_width=True)
        st.button('3', on_click=partial(handle_digit, '3'), use_container_width=True)
        st.button('=', on_click=handle_equals, use_container_width=True)

    with col4:
        st.button('/', on_click=partial(handle_operator, '/'), use_container_width=True)
        st.button('Mul', on_click=partial(handle_operator, '*'), use_container_width=True)
        st.button('Sub', on_click=partial(handle_operator, '-'), use_container_width=True)
        st.button('Add', on_click=partial(handle_operator, '+'), use_container_width=True)
        
    st.button('C', on_click=handle_clear, use_container_width=True)