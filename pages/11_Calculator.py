import streamlit as st
import utils
from functools import partial

st.set_page_config(page_title="Calculator - Visio AI", page_icon="🧮", layout="centered")
utils.init_session_state()
utils.load_css()
utils.sidebar_nav()

st.markdown("<h2 style='text-align: center;'>🧮 Calculator</h2>", unsafe_allow_html=True)

# State init
if 'calc_display' not in st.session_state:
    st.session_state.calc_display = '0'
    st.session_state.first_op = None
    st.session_state.operator = None
    st.session_state.reset_next = False

def update(val):
    if st.session_state.reset_next:
        st.session_state.calc_display = val
        st.session_state.reset_next = False
    else:
        current = st.session_state.calc_display
        st.session_state.calc_display = val if current == '0' else current + val

def operate(op):
    st.session_state.first_op = float(st.session_state.calc_display)
    st.session_state.operator = op
    st.session_state.reset_next = True

def calculate():
    if st.session_state.operator and st.session_state.first_op is not None:
        second = float(st.session_state.calc_display)
        res = 0
        if st.session_state.operator == '+': res = st.session_state.first_op + second
        elif st.session_state.operator == '-': res = st.session_state.first_op - second
        elif st.session_state.operator == '*': res = st.session_state.first_op * second
        elif st.session_state.operator == '/': res = st.session_state.first_op / second if second != 0 else "Error"
        
        st.session_state.calc_display = str(res)
        st.session_state.reset_next = True

def clear():
    st.session_state.calc_display = '0'
    st.session_state.first_op = None
    st.session_state.operator = None

st.text_input("Display", value=st.session_state.calc_display, key="disp", disabled=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button('7', use_container_width=True): update('7')
    if st.button('4', use_container_width=True): update('4')
    if st.button('1', use_container_width=True): update('1')
    if st.button('C', type="primary", use_container_width=True): clear()
with c2:
    if st.button('8', use_container_width=True): update('8')
    if st.button('5', use_container_width=True): update('5')
    if st.button('2', use_container_width=True): update('2')
    if st.button('0', use_container_width=True): update('0')
with c3:
    if st.button('9', use_container_width=True): update('9')
    if st.button('6', use_container_width=True): update('6')
    if st.button('3', use_container_width=True): update('3')
    if st.button('=', type="primary", use_container_width=True): calculate()
with c4:
    if st.button('+', use_container_width=True): operate('+')
    if st.button('-', use_container_width=True): operate('-')
    if st.button('×', use_container_width=True): operate('*')
    if st.button('÷', use_container_width=True): operate('/')
