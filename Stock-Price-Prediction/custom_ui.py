import streamlit as st
from pathlib import Path

def local_css(file_name: str):
    """
    Loads a local CSS file and injects its styles into the Streamlit app.
    
    Args:
        file_name (str): The path to your CSS file.
    """
    css_path = Path(file_name)
    if css_path.is_file():
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.error(f"CSS file not found: {css_path.resolve()}")

def custom_header(title: str, subtitle: str = ""):
    """
    Displays a custom header with a title and an optional subtitle.
    
    Args:
        title (str): The header title.
        subtitle (str): Optional subtitle.
    """
    st.markdown(f"""
        <div class="custom-header" style="
            background-color: var(--primary-color, #1e90ff);
            padding: 2rem;
            text-align: center;
            border-radius: 8px;
            margin-bottom: 1rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
        ">
            <h1 style="color: var(--text-color, #ffffff);">{title}</h1>
            <p style="color: var(--text-color, #ffffff);">{subtitle}</p>
        </div>
    """, unsafe_allow_html=True)

def custom_footer(text: str):
    """
    Displays a custom footer.
    
    Args:
        text (str): Footer text.
    """
    st.markdown(f"""
        <div class="custom-footer" style="
            background-color: var(--card-background, #1a1a1a);
            padding: 1rem;
            text-align: center;
            font-size: 0.9rem;
            border-top: 1px solid var(--border-color, #333333);
            margin-top: 2rem;
        ">
            <p style="color: var(--text-color, #e0e0e0);">{text}</p>
        </div>
    """, unsafe_allow_html=True)
