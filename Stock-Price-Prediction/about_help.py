import streamlit as st

def about_help():
    st.title("About, Help & Documentation")
    
    st.subheader("App Information")
    st.write("Version: 1.0.0")
    st.write("Changelog: Initial release of the AI Stock Forecasting Suite.")

    st.subheader("Documentation & Roadmap")
    st.write("Detailed documentation and a roadmap for future enhancements will be provided here.")

    st.subheader("Support")
    support_email = st.text_input("Support Email", value="support@example.com")
    support_message = st.text_area("Your Message", "Enter your query or feedback here...")
    if st.button("Submit"):
        st.success("Thank you for your feedback. We will get back to you soon.")

    st.markdown("---")
    st.caption("AI assistant temporarily disabled due to deployment environment limitations.")

if __name__ == "__main__":
    about_help()
