import streamlit as st
from transformers import pipeline, TFAutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"  # GPT-2 has a TF version

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForCausalLM.from_pretrained(model_name)  # Load TF model

# Create the text-generation pipeline using TensorFlow
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, framework="tf")

def get_gpt_response(query: str) -> str:
    responses = generator(query, max_length=100, num_return_sequences=1)
    return responses[0]['generated_text']

def ai_boot_gpt():
    st.subheader("AI Boot GPT")
    user_query = st.text_input("Ask the AI Boot GPT something:")
    if st.button("Get Response"):
        if user_query.strip() == "":
            st.error("Please enter a query.")
        else:
            response = get_gpt_response(user_query)
            st.write(response)

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
    ai_boot_gpt()

if __name__ == "__main__":
    about_help()
