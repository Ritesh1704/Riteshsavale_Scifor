import streamlit as st

def main():
    # Add CSS for background
    set_background()

    st.title("Grocery Recommendation System")

    # link to the file
    file_link = "https://github.com/Ritesh1704/Riteshsavale_Scifor/tree/main"  

    # Button to view the file
    if st.button("Model"):
        view_file(file_link)

def view_file(file_link):
    # Redirect to the file link
    st.write("Redirecting to the file...")
    st.markdown(f'<a href="{file_link}" target="_blank">Click here to view the file</a>', unsafe_allow_html=True)




if __name__ == "__main__":
    main()
