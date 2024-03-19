import streamlit as st
from tempfile import NamedTemporaryFile
from langchain.document_loaders import PyPDFLoader
from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


def main():
    st.header("MindZen")
    st.title("Insurance Policy Extractor📄")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        pages = loader.load()

        st.write(f"Number of pages: {len(pages)}")
        prog_bar = st.progress(0, text="Reading PDF and Generating JSON...")



        llm = CTransformers(model="llama-2-13b-chat.ggmlv3.q4_1.bin", model_type="llama",
                            config={'max_new_tokens': 3072,  'context_length': 4096, 'temperature': 0.01})

        template = """Extract these: 
        
        - Insured Name/Policyholder
        - Company/Insurer
        - Company Branch/Company Address
        - Policy Vehicle(Example: Scooter, Car etc.)
        - Policy Number
        - Policy Date/Period of Insurance
        - Policy Expiry 
        - Date/Period of Insurance
        - Total Value in Insured's Declared Value
        - Gross OD/Gross Own Damage
        - Gross TP in Liability
        - Vehicle No/Registration No
        - Receipt Total
        - Receipt Number
        - Receipt Date {pages} 
        
        
        'Label': value 
        
        in JSON format
        """
        prompt_template = PromptTemplate(input_variables=["pages"], template=template)
        chain = LLMChain(llm=llm, prompt=prompt_template)

        prog_bar.progress(50, text="Reading PDF and Generating JSON...")

        result = chain.run(pages=pages[0].page_content)

        prog_bar.progress(100, text="JSON Generated")

        st.sidebar.header("Extracted entities:")

        st.sidebar.write(result)




if __name__ == "__main__":
    main()



