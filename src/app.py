import streamlit as st
from stqdm import stqdm
from setup_and_parse import FolderPaths
from compare import compare_models
from setup_and_parse import initialize_setup
from data_processing import load_prepared_data
from generate_questions import generate_questions
from summarize_documents import summarize_documents
import os
from train_model import train_model
from deploy import deploy_models
import json

def main():
    args, summary_choices = initialize_setup()
    folders = FolderPaths(args)
    
    # Clear folders before starting the app
    folders.clear_folders()
    st.set_page_config(
        page_title="Automated Learning Amplifier",
        layout="wide"
    )
    st.title("Automated Learning Amplifier")
    css = '''
    <style>
        section.main > div {max-width:75rem}
    </style>
    '''
    st.markdown(css, unsafe_allow_html=True)

    # Upload Documents
    uploaded_files = st.file_uploader("Upload Documents", type=["pdf", "txt", "docx", "pptx", "html"], accept_multiple_files=True)
    if uploaded_files:
        # Save uploaded files to a directory
        save_path = folders.documents_folder
        for uploaded_file in uploaded_files:
            file_path = os.path.join(save_path, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.toast(f"Saved {len(uploaded_files)} files.")

    # Checkboxes and Sliders
    col1, col2 = st.columns(2)
    with col1:
        args.images = st.checkbox("Caption Images in Documents", value=False)
        add_summary_checked = st.checkbox("Add Summary")  # Store checkbox state in a variable
        args.verify = st.checkbox("Verify")
        args.compare = st.checkbox("Compare", value=True)
        args.deploy = st.checkbox("Deploy with mlx-ui", value=False)
        # Add question amount slider
        args.question_amount = st.slider("Number of Questions", min_value=1, max_value=10, value=5, step=1)

    with col2:
        # Add focus radio buttons side by side
        focus_options = [None, 'Processes', 'Knowledge', 'Formulas']
        selected_focus = st.radio("Focus Area", focus_options, index=0, key="focus_area")
        if selected_focus:
            args.focus = selected_focus.lower()
        else:
            args.focus = None
        synthetic_data = st.radio("Generate synthetic data with:", ("Local", "OpenRouter", "Claude"), key="synthetic_data")
        args.local = synthetic_data == "Local"
        args.openrouter = synthetic_data == "OpenRouter"
        args.claude = synthetic_data == "Claude"

    # Conditional display of summary_batch_size slider
    if add_summary_checked:
        with col1:
            args.summary_batch_size = st.slider("Summary Batch Size", min_value=2, max_value=10, value=5, step=1)

        # Conditional display of add_summary choices
        with col1:
            args.add_summary = st.radio("Choose a summary topic:", summary_choices, key="summary_topic")
    else:
        args.add_summary = None

    # Add word limit slider (independent adjustment)
    min_word_limit = 100
    max_word_limit = 1000
    col3, col4 = st.columns(2)
    with col3:
        # Calculate suggested word limit based on the number of questions
        suggested_word_limit = args.question_amount * 100
        args.word_limit = st.slider("Word Limit", min_value=min_word_limit, max_value=max_word_limit, value=min(suggested_word_limit, max_word_limit), step=100)
        st.write(f"Suggested word limit: {suggested_word_limit} words")

    with col4:
        args.overlap = st.slider("Overlap", min_value=0.0, max_value=0.2, value=0.1, step=0.01)
        st.write(f"Overlap both top and bottom: {args.overlap*50}%")
        
    col5, col6, col7 = st.columns(3)
    with col5:
        ft_type = st.radio("LoRA Type", ("QLoRA", "LoRA"), index=0, key="ft_type")
        args.ft_type = ft_type.lower()
    with col6:
        model_format = st.radio("Run inference with:", ("MLX", "GGUF"), index=0, key="model_format")
        args.gguf = model_format == "GGUF"
    with col7:
        inference_type = st.radio("Run inference in:", ("Q4", "FP16"), index=0, key="inference_type")
        args.q4 = inference_type == "Q4"
        args.fp16 = inference_type == "FP16"
        # Switch LoRA type based on inference type
        if args.fp16:
            args.ft_type = "lora"
        elif args.q4:
            args.ft_type = "qlora"

    col8, col9 = st.columns(2)
    with col8:
        lora_layer_options = [8, 16, 32, 64]
        args.lora_layers = st.select_slider(f"Number of {args.ft_type} layers", options=lora_layer_options, value=16)
    with col9:
        args.val_set_size = st.slider("Validation Set Size", min_value=0.01, max_value=0.2, value=0.1, step=0.01)

    col10, col11 = st.columns(2)
    with col10:
        args.epochs = st.number_input("Number of Epochs", min_value=1, max_value=10, value=2, step=1)
    with col11:
        initial_value = 3e-6
        args.learning_rate = st.number_input("Learning Rate", min_value=1e-7, max_value=1e-2, value=initial_value, step=1e-6, format="%.1e")

    folders = FolderPaths(args)  # Initialize folders
    if uploaded_files:
        if st.button(label="Launch!", type="primary"):
            # Setup logging, create folders, and parse arguments
            with st.spinner("Processing documents..."):
                all_data, file_contents, chunk_count = load_prepared_data(args, folders)
                if not all_data:
                    st.error("No data processed. Please check your inputs and settings.")
                else:
                    if chunk_count > 1:
                        st.toast(f"{chunk_count} chunks processed successfully.")
                    else:
                        st.toast("Document processed successfully.")

            if args.add_summary:
                with st.spinner("Summarizing data..."):
                    summary = summarize_documents(all_data, args, folders)
                    st.write("Summary:", summary)

            with st.spinner("Generating questions..."):
                generate_questions(args, folders, file_contents)

            with st.spinner("Training model..."):
                train_model(args)
            
            if args.compare:
                val_file_path = os.path.join(folders.finetune_data_folder, "valid.jsonl")
                with open(val_file_path, 'r') as file:
                    total_lines = sum(1 for _ in file)
                    file.seek(0)  # Reset file pointer to the beginning
                    for i, line in stqdm(enumerate(file), total=total_lines, desc="Comparing evaluation samples..."):
                        sample = json.loads(line)
                        comparison_sample, initial_model_response, finetuned_model_response = compare_models(args, sample)
                        # Make sure comparison_sample is json
                        st.subheader(f"Question {i+1}: {comparison_sample['question']}")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Initial Model Response:")
                            st.write(initial_model_response)
                        with col2:
                            st.write("Finetuned Model Response:")
                            st.write(finetuned_model_response)
            if args.deploy:
                with st.spinner("Deploying initital and finetuned models for chat..."):
                    deploy_models(args)
    else:
        st.warning("Please upload documents first.")

if __name__ == "__main__":
    main()